import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 
from dataclasses import dataclass 
from typing import Optional 

@dataclass 
class ModelArgs:
    dim = 4096 
    n_layers = 32 
    n_heads = 32 #number of heads for the query
    n_kv_heads = None #number of heads for the key and value
    vocab_size = -1 # will be set when we load the tokenizer
    multiple_of = 256 
    ffn_dim_multiplier = None 
    norm_eps = 1e-5 

    #Needed for kv cache 
    max_batch_size = 32 
    max_seq_len = 2048 

    device = None 

def apply_rotary_embeddings(x, freqs_complex, device):
    #batch, seq_len, h, head_dim / 2
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    #1, seq_len , 1, head_dim / 2
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    #batch, seq_len, h, head_dim / 2
    x_rotated = x_complex * freqs_complex
    #batch, seq_len, h , head_dim / 2, 2
    x_out = torch.view_as_real(x_rotated)
    #batch, seq_len, h, head_dim
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)



def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta=10000.0):
    # the dimension of the embedding must be even
    assert head_dim % 2 == 0, "head dim must be even"
    # build the theta parameters 
    #theta_shape : head_dim / 2 
    #10000 ^ (-2(i-1)/dim) for i = [1,2,...,dim/2]
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # head_dim / 2
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    #construct the position (the "m" parameter)
    #shape: seq_len
    m = torch.arange(seq_len, device=device)
    #multiply each theta by each position using outer product
    #--> seq_len, head_dim / 2
    freqs = torch.outer(m , theta).float()
    #computer complex numbers in the polar form c = R * exp(i*m*theta), where R = 1
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def repeat_kv(x, n_rep):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .rehape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        #rsqrt : 1/ sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        #dim * (b, seq_len, dim) = b, seq_len, dim
        return self.weight * self.norm(x.float()).type_as(x)
    
class SelfAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads 
        self.n_heads_q = args.n_heads 
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self. wq = nn.Linear(args.dim , args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias = False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x, start_pos, freqs_complex):
        #batch, 1 , dim
        batch_size, seq_len, _ = x.shape
        #batch, 1, dim -> batch, 1, H_Q * head_dim
        xq = self.wq(x)
        #batch, 1, dim -> batch, 1, H_kv * head_dim, may make the dimnesion lower
        xk = self.wk(x)
        xv = self.wv(x)

        #B, 1, H_Q, head_dim
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        #B, 1, H_kv, head_dim
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex, x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, x.device)

        #replace the entry in the cache for this token 
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        #B, seq_len_kv, H_kv, head_dim
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]

        #repeat the heads of the k and v to reach the number of heads of the query
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # B, 1, H_Q, head_dim -> B, H_Q, 1, head_dim
        xq = xq.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        #B, H_Q, 1, seq_len_kv
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).continuous().view(batch_size, seq_len, -1)
        return self.wo(output)



class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        hidden_dim = 4 * args.dim
        hidden_dim = int(2*hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * args.dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V 
        x = self.w2(x)
        return x



class EncoderBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads 
        self.dim = args.dim 
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        #norm BEFORE the attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        #norm BEFORE the ff
        self.ffn_norm = RMSNorm(args.dim, eps = args.norm_eps)
    def forward(self, x, start_pos, freqs_complex):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args 
        self.vocab_size = args.vocab_size 
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=None)

        self.freqs_complex = precompute_theta_pos_frequencies(args.dim // args.n_heads, args.max_seq_len * 2, device=args.device)

    def forward(self, token, start_pos):
        #batch, seq_len
        batch_size, seq_len = token.shape 
        assert seq_len == 1, "only one token at a time can be processed"

        #batch, seq_len, dim 
        h = self.tok_embeddings(token)

        # retrieve the pairs (m, theta) corresponding to the positions (start_pos, start_pos + seq_len)
        freqs_complex = self.freqs_complex[start_pos, start_pos+seq_len]
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        return self.output(h).float()