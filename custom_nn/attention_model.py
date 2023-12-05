from dataset import Dataset

import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionHead(nn.Module):

    def __init__(
        self, 
        head_size: int, 
        n_embd: int, 
        block_size: int, 
        dropout: float = 0.2
    ) -> None:

        super().__init__()

        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)
        q = self.key(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out
    
class MultiHeadAttention(nn.Module):

    def __init__(
        self, 
        num_heads: int,  
        head_size: int, 
        n_embd: int, 
        block_size: int, 
        dropout: float = 0.2
    ) -> None:

        super().__init__()

        
        self.heads = nn.ModuleList([AttentionHead(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd: int, dropout: float = 0.2) -> None:
        super().__init__()

        self.n_embd = n_embd

        self.net = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            nn.ReLU(),
            nn.Linear(4 * self.n_embd, self.n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformBlock(nn.Module):

    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:

        super().__init__()

        head_size = n_embd // n_head

        self.sa_heads = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x

class LanguageModel(nn.Module):
    
    def __init__(
        self, 
        vocab_size: int, 
        n_embed: int, 
        block_size: int, 
        eval_iters: int, 
        batch_size: int,
        n_layers: int,
        n_head: int
    ) -> None:

        super().__init__()

        self.block_size = block_size
        self.eval_iters = eval_iters
        self.batch_size = batch_size

        self.token_embedding_table = nn.Embedding(
            vocab_size, 
            n_embed
        )

        self.position_embedding_table = nn.Embedding(
            self.block_size, 
            n_embed
        )

        self.blocks = nn.Sequential(*[TransformBlock(n_embed, n_head=n_head, block_size=self.block_size) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(n_embed)

        self.lm_head = nn.Linear(
            n_embed, 
            vocab_size
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:

        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    def __mean_loss(
        self, 
        data: torch.Tensor, 
    ) -> torch.Tensor:

        losses = torch.zeros(self.eval_iters)

        for k in range(self.eval_iters):
            X, Y = Dataset.get_batch(
                data, 
                block_size=self.block_size, 
                batch_size=self.batch_size
                )
            _, loss = self(X, Y)
            losses[k] = loss.item()
        
        return losses.mean()
    
    @torch.no_grad()
    def estimate_loss(
        self, 
        train: torch.Tensor, 
        eval: torch.Tensor, 
    ) -> dict[str, float]:

        self.eval()

        out = {'train': self.__mean_loss(train),
                'eval': self.__mean_loss(eval)}
        
        self.train()

        return out
        

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size: int, n_embed: int) -> None:
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:

        logits = self.token_embedding_table(idx)

        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))

        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:

        for _ in range(max_new_tokens):

            logits, _ = self(idx)

            logits = logits[-1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=0)

        return idx
    
    def __mean_loss(
        self, 
        data: torch.Tensor, 
        eval_iters: int, 
        block_size: int, 
        batch_size: int
    ) -> torch.Tensor:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = Dataset.get_batch(
                data, 
                block_size=block_size, 
                batch_size=batch_size
                )
            _, loss = self(X, Y)
            losses[k] = loss.item()
        
        return losses.mean()
    
    @torch.no_grad()
    def estimate_loss(
        self, 
        train: torch.Tensor, 
        eval: torch.Tensor, 
        eval_interval: int = 10,
        block_size: int = 8,
        batch_size: int = 4
    ) -> dict[str, float]:

        self.eval()

        out = {'train': self.__mean_loss(train, eval_interval, block_size, batch_size),
                'eval': self.__mean_loss(eval, eval_interval, block_size, batch_size)}
        
        self.train()

        return out
        