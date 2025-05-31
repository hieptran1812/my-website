---
title: "Transformer Architecture: Building GPT from Scratch"
excerpt: "A comprehensive implementation guide to building a GPT-style transformer model from scratch using PyTorch, with detailed explanations of attention mechanisms."
date: "2024-03-25"
readTime: "20 min read"
tags: ["Transformers", "GPT", "PyTorch", "NLP"]
level: "Advanced"
type: "Tutorial"
featured: true
category: "machine-learning"
---

# Transformer Architecture: Building GPT from Scratch

The Transformer architecture, introduced in the groundbreaking paper "Attention Is All You Need," has revolutionized natural language processing. In this comprehensive guide, we'll build a GPT-style transformer model from scratch using PyTorch.

## Understanding the Architecture

The transformer consists of several key components:

- Multi-head self-attention mechanisms
- Position encodings
- Feed-forward networks
- Layer normalization
- Residual connections

## Self-Attention Mechanism

The core innovation of transformers is the self-attention mechanism, which allows the model to focus on different parts of the input sequence when processing each token.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Linear transformations
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )

        return self.out_linear(attention_output)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)
```

## Position Encoding

Since transformers don't have built-in notion of sequence order, we need to add positional information:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super().__init__()

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
```

## Complete GPT Model

Now let's build the complete GPT model:

```python
class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_length=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Create causal mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)

        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.head(x)
```

## Training the Model

Here's how to train the GPT model:

```python
def train_gpt(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)

        # Reshape for loss computation
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)

        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

    return total_loss / len(train_loader)

# Model configuration
config = {
    'vocab_size': 50257,  # GPT-2 tokenizer vocab size
    'd_model': 768,
    'n_heads': 12,
    'n_layers': 12,
    'd_ff': 3072,
    'max_length': 1024
}

# Initialize model
model = GPT(**config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    avg_loss = train_gpt(model, train_loader, optimizer, criterion, device)
    print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
```

## Key Insights

1. **Attention Mechanisms**: The self-attention allows the model to dynamically focus on relevant parts of the input
2. **Scalability**: Transformer architecture scales well with increased model size and data
3. **Parallelization**: Unlike RNNs, transformers can be trained in parallel across sequence positions
4. **Transfer Learning**: Pre-trained transformers can be fine-tuned for various downstream tasks

## Conclusion

Building a GPT model from scratch helps understand the fundamental concepts behind modern language models. The transformer architecture's elegance lies in its simplicity and effectiveness, making it the foundation for most current NLP breakthroughs.

The complete implementation provides a solid foundation for understanding how large language models work under the hood and can be extended for various applications.
