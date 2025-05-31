---
title: Fine-tuning Large Language Models with LoRA
excerpt: >-
  Learn how to efficiently fine-tune large language models using Low-Rank
  Adaptation (LoRA) for specific tasks while maintaining performance.
date: '2024-03-20'
readTime: 6 min read
tags:
  - LLM
  - Fine-tuning
  - LoRA
  - Efficiency
level: Intermediate
type: Tutorial
category: machine-learning
---

# Fine-tuning Large Language Models with LoRA

Low-Rank Adaptation (LoRA) has emerged as a revolutionary technique for efficiently fine-tuning large language models without modifying the original model weights. This guide explores how to implement and use LoRA for custom tasks.

## What is LoRA?

LoRA works by decomposing weight updates into low-rank matrices, significantly reducing the number of trainable parameters while maintaining model performance. Instead of updating all model weights, LoRA adds trainable low-rank matrices alongside frozen pre-trained weights.

## Mathematical Foundation

The core idea is to represent weight updates as a product of two low-rank matrices:

```
ΔW = BA
```

Where:

- B ∈ R^(d×r)
- A ∈ R^(r×k)
- r << min(d,k) (rank is much smaller than original dimensions)

## Implementation with PyTorch

Let's implement LoRA from scratch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply LoRA: x @ A.T @ B.T * (alpha / rank)
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result * (self.alpha / self.rank)

class LoRALinear(nn.Module):
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.requires_grad_(False)  # Freeze original weights

        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original_layer(x)
        lora_output = self.lora(x)
        return original_output + lora_output
```

## Applying LoRA to Transformers

Here's how to apply LoRA to a transformer model:

```python
def apply_lora_to_model(model, rank=4, alpha=1.0, target_modules=None):
    """
    Apply LoRA to specified modules in a transformer model
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA version
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                parent_module = model
                for part in parent_name.split("."):
                    if part:
                        parent_module = getattr(parent_module, part)

                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent_module, child_name, lora_layer)

    return model

# Example usage with a pre-trained model
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply LoRA
lora_model = apply_lora_to_model(model, rank=8, alpha=16)

# Count trainable parameters
total_params = sum(p.numel() for p in lora_model.parameters())
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
```

## Training with LoRA

Here's a complete training setup:

```python
class LoRATrainer:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def prepare_data(self, texts, labels, max_length=512):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels)
        )

        return dataset

    def train(self, train_dataset, val_dataset, epochs=3, batch_size=16, lr=1e-4):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size
        )

        # Only optimize LoRA parameters
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr
        )

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0

            for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token

                # Add classification head if needed
                if not hasattr(self.model, 'classifier'):
                    self.model.classifier = nn.Linear(
                        logits.size(-1), len(set(labels.cpu().numpy()))
                    ).to(self.device)

                predictions = self.model.classifier(logits)
                loss = criterion(predictions, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')

            # Validation
            val_accuracy = self.evaluate(val_loader)
            print(f'Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}')

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in data_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.last_hidden_state[:, 0, :]
                predictions = self.model.classifier(logits)

                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
```

## Advantages of LoRA

1. **Parameter Efficiency**: Reduces trainable parameters by 99%+
2. **Memory Efficiency**: Lower GPU memory requirements
3. **Fast Training**: Faster convergence due to fewer parameters
4. **Modularity**: Easy to swap different LoRA adapters
5. **No Inference Overhead**: Can be merged with original weights

## Best Practices

1. **Rank Selection**: Start with rank 4-8, increase if needed
2. **Alpha Scaling**: Use alpha = 2 × rank as starting point
3. **Target Modules**: Focus on attention layers for best results
4. **Learning Rate**: Use higher learning rates (1e-4 to 5e-4)
5. **Regularization**: Apply dropout to prevent overfitting

## Saving and Loading LoRA Adapters

```python
def save_lora_adapters(model, path):
    """Save only LoRA parameters"""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            lora_state_dict[name] = param.cpu()

    torch.save(lora_state_dict, path)

def load_lora_adapters(model, path):
    """Load LoRA parameters"""
    lora_state_dict = torch.load(path)

    for name, param in model.named_parameters():
        if name in lora_state_dict:
            param.data = lora_state_dict[name].to(param.device)

    return model

# Usage
save_lora_adapters(lora_model, "lora_adapters.pt")
loaded_model = load_lora_adapters(lora_model, "lora_adapters.pt")
```

## Conclusion

LoRA provides an elegant solution for efficient fine-tuning of large language models. By decomposing weight updates into low-rank matrices, it achieves comparable performance to full fine-tuning while using a fraction of the computational resources.

This technique is particularly valuable for:

- Resource-constrained environments
- Multi-task scenarios requiring multiple adapters
- Rapid prototyping and experimentation
- Domain adaptation tasks

The implementation shown here provides a foundation for applying LoRA to various transformer architectures and can be extended for specific use cases.
