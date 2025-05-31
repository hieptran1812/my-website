---
title: "Optimization Algorithms for Machine Learning: From SGD to Adam and Beyond"
publishDate: "2024-03-24"
readTime: "16 min read"
category: "machine-learning"
subcategory: "Optimization"
author: "Hiep Tran"
featured: true
tags:
  - Optimization
  - SGD
  - Adam
  - Gradient Descent
  - Learning Rate
  - Convergence
date: "2024-03-24"
image: "/blog-placeholder.jpg"
excerpt: >-
  Deep dive into optimization algorithms that power machine learning, from
  classical gradient descent to modern adaptive methods like Adam, RMSprop,
  and cutting-edge optimizers.
---

# Optimization Algorithms for Machine Learning: From SGD to Adam and Beyond

![Optimization Landscape Visualization](/blog-placeholder.jpg)

Optimization lies at the heart of machine learning, determining how effectively models learn from data. This comprehensive guide explores the evolution of optimization algorithms, from classical gradient descent to modern adaptive methods that power today's deep learning systems.

## The Optimization Problem in Machine Learning

Machine learning optimization involves finding parameters θ that minimize a loss function L(θ):

```
θ* = argmin L(θ)
```

Where L(θ) typically represents the average loss over training data:

```
L(θ) = (1/n) Σ l(f(x_i; θ), y_i)
```

### Challenges in ML Optimization

1. **Non-convexity**: Most ML loss functions have multiple local minima
2. **High dimensionality**: Modern models have millions/billions of parameters
3. **Stochastic nature**: Mini-batch training introduces noise
4. **Computational constraints**: Limited time and memory resources

## Classical Gradient Descent Methods

### Batch Gradient Descent

The foundational optimization algorithm:

```python
import numpy as np

def batch_gradient_descent(X, y, theta, learning_rate, iterations):
    """
    Batch Gradient Descent implementation
    """
    m = len(y)
    cost_history = []

    for i in range(iterations):
        # Forward pass
        predictions = X.dot(theta)

        # Compute cost
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

        # Compute gradients
        gradients = (1/m) * X.T.dot(predictions - y)

        # Update parameters
        theta = theta - learning_rate * gradients

    return theta, cost_history
```

**Advantages:**

- Guaranteed convergence for convex functions
- Stable gradient estimates
- Theoretical guarantees

**Disadvantages:**

- Computationally expensive for large datasets
- Slow convergence
- Memory intensive

### Stochastic Gradient Descent (SGD)

SGD processes one sample at a time:

```python
def stochastic_gradient_descent(X, y, theta, learning_rate, iterations):
    """
    Stochastic Gradient Descent implementation
    """
    m = len(y)
    cost_history = []

    for i in range(iterations):
        for j in range(m):
            # Random sample selection
            random_index = np.random.randint(0, m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            # Forward pass
            prediction = xi.dot(theta)

            # Compute gradient for single sample
            gradient = xi.T.dot(prediction - yi)

            # Update parameters
            theta = theta - learning_rate * gradient

        # Compute cost on full dataset
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

    return theta, cost_history
```

### Mini-Batch Gradient Descent

The practical middle ground:

```python
def mini_batch_gradient_descent(X, y, theta, learning_rate, iterations, batch_size=32):
    """
    Mini-batch Gradient Descent implementation
    """
    m = len(y)
    cost_history = []

    for i in range(iterations):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for j in range(0, m, batch_size):
            # Get mini-batch
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            # Forward pass
            predictions = X_batch.dot(theta)

            # Compute gradients
            gradients = (1/batch_size) * X_batch.T.dot(predictions - y_batch)

            # Update parameters
            theta = theta - learning_rate * gradients

        # Compute cost
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

    return theta, cost_history
```

## Momentum-Based Methods

### Classical Momentum

Momentum accelerates convergence by accumulating gradients:

```python
def sgd_with_momentum(X, y, theta, learning_rate, momentum, iterations, batch_size=32):
    """
    SGD with Momentum implementation
    """
    m = len(y)
    velocity = np.zeros_like(theta)
    cost_history = []

    for i in range(iterations):
        # Shuffle and create mini-batches
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            # Compute gradients
            predictions = X_batch.dot(theta)
            gradients = (1/batch_size) * X_batch.T.dot(predictions - y_batch)

            # Update velocity
            velocity = momentum * velocity + learning_rate * gradients

            # Update parameters
            theta = theta - velocity

        # Compute cost
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

    return theta, cost_history
```

**Key Benefits:**

- Faster convergence in ravines
- Reduces oscillations
- Better handling of noisy gradients

### Nesterov Accelerated Gradient (NAG)

NAG looks ahead before computing gradients:

```python
def nesterov_accelerated_gradient(X, y, theta, learning_rate, momentum, iterations, batch_size=32):
    """
    Nesterov Accelerated Gradient implementation
    """
    m = len(y)
    velocity = np.zeros_like(theta)
    cost_history = []

    for i in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            # Look ahead
            theta_lookahead = theta - momentum * velocity

            # Compute gradients at lookahead position
            predictions = X_batch.dot(theta_lookahead)
            gradients = (1/batch_size) * X_batch.T.dot(predictions - y_batch)

            # Update velocity and parameters
            velocity = momentum * velocity + learning_rate * gradients
            theta = theta - velocity

        # Compute cost
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

    return theta, cost_history
```

## Adaptive Learning Rate Methods

### AdaGrad: Adaptive Gradient Algorithm

AdaGrad adapts learning rates based on historical gradients:

```python
def adagrad(X, y, theta, learning_rate, iterations, batch_size=32, epsilon=1e-8):
    """
    AdaGrad implementation
    """
    m = len(y)
    G = np.zeros_like(theta)  # Accumulated squared gradients
    cost_history = []

    for i in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            # Compute gradients
            predictions = X_batch.dot(theta)
            gradients = (1/batch_size) * X_batch.T.dot(predictions - y_batch)

            # Accumulate squared gradients
            G += gradients ** 2

            # Adaptive learning rate update
            adjusted_gradients = gradients / (np.sqrt(G) + epsilon)
            theta = theta - learning_rate * adjusted_gradients

        # Compute cost
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

    return theta, cost_history
```

**Advantages:**

- Automatically adapts learning rates
- Works well for sparse features
- No manual learning rate tuning

**Disadvantages:**

- Learning rate monotonically decreases
- Can stop learning too early

### RMSprop: Root Mean Square Propagation

RMSprop fixes AdaGrad's diminishing learning rate problem:

```python
def rmsprop(X, y, theta, learning_rate, decay_rate, iterations, batch_size=32, epsilon=1e-8):
    """
    RMSprop implementation
    """
    m = len(y)
    E_g2 = np.zeros_like(theta)  # Exponential average of squared gradients
    cost_history = []

    for i in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            # Compute gradients
            predictions = X_batch.dot(theta)
            gradients = (1/batch_size) * X_batch.T.dot(predictions - y_batch)

            # Update exponential average of squared gradients
            E_g2 = decay_rate * E_g2 + (1 - decay_rate) * gradients ** 2

            # Update parameters
            theta = theta - learning_rate * gradients / (np.sqrt(E_g2) + epsilon)

        # Compute cost
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

    return theta, cost_history
```

### Adam: Adaptive Moment Estimation

Adam combines momentum and adaptive learning rates:

```python
def adam_optimizer(X, y, theta, learning_rate=0.001, beta1=0.9, beta2=0.999,
                   iterations=1000, batch_size=32, epsilon=1e-8):
    """
    Adam optimizer implementation
    """
    m = len(y)
    moment1 = np.zeros_like(theta)  # First moment (momentum)
    moment2 = np.zeros_like(theta)  # Second moment (RMSprop)
    cost_history = []

    for i in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            # Compute gradients
            predictions = X_batch.dot(theta)
            gradients = (1/batch_size) * X_batch.T.dot(predictions - y_batch)

            # Update moments
            moment1 = beta1 * moment1 + (1 - beta1) * gradients
            moment2 = beta2 * moment2 + (1 - beta2) * gradients ** 2

            # Bias correction
            moment1_corrected = moment1 / (1 - beta1 ** (i + 1))
            moment2_corrected = moment2 / (1 - beta2 ** (i + 1))

            # Update parameters
            theta = theta - learning_rate * moment1_corrected / (np.sqrt(moment2_corrected) + epsilon)

        # Compute cost
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

    return theta, cost_history
```

**Adam's Key Features:**

- Combines benefits of momentum and RMSprop
- Bias correction for initial time steps
- Generally robust across different problems
- Default choice for many deep learning applications

## Advanced Optimization Algorithms

### AdamW: Adam with Weight Decay

AdamW decouples weight decay from gradient-based updates:

```python
def adamw_optimizer(X, y, theta, learning_rate=0.001, beta1=0.9, beta2=0.999,
                    weight_decay=0.01, iterations=1000, batch_size=32, epsilon=1e-8):
    """
    AdamW optimizer implementation
    """
    m = len(y)
    moment1 = np.zeros_like(theta)
    moment2 = np.zeros_like(theta)
    cost_history = []

    for i in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            # Compute gradients
            predictions = X_batch.dot(theta)
            gradients = (1/batch_size) * X_batch.T.dot(predictions - y_batch)

            # Update moments
            moment1 = beta1 * moment1 + (1 - beta1) * gradients
            moment2 = beta2 * moment2 + (1 - beta2) * gradients ** 2

            # Bias correction
            moment1_corrected = moment1 / (1 - beta1 ** (i + 1))
            moment2_corrected = moment2 / (1 - beta2 ** (i + 1))

            # AdamW update with decoupled weight decay
            theta = theta - learning_rate * (
                moment1_corrected / (np.sqrt(moment2_corrected) + epsilon) +
                weight_decay * theta
            )

        # Compute cost
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

    return theta, cost_history
```

### RAdam: Rectified Adam

RAdam addresses the variance problem in early training stages:

```python
def radam_optimizer(X, y, theta, learning_rate=0.001, beta1=0.9, beta2=0.999,
                    iterations=1000, batch_size=32, epsilon=1e-8):
    """
    RAdam optimizer implementation
    """
    m = len(y)
    moment1 = np.zeros_like(theta)
    moment2 = np.zeros_like(theta)
    cost_history = []

    # RAdam specific parameters
    rho_inf = 2.0 / (1 - beta2) - 1

    for i in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for j in range(0, m, batch_size):
            t = i * (m // batch_size) + j // batch_size + 1  # Time step

            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            # Compute gradients
            predictions = X_batch.dot(theta)
            gradients = (1/batch_size) * X_batch.T.dot(predictions - y_batch)

            # Update moments
            moment1 = beta1 * moment1 + (1 - beta1) * gradients
            moment2 = beta2 * moment2 + (1 - beta2) * gradients ** 2

            # Compute bias correction
            moment1_corrected = moment1 / (1 - beta1 ** t)

            # Compute length of SMA
            rho_t = rho_inf - 2 * t * (beta2 ** t) / (1 - beta2 ** t)

            if rho_t > 4:
                # Compute variance rectification term
                l_t = np.sqrt((1 - beta2 ** t) / moment2)
                r_t = np.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) /
                             ((rho_inf - 4) * (rho_inf - 2) * rho_t))

                theta = theta - learning_rate * r_t * moment1_corrected * l_t
            else:
                # Use SGD-style update
                theta = theta - learning_rate * moment1_corrected

        # Compute cost
        predictions = X.dot(theta)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

    return theta, cost_history
```

### Lookahead Optimizer

Lookahead can be combined with any base optimizer:

```python
def lookahead_optimizer(base_optimizer, X, y, theta, k=5, alpha=0.5, **optimizer_kwargs):
    """
    Lookahead wrapper for any base optimizer
    """
    phi = theta.copy()  # Slow weights
    cost_history = []

    for i in range(optimizer_kwargs['iterations']):
        # Run base optimizer for k steps
        theta_updated, _ = base_optimizer(X, y, theta, iterations=k, **optimizer_kwargs)

        # Lookahead update
        phi = phi + alpha * (theta_updated - phi)
        theta = phi.copy()

        # Compute cost
        predictions = X.dot(theta)
        cost = (1/(2*len(y))) * np.sum((predictions - y)**2)
        cost_history.append(cost)

    return theta, cost_history
```

## Learning Rate Scheduling

### Step Decay

```python
def step_decay_schedule(initial_lr, decay_factor=0.1, step_size=10):
    """
    Step decay learning rate schedule
    """
    def schedule(epoch):
        return initial_lr * (decay_factor ** (epoch // step_size))
    return schedule
```

### Exponential Decay

```python
def exponential_decay_schedule(initial_lr, decay_rate=0.95):
    """
    Exponential decay learning rate schedule
    """
    def schedule(epoch):
        return initial_lr * (decay_rate ** epoch)
    return schedule
```

### Cosine Annealing

```python
def cosine_annealing_schedule(initial_lr, T_max, eta_min=0):
    """
    Cosine annealing learning rate schedule
    """
    def schedule(epoch):
        return eta_min + (initial_lr - eta_min) * (
            1 + np.cos(np.pi * epoch / T_max)
        ) / 2
    return schedule
```

### Warm Restarts

```python
def cosine_annealing_warm_restarts(initial_lr, T_0, T_mult=1, eta_min=0):
    """
    Cosine annealing with warm restarts
    """
    def schedule(epoch):
        T_cur = epoch % T_0
        return eta_min + (initial_lr - eta_min) * (
            1 + np.cos(np.pi * T_cur / T_0)
        ) / 2
    return schedule
```

## Practical Implementation with PyTorch

### Using Built-in Optimizers

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Various optimizers
optimizers = {
    'SGD': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'Adam': optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999)),
    'AdamW': optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
    'RMSprop': optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99),
}

# Learning rate schedulers
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizers['Adam'], T_max=100, eta_min=1e-6
)
```

### Custom Training Loop

```python
def train_with_optimizer(model, optimizer, train_loader, criterion, epochs):
    """
    Training loop with specified optimizer
    """
    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')

    return loss_history
```

## Optimization for Different Scenarios

### Computer Vision Models

```python
# ResNet-style optimization
optimizer = optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],
    gamma=0.1
)
```

### Natural Language Processing

```python
# Transformer optimization
optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-4,
    betas=(0.9, 0.98),
    eps=1e-6,
    weight_decay=0.01
)

# Warmup + cosine decay
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-4,
    steps_per_epoch=len(train_loader),
    epochs=epochs
)
```

### Reinforcement Learning

```python
# Policy gradient optimization
optimizer = optim.Adam(
    policy_network.parameters(),
    lr=3e-4,
    eps=1e-5
)

# Value function optimization
value_optimizer = optim.Adam(
    value_network.parameters(),
    lr=1e-3
)
```

## Hyperparameter Tuning for Optimizers

### Learning Rate Finding

```python
def find_learning_rate(model, optimizer, criterion, train_loader,
                      init_lr=1e-8, final_lr=10, beta=0.98):
    """
    Learning rate range test
    """
    num_batches = len(train_loader)
    mult = (final_lr / init_lr) ** (1 / num_batches)
    lr = init_lr

    avg_loss = 0.
    best_loss = 0.
    losses = []
    lrs = []

    for batch_idx, (data, target) in enumerate(train_loader):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Compute smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (batch_idx + 1))

        # Stop if loss explodes
        if batch_idx > 0 and smoothed_loss > 4 * best_loss:
            break

        # Record best loss
        if smoothed_loss < best_loss or batch_idx == 0:
            best_loss = smoothed_loss

        # Store values
        losses.append(smoothed_loss)
        lrs.append(lr)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update learning rate
        lr *= mult

    return lrs, losses
```

### Grid Search for Optimizer Hyperparameters

```python
def optimizer_grid_search(model_class, train_loader, val_loader):
    """
    Grid search for optimizer hyperparameters
    """
    results = []

    # Define search space
    optimizers_config = {
        'Adam': {
            'lr': [1e-4, 3e-4, 1e-3, 3e-3],
            'betas': [(0.9, 0.999), (0.9, 0.99), (0.95, 0.999)],
            'weight_decay': [0, 1e-5, 1e-4, 1e-3]
        },
        'SGD': {
            'lr': [1e-3, 1e-2, 1e-1],
            'momentum': [0.9, 0.95, 0.99],
            'weight_decay': [0, 1e-5, 1e-4, 1e-3]
        }
    }

    for opt_name, param_grid in optimizers_config.items():
        # Generate all combinations
        param_combinations = list(itertools.product(*param_grid.values()))

        for params in param_combinations:
            # Create model and optimizer
            model = model_class()

            if opt_name == 'Adam':
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=params[0],
                    betas=params[1],
                    weight_decay=params[2]
                )
            elif opt_name == 'SGD':
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=params[0],
                    momentum=params[1],
                    weight_decay=params[2]
                )

            # Train and evaluate
            val_acc = train_and_evaluate(model, optimizer, train_loader, val_loader)

            results.append({
                'optimizer': opt_name,
                'params': dict(zip(param_grid.keys(), params)),
                'val_accuracy': val_acc
            })

    return sorted(results, key=lambda x: x['val_accuracy'], reverse=True)
```

## Common Optimization Challenges and Solutions

### Gradient Clipping

```python
# Gradient norm clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Gradient value clipping
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

### Batch Normalization and Optimization

```python
# Proper batch norm initialization affects optimization
def init_batch_norm(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
```

### Handling Exploding/Vanishing Gradients

```python
def check_gradients(model):
    """
    Monitor gradient norms during training
    """
    total_norm = 0
    param_count = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1

    total_norm = total_norm ** (1. / 2)
    return total_norm, param_count
```

## Recent Advances and Future Directions

### Sharpness-Aware Minimization (SAM)

```python
def sam_step(model, optimizer, loss_fn, data, target, rho=0.05):
    """
    Sharpness-Aware Minimization step
    """
    # First forward-backward pass
    loss = loss_fn(model(data), target)
    loss.backward()

    # Compute SAM gradient
    grad_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm() ** 2
    grad_norm = grad_norm ** 0.5

    # Ascent step
    for p in model.parameters():
        if p.grad is not None:
            e_w = rho * p.grad / (grad_norm + 1e-8)
            p.add_(e_w)

    # Second forward-backward pass
    loss = loss_fn(model(data), target)
    optimizer.zero_grad()
    loss.backward()

    # Descent step
    for p in model.parameters():
        if p.grad is not None:
            e_w = rho * p.grad / (grad_norm + 1e-8)
            p.sub_(e_w)

    optimizer.step()
    return loss
```

### Lion Optimizer

Lion is a recent optimizer that uses only the sign of gradients:

```python
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Update parameters
                update = exp_avg.mul(beta1).add(grad, alpha=1-beta1).sign_()
                p.data.add_(update, alpha=-group['lr'])

                # Update exponential moving average
                exp_avg.mul_(beta2).add_(grad, alpha=1-beta2)

        return loss
```

## Best Practices and Recommendations

### Optimizer Selection Guidelines

1. **Default Choice**: Start with Adam or AdamW
2. **Computer Vision**: SGD with momentum often works better
3. **NLP/Transformers**: AdamW with warmup and decay
4. **Fine-tuning**: Lower learning rates, often AdamW
5. **Resource Constrained**: SGD with momentum

### Learning Rate Guidelines

```python
# Common learning rate ranges by optimizer
lr_guidelines = {
    'SGD': 1e-2,
    'Adam': 1e-3,
    'AdamW': 5e-4,
    'RMSprop': 1e-3,
}

# Scale learning rate with batch size
def scale_lr_with_batch_size(base_lr, batch_size, base_batch_size=32):
    return base_lr * (batch_size / base_batch_size)
```

### Debugging Optimization

```python
def optimization_diagnostics(model, train_loader, optimizer, epochs=5):
    """
    Run optimization diagnostics
    """
    diagnostics = {
        'loss_history': [],
        'grad_norms': [],
        'param_norms': [],
        'lr_history': []
    }

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = F.cross_entropy(output, target)

            # Backward pass
            loss.backward()

            # Record diagnostics
            diagnostics['loss_history'].append(loss.item())

            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            diagnostics['grad_norms'].append(grad_norm)

            param_norm = sum(p.norm().item() for p in model.parameters())
            diagnostics['param_norms'].append(param_norm)

            current_lr = optimizer.param_groups[0]['lr']
            diagnostics['lr_history'].append(current_lr)

            optimizer.step()

    return diagnostics
```

## Conclusion

Optimization algorithms are the engines that drive machine learning model training. Understanding the evolution from simple gradient descent to sophisticated adaptive methods like Adam and beyond is crucial for effective model development.

Key takeaways:

1. **No universal optimizer**: Different problems may require different optimization strategies
2. **Learning rate is critical**: Often more important than the optimizer choice
3. **Modern adaptive methods**: Adam/AdamW are good default choices
4. **Problem-specific tuning**: Computer vision vs. NLP may need different approaches
5. **Monitor training dynamics**: Use diagnostics to understand optimization behavior

The field continues to evolve with new algorithms like Lion, SAM, and others addressing specific challenges in modern deep learning. Staying informed about these developments while mastering the fundamentals provides the best foundation for tackling optimization challenges in machine learning.

---

_Explore more optimization topics in our related articles on [gradient descent variants](link-to-gradient-article) and [learning rate scheduling strategies](link-to-scheduling-article)._
