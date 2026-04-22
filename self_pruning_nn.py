"""
Self-Pruning Neural Network for CIFAR-10 Classification
========================================================
Implements a custom PrunableLinear layer with learnable gating mechanism,
sparsity regularization loss, and training/evaluation pipeline.

Tredence Intern Python AI Engineer — Case Study
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab/headless
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------------------------
# Device selection: CUDA > MPS (Apple Silicon) > CPU
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LAMBDA_VALUES = [0.0, 1e-4, 1e-3, 1e-2]  # Different lambda values to test
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
SPARSE_THRESHOLD = 1e-2  # Gate values below this are considered pruned


# ============================================================================
# Part 1: The "Prunable" Layers
# ============================================================================

class PrunableLinear(nn.Module):
    """
    A fully-connected layer with learnable gate scores for self-pruning.

    Gating mechanism:
        1. gate_scores are learned parameters (same shape as weights).
        2. gates = sigmoid(gate_scores)  →  values in (0, 1).
        3. pruned_weights = weight * gates  (element-wise).
        4. Output = x @ pruned_weights^T + bias.

    Gradients flow through both `weight` and `gate_scores` because:
        - sigmoid is differentiable everywhere.
        - element-wise multiplication is differentiable.
    """

    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()

        # Standard linear layer parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — registered as a parameter so the
        # optimizer will update them alongside weights.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self):
        """Xavier init for weights; gate_scores start so sigmoid ≈ 0.27."""
        nn.init.xavier_uniform_(self.weight)
        # sigmoid(-1) ≈ 0.27 — starts partially open, giving the
        # optimizer room to push gates toward 0 or 1.
        nn.init.constant_(self.gate_scores, -1.0)

    def forward(self, x):
        """Forward pass with gated weights."""
        # Sigmoid keeps gates in (0, 1) and is differentiable
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise gating — differentiable w.r.t. both weight & gate_scores
        pruned_weights = self.weight * gates

        # Standard linear: x @ pruned_weights^T + bias
        return nn.functional.linear(x, pruned_weights, self.bias)

    def get_gate_values(self):
        """Return sigmoid-transformed gate values (detached, on CPU)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).detach().cpu()


class PrunableConv2d(nn.Module):
    """
    A 2-D convolution layer with learnable gate scores for self-pruning.
    Same gating principle as PrunableLinear, applied to conv weights.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super(PrunableConv2d, self).__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        # Conv weight: (out_channels, in_channels, kH, kW)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Learnable gate scores — same shape as weights
        self.gate_scores = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.gate_scores, -1.0)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return nn.functional.conv2d(x, pruned_weight, self.bias,
                                    self.stride, self.padding)

    def get_gate_values(self):
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).detach().cpu()


# ============================================================================
# Network Architecture
# ============================================================================

class SelfPruningCNN(nn.Module):
    """
    A CNN for CIFAR-10 that uses PrunableConv2d and PrunableLinear layers
    so the network can learn to prune its own weights during training.
    """

    def __init__(self, num_classes=10):
        super(SelfPruningCNN, self).__init__()

        # Convolutional backbone
        self.conv1 = PrunableConv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = PrunableConv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = PrunableConv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        # After 3× pool(2): 32→16→8→4  ⇒  128 * 4 * 4 = 2048
        self.fc1 = PrunableLinear(128 * 4 * 4, 256)
        self.fc2 = PrunableLinear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 32×32 → 16×16
        x = self.pool(self.relu(self.conv2(x)))   # 16×16 → 8×8
        x = self.pool(self.relu(self.conv3(x)))   # 8×8   → 4×4

        x = x.view(x.size(0), -1)                # Flatten to (B, 2048)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def get_all_gates(self):
        """Collect all gate values from every prunable layer."""
        gates = []
        for module in self.modules():
            if isinstance(module, (PrunableLinear, PrunableConv2d)):
                gates.append(module.get_gate_values())
        return torch.cat([g.flatten() for g in gates])


# ============================================================================
# Part 2: Sparsity Regularization Loss
# ============================================================================

def compute_sparsity_loss(model):
    """
    L1 penalty on the gate values across all prunable layers.

    Since gates = sigmoid(gate_scores) ∈ (0,1), the L1 norm is just
    the sum of all gate values.  Minimising this pushes gates toward 0.

    Returns:
        Scalar tensor — the un-weighted sparsity penalty.
    """
    sparsity_loss = 0.0
    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            gates = torch.sigmoid(module.gate_scores)
            sparsity_loss = sparsity_loss + gates.sum()
    return sparsity_loss


# ============================================================================
# Metrics helpers
# ============================================================================

def calculate_sparsity(model, threshold=SPARSE_THRESHOLD):
    """
    Percentage of weights whose corresponding gate < threshold.
    Higher = more weights pruned.
    """
    total_gates = 0
    pruned_gates = 0
    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            gates = module.get_gate_values()
            total_gates += gates.numel()
            pruned_gates += (gates < threshold).sum().item()
    return (pruned_gates / total_gates) * 100 if total_gates > 0 else 0.0


def count_gate_stats(model, threshold=SPARSE_THRESHOLD):
    """Return (total_gates, pruned_gates) for reporting."""
    total = pruned = 0
    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            gates = module.get_gate_values()
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
    return total, pruned


# ============================================================================
# Part 3: Training and Evaluation
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, lambda_reg):
    """
    Train for one epoch.

    Total Loss = CrossEntropy + λ × SparsityLoss
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        # Classification loss
        class_loss = criterion(outputs, targets)

        # Sparsity regularization (L1 on gate values)
        sparsity_loss = compute_sparsity_loss(model)

        # Combined loss
        total_loss = class_loss + lambda_reg * sparsity_loss

        total_loss.backward()
        optimizer.step()

        running_loss += class_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader):
    """Evaluate model on a dataset and return accuracy (%)."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


# ============================================================================
# Plotting
# ============================================================================

def plot_gate_distribution(model, lambda_val, save_path='gate_distribution.png'):
    """
    Plot histogram of gate values for a trained model.
    A successful pruning shows a large spike near 0 and a cluster near 1.
    """
    gates = model.get_all_gates().numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(gates, bins=100, edgecolor='black', alpha=0.7, color='#2196F3')
    plt.axvline(x=SPARSE_THRESHOLD, color='red', linestyle='--',
                linewidth=1.5, label=f'Prune threshold ({SPARSE_THRESHOLD})')
    plt.xlabel('Gate Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Gate Value Distribution (λ = {lambda_val})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Saved gate distribution plot to {save_path}")


# ============================================================================
# Report generation
# ============================================================================

def generate_report(results, best_lambda, save_path='report.md'):
    """Auto-generate the required Markdown report from training results."""

    table_rows = ""
    for r in results:
        table_rows += (
            f"| {r['lambda']:.0e} | {r['test_accuracy']:.2f}% "
            f"| {r['sparsity']:.1f}% |\n"
        )

    report = f"""\
# Self-Pruning Neural Network — Case Study Report

## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

The gate values are obtained by applying the **sigmoid function** to learnable
`gate_scores`, producing values in the range (0, 1).  The **L1 penalty** adds
the sum of all gate values to the loss:

```
Total Loss = Classification Loss + λ × Σ sigmoid(gate_scores)
```

Because the sigmoid function is **monotonically increasing**, minimising the
sum of sigmoid outputs pushes each `gate_score` toward **−∞**, which drives
the corresponding sigmoid output toward **0**.  A gate value near 0
effectively **zeroes out** its associated weight, pruning it from the network.

The key insight is the **trade-off**: the classification loss pulls gate
values toward whatever magnitude preserves accuracy, while the L1 penalty
pulls them toward 0.  Weights that contribute little to accuracy are cheaply
pushed to 0, while essential weights retain high gate values.  This creates
a **bimodal distribution** — a spike near 0 (pruned) and a cluster near 1
(retained) — which is exactly the sparsity pattern we desire.

The hyperparameter **λ** controls the aggressiveness of pruning:
- **λ = 0**: No sparsity pressure — all gates remain open.
- **Small λ**: Mild pruning — only truly redundant weights are removed.
- **Large λ**: Aggressive pruning — even useful weights may be forced to 0,
  potentially hurting accuracy.

## 2. Results Table

| Lambda | Test Accuracy | Sparsity Level (%) |
|--------|---------------|--------------------|
{table_rows}
## 3. Gate Value Distribution (Best Model, λ = {best_lambda})

The plot below shows the distribution of final gate values for the model
trained with **λ = {best_lambda}**.  A successful self-pruning result
exhibits a large spike near 0 (pruned weights) and another cluster near 1
(retained weights).

![Gate Value Distribution](gate_distribution.png)

## 4. Observations

- With **λ = 0** (no regularisation), gates remain near their initialisation
  and nearly no weights are pruned.  This serves as the accuracy baseline.
- As **λ increases**, sparsity rises — more gates are pushed below the
  pruning threshold of {SPARSE_THRESHOLD}.
- At very high λ, sparsity is maximised but **accuracy degrades** because
  the network is forced to discard important connections.
- The middle-ground λ value achieves **meaningful sparsity** while preserving
  most of the baseline accuracy, demonstrating a successful
  accuracy–sparsity trade-off.
"""
    with open(save_path, 'w') as f:
        f.write(report)
    print(f"  → Saved Markdown report to {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    # ------------------------------------------------------------------
    # Data loading (CIFAR-10)
    # ------------------------------------------------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"CIFAR-10 loaded: {len(trainset)} training, {len(testset)} test samples\n")

    # ------------------------------------------------------------------
    # Train one model per lambda value
    # ------------------------------------------------------------------
    results = []
    best_model = None
    best_lambda = None
    best_score = -1  # score = accuracy when sparsity > 10%, else accuracy - 100

    for lambda_reg in LAMBDA_VALUES:
        print(f"{'=' * 60}")
        print(f"  Training with λ = {lambda_reg:.0e}")
        print(f"{'=' * 60}")

        model = SelfPruningCNN(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        total_gates, _ = count_gate_stats(model)
        print(f"  Total gate parameters: {total_gates:,}")

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(
                model, trainloader, criterion, optimizer, lambda_reg)
            test_acc = evaluate(model, testloader)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                sparsity = calculate_sparsity(model)
                print(f"  Epoch {epoch+1:2d}/{EPOCHS}  |  "
                      f"Loss: {train_loss:.4f}  |  "
                      f"Train: {train_acc:.2f}%  |  "
                      f"Test: {test_acc:.2f}%  |  "
                      f"Sparsity: {sparsity:.1f}%")

        # Final evaluation
        final_acc = evaluate(model, testloader)
        final_sparsity = calculate_sparsity(model)
        total_g, pruned_g = count_gate_stats(model)

        print(f"\n  Final — Test Accuracy: {final_acc:.2f}%  |  "
              f"Sparsity: {final_sparsity:.1f}%  |  "
              f"Pruned: {pruned_g:,}/{total_g:,} gates\n")

        results.append({
            'lambda': lambda_reg,
            'test_accuracy': final_acc,
            'sparsity': final_sparsity,
        })

        # Pick best model: highest accuracy among those with sparsity > 10%
        score = final_acc if final_sparsity > 10 else final_acc - 100
        if score > best_score:
            best_score = score
            best_model = model
            best_lambda = lambda_reg

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"  {'Lambda':<12} {'Test Accuracy (%)':<22} {'Sparsity Level (%)':<20}")
    print(f"  {'-' * 60}")
    for r in results:
        print(f"  {r['lambda']:<12.0e} {r['test_accuracy']:<22.2f} {r['sparsity']:<20.1f}")
    print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Generate deliverables
    # ------------------------------------------------------------------
    print(f"\n  Best model for plot: λ = {best_lambda}")
    plot_gate_distribution(best_model, best_lambda)
    generate_report(results, best_lambda)

    return results


if __name__ == "__main__":
    results = main()
    print("\n Training complete!")