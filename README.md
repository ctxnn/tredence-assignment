# Self-Pruning Neural Network for CIFAR-10

## Introduction

This report presents the results of implementing a self-pruning convolutional neural network (CNN) for image classification on the CIFAR-10 dataset. The network incorporates learnable gate parameters that allow it to identify and eliminate redundant connections during training, rather than relying on post-hoc pruning. The approach is evaluated across four values of the sparsity hyperparameter $\lambda$ to study the trade-off between model accuracy and network sparsity.

---

## Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

Each weight $w_{ij}$ in the network is paired with a learnable gate score $s_{ij}$. During the forward pass, the gate score is transformed via the sigmoid function to produce a gate value:

$$g_{ij} = \sigma(s_{ij}) = \frac{1}{1 + e^{-s_{ij}}}$$

This gate value lies in the interval $(0, 1)$ and is multiplied element-wise with the weight:

$$\tilde{w}_{ij} = w_{ij} \cdot g_{ij}$$

A gate value near zero effectively disables the corresponding connection.

The total loss function augments the standard classification loss with an L1 penalty on all gate values:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(y, \hat{y}) + \lambda \sum_{l} \sum_{i,j} g_{ij}^{(l)}$$

where $\mathcal{L}_{\text{CE}}$ is the cross-entropy loss, $\lambda$ is the regularisation strength, and the double sum runs over all gate values across all prunable layers $l$.

Since the sigmoid function is monotonically increasing, minimising the sum of gate values pushes each gate score $s_{ij}$ toward $-\infty$, which drives $\sigma(s_{ij})$ toward zero. However, for weights that are critical to maintaining classification accuracy, the gradient from $\mathcal{L}_{\text{CE}}$ opposes this pressure and keeps the gate open. This creates a natural selection process: important connections survive while redundant ones are pruned.

The result is a separation in gate values — a large group near zero (pruned connections) and a smaller group retaining non-zero values (surviving connections). The hyperparameter $\lambda$ controls the strength of this effect: larger values impose a stronger per-gate cost, leading to more aggressive pruning at the potential expense of accuracy.

---

## Architecture

The model is a three-layer convolutional neural network. Both the convolutional and fully connected layers use the custom gating mechanism described above.

| Layer | Type | Output Shape | Gated Weights |
|-------|------|-------------|---------------|
| conv1 | PrunableConv2d(3 $\to$ 32, 3×3) | 32 × 16 × 16 | 864 |
| conv2 | PrunableConv2d(32 $\to$ 64, 3×3) | 64 × 8 × 8 | 18,432 |
| conv3 | PrunableConv2d(64 $\to$ 128, 3×3) | 128 × 4 × 4 | 73,728 |
| fc1 | PrunableLinear(2048 $\to$ 256) | 256 | 524,288 |
| fc2 | PrunableLinear(256 $\to$ 10) | 10 | 2,560 |
| **Total** | | | **619,872** |

Dropout ($p = 0.5$) is applied before the final classification layer. Batch normalisation is intentionally omitted so that the effect of the gating mechanism is not confounded by other normalisation techniques.

---

## Experimental Setup

- **Dataset:** CIFAR-10 (50,000 training images, 10,000 test images, 10 classes)
- **Optimiser:** Adam with learning rate $\alpha = 10^{-3}$
- **Epochs:** 20
- **Batch size:** 128
- **Pruning threshold:** gate values below $\tau = 10^{-2}$ are considered pruned
- **Lambda values tested:** $\lambda \in \{0,\ 10^{-4},\ 10^{-3},\ 10^{-2}\}$
- **Hardware:** NVIDIA T4 GPU (Google Colab)

---

## Results

### Summary Table

| $\lambda$ | Test Accuracy (%) | Sparsity Level (%) | Pruned / Total Gates |
|-----------|-------------------|---------------------|----------------------|
| $0$       | 72.76             | 0.0                 | 0 / 619,872          |
| $10^{-4}$ | 58.57             | 74.1                | 459,321 / 619,872    |
| $10^{-3}$ | 50.56             | 96.5                | 597,944 / 619,872    |
| $10^{-2}$ | 39.29             | 99.8                | 618,875 / 619,872    |

### Gate Value Distribution

The figure below shows the histogram of all 619,872 gate values after training with $\lambda = 10^{-4}$, which achieved the best accuracy among the pruned models. The distribution shows a dominant peak near zero, corresponding to the 74.1% of gates that have been driven below the pruning threshold. The remaining gates form a decaying tail extending to approximately 0.4, representing the surviving connections that the network has deemed necessary for classification.

![Gate Value Distribution](gate_distribution.png)

---

## Analysis

**Baseline** ($\lambda = 0$). Without any sparsity regularisation, the network achieves 72.76% test accuracy. All gate values remain above the pruning threshold, resulting in 0% sparsity. This establishes the accuracy upper bound for the architecture.

**Mild regularisation** ($\lambda = 10^{-4}$). Even a small penalty prunes 74.1% of all gated connections (459,321 out of 619,872). Accuracy decreases to 58.57%, a reduction of roughly 14 percentage points. This demonstrates that a large fraction of the network's weights are redundant — the model can still classify images meaningfully with only about a quarter of its original connections.

**Moderate regularisation** ($\lambda = 10^{-3}$). Sparsity increases to 96.5%, meaning fewer than 22,000 gates remain active out of nearly 620,000. Accuracy is 50.56%, which is still well above random chance (10% for 10 classes). The network has identified a small but effective subset of connections.

**Aggressive regularisation** ($\lambda = 10^{-2}$). At this level, 99.8% of all gates are pruned. Only approximately 1,000 connections survive. Accuracy degrades to 39.29%, confirming that excessive sparsity destroys the network's representational capacity, though it remains far above the 10% baseline of random guessing.

**Transition dynamics.** An interesting observation is that sparsity remains near 0% for most of training and then rises sharply in the final epochs. This is not a defect but a consequence of the sigmoid function's shape. The gate scores decrease steadily throughout training, but the sigmoid output $\sigma(s)$ only drops below the pruning threshold $\tau = 0.01$ once $s$ passes approximately $-4.6$. The apparent jump is therefore a thresholding artefact on a smoothly evolving underlying process.

---

## Conclusion

The self-pruning mechanism works as intended. The learnable gates, combined with L1 regularisation on their sigmoid-transformed values, allow the network to selectively eliminate connections during training without any external pruning step. The hyperparameter $\lambda$ provides effective control over the sparsity-accuracy trade-off, and the results confirm a clear monotonic relationship: increasing $\lambda$ yields higher sparsity at the cost of lower accuracy. The gate value distribution for the best pruned model confirms that the network successfully distinguishes between essential and redundant weights.
