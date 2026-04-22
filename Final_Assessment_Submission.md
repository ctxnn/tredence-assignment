# Case Study Submission: Self-Pruning Neural Network
**Role:** Python AI Engineer Intern  
**Candidate:** Chirag Taneja  
**Repository:** [https://github.com/ctxnn/tredence-assignment](https://github.com/ctxnn/tredence-assignment)

---

## 1. Executive Summary

This submission details the design and implementation of a **Self-Pruning Neural Network** for image classification on the CIFAR-10 dataset. Unlike traditional post-training pruning, this model utilizes learnable gating mechanisms and L1 regularization to dynamically identify and remove redundant connections during the training phase. 

The project demonstrates a successful trade-off between model complexity and classification performance, achieving high levels of sparsity (over 70%) while maintaining functional accuracy.

---

## 2. Technical Implementation

### 2.1 The Prunable Architecture
The network is built using custom `PrunableLinear` and `PrunableConv2d` layers. Each weight in these layers is associated with a learnable gate score. 

*   **Gating Function:** gate = sigmoid(gate_score)
*   **Effective Weight:** pruned_weight = weight * gate

The total loss function augments the standard classification loss with an L1 penalty on all gate values:

**Total Loss = Classification Loss + lambda * sum(gates)**

Since the sigmoid function is monotonically increasing, minimising the sum of gate values pushes each gate score toward negative infinity, which drives the sigmoid output toward zero. However, for weights that are critical to maintaining classification accuracy, the gradient from the classification loss opposes this pressure and keeps the gate open. This creates a natural selection process: important connections survive while redundant ones are pruned.

The hyperparameter lambda controls the strength of this effect: larger values impose a stronger per-gate cost, leading to more aggressive pruning at the potential expense of accuracy.

---

## 3. Results and Analysis

A series of experiments were conducted to evaluate the effect of the regularisation strength (lambda) on both accuracy and sparsity.

### 3.1 Accuracy vs. Sparsity Trade-off

| Regularization (lambda) | Test Accuracy (%) | Sparsity Level (%) | Active Parameters (approx.) |
|--------------------------|-------------------|---------------------|-----------------------------|
| 0 (Baseline)             | 72.76             | 0.0                 | 619,872                     |
| 1e-4 (Optimal)           | **58.57**         | **74.1**            | **160,551**                 |
| 1e-3                     | 50.56             | 96.5                | 21,928                      |
| 1e-2                     | 39.29             | 99.8                | 997                         |

### 3.2 Training Logs

The full training output from Google Colab (NVIDIA T4 GPU) is reproduced below:

```
Using device: cuda

CIFAR-10 loaded: 50000 training, 10000 test samples

============================================================
  Training with lambda = 0e+00
============================================================
  Total gate parameters: 619,872
  Epoch  1/20  |  Loss: 1.9508  |  Train: 27.26%  |  Test: 37.96%  |  Sparsity: 0.0%
  Epoch  5/20  |  Loss: 1.4193  |  Train: 48.66%  |  Test: 54.56%  |  Sparsity: 0.0%
  Epoch 10/20  |  Loss: 1.1926  |  Train: 57.52%  |  Test: 63.22%  |  Sparsity: 0.0%
  Epoch 15/20  |  Loss: 1.0248  |  Train: 63.76%  |  Test: 68.79%  |  Sparsity: 0.0%
  Epoch 20/20  |  Loss: 0.9111  |  Train: 68.23%  |  Test: 72.76%  |  Sparsity: 0.0%

  Final -- Test Accuracy: 72.76%  |  Sparsity: 0.0%  |  Pruned: 0/619,872 gates

============================================================
  Training with lambda = 1e-04
============================================================
  Total gate parameters: 619,872
  Epoch  1/20  |  Loss: 1.9888  |  Train: 26.29%  |  Test: 37.06%  |  Sparsity: 0.0%
  Epoch  5/20  |  Loss: 1.5577  |  Train: 43.07%  |  Test: 48.82%  |  Sparsity: 0.0%
  Epoch 10/20  |  Loss: 1.4389  |  Train: 47.73%  |  Test: 53.39%  |  Sparsity: 0.0%
  Epoch 15/20  |  Loss: 1.3640  |  Train: 50.96%  |  Test: 56.64%  |  Sparsity: 0.0%
  Epoch 20/20  |  Loss: 1.3030  |  Train: 53.43%  |  Test: 58.57%  |  Sparsity: 74.1%

  Final -- Test Accuracy: 58.57%  |  Sparsity: 74.1%  |  Pruned: 459,321/619,872 gates

============================================================
  Training with lambda = 1e-03
============================================================
  Total gate parameters: 619,872
  Epoch  1/20  |  Loss: 1.9673  |  Train: 26.45%  |  Test: 37.38%  |  Sparsity: 0.0%
  Epoch  5/20  |  Loss: 1.6211  |  Train: 40.47%  |  Test: 44.80%  |  Sparsity: 0.0%
  Epoch 10/20  |  Loss: 1.5716  |  Train: 42.56%  |  Test: 47.31%  |  Sparsity: 0.0%
  Epoch 15/20  |  Loss: 1.5380  |  Train: 44.06%  |  Test: 49.05%  |  Sparsity: 0.0%
  Epoch 20/20  |  Loss: 1.5012  |  Train: 45.32%  |  Test: 50.56%  |  Sparsity: 96.5%

  Final -- Test Accuracy: 50.56%  |  Sparsity: 96.5%  |  Pruned: 597,944/619,872 gates

============================================================
  Training with lambda = 1e-02
============================================================
  Total gate parameters: 619,872
  Epoch  1/20  |  Loss: 1.9972  |  Train: 25.76%  |  Test: 35.72%  |  Sparsity: 0.0%
  Epoch  5/20  |  Loss: 1.7206  |  Train: 36.42%  |  Test: 40.07%  |  Sparsity: 0.0%
  Epoch 10/20  |  Loss: 1.7718  |  Train: 34.59%  |  Test: 38.86%  |  Sparsity: 0.0%
  Epoch 15/20  |  Loss: 1.7947  |  Train: 33.95%  |  Test: 38.05%  |  Sparsity: 0.0%
  Epoch 20/20  |  Loss: 1.7871  |  Train: 34.14%  |  Test: 39.29%  |  Sparsity: 99.8%

  Final -- Test Accuracy: 39.29%  |  Sparsity: 99.8%  |  Pruned: 618,875/619,872 gates


======================================================================
  SUMMARY TABLE
======================================================================
  Lambda       Test Accuracy (%)      Sparsity Level (%)
  ------------------------------------------------------------
  0e+00        72.76                  0.0
  1e-04        58.57                  74.1
  1e-03        50.56                  96.5
  1e-02        39.29                  99.8
======================================================================

  Best model for plot: lambda = 0.0001
  Saved gate distribution plot to gate_distribution.png
  Saved Markdown report to report.md

 Training complete!
```

### 3.3 Visual Analysis
The following distribution plot for the lambda = 1e-4 model confirms the success of the pruning mechanism. 

![Gate Value Distribution](gate_distribution.png)

*The plot exhibits a clear spike near 0.0, representing the 74.1% of weights that have been successfully pruned, and a secondary distribution of active weights that preserve the model's predictive power.*

---

## 4. Analysis

**Baseline (lambda = 0).** Without any sparsity regularisation, the network reaches 72.76% test accuracy. All gate values remain above the pruning threshold, resulting in 0% sparsity. This serves as the upper bound on accuracy for this architecture.

**Mild regularisation (lambda = 1e-4).** Even a small penalty is sufficient to prune 74.1% of all gated connections (459,321 out of 619,872). The accuracy drops to 58.57%, a reduction of roughly 14 percentage points. This demonstrates that a large fraction of the network's weights are redundant.

**Moderate regularisation (lambda = 1e-3).** Sparsity increases to 96.5%, meaning fewer than 22,000 gates remain active out of nearly 620,000. The accuracy is 50.56%, which is still well above random chance (10% for 10 classes). The network has identified a small but effective subset of connections.

**Aggressive regularisation (lambda = 1e-2).** At this level, 99.8% of all gates are pruned. Only approximately 1,000 connections survive. The accuracy degrades to 39.29%, confirming that excessive sparsity destroys the network's representational capacity, though it remains far above random guessing.

**Transition dynamics.** Sparsity remains near 0% for most of training and then rises sharply in the final epochs. This is not a defect but a consequence of the sigmoid function's shape: gate scores decrease steadily during training, but the sigmoid output only drops below the pruning threshold (0.01) once the score passes approximately -4.6. The apparent jump is a thresholding artefact on a smoothly evolving underlying process.

---

## 5. Conclusion

The self-pruning implementation successfully identifies redundant parameters within the network. The experiments reveal that:
1.  **Redundancy:** Nearly 75% of the network's weights can be removed while retaining a significant portion of its predictive accuracy.
2.  **Control:** The lambda hyperparameter provides a precise "knob" for hardware-constrained deployments, allowing developers to choose the optimal point on the accuracy-vs-efficiency curve.
3.  **Efficiency:** The learnable gating mechanism is a robust alternative to manually tuned post-processing pruning steps.

---

## 6. Repository Structure
The full source code and results can be accessed at the GitHub link provided above.
- `self_pruning_nn.py`: The core implementation and training script.
- `requirements.txt`: Environment configuration.
- `gate_distribution.png`: Pruning visualization.
