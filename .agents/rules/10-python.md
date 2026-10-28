---
description: Python code standards for types, docstrings, numerics, and device safety
globs: "**/*.py"
alwaysApply: false
---

# Python Coding & Numerics Standards

1. **Type Hints & Static Checks:**
   - Every function and method signature must carry complete type annotations (`typing` / `torch.Tensor` / Python 3.10+ union syntax).
   - Use `torch.Tensor` for tensor inputs and outputs, annotating expected shapes in docstrings.

2. **Mathematical Docstrings & Paper Citations:**
   - Every neural network module, layer, pooling block, and loss function must cite the corresponding equation and section from the paper (*Alam et al., IJCNN 2025*) in its docstring:
     ```python
     def forward(self, x: torch.Tensor) -> torch.Tensor:
         """Compute frequency-wise selective kernel attention.
         
         Implements fwSKA as described in Alam et al. (IJCNN 2025, Section II-A).
         Splits input into two conv branches, pools across time and channels,
         and computes soft attention weights across frequency bins.
         
         Args:
             x: Input tensor of shape (B, C, F, T).
         Returns:
             Attended tensor of shape (B, C, F, T).
         """
     ```

3. **No Hidden Global State:**
   - Modules and functions must be pure and self-contained; no reliance on mutable global variables or hidden state outside class instances.
   - All hyperparameters must be passed explicitly via constructor arguments or typed configuration objects.

4. **Numerics & Local Statistics Rules:**
   - **fp32 for Local Statistics:** Sliding-window local statistics ($\mu_t, \sigma_t$) must always be computed in `torch.float32`.
   - **Variance Clamping:** Clamp variance before computing square root: `torch.clamp(var, min=1e-5)` to prevent negative variance or zero-gradient singularity.
   - **Epsilon in Division:** Always include $\epsilon = 10^{-8}$ when normalizing vectors or computing cosine similarity.

5. **Device-Agnostic Execution & MPS Safety:**
   - Code must run seamlessly across `mps`, `cuda`, and `cpu`. Instantiate devices dynamically:
     ```python
     device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
     ```
   - **No CUDA GradScaler on MPS:** `torch.cuda.amp.GradScaler` is strictly prohibited on `mps`. On Apple Silicon, run model training in `torch.float32`.
   - Loss functions (AAMSoftmax, IAAMSoftmax) and prototype matrix operations must always execute in `torch.float32`.
