# ternary-pca

Principal component analysis for ternary-valued data `{-1, 0, +1}`, using fixed-point arithmetic for embedded-friendly computation with no floating-point hardware required.

## Why This Exists

Standard PCA assumes continuous floating-point data and relies on BLAS/LAPACK for eigenvalue decomposition. Ternary data has special structure — only three possible values per dimension — and often appears in resource-constrained environments (microcontrollers, FPGAs). This crate performs PCA entirely in fixed-point arithmetic (i32 with 8.8 scaling), making it suitable for no-std embedded targets while still providing useful dimensionality reduction.

## Core Concepts

- **TernaryCovariance** — Fixed-point covariance matrix computed from ternary samples
- **EigenDecomp** — Top-k eigenvalues and eigenvectors via power iteration with deflation
- **TernaryPCA** — Full PCA pipeline: fit, project, reconstruct, and variance analysis
- **Fixed-Point Arithmetic** — All internal math uses i32 with SCALE=256 (8 integer + 8 fractional bits)

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-pca = "0.1"
```

```rust
use ternary_pca::*;

// Your ternary dataset: rows = samples, cols = dimensions
let data: Vec<Vec<Trit>> = vec![
    vec![ 1,  1, -1],
    vec![-1, -1,  1],
    vec![ 1,  1, -1],
    vec![-1, -1,  1],
    vec![ 0,  1, -1],
];

// Compute covariance matrix
let cov = TernaryCovariance::from_data(&data);
println!("Dimensions: {}", cov.dim());
println!("Variance per dim: {:?}", (0..cov.dim()).map(|i| cov.variance(i)).collect::<Vec<_>>());
println!("Total variance: {:.4}", cov.total_variance());

// Fit PCA with 2 components
let pca = TernaryPCA::fit(&data, 2);
println!("Components: {}", pca.num_components());

// Project a sample into reduced space
let projected = pca.project(&[1, 1, -1]);
println!("Projected: {:?}", projected);

// Reconstruct and check error
let reconstructed = pca.reconstruct(&projected);
let error = pca.reconstruction_error(&[1, 1, -1]);
println!("Reconstruction error: {:.4}", error);

// Variance analysis
println!("Variance explained: {:?}", pca.variance_explained());
println!("Cumulative (k=2): {:.4}", pca.cumulative_variance(2));
```

## API Overview

| Type / Function | Description |
|---|---|
| `TernaryCovariance::from_data` | Compute covariance from ternary samples |
| `TernaryCovariance::variance` / `total_variance` | Per-dimension and total variance |
| `EigenDecomp::compute` | Top-k eigenvalues/vectors via power iteration |
| `EigenDecomp::variance_explained` | Fraction of variance per component |
| `TernaryPCA::fit` | Fit PCA model on training data |
| `TernaryPCA::project` | Reduce a sample to k dimensions |
| `TernaryPCA::reconstruct` | Reconstruct from reduced space |
| `TernaryPCA::reconstruction_error` | MSE between original and reconstructed |

## How It Works

1. **Fixed-point representation**: All values stored as `i32` with scale factor 256. Multiply uses 64-bit intermediates to prevent overflow. Newton's method computes fixed-point square roots.

2. **Covariance**: Means computed in fixed-point. Covariance matrix calculated as `E[(X-μ)(X-μ)ᵀ]` with Bessel's correction (dividing by n−1).

3. **Power iteration**: For each requested eigenvalue, iterates `v ← Av / ‖Av‖` to converge on the dominant eigenvector, then deflates the matrix by subtracting `λvvᵀ` before finding the next component.

4. **Projection**: Centers the input by subtracting means, then computes dot products with eigenvectors. All projection math uses f64 for the final output stage.

## Use Cases

1. **Embedded dimensionality reduction** — Reduce high-dimensional ternary sensor data on microcontrollers without FPU
2. **Ternary neural network analysis** — Analyze the variance structure of quantized weight matrices
3. **Feature extraction from ternary signals** — Find the principal directions in thresholded measurement data
4. **Anomaly detection** — High reconstruction error after PCA projection indicates outliers

## Ecosystem

Part of the **SuperInstance** ternary computing crate family:

- `ternary-compression-v2` — Multi-algorithm ternary compression
- `ternary-hash` — Hashing and fingerprinting for ternary data
- `ternary-matrix` — Compact ternary matrix operations
- `ternary-ga` — Genetic algorithms with ternary genomes
- `ternary-reservoir` — Echo state networks with ternary nodes
- `ternary-evolution-advanced` — Advanced evolutionary optimization
- `ternary-geometry` — Geometric algorithms in ternary space
- `ternary-causality` — Causal inference for ternary systems
- `ternary-consensus` — Distributed consensus for ternary agents

## License

MIT
