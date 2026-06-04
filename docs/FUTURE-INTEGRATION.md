# Future Integration: ternary-pca

## Current State
Implements PCA for ternary data using fixed-point arithmetic (i32 with 256x scale factor). Provides `TernaryCovariance` matrix computation, eigenvalue decomposition via power iteration, dimensionality reduction, variance explained ratios, and reconstruction — all without floating-point hardware.

## Integration Opportunities

### With ternary-cell / construct-core
PCA compresses high-dimensional cell state. A room with 64 ternary parameters (sensors, actuator states, neighbor summaries) can be projected to 8 principal components via `project()`. The `TernaryCovariance` matrix is computed from historical room data, then the projection runs on every tick. At Layer 0 (ESP32), the fixed-point arithmetic means no FPU needed — the entire PCA runs in integer math.

### With ternary-matrix
`TernaryCovariance` stores its matrix as `Vec<Vec<i32>>` in fixed-point. `ternary-matrix`'s compact 2-bit-per-trit storage could compress the input data, while PCA's covariance computation would use the unpacked form. The `TernaryMatrix::multiply()` handles the bulk linear algebra.

### With ternary-projection
Both crates do dimensionality reduction. `ternary-pca` uses covariance-based eigenvectors (optimal for variance preservation); `ternary-projection` uses random projections (faster, approximate). For real-time cell tick cycles, random projections may suffice. For offline analysis and visualization, PCA is superior. Both should share a common `Projection` trait.

## Potential in Mature Systems
In PLATO's fleet management, PCA identifies the dominant modes of variation across all rooms. The first few principal components capture the "personality" of each room — is it temperature-dominated, occupancy-dominated, or energy-dominated? `variance_explained()` quantifies how many dimensions matter. Rooms with similar PCA projections are candidates for shared ensign assignment. The fixed-point implementation runs on any platform.

## Cross-Pollination Ideas
**Music × PCA:** Decompose a corpus of ternary chord progressions. PCA reveals the latent dimensions of harmonic variation — perhaps one axis is tension/resolution, another is brightness/darkness. `project()` maps any chord sequence to a point in this latent space. Clustering in PCA space groups similar musical passages. Connects to `ternary-music`.

**Topology × PCA:** The principal components of a ternary dataset define its intrinsic dimensionality. This connects to `ternary-topology` — the manifold dimension estimated by PCA should match the topological dimension. Discrepancies indicate non-linear structure.

## Dependencies for Next Steps
- Shared `Projection` trait with `ternary-projection`
- Incremental PCA for streaming room data (update covariance without storing all history)
- Benchmark fixed-point vs. floating-point PCA accuracy on real room data
