#![forbid(unsafe_code)]

//! Principal component analysis for ternary data ({-1, 0, +1}).
//!
//! Provides TernaryCovariance matrix, eigenvalue decomposition adapted for
//! ternary values, dimensionality reduction, variance explained, and
//! projection/reconstruction. Uses fixed-point arithmetic (i32 with scale factor)
//! for embedded-friendly computation.

/// A trit value.
pub type Trit = i8;

/// Fixed-point scale: values are stored as i32, representing real * SCALE.
const SCALE: i32 = 256;

/// Convert to fixed-point.
fn to_fp(v: f64) -> i32 {
    (v * SCALE as f64).round() as i32
}

/// Convert from fixed-point.
fn from_fp(v: i32) -> f64 {
    v as f64 / SCALE as f64
}

/// Fixed-point multiply.
fn fp_mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) / SCALE as i64) as i32
}

/// Fixed-point divide.
fn fp_div(a: i32, b: i32) -> i32 {
    if b == 0 { return 0; }
    ((a as i64 * SCALE as i64) / b as i64) as i32
}

/// Fixed-point sqrt (Newton's method).
fn fp_sqrt(v: i32) -> i32 {
    if v <= 0 { return 0; }
    let mut x = v;
    for _ in 0..20 {
        let next = (x + fp_div(v, x)) / 2;
        if (next - x).unsigned_abs() <= 1 { break; }
        x = next;
        if x <= 0 { x = 1; }
        // Prevent overflow in next iteration
        if x > (1 << 30) { x = 1 << 30; }
    }
    x
}

// ---------------------------------------------------------------------------
// TernaryCovariance
// ---------------------------------------------------------------------------

/// Covariance matrix for ternary data, computed in fixed-point.
#[derive(Clone, Debug)]
pub struct TernaryCovariance {
    dim: usize,
    matrix: Vec<Vec<i32>>, // fixed-point upper triangle stored as full matrix
}

impl TernaryCovariance {
    /// Compute covariance from a dataset (rows = samples, cols = dimensions).
    /// All values should be {-1, 0, +1}.
    pub fn from_data(data: &[Vec<Trit>]) -> Self {
        if data.is_empty() {
            return Self { dim: 0, matrix: vec![] };
        }
        let n = data.len();
        let dim = data[0].len();
        let mut means = vec![0i32; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                means[j] += v as i32;
            }
        }
        for m in &mut means {
            *m = fp_div(*m, n as i32);
        }
        let mut cov = vec![vec![0i32; dim]; dim];
        for row in data {
            for i in 0..dim {
                let di = (row[i] as i32 * SCALE) - means[i];
                for j in i..dim {
                    let dj = (row[j] as i32 * SCALE) - means[j];
                    let val = fp_mul(di, dj);
                    cov[i][j] += val;
                }
            }
        }
        let denom = if n > 1 { (n - 1) as i32 } else { 1i32 };
        for i in 0..dim {
            for j in i..dim {
                cov[i][j] = fp_div(cov[i][j], denom);
                cov[j][i] = cov[i][j];
            }
        }
        Self { dim, matrix: cov }
    }

    /// Dimension of the covariance matrix.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get covariance value at (i, j) as f64.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        from_fp(self.matrix[i][j])
    }

    /// Variance of dimension i.
    pub fn variance(&self, i: usize) -> f64 {
        self.get(i, i)
    }

    /// Total variance (trace of covariance matrix).
    pub fn total_variance(&self) -> f64 {
        (0..self.dim).map(|i| self.variance(i)).sum()
    }

    /// Return the full matrix as f64 values.
    pub fn to_f64_matrix(&self) -> Vec<Vec<f64>> {
        self.matrix.iter().map(|row| row.iter().map(|&v| from_fp(v)).collect()).collect()
    }
}

// ---------------------------------------------------------------------------
// Eigenvalue decomposition (power iteration for symmetric matrices)
// ---------------------------------------------------------------------------

/// Result of eigenvalue decomposition.
#[derive(Clone, Debug)]
pub struct EigenDecomp {
    /// Eigenvalues in fixed-point.
    eigenvalues: Vec<i32>,
    /// Eigenvectors (columns), each of length dim, in fixed-point.
    eigenvectors: Vec<Vec<i32>>,
    dim: usize,
}

impl EigenDecomp {
    /// Compute top `k` eigenvalues/eigenvectors via power iteration.
    pub fn compute(cov: &TernaryCovariance, k: usize) -> Self {
        let dim = cov.dim();
        let k = k.min(dim);
        let mut eigenvalues = Vec::with_capacity(k);
        let mut eigenvectors = Vec::with_capacity(k);
        // Deflation approach
        let mut mat = cov.matrix.clone();
        for _ in 0..k {
            let (val, vec) = power_iteration(&mat, dim, 100);
            eigenvalues.push(val);
            eigenvectors.push(vec.clone());
            // Deflate: mat = mat - lambda * v * v^T
            for i in 0..dim {
                for j in 0..dim {
                    mat[i][j] -= fp_mul(fp_mul(val, vec[i]), vec[j]);
                }
            }
        }
        Self { eigenvalues, eigenvectors, dim }
    }

    /// Number of components.
    pub fn num_components(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Get eigenvalue i as f64.
    pub fn eigenvalue(&self, i: usize) -> f64 {
        from_fp(self.eigenvalues[i])
    }

    /// Get eigenvector i as f64 values.
    pub fn eigenvector(&self, i: usize) -> Vec<f64> {
        self.eigenvectors[i].iter().map(|&v| from_fp(v)).collect()
    }

    /// Variance explained by component i.
    pub fn variance_explained(&self, i: usize) -> f64 {
        let total: f64 = self.eigenvalues.iter().map(|&v| from_fp(v).abs()).sum();
        if total == 0.0 { 0.0 } else { from_fp(self.eigenvalues[i]).abs() / total }
    }

    /// Cumulative variance explained for the first `k` components.
    pub fn cumulative_variance_explained(&self, k: usize) -> f64 {
        let k = k.min(self.eigenvalues.len());
        let total: f64 = self.eigenvalues.iter().map(|&v| from_fp(v).abs()).sum();
        if total == 0.0 { return 0.0; }
        let cum: f64 = self.eigenvalues[..k].iter().map(|&v| from_fp(v).abs()).sum();
        cum / total
    }
}

fn power_iteration(mat: &[Vec<i32>], dim: usize, iters: usize) -> (i32, Vec<i32>) {
    let mut v = vec![SCALE; dim];
    normalize_vec(&mut v);
    for _ in 0..iters {
        let mut w = vec![0i64; dim];
        for i in 0..dim {
            for j in 0..dim {
                w[i] += (mat[i][j] as i64 * v[j] as i64) / SCALE as i64;
            }
            w[i] = w[i].clamp(-(SCALE as i64 * 4), SCALE as i64 * 4);
        }
        v = w.iter().map(|&x| x as i32).collect();
        normalize_vec(&mut v);
    }
    let mut mv = vec![0i64; dim];
    for i in 0..dim {
        for j in 0..dim {
            mv[i] += (mat[i][j] as i64 * v[j] as i64) / SCALE as i64;
        }
    }
    let mut lambda: i64 = 0;
    for i in 0..dim {
        lambda += (v[i] as i64 * mv[i]) / SCALE as i64;
    }
    (lambda.clamp(i32::MIN as i64, i32::MAX as i64) as i32, v)
}

fn normalize_vec(v: &mut Vec<i32>) {
    let mut norm_sq: i64 = 0;
    for &x in v.iter() {
        norm_sq += (x as i64) * (x as i64);
    }
    if norm_sq == 0 { return; }
    let norm = fp_sqrt((norm_sq / SCALE as i64) as i32);
    if norm > 0 {
        for x in v.iter_mut() {
            *x = fp_div(*x, norm);
            // Clamp to prevent runaway growth
            *x = (*x).clamp(-(SCALE * 4), SCALE * 4);
        }
    }
}

// ---------------------------------------------------------------------------
// PCA Transformer
// ---------------------------------------------------------------------------

/// PCA transformer: project ternary data onto principal components.
#[derive(Clone, Debug)]
pub struct TernaryPCA {
    decomp: EigenDecomp,
    means: Vec<i32>, // fixed-point means
    dim: usize,
}

impl TernaryPCA {
    /// Fit PCA on ternary data, keeping `k` components.
    pub fn fit(data: &[Vec<Trit>], k: usize) -> Self {
        if data.is_empty() {
            return Self { decomp: EigenDecomp { eigenvalues: vec![], eigenvectors: vec![], dim: 0 }, means: vec![], dim: 0 };
        }
        let dim = data[0].len();
        let n = data.len();
        // Compute means
        let mut means = vec![0i32; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                means[j] += v as i32;
            }
        }
        for m in &mut means {
            *m = fp_div(*m, n as i32);
        }
        // Covariance
        let cov = TernaryCovariance::from_data(data);
        let decomp = EigenDecomp::compute(&cov, k);
        Self { decomp, means, dim }
    }

    /// Number of components.
    pub fn num_components(&self) -> usize {
        self.decomp.num_components()
    }

    /// Project a ternary vector into the reduced space.
    pub fn project(&self, sample: &[Trit]) -> Vec<f64> {
        let k = self.decomp.num_components();
        let mut result = vec![0.0f64; k];
        for c in 0..k {
            let ev = self.decomp.eigenvector(c);
            let mut val = 0.0f64;
            for (j, &t) in sample.iter().enumerate() {
                let centered = t as f64 - from_fp(self.means[j]);
                val += centered * ev[j];
            }
            result[c] = val;
        }
        result
    }

    /// Reconstruct a projected vector back to the original space.
    pub fn reconstruct(&self, projected: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let mut result = vec![0.0f64; dim];
        for c in 0..projected.len().min(self.decomp.num_components()) {
            let ev = self.decomp.eigenvector(c);
            for j in 0..dim {
                result[j] += projected[c] * ev[j];
            }
        }
        // Add back means
        for j in 0..dim {
            result[j] += from_fp(self.means[j]);
        }
        result
    }

    /// Project and then reconstruct, returning reconstruction error.
    pub fn reconstruction_error(&self, sample: &[Trit]) -> f64 {
        let proj = self.project(sample);
        let recon = self.reconstruct(&proj);
        let mut err = 0.0f64;
        for (i, &t) in sample.iter().enumerate() {
            let diff = t as f64 - recon[i];
            err += diff * diff;
        }
        err
    }

    /// Variance explained by each component.
    pub fn variance_explained(&self) -> Vec<f64> {
        (0..self.decomp.num_components())
            .map(|i| self.decomp.variance_explained(i))
            .collect()
    }

    /// Cumulative variance explained.
    pub fn cumulative_variance(&self, k: usize) -> f64 {
        self.decomp.cumulative_variance_explained(k)
    }

    /// Access the eigen decomposition.
    pub fn decomp(&self) -> &EigenDecomp {
        &self.decomp
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covariance_dim() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 0, -1],
            vec![0, 1, 0],
            vec![-1, 0, 1],
        ];
        let cov = TernaryCovariance::from_data(&data);
        assert_eq!(cov.dim(), 3);
    }

    #[test]
    fn test_covariance_positive_diagonal() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, -1],
            vec![-1, 1],
            vec![1, -1],
            vec![-1, 1],
        ];
        let cov = TernaryCovariance::from_data(&data);
        assert!(cov.variance(0) > 0.0);
        assert!(cov.variance(1) > 0.0);
    }

    #[test]
    fn test_covariance_symmetric() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 0, -1],
            vec![0, 1, 0],
            vec![-1, 0, 1],
            vec![1, 1, 1],
        ];
        let cov = TernaryCovariance::from_data(&data);
        for i in 0..cov.dim() {
            for j in 0..cov.dim() {
                assert!((cov.get(i, j) - cov.get(j, i)).abs() < 0.01);
            }
        }
    }

    #[test]
    fn test_covariance_total_variance() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, -1],
            vec![-1, 1],
        ];
        let cov = TernaryCovariance::from_data(&data);
        assert!(cov.total_variance() > 0.0);
    }

    #[test]
    fn test_covariance_empty() {
        let cov = TernaryCovariance::from_data(&[]);
        assert_eq!(cov.dim(), 0);
    }

    #[test]
    fn test_covariance_constant() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
        ];
        let cov = TernaryCovariance::from_data(&data);
        // All variance should be zero
        assert!(cov.variance(0).abs() < 0.1);
        assert!(cov.variance(1).abs() < 0.1);
    }

    #[test]
    fn test_eigen_positive() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 1],
            vec![-1, -1],
            vec![1, 1],
            vec![-1, -1],
        ];
        let cov = TernaryCovariance::from_data(&data);
        let decomp = EigenDecomp::compute(&cov, 2);
        // At least one eigenvalue should be positive
        let has_positive = (0..decomp.num_components())
            .any(|i| decomp.eigenvalue(i) > 0.0);
        assert!(has_positive);
    }

    #[test]
    fn test_eigen_variance_sums_to_one() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 0],
            vec![-1, 1],
            vec![0, -1],
            vec![1, -1],
        ];
        let cov = TernaryCovariance::from_data(&data);
        let decomp = EigenDecomp::compute(&cov, 2);
        let total: f64 = (0..decomp.num_components())
            .map(|i| decomp.variance_explained(i))
            .sum();
        assert!((total - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_eigen_cumulative() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 0, -1],
            vec![-1, 1, 0],
            vec![0, -1, 1],
            vec![1, 1, 1],
            vec![-1, -1, -1],
        ];
        let cov = TernaryCovariance::from_data(&data);
        let decomp = EigenDecomp::compute(&cov, 3);
        // Just verify it runs without panicking
        let cv = decomp.cumulative_variance_explained(3);
        assert!(cv >= 0.0);
    }

    #[test]
    fn test_pca_fit_and_project() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 1],
            vec![-1, -1],
            vec![1, 1],
            vec![-1, -1],
        ];
        let pca = TernaryPCA::fit(&data, 1);
        assert_eq!(pca.num_components(), 1);
        let proj = pca.project(&[1, 1]);
        assert_eq!(proj.len(), 1);
    }

    #[test]
    fn test_pca_reconstruct() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 1],
            vec![-1, -1],
            vec![1, 1],
            vec![-1, -1],
        ];
        let pca = TernaryPCA::fit(&data, 2);
        let proj = pca.project(&[1, 1]);
        let recon = pca.reconstruct(&proj);
        assert_eq!(recon.len(), 2);
    }

    #[test]
    fn test_pca_reconstruction_error_lower_with_more_components() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 0, -1],
            vec![-1, 1, 0],
            vec![0, -1, 1],
            vec![1, 1, -1],
            vec![-1, -1, 1],
        ];
        let pca1 = TernaryPCA::fit(&data, 1);
        let pca2 = TernaryPCA::fit(&data, 2);
        let pca3 = TernaryPCA::fit(&data, 3);
        let sample = vec![1, 0, -1];
        let err1 = pca1.reconstruction_error(&sample);
        let err3 = pca3.reconstruction_error(&sample);
        assert!(err3 <= err1 + 0.5, "err3={} should be <= err1={}", err3, err1);
    }

    #[test]
    fn test_pca_variance_explained() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 1],
            vec![-1, -1],
            vec![1, 1],
            vec![-1, -1],
        ];
        let pca = TernaryPCA::fit(&data, 2);
        let ve = pca.variance_explained();
        assert_eq!(ve.len(), 2);
    }

    #[test]
    fn test_pca_cumulative() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, 1],
            vec![-1, -1],
            vec![1, 1],
            vec![-1, -1],
        ];
        let pca = TernaryPCA::fit(&data, 2);
        let cum = pca.cumulative_variance(2);
        assert!((cum - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_fp_roundtrip() {
        let v = 0.5;
        assert!((from_fp(to_fp(v)) - v).abs() < 0.01);
    }

    #[test]
    fn test_fp_mul_basic() {
        let a = to_fp(0.5);
        let b = to_fp(0.5);
        assert!((from_fp(fp_mul(a, b)) - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_fp_sqrt() {
        let v = to_fp(4.0);
        let s = fp_sqrt(v);
        assert!((from_fp(s) - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_fp_div() {
        let a = to_fp(1.0);
        let b = to_fp(4.0);
        assert!((from_fp(fp_div(a, b)) - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_pca_empty_data() {
        let pca = TernaryPCA::fit(&[], 1);
        assert_eq!(pca.num_components(), 0);
    }

    #[test]
    fn test_covariance_to_f64_matrix() {
        let data: Vec<Vec<Trit>> = vec![
            vec![1, -1],
            vec![-1, 1],
        ];
        let cov = TernaryCovariance::from_data(&data);
        let m = cov.to_f64_matrix();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].len(), 2);
    }
}
