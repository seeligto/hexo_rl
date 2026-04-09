/// Dirichlet noise sampling for MCTS root exploration.
///
/// Used by the Rust training path (`game_runner.rs`) to inject exploration
/// noise into root priors after the first NN expansion, matching the AlphaZero
/// recipe (Silver et al. 2018 §2.1).
///
/// Approach: draw n independent Gamma(alpha, 1.0) samples, normalize by sum.
/// This is mathematically equivalent to sampling from Dir(alpha, …, alpha).
use rand::Rng;
use rand_distr::{Distribution, Gamma};

/// Sample a symmetric Dirichlet(alpha) vector of length `n` using the
/// per-worker RNG already in scope. The caller holds the `&mut impl Rng`.
///
/// Falls back to uniform distribution if the Gamma samples sum to < 1e-8
/// (degenerate case at very small alpha with float underflow).
pub fn sample_dirichlet(alpha: f32, n: usize, rng: &mut impl Rng) -> Vec<f32> {
    debug_assert!(alpha > 0.0, "Dirichlet alpha must be positive");
    debug_assert!(n > 0, "Dirichlet n must be positive");

    let dist = Gamma::new(alpha as f64, 1.0).expect("Dirichlet: invalid alpha");
    let mut samples: Vec<f32> = (0..n).map(|_| dist.sample(rng) as f32).collect();

    let sum: f32 = samples.iter().sum();
    if sum > 1e-8 {
        for s in &mut samples {
            *s /= sum;
        }
    } else {
        // Degenerate: all Gamma draws underflowed to zero. Fall back to uniform.
        let u = 1.0 / n as f32;
        for s in &mut samples {
            *s = u;
        }
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirichlet_sums_to_one() {
        let mut rng = rand::rng();
        for _ in 0..20 {
            let v = sample_dirichlet(0.3, 25, &mut rng);
            let sum: f32 = v.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Dirichlet sum {sum} not within 1e-5 of 1.0"
            );
        }
    }

    #[test]
    fn test_dirichlet_non_negative() {
        let mut rng = rand::rng();
        for _ in 0..20 {
            let v = sample_dirichlet(0.3, 25, &mut rng);
            for &s in &v {
                assert!(s >= 0.0, "Dirichlet sample {s} is negative");
            }
        }
    }

    #[test]
    fn test_dirichlet_different_draws() {
        // Two independent draws from the same alpha/n should almost certainly differ.
        let mut rng = rand::rng();
        let v1 = sample_dirichlet(0.3, 25, &mut rng);
        let v2 = sample_dirichlet(0.3, 25, &mut rng);
        let max_diff: f32 = v1
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "Two sequential Dirichlet draws were identical (max_diff={max_diff})"
        );
    }

    #[test]
    fn test_dirichlet_sparse_at_low_alpha() {
        // At alpha=0.3, n=25, the effective support (exp(H)) should be sparse.
        // Concretely: average over 100 trials should have fewer than 10 components
        // receiving > 1/n weight (i.e. > 0.04).
        let mut rng = rand::rng();
        let n = 25usize;
        let threshold = 1.0 / n as f32; // 0.04
        let trials = 100;

        let total_above: f32 = (0..trials)
            .map(|_| {
                let v = sample_dirichlet(0.3, n, &mut rng);
                v.iter().filter(|&&s| s > threshold).count() as f32
            })
            .sum();
        let avg_above = total_above / trials as f32;

        assert!(
            avg_above < 10.0,
            "alpha=0.3 n=25: expected avg effective support < 10, got {avg_above:.2}"
        );
    }
}
