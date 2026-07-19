//! Deterministic, counter-based seeding for Monte Carlo and path simulation.
//!
//! Reproducibility is a correctness property for the simulation engines: a given
//! `(seed, paths, steps)` must produce the same price and standard error
//! regardless of how many threads `rayon` happens to use. The trick is to give
//! every independent unit of work (an antithetic path pair, indexed by `i`) its
//! own RNG stream seeded *only* from `(seed, i)` — never from a shared counter
//! that thread scheduling could reorder. [`path_seed`] derives that per-index
//! seed; `SmallRng::seed_from_u64(path_seed(seed, i))` then yields an independent,
//! reproducible stream.

/// SplitMix64 — a fast, well-distributed 64-bit mixing finalizer.
///
/// Used to decorrelate seeds; not a general-purpose PRNG on its own. This is the
/// reference SplitMix64 finalizer (the increment + two xor-shift/multiply rounds
/// from Steele, Lea & Flood, "Fast Splittable Pseudorandom Number Generators").
pub fn splitmix64(z: u64) -> u64 {
    let mut x = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Derive an independent per-unit RNG seed from `(seed, index)`.
///
/// Two `splitmix64` passes decorrelate even sequential indices, so the streams
/// for `index = 0, 1, 2, …` are independent and reproducible across thread
/// counts. This is the seeding used by the Monte Carlo and Longstaff-Schwartz
/// engines and by the stochastic-process path simulator.
pub fn path_seed(seed: u64, index: usize) -> u64 {
    splitmix64(seed ^ splitmix64(index as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_seeds_are_distinct_for_sequential_indices() {
        let seeds: Vec<u64> = (0..1000).map(|i| path_seed(42, i)).collect();
        let mut unique = seeds.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), seeds.len(), "path seeds collided");
    }

    #[test]
    fn path_seed_is_deterministic() {
        assert_eq!(path_seed(7, 123), path_seed(7, 123));
        assert_ne!(path_seed(7, 123), path_seed(8, 123));
    }

    #[test]
    fn splitmix64_matches_reference_vector() {
        // First output of SplitMix64 with state 0 (reference implementation).
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
    }
}
