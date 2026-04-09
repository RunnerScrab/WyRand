
pub mod traits;
pub mod distributions;

pub use traits::ParamSource;
pub use distributions::*;

/// WyRand is a high-performance, non-cryptographic random number generator
/// designed for scientific simulations and procedural generation.
/// 
/// This implementation features a **Unified Generic API** that supports both 
/// scalar and bulk generation using SIMD-accelerated static dispatch. 
/// Parameters can be passed as either constants (scalars) or varying columns 
/// (slices/vectors) without runtime overhead.
///
/// # Examples
///
/// ### Simple Scalar Generation
/// ```
/// use wyrand::WyRand;
///
/// let mut rng = WyRand::new(42);
/// let rv = rng.next_uniform_f32();
/// let in_range = rng.next_uniform_in_range_f32(10.0, 20.0);
/// ```
///
/// ### High-Throughput Bulk Generation (SIMD)
/// ```
/// use wyrand::WyRand;
///
/// let mut rng = WyRand::new(42);
/// let mut buffer = vec![0.0f32; 1024];
/// 
/// // Bulk uniform generation
/// rng.fill_uniform_f32(&mut buffer);
///
/// // Bulk generation with varying parameters (Columnar)
/// let mins = vec![0.0; 1024];
/// let maxs = vec![1.0; 1024];
/// rng.fill_uniform_in_range_f32(&mut buffer, &mins, &maxs);
///
/// // Hybrid usage: Constant mode with varying sigmas
/// let modes = 10.0;
/// let sigmas = vec![1.5; 1024];
/// rng.fill_normal_f32(&mut buffer, modes, &sigmas);
///
/// // Bulk Poisson sampling with varying rates
/// let mut p_buffer = vec![0u32; 1024];
/// let lambdas = vec![2.0; 1024];
/// rng.fill_poisson_u32(&mut p_buffer, &lambdas);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct WyRand(u64);

impl From<u64> for WyRand {
    fn from(val: u64) -> Self {
        WyRand(val)
    }
}

impl WyRand {
    pub(crate) const INC: u64 = 0x60bee2bee120fc15;
    pub(crate) const FLOAT32_MASK: u64 = u64::MAX ^ 0xFFFF_FFFF_F800_0000;

    /// Creates a new WyRand generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Returns the next 64 bits of randomness.
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(Self::INC);

        // First mix
        let tmp = (self.0 as u128).wrapping_mul(0xa3b195354a39b70d);
        let m1 = ((tmp.wrapping_shr(64)) as u64) ^ (tmp as u64);

        // Second mix
        let tmp2 = (m1 as u128).wrapping_mul(0x1b03738712fad5c9);
        ((tmp2.wrapping_shr(64)) as u64) ^ (tmp2 as u64)
    }
}
