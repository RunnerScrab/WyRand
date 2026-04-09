#![allow(unused)]
use fptricks::*;

/// A trait for types that can provide 8-element SIMD chunks of parameters.
/// 
/// This trait is implemented for scalars (providing a broadcasted chunk) and
/// slices/vectors (providing a direct memory load).
pub trait ParamSource8<T: Copy>: Copy {
    /// Returns a chunk of 8 elements starting at the given offset.
    fn chunk8(&self, offset: usize) -> [T; 8];
    /// Returns a chunk of 16 elements starting at the given offset.
    fn chunk16(&self, offset: usize) -> [T; 16];
    /// Returns a single element at the given index.
    fn get(&self, idx: usize) -> T;
}

/// A trait for types that can provide 4-element SIMD chunks of parameters.
/// 
/// This trait is implemented for scalars (providing a broadcasted chunk) and
/// slices/vectors (providing a direct memory load).
pub trait ParamSource4<T: Copy>: Copy {
    /// Returns a chunk of 4 elements starting at the given offset.
    fn chunk4(&self, offset: usize) -> [T; 4];
    /// Returns a chunk of 8 elements starting at the given offset.
    fn chunk8(&self, offset: usize) -> [T; 8];
    /// Returns a single element at the given index.
    fn get(&self, idx: usize) -> T;
}

impl ParamSource8<f32> for f32 {
    #[inline(always)]
    fn chunk8(&self, _: usize) -> [f32; 8] { [*self; 8] }
    #[inline(always)]
    fn chunk16(&self, _: usize) -> [f32; 16] { [*self; 16] }
    #[inline(always)]
    fn get(&self, _: usize) -> f32 { *self }
}

impl<'a> ParamSource8<f32> for &'a [f32] {
    #[inline(always)]
    fn chunk8(&self, offset: usize) -> [f32; 8] {
        let mut arr = [0.0; 8];
        arr.copy_from_slice(&self[offset..offset+8]);
        arr
    }
    #[inline(always)]
    fn chunk16(&self, offset: usize) -> [f32; 16] {
        let mut arr = [0.0; 16];
        arr.copy_from_slice(&self[offset..offset+16]);
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> f32 { self[idx] }
}

impl<'a> ParamSource8<f32> for &'a Vec<f32> {
    #[inline(always)]
    fn chunk8(&self, offset: usize) -> [f32; 8] {
        let mut arr = [0.0; 8];
        arr.copy_from_slice(&self[offset..offset+8]);
        arr
    }
    #[inline(always)]
    fn chunk16(&self, offset: usize) -> [f32; 16] {
        let mut arr = [0.0; 16];
        arr.copy_from_slice(&self[offset..offset+16]);
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> f32 { self[idx] }
}

impl ParamSource4<f64> for f64 {
    #[inline(always)]
    fn chunk4(&self, _: usize) -> [f64; 4] { [*self; 4] }
    #[inline(always)]
    fn chunk8(&self, _: usize) -> [f64; 8] { [*self; 8] }
    #[inline(always)]
    fn get(&self, _: usize) -> f64 { *self }
}

impl<'a> ParamSource4<f64> for &'a [f64] {
    #[inline(always)]
    fn chunk4(&self, offset: usize) -> [f64; 4] {
        let mut arr = [0.0; 4];
        arr.copy_from_slice(&self[offset..offset+4]);
        arr
    }
    #[inline(always)]
    fn chunk8(&self, offset: usize) -> [f64; 8] {
        let mut arr = [0.0; 8];
        arr.copy_from_slice(&self[offset..offset+8]);
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> f64 { self[idx] }
}

impl<'a> ParamSource4<f64> for &'a Vec<f64> {
    #[inline(always)]
    fn chunk4(&self, offset: usize) -> [f64; 4] {
        let mut arr = [0.0; 4];
        arr.copy_from_slice(&self[offset..offset+4]);
        arr
    }
    #[inline(always)]
    fn chunk8(&self, offset: usize) -> [f64; 8] {
        let mut arr = [0.0; 8];
        arr.copy_from_slice(&self[offset..offset+8]);
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> f64 { self[idx] }
}
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
/// let rv = rng.next_f32();
/// let in_range = rng.next_range_f32(10.0, 20.0);
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
/// rng.fill_f32(&mut buffer);
///
/// // Bulk generation with varying parameters (Columnar)
/// let mins = vec![0.0; 1024];
/// let maxs = vec![1.0; 1024];
/// rng.fill_range_f32(&mut buffer, &mins, &maxs);
///
/// // Hybrid usage: Constant mode with varying sigmas
/// let modes = 10.0;
/// let sigmas = vec![1.5; 1024];
/// rng.fill_sym_f32(&mut buffer, modes, &sigmas);
/// ```

#[derive(Clone, Copy, Debug)]
pub struct WyRand(u64);

impl From<u64> for WyRand {
    fn from(val: u64) -> Self {
        WyRand(val)
    }
}

impl WyRand {
    // Implementation of a variant of the Wyhash algorithm by "Vladimir Makarov"
    // This is obviously not cryptography grade
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x60bee2bee120fc15);

        // First mix
        let tmp = (self.0 as u128).wrapping_mul(0xa3b195354a39b70d);
        let m1 = ((tmp.wrapping_shr(64)) as u64) ^ (tmp as u64);

        // Second mix
        let tmp2 = (m1 as u128).wrapping_mul(0x1b03738712fad5c9);
        ((tmp2.wrapping_shr(64)) as u64) ^ (tmp2 as u64)
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64().wrapping_shr(32) as u32
    }

    #[inline(always)]
    pub fn next_f64(&mut self) -> f64 {
        let rv = self.next_u64();
        let bits = (rv >> 12) | 0x3FF0_0000_0000_0000;
        f64::from_bits(bits) - 1.0
    }

    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        let rv = self.next_u64();
        const MASK: u64 = u64::MAX ^ 0xFFFF_FFFF_F800_0000;
        f32::from_bits(((rv & MASK) | 0x3F80_0000) as u32) - 1.0
    }

    /// Generates a uniform f32 in the range [min, max].
    #[inline(always)]
    pub fn next_range_f32(&mut self, min: f32, max: f32) -> f32 {
        let rv = self.next_f32();
        rv.mul_add(max - min, min)
    }

    /// Generates a uniform f64 in the range [min, max].
    #[inline(always)]
    pub fn next_range_f64(&mut self, min: f64, max: f64) -> f64 {
        let rv = self.next_f64();
        rv.mul_add(max - min, min)
    }

    #[inline(always)]
    pub fn next_u64_in_range(&mut self, min: u64, max: u64) -> u64 {
        let rv: u128 = self.next_u64() as u128;
        let span: u128 = (max - min) as u128;
        min + (rv * span).wrapping_shr(64) as u64
    }

    #[inline(always)]
    pub fn next_usize_rv_in_range(&mut self, max: usize) -> usize {
        self.next_u64_in_range(0, max as u64) as usize
    }

    const TWO_PI_F32: f32 = 2.0 * std::f32::consts::PI;
    const TWO_PI_F64: f64 = 2.0 * std::f64::consts::PI;

    // Generates a standard normal random variable (mean=0, std_dev=1)
    // using the Box-Muller transform
    #[inline(always)]
    pub fn next_gaussian_f32(&mut self) -> f32 {
        // Box-Muller uses rvs in (0, 1]; subtracting a rv on [0, 1) from 1 gives an rv in (0, 1]
        let u1 = 1.0 - self.next_f32();
        let u2 = 1.0 - self.next_f32();
        let r = (-u1.approx_ln().fast_mul2()).approx_sqrt();
        //r * (Self::TWO_PI_F32 * u1).sin()
        r * (Self::TWO_PI_F32 * u2).approx_cos()
    }

    // Generates a standard normal random variable (mean=0, std_dev=1)
    // using the Box-Muller transform
    pub fn next_gaussian_f64(&mut self) -> f64 {
        // Box-Muller uses rvs in (0, 1]; subtracting a rv on [0, 1) from 1 gives an rv in (0, 1]
        let u1 = 1.0 - self.next_f64();
        let u2 = 1.0 - self.next_f64();
        let r = (-u1.approx_ln().fast_mul2()).approx_sqrt();
        //r * (Self::TWO_PI_F64 * u2).sin()
        r * (Self::TWO_PI_F64 * u2).approx_cos()
    }

    // Generates a pair of independent standard normal random variables
    // (mean=0, std_dev=1) using the Box-Muller transform
    pub fn next_gaussian_pair_f32(&mut self) -> (f32, f32) {
        let u1 = 1.0 - self.next_f32();
        let u2 = 1.0 - self.next_f32();
        let r = (-u1.approx_ln().fast_mul2()).approx_sqrt();
        let (s, c) = (Self::TWO_PI_F32 * u2).approx_sin_cos();
        (r * s, r * c)
    }

    // Generates a pair of independent standard normal random variables
    // (mean=0, std_dev=1) using the Box-Muller transform
    pub fn next_gaussian_pair_f64(&mut self) -> (f64, f64) {
        let u1 = 1.0 - self.next_f64();
        let u2 = 1.0 - self.next_f64();
        let r = (-u1.approx_ln().fast_mul2()).approx_sqrt();
        let (s, c) = (Self::TWO_PI_F64 * u2).approx_sin_cos();
        (r * s, r * c)
    }

    // Symmetric uncertainty: returns a value shifted by a Gaussian distribution.
    // Result = mean +/- (sigma * Gaussian)
    /// Generates a symmetric uncertainty value with the given mode and sigma.
    /// Result = mode + (sigma * Gaussian)
    #[inline(always)]
    pub fn next_sym_f32(&mut self, mode: f32, sigma: f32) -> f32 {
        self.next_gaussian_f32().mul_add(sigma, mode)
    }

    /// Generates a symmetric uncertainty value with the given mode and sigma (f64).
    #[inline(always)]
    pub fn next_sym_f64(&mut self, mode: f64, sigma: f64) -> f64 {
        self.next_gaussian_f64().mul_add(sigma, mode)
    }

    // Asymmetric uncertainty: uses a split-normal distribution. sigma_low_mag must be
    // an absolute value
    /// Generates an asymmetric uncertainty value.
    /// 
    /// If the internal Gaussian sample is negative, `sigma_low_mag` is used;
    /// otherwise, `sigma_high_mag` is used.
    pub fn next_asym_f32(
        &mut self,
        mode: f32,
        sigma_low_mag: f32,
        sigma_high_mag: f32,
    ) -> f32 {
        let z = self.next_gaussian_f32();
        let zlt_mask: u32 = ((z < 0.0) as u32).wrapping_neg();
        let zgeq_mask: u32 = ((z >= 0.0) as u32).wrapping_neg();

        let sigma = (sigma_low_mag.to_bits() & zlt_mask) | (sigma_high_mag.to_bits() & zgeq_mask);
        z.mul_add(f32::from_bits(sigma), mode)
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_asym_f64(
        &mut self,
        mode: f64,
        sigma_low_mag: f64,
        sigma_high_mag: f64,
    ) -> f64 {
        let z = self.next_gaussian_f64();
        let zlt_mask: u64 = ((z < 0.0) as u64).wrapping_neg();
        let zgeq_mask: u64 = ((z >= 0.0) as u64).wrapping_neg();

        let sigma = (sigma_low_mag.to_bits() & zlt_mask) | (sigma_high_mag.to_bits() & zgeq_mask);
        z.mul_add(f64::from_bits(sigma), mode)
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_sym_clamped_f32(
        &mut self,
        mode: f32,
        sigma: f32,
        limit: f32,
    ) -> f32 {
        let z = self.next_gaussian_f32().clamp(-limit, limit);
        z.mul_add(sigma, mode)
    }

    pub fn next_sym_clamped_f64(
        &mut self,
        mode: f64,
        sigma: f64,
        limit: f64,
    ) -> f64 {
        let z = self.next_gaussian_f64().clamp(-limit, limit);
        z.mul_add(sigma, mode)
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    /// Generates a log-normal symmetric distribution sample (f32).
    /// This uses approximate math functions.
    pub fn next_ln_sym_f32(&mut self, ln_mode: f32, sigma_ln: f32) -> f32 {
        let exponent = self.next_sym_f32(ln_mode, sigma_ln);
        exponent.exp()
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_ln_sym_f64(&mut self, ln_mode: f64, sigma_ln: f64) -> f64 {
        let exponent = self.next_sym_f64(ln_mode, sigma_ln);
        exponent.exp()
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_log_sym_f32(&mut self, log_mode: f32, sigma_log: f32) -> f32 {
        let exponent = self.next_sym_f32(log_mode, sigma_log);
        10.0_f32.powf(exponent)
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_log_sym_f64(&mut self, log_mode: f64, sigma_log: f64) -> f64 {
        let exponent = self.next_sym_f64(log_mode, sigma_log);
        10.0_f64.powf(exponent)
    }

    #[inline(always)]
    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_rayleigh_f32(&mut self, sigma: f32) -> f32 {
        let u = 1.0 - self.next_f32();
        let r = (-u.approx_ln().fast_mul2()).approx_sqrt();
        r * sigma
    }

    #[inline(always)]
    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_rayleigh_f64(&mut self, sigma: f64) -> f64 {
        let u = 1.0 - self.next_f64();
        let r = (-u.approx_ln().fast_mul2()).approx_sqrt();
        r * sigma
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_gamma_f32(&mut self, alpha: f32) -> f32 {
        if alpha <= 0.0 {
            return 0.0;
        }
        if alpha < 1.0 {
            let u1 = 1.0 - self.next_f32();
            let gamma = self.next_gamma_f32(alpha + 1.0);
            return gamma * (u1.approx_ln() / alpha).approx_exp();
        }

        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).approx_sqrt();
        loop {
            let z = self.next_gaussian_f32();
            let v = 1.0 + c * z;
            if v <= 0.0 {
                continue;
            }
            let v = v.approx_powi(3);

            let u = 1.0 - self.next_f32();
            let z_sq = z * z;

            if u < 1.0 - 0.0331 * z_sq * z_sq {
                return d * v;
            }
            if u.approx_ln() < 0.5 * z_sq + d * (1.0 - v + v.approx_ln()) {
                return d * v;
            }
        }
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_gamma_f64(&mut self, alpha: f64) -> f64 {
        if alpha <= 0.0 {
            return 0.0;
        }
        if alpha < 1.0 {
            let u1 = 1.0 - self.next_f64();
            let gamma = self.next_gamma_f64(alpha + 1.0);
            return gamma * (u1.approx_ln() / alpha).approx_exp();
        }

        let d = alpha - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).approx_sqrt();
        loop {
            let z = self.next_gaussian_f64();
            let v = 1.0 + c * z;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;

            let u = 1.0 - self.next_f64();
            let z_sq = z * z;

            if u < 1.0 - 0.0331 * z_sq * z_sq {
                return d * v;
            }
            if u.approx_ln() < 0.5 * z_sq + d * (1.0 - v + v.approx_ln()) {
                return d * v;
            }
        }
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_beta_f32(&mut self, alpha: f32, beta: f32) -> f32 {
        let gamma_a = self.next_gamma_f32(alpha);
        let gamma_b = self.next_gamma_f32(beta);
        let sum = gamma_a + gamma_b;
        let sumz = ((sum != 0.0) as u32).wrapping_neg();
        f32::from_bits((gamma_a / sum).to_bits() & sumz)
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_beta_f64(&mut self, alpha: f64, beta: f64) -> f64 {
        let gamma_a = self.next_gamma_f64(alpha);
        let gamma_b = self.next_gamma_f64(beta);
        let sum = gamma_a + gamma_b;
        let sumz = ((sum != 0.0) as u64).wrapping_neg();
        f64::from_bits((gamma_a / sum).to_bits() & sumz)
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_chi_squared_f32(&mut self, k: f32) -> f32 {
        self.next_gamma_f32(k * 0.5).fast_mul2()
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_chi_squared_f64(&mut self, k: f64) -> f64 {
        self.next_gamma_f64(k * 0.5).fast_mul2()
    }

    // --- BULK GENERATORS ---

    #[inline(always)]
    pub fn fill_f32(&mut self, buf: &mut [f32]) {
        for val in buf.iter_mut() {
            *val = self.next_f32();
        }
    }

    #[inline(always)]
    pub fn fill_f64(&mut self, buf: &mut [f64]) {
        for val in buf.iter_mut() {
            *val = self.next_f64();
        }
    }

    /// Fills a buffer with uniform f32 values in the range [min, max].
    /// 
    /// Parameters `min` and `max` can be either scalars (broadcasting to the 
    /// entire buffer) or slices/vectors (providing per-element values).
    pub fn fill_range_f32<MIN, MAX>(&mut self, buf: &mut [f32], min: MIN, max: MAX)
    where
        MIN: ParamSource8<f32>,
        MAX: ParamSource8<f32>,
    {
        let total_len = buf.len();
        let mut iter = buf.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i * 8;
            let mut arr = [0.0; 8];
            for j in 0..8 { arr[j] = self.next_f32(); }
            
            let min_arr = min.chunk8(offset);
            let max_arr = max.chunk8(offset);
            let mut diff_arr = [0.0; 8];
            for j in 0..8 { diff_arr[j] = max_arr[j] - min_arr[j]; }
            
            let res = fptricks::batch_fmadd_cols_f32(arr, diff_arr, min_arr);
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = (total_len / 8) * 8;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_range_f32(min.get(offset + i), max.get(offset + i));
        }
    }

    pub fn fill_range_f64<MIN, MAX>(&mut self, buf: &mut [f64], min: MIN, max: MAX)
    where
        MIN: ParamSource4<f64>,
        MAX: ParamSource4<f64>,
    {
        let total_len = buf.len();
        let mut iter = buf.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i * 4;
            let mut arr = [0.0; 4];
            for j in 0..4 { arr[j] = self.next_f64(); }
            
            let min_arr = min.chunk4(offset);
            let max_arr = max.chunk4(offset);
            let mut diff_arr = [0.0; 4];
            for j in 0..4 { diff_arr[j] = max_arr[j] - min_arr[j]; }
            
            let res = fptricks::batch_fmadd_cols_f64(arr, diff_arr, min_arr);
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = (total_len / 4) * 4;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_range_f64(min.get(offset + i), max.get(offset + i));
        }
    }


    // Generates Box-Muller normal variables in bulk using SIMD batches.
    // Extremely fast: processes 8 sines and 8 cosines per 16 iterations.
    pub fn fill_gaussian_f32(&mut self, buf: &mut [f32]) {
        let mut iter = buf.chunks_exact_mut(16);
        for chunk in iter.by_ref() {
            let mut u1 = [0.0; 8];
            let mut u2 = [0.0; 8];
            for j in 0..8 {
                u1[j] = 1.0 - self.next_f32();
                u2[j] = 1.0 - self.next_f32();
            }
            let batch_ln = fptricks::batch_approx_ln_f32(u1);
            let r_input = fptricks::batch_fmadd_f32(batch_ln, -2.0, 0.0);
            let r = fptricks::batch_approx_sqrt_f32(r_input);

            let u2_scaled = fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0);
            let (s, c) = fptricks::batch_approx_sin_cos_f32(u2_scaled);

            for j in 0..8 {
                chunk[j * 2] = r[j] * s[j];
                chunk[j * 2 + 1] = r[j] * c[j];
            }
        }
        
        let remainder = iter.into_remainder();
        let mut i = 0;
        while i + 1 < remainder.len() {
            let (s, c) = self.next_gaussian_pair_f32();
            remainder[i] = s;
            remainder[i + 1] = c;
            i += 2;
        }
        if i < remainder.len() {
            remainder[i] = self.next_gaussian_f32();
        }
    }

    pub fn fill_gaussian_f64(&mut self, buf: &mut [f64]) {
        let mut iter = buf.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let mut u1 = [0.0; 4];
            let mut u2 = [0.0; 4];
            for j in 0..4 {
                u1[j] = 1.0 - self.next_f64();
                u2[j] = 1.0 - self.next_f64();
            }
            let batch_ln = fptricks::batch_approx_ln_f64(u1);
            let r_input = fptricks::batch_fmadd_f64(batch_ln, -2.0, 0.0);
            let r = fptricks::batch_approx_sqrt_f64(r_input);

            let u2_scaled = fptricks::batch_fmadd_f64(u2, Self::TWO_PI_F64, 0.0);
            let (s, c) = fptricks::batch_approx_sin_cos_f64(u2_scaled);

            for j in 0..4 {
                chunk[j * 2] = r[j] * s[j];
                chunk[j * 2 + 1] = r[j] * c[j];
            }
        }
        
        let remainder = iter.into_remainder();
        let mut i = 0;
        while i + 1 < remainder.len() {
            let (s, c) = self.next_gaussian_pair_f64();
            remainder[i] = s;
            remainder[i + 1] = c;
            i += 2;
        }
        if i < remainder.len() {
            remainder[i] = self.next_gaussian_f64();
        }
    }
    /// Fills a buffer with symmetric uncertainty samples.
    /// 
    /// Supports hybrid parameter sources (e.g., constant mode, varying sigma).
    pub fn fill_sym_f32<M, S>(&mut self, buf: &mut [f32], mode: M, sigma: S)
    where
        M: ParamSource8<f32>,
        S: ParamSource8<f32>,
    {
        let total_len = buf.len();
        self.fill_gaussian_f32(buf);
        let mut iter = buf.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i * 8;
            let mut arr = [0.0; 8];
            arr.copy_from_slice(chunk);
            let res = fptricks::batch_fmadd_cols_f32(arr, sigma.chunk8(offset), mode.chunk8(offset));
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = (total_len / 8) * 8;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = val.mul_add(sigma.get(offset + i), mode.get(offset + i));
        }
    }

    pub fn fill_sym_f64<M, S>(&mut self, buf: &mut [f64], mode: M, sigma: S)
    where
        M: ParamSource4<f64>,
        S: ParamSource4<f64>,
    {
        let total_len = buf.len();
        self.fill_gaussian_f64(buf);
        let mut iter = buf.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i * 4;
            let mut arr = [0.0; 4];
            arr.copy_from_slice(chunk);
            let res = fptricks::batch_fmadd_cols_f64(arr, sigma.chunk4(offset), mode.chunk4(offset));
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = (total_len / 4) * 4;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = val.mul_add(sigma.get(offset + i), mode.get(offset + i));
        }
    }

    pub fn fill_sym_clamped_f32<M, S, L>(&mut self, buf: &mut [f32], mode: M, sigma: S, limit: L)
    where
        M: ParamSource8<f32>,
        S: ParamSource8<f32>,
        L: ParamSource8<f32>,
    {
        let total_len = buf.len();
        self.fill_gaussian_f32(buf);
        let mut iter = buf.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i * 8;
            let lim = limit.chunk8(offset);
            let mut arr = [0.0; 8];
            arr.copy_from_slice(chunk);
            for j in 0..8 {
                arr[j] = arr[j].clamp(-lim[j], lim[j]);
            }
            let res = fptricks::batch_fmadd_cols_f32(arr, sigma.chunk8(offset), mode.chunk8(offset));
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = (total_len / 8) * 8;
        for (i, val) in rem.iter_mut().enumerate() {
            let lim = limit.get(offset + i);
            *val = val.clamp(-lim, lim).mul_add(sigma.get(offset + i), mode.get(offset + i));
        }
    }

    pub fn fill_sym_clamped_f64<M, S, L>(&mut self, buf: &mut [f64], mode: M, sigma: S, limit: L)
    where
        M: ParamSource4<f64>,
        S: ParamSource4<f64>,
        L: ParamSource4<f64>,
    {
        let total_len = buf.len();
        self.fill_gaussian_f64(buf);
        let mut iter = buf.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i * 4;
            let lim = limit.chunk4(offset);
            let mut arr = [0.0; 4];
            arr.copy_from_slice(chunk);
            for j in 0..4 {
                arr[j] = arr[j].clamp(-lim[j], lim[j]);
            }
            let res = fptricks::batch_fmadd_cols_f64(arr, sigma.chunk4(offset), mode.chunk4(offset));
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = (total_len / 4) * 4;
        for (i, val) in rem.iter_mut().enumerate() {
            let lim = limit.get(offset + i);
            *val = val.clamp(-lim, lim).mul_add(sigma.get(offset + i), mode.get(offset + i));
        }
    }

    pub fn fill_asym_f32<M, SLO, SHI>(&mut self, buf: &mut [f32], mode: M, sigma_lo: SLO, sigma_hi: SHI)
    where
        M: ParamSource8<f32>,
        SLO: ParamSource8<f32>,
        SHI: ParamSource8<f32>,
    {
        let total_len = buf.len();
        self.fill_gaussian_f32(buf);
        let mut iter = buf.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i * 8;
            let mut arr = [0.0; 8];
            arr.copy_from_slice(chunk);
            let res = fptricks::batch_asymmetric_fma_cols_f32(arr, mode.chunk8(offset), 
                sigma_lo.chunk8(offset), sigma_hi.chunk8(offset));
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = (total_len / 8) * 8;
        for (i, val) in rem.iter_mut().enumerate() {
            let sig_lo = sigma_lo.get(offset + i);
            let sig_hi = sigma_hi.get(offset + i);
            let sig_lo_bits = sig_lo.to_bits();
            let sig_hi_bits = sig_hi.to_bits();
            let zlt_mask: u32 = ((*val < 0.0) as u32).wrapping_neg();
            let zgeq_mask: u32 = ((*val >= 0.0) as u32).wrapping_neg();
            let sigma = (sig_lo_bits & zlt_mask) | (sig_hi_bits & zgeq_mask);
            *val = val.mul_add(f32::from_bits(sigma), mode.get(offset + i));
        }
    }

    pub fn fill_asym_f64<M, SLO, SHI>(&mut self, buf: &mut [f64], mode: M, sigma_lo: SLO, sigma_hi: SHI)
    where
        M: ParamSource4<f64>,
        SLO: ParamSource4<f64>,
        SHI: ParamSource4<f64>,
    {
        let total_len = buf.len();
        self.fill_gaussian_f64(buf);
        let mut iter = buf.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i * 4;
            let mut arr = [0.0; 4];
            arr.copy_from_slice(chunk);
            let res = fptricks::batch_asymmetric_fma_cols_f64(arr, mode.chunk4(offset), 
                sigma_lo.chunk4(offset), sigma_hi.chunk4(offset));
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = (total_len / 4) * 4;
        for (i, val) in rem.iter_mut().enumerate() {
            let sig_lo = sigma_lo.get(offset + i);
            let sig_hi = sigma_hi.get(offset + i);
            let sig_lo_bits = sig_lo.to_bits();
            let sig_hi_bits = sig_hi.to_bits();
            let zlt_mask: u64 = ((*val < 0.0) as u64).wrapping_neg();
            let zgeq_mask: u64 = ((*val >= 0.0) as u64).wrapping_neg();
            let sigma = (sig_lo_bits & zlt_mask) | (sig_hi_bits & zgeq_mask);
            *val = val.mul_add(f64::from_bits(sigma), mode.get(offset + i));
        }
    }

    pub fn fill_ln_sym_f32<M, S>(&mut self, buf: &mut [f32], ln_mode: M, sigma_ln: S)
    where
        M: ParamSource8<f32>,
        S: ParamSource8<f32>,
    {
        self.fill_sym_f32(buf, ln_mode, sigma_ln);
        let mut iter = buf.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let mut arr = [0.0; 8];
            arr.copy_from_slice(chunk);
            let res = fptricks::batch_approx_exp_f32(arr);
            chunk.copy_from_slice(&res);
        }
        for val in iter.into_remainder() {
            *val = val.approx_exp();
        }
    }

    pub fn fill_ln_sym_f64<M, S>(&mut self, buf: &mut [f64], ln_mode: M, sigma_ln: S)
    where
        M: ParamSource4<f64>,
        S: ParamSource4<f64>,
    {
        self.fill_sym_f64(buf, ln_mode, sigma_ln);
        let mut iter = buf.chunks_exact_mut(4);
        for chunk in iter.by_ref() {
            let mut arr = [0.0; 4];
            arr.copy_from_slice(chunk);
            let res = fptricks::batch_approx_exp_f64(arr);
            chunk.copy_from_slice(&res);
        }
        for val in iter.into_remainder() {
            *val = val.approx_exp();
        }
    }

    pub fn fill_log_sym_f32<M, S>(&mut self, buf: &mut [f32], log_mode: M, sigma_log: S)
    where
        M: ParamSource8<f32>,
        S: ParamSource8<f32>,
    {
        self.fill_sym_f32(buf, log_mode, sigma_log);
        let mut iter = buf.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let mut arr = [0.0; 8];
            arr.copy_from_slice(chunk);
            let res = fptricks::batch_approx_powf_f32(10.0, arr);
            chunk.copy_from_slice(&res);
        }
        for val in iter.into_remainder() {
            *val = 10.0_f32.powf(*val);
        }
    }

    pub fn fill_log_sym_f64<M, S>(&mut self, buf: &mut [f64], log_mode: M, sigma_log: S)
    where
        M: ParamSource4<f64>,
        S: ParamSource4<f64>,
    {
        self.fill_sym_f64(buf, log_mode, sigma_log);
        let mut iter = buf.chunks_exact_mut(4);
        for chunk in iter.by_ref() {
            let mut arr = [0.0; 4];
            arr.copy_from_slice(chunk);
            let res = fptricks::batch_approx_powf_f64(10.0, arr);
            chunk.copy_from_slice(&res);
        }
        for val in iter.into_remainder() {
            *val = 10.0_f64.powf(*val);
        }
    }


    /// Fills a buffer with Rayleigh-distributed samples.
    /// 
    /// The `sigma` parameter can be a scalar or a column.
    pub fn fill_rayleigh_f32<S>(&mut self, buf: &mut [f32], sigma: S)
    where
        S: ParamSource8<f32>,
    {
        let total_len = buf.len();
        let mut iter = buf.chunks_exact_mut(16);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i * 16;
            let mut u1 = [0.0; 8];
            let mut u2 = [0.0; 8];
            for j in 0..8 {
                u1[j] = 1.0 - self.next_f32();
                u2[j] = 1.0 - self.next_f32();
            }
            let ln1 = fptricks::batch_approx_ln_f32(u1);
            let ln2 = fptricks::batch_approx_ln_f32(u2);
            
            let s_chunk = sigma.chunk16(offset);
            let mut neg_two_sigma_sq1 = [0.0; 8];
            let mut neg_two_sigma_sq2 = [0.0; 8];
            for j in 0..8 {
                let s1 = s_chunk[j];
                let s2 = s_chunk[j+8];
                neg_two_sigma_sq1[j] = -2.0 * s1 * s1;
                neg_two_sigma_sq2[j] = -2.0 * s2 * s2;
            }
            
            let r_in1 = fptricks::batch_fmadd_cols_f32(ln1, neg_two_sigma_sq1, [0.0; 8]);
            let r_in2 = fptricks::batch_fmadd_cols_f32(ln2, neg_two_sigma_sq2, [0.0; 8]);
            
            let r1 = fptricks::batch_approx_sqrt_f32(r_in1);
            let r2 = fptricks::batch_approx_sqrt_f32(r_in2);
            
            chunk[0..8].copy_from_slice(&r1);
            chunk[8..16].copy_from_slice(&r2);
        }
        let rem = iter.into_remainder();
        let offset = (total_len / 16) * 16;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_rayleigh_f32(sigma.get(offset + i));
        }
    }

    pub fn fill_rayleigh_f64<S>(&mut self, buf: &mut [f64], sigma: S)
    where
        S: ParamSource4<f64>,
    {
        let total_len = buf.len();
        let mut iter = buf.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i * 8;
            let mut u1 = [0.0; 4];
            let mut u2 = [0.0; 4];
            for j in 0..4 {
                u1[j] = 1.0 - self.next_f64();
                u2[j] = 1.0 - self.next_f64();
            }
            let ln1 = fptricks::batch_approx_ln_f64(u1);
            let ln2 = fptricks::batch_approx_ln_f64(u2);
            
            let s_chunk = sigma.chunk8(offset);
            let mut neg_two_sigma_sq1 = [0.0; 4];
            let mut neg_two_sigma_sq2 = [0.0; 4];
            for j in 0..4 {
                let s1 = s_chunk[j];
                let s2 = s_chunk[j+4];
                neg_two_sigma_sq1[j] = -2.0 * s1 * s1;
                neg_two_sigma_sq2[j] = -2.0 * s2 * s2;
            }
            
            let r_in1 = fptricks::batch_fmadd_cols_f64(ln1, neg_two_sigma_sq1, [0.0; 4]);
            let r_in2 = fptricks::batch_fmadd_cols_f64(ln2, neg_two_sigma_sq2, [0.0; 4]);
            
            let r1 = fptricks::batch_approx_sqrt_f64(r_in1);
            let r2 = fptricks::batch_approx_sqrt_f64(r_in2);
            
            chunk[0..4].copy_from_slice(&r1);
            chunk[4..8].copy_from_slice(&r2);
        }
        let rem = iter.into_remainder();
        let offset = (total_len / 8) * 8;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_rayleigh_f64(sigma.get(offset + i));
        }
    }

    pub fn fill_gamma_f32<A>(&mut self, buf: &mut [f32], alpha: A)
    where
        A: ParamSource8<f32>,
    {
        for (i, val) in buf.iter_mut().enumerate() {
            *val = self.next_gamma_f32(alpha.get(i));
        }
    }

    pub fn fill_gamma_f64<A>(&mut self, buf: &mut [f64], alpha: A)
    where
        A: ParamSource4<f64>,
    {
        for (i, val) in buf.iter_mut().enumerate() {
            *val = self.next_gamma_f64(alpha.get(i));
        }
    }

    /// Fills a buffer with Beta-distributed samples.
    pub fn fill_beta_f32<A, B>(&mut self, buf: &mut [f32], alpha: A, beta: B)
    where
        A: ParamSource8<f32>,
        B: ParamSource8<f32>,
    {
        for (i, val) in buf.iter_mut().enumerate() {
            *val = self.next_beta_f32(alpha.get(i), beta.get(i));
        }
    }

    pub fn fill_beta_f64<A, B>(&mut self, buf: &mut [f64], alpha: A, beta: B)
    where
        A: ParamSource4<f64>,
        B: ParamSource4<f64>,
    {
        for (i, val) in buf.iter_mut().enumerate() {
            *val = self.next_beta_f64(alpha.get(i), beta.get(i));
        }
    }

    pub fn fill_chi_squared_f32<K>(&mut self, buf: &mut [f32], k: K)
    where
        K: ParamSource8<f32>,
    {
        for (i, val) in buf.iter_mut().enumerate() {
            *val = self.next_chi_squared_f32(k.get(i));
        }
    }

    pub fn fill_chi_squared_f64<K>(&mut self, buf: &mut [f64], k: K)
    where
        K: ParamSource4<f64>,
    {
        for (i, val) in buf.iter_mut().enumerate() {
            *val = self.next_chi_squared_f64(k.get(i));
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_rand_f32() {
        let mut rng = WyRand::new(1);
        for _ in 0..128 {
            let rv = rng.next_range_f32(1.0, 12.5);
            assert!(rv >= 1.0 && rv <= 12.5);
        }
    }
    #[test]
    fn test_rand_f64() {
        let mut rng = WyRand::new(1);
        for _ in 0..128 {
            let rv = rng.next_f64();
            assert!(rv >= 0.0 && rv <= 1.0);
        }
    }

    fn calculate_stats<F>(mut generator: F, n: usize) -> (f64, f64)
    where
        F: FnMut() -> f64,
    {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let val = generator();
            sum += val;
            sum_sq += val * val;
        }
        let mean = sum / n as f64;
        let variance = sum_sq / n as f64 - mean * mean;
        (mean, variance)
    }

    #[test]
    fn test_gaussian_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let (mean, var) = calculate_stats(|| rng.next_gaussian_f32() as f64, n);
        println!("Gaussian f32 | mean: {}, var: {}", mean, var);
        assert!(mean.abs() < 0.1);
        assert!((var - 1.0).abs() < 0.15);

        let (mean, var) = calculate_stats(|| rng.next_gaussian_f64(), n);
        println!("Gaussian f64 | mean: {}, var: {}", mean, var);
        assert!(mean.abs() < 0.1);
        assert!((var - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_gaussian_pair_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;

        let mut sum_s = 0.0;
        let mut sum_c = 0.0;
        let mut sum_sq_s = 0.0;
        let mut sum_sq_c = 0.0;

        for _ in 0..n {
            let (s, c) = rng.next_gaussian_pair_f32();
            sum_s += s as f64;
            sum_c += c as f64;
            sum_sq_s += (s * s) as f64;
            sum_sq_c += (c * c) as f64;
        }

        let mean_s = sum_s / n as f64;
        let var_s = sum_sq_s / n as f64 - mean_s * mean_s;
        let mean_c = sum_c / n as f64;
        let var_c = sum_sq_c / n as f64 - mean_c * mean_c;

        println!("Gaussian Pair f32 | mean_s: {}, var_s: {}, mean_c: {}, var_c: {}", mean_s, var_s, mean_c, var_c);
        assert!(mean_s.abs() < 0.1);
        assert!((var_s - 1.0).abs() < 0.15);
        assert!(mean_c.abs() < 0.1);
        assert!((var_c - 1.0).abs() < 0.15);

        let mut sum_s = 0.0;
        let mut sum_c = 0.0;
        let mut sum_sq_s = 0.0;
        let mut sum_sq_c = 0.0;

        for _ in 0..n {
            let (s, c) = rng.next_gaussian_pair_f64();
            sum_s += s;
            sum_c += c;
            sum_sq_s += s * s;
            sum_sq_c += c * c;
        }

        let mean_s = sum_s / n as f64;
        let var_s = sum_sq_s / n as f64 - mean_s * mean_s;
        let mean_c = sum_c / n as f64;
        let var_c = sum_sq_c / n as f64 - mean_c * mean_c;

        println!("Gaussian Pair f64 | mean_s: {}, var_s: {}, mean_c: {}, var_c: {}", mean_s, var_s, mean_c, var_c);
        assert!(mean_s.abs() < 0.1);
        assert!((var_s - 1.0).abs() < 0.1);
        assert!(mean_c.abs() < 0.1);
        assert!((var_c - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_gamma_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let alphas = [0.5, 1.0, 2.0, 5.0];

        for &alpha in &alphas {
            let (mean, var) = calculate_stats(|| rng.next_gamma_f32(alpha as f32) as f64, n);
            println!("Gamma f32 (alpha={}) | mean: {}, var: {}", alpha, mean, var);
            assert!((mean - alpha).abs() < alpha * 0.15);
            assert!((var - alpha).abs() < alpha * 0.15);

            let (mean, var) = calculate_stats(|| rng.next_gamma_f64(alpha), n);
            println!("Gamma f64 (alpha={}) | mean: {}, var: {}", alpha, mean, var);
            assert!((mean - alpha).abs() < alpha * 0.1);
            assert!((var - alpha).abs() < alpha * 0.1);
        }
    }

    #[test]
    fn test_beta_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let pairs = [(0.5, 0.5), (1.0, 3.0), (2.0, 2.0)];

        for &(a, b) in &pairs {
            let expected_mean = a / (a + b);
            let expected_var = (a * b) / ((a + b) * (a + b) * (a + b + 1.0));

            let (mean, var) = calculate_stats(|| rng.next_beta_f32(a as f32, b as f32) as f64, n);
            println!("Beta f32 (a={}, b={}) | mean: {}, var: {}", a, b, mean, var);
            assert!((mean - expected_mean).abs() < 0.1);
            assert!((var - expected_var).abs() < 0.1);

            let (mean, var) = calculate_stats(|| rng.next_beta_f64(a, b), n);
            println!("Beta f64 (a={}, b={}) | mean: {}, var: {}", a, b, mean, var);
            assert!((mean - expected_mean).abs() < 0.05);
            assert!((var - expected_var).abs() < 0.05);
        }
    }

    #[test]
    fn test_rayleigh_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let sigma = 2.0;
        let pi = std::f64::consts::PI;
        let expected_mean = sigma * (pi / 2.0).sqrt();
        let expected_var = (4.0 - pi) / 2.0 * sigma * sigma;

        let (mean, var) = calculate_stats(|| rng.next_rayleigh_f32(sigma as f32) as f64, n);
        println!("Rayleigh f32 | mean: {}, var: {}", mean, var);
        assert!((mean - expected_mean).abs() < 0.15);
        assert!((var - expected_var).abs() < 0.15);

        let (mean, var) = calculate_stats(|| rng.next_rayleigh_f64(sigma), n);
        println!("Rayleigh f64 | mean: {}, var: {}", mean, var);
        assert!((mean - expected_mean).abs() < 0.1);
        assert!((var - expected_var).abs() < 0.1);
    }

    #[test]
    fn test_batch_fill_gaussian() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let mut buf = vec![0.0f32; n];
        rng.fill_gaussian_f32(&mut buf);
        
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for &val in &buf {
            sum += val as f64;
            sum_sq += (val * val) as f64;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        println!("Bulk Gaussian f32 | mean: {}, var: {}", mean, var);
        assert!(mean.abs() < 0.1);
        assert!((var - 1.0).abs() < 0.15);
    }

    #[test]
    fn test_batch_fill_rayleigh() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let sigma = 2.0;
        let mut buf = vec![0.0f32; n];
        rng.fill_rayleigh_f32(&mut buf, sigma);
        
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for &val in &buf {
            sum += val as f64;
            sum_sq += (val * val) as f64;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        
        let pi = std::f64::consts::PI;
        let expected_mean = (sigma as f64) * (pi / 2.0).sqrt();
        let expected_var = (4.0 - pi) / 2.0 * (sigma as f64) * (sigma as f64);
        
        println!("Bulk Rayleigh f32 | mean: {}, var: {}", mean, var);
        assert!((mean - expected_mean).abs() < 0.15);
        assert!((var - expected_var).abs() < 0.15);
    }

    #[test]
    fn test_cols_fill() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let mut buf = vec![0.0f32; n];
        let mut modes = vec![5.0f32; n];
        let mut sigmas = vec![2.0f32; n];
        
        // Vary half of them to make it interesting
        for i in 0..n/2 {
            modes[i] = 10.0;
            sigmas[i] = 1.0;
        }
        
        rng.fill_sym_f32(&mut buf, &modes, &sigmas);
        
        // Check first half (mode 10, sigma 1)
        let mut sum1 = 0.0;
        let mut sum_sq1 = 0.0;
        for i in 0..n/2 {
            let val = buf[i] as f64;
            sum1 += val;
            sum_sq1 += val * val;
        }
        let mean1 = sum1 / (n/2) as f64;
        let var1 = sum_sq1 / (n/2) as f64 - mean1 * mean1;
        assert!((mean1 - 10.0).abs() < 0.1);
        assert!((var1 - 1.0).abs() < 0.15);
        
        // Check second half (mode 5, sigma 2)
        let mut sum2 = 0.0;
        let mut sum_sq2 = 0.0;
        for i in n/2..n {
            let val = buf[i] as f64;
            sum2 += val;
            sum_sq2 += val * val;
        }
        let mean2 = sum2 / (n/2) as f64;
        let var2 = sum_sq2 / (n/2) as f64 - mean2 * mean2;
        assert!((mean2 - 5.0).abs() < 0.1);
        assert!((var2 - 4.0).abs() < 0.2);
    }

    #[test]
    fn test_asymmetric_cols_fill() {
        let mut rng = WyRand::new(1);
        let n = 1000;
        let mut buf = vec![0.0f32; n];
        let mut modes = vec![0.0; n];
        let mut slounds = vec![1.0; n];
        let mut shounds = vec![2.0; n];
        
        rng.fill_asym_f32(&mut buf, &modes, &slounds, &shounds);
        
        // Asymmetric gaussian with mode 0, sigma_lo 1, sigma_hi 2
        // Mean should be approx (2-1) / sqrt(2*pi) = 0.398
        let mut sum = 0.0;
        for &val in &buf { sum += val as f64; }
        let mean = sum / n as f64;
        println!("Asymmetric Cols Mean: {}", mean);
        assert!((mean - 0.398).abs() < 0.2);
    }

    #[test]
    fn test_range_cols_fill() {
        let mut rng = WyRand::new(1);
        let n = 1000;
        let mut buf = vec![0.0f32; n];
        let mut mins = vec![10.0; n];
        let mut maxs = vec![20.0; n];
        
        rng.fill_range_f32(&mut buf, &mins, &maxs);
        for &val in &buf {
            assert!(val >= 10.0 && val <= 20.0);
        }
    }

    #[test]
    fn test_rayleigh_cols_fill() {
        let mut rng = WyRand::new(1);
        let n = 1000;
        let mut buf = vec![0.0f32; n];
        let mut sigmas = vec![2.0; n];
        
        rng.fill_rayleigh_f32(&mut buf, &sigmas);
        let mut sum = 0.0;
        for &val in &buf { sum += val as f64; }
        let mean = sum / n as f64;
        let pi = std::f64::consts::PI;
        let expected_mean = 2.0 * (pi / 2.0).sqrt();
        assert!((mean - expected_mean).abs() < 0.2);
    }
}
