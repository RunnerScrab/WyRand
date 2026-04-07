///This library uses approximate math functions for procedural generation, and is not
///appropriate when high accuracy or cryptographically secure random numbers are required.
///
///# Examples
///
///```
///use wyrand::WyRand;
///
///let mut rng = WyRand::new(42);
///let rv = rng.next_f32();
///```
#![allow(unused)]
use fptricks::*;

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

    #[inline(always)]
    pub fn next_f32_in_range(&mut self, min: f32, max: f32) -> f32 {
        let rv = self.next_f32();
        rv.mul_add(max - min, min)
    }

    #[inline(always)]
    pub fn next_f64_in_range(&mut self, min: f64, max: f64) -> f64 {
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
        let rv: u64 = self.next_u64();
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
        //r * (Self::TWO_PI_F64 * u1).sin()
        r * (Self::TWO_PI_F64 * u2).approx_cos()
    }

    // Symmetric uncertainty: returns a value shifted by a Gaussian distribution.
    // Result = mean +/- (sigma * Gaussian)
    #[inline(always)]
    pub fn next_symmetric_uncertainty_f32(&mut self, mode: f32, sigma: f32) -> f32 {
        self.next_gaussian_f32().mul_add(sigma, mode)
    }

    #[inline(always)]
    pub fn next_symmetric_uncertainty_f64(&mut self, mode: f64, sigma: f64) -> f64 {
        self.next_gaussian_f64().mul_add(sigma, mode)
    }

    // Asymmetric uncertainty: uses a split-normal distribution. sigma_low_mag must be
    // an absolute value
    pub fn next_asymmetric_uncertainty_f32(
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
    pub fn next_asymmetric_uncertainty_f64(
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
    pub fn next_clamped_symmetric_uncertainty_f32(
        &mut self,
        mode: f32,
        sigma: f32,
        limit: f32,
    ) -> f32 {
        let z = self.next_gaussian_f32().clamp(-limit, limit);
        z.mul_add(sigma, mode)
    }

    pub fn next_clamped_symmetric_uncertainty_f64(
        &mut self,
        mode: f64,
        sigma: f64,
        limit: f64,
    ) -> f64 {
        let z = self.next_gaussian_f64().clamp(-limit, limit);
        z.mul_add(sigma, mode)
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_ln_symmetric_f32(&mut self, ln_mode: f32, sigma_ln: f32) -> f32 {
        let exponent = self.next_symmetric_uncertainty_f32(ln_mode, sigma_ln);
        exponent.exp()
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_ln_symmetric_f64(&mut self, ln_mode: f64, sigma_ln: f64) -> f64 {
        let exponent = self.next_symmetric_uncertainty_f64(ln_mode, sigma_ln);
        exponent.exp()
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_log_symmetric_f32(&mut self, log_mode: f32, sigma_log: f32) -> f32 {
        let exponent = self.next_symmetric_uncertainty_f32(log_mode, sigma_log);
        10.0_f32.powf(exponent)
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_log_symmetric_f64(&mut self, log_mode: f64, sigma_log: f64) -> f64 {
        let exponent = self.next_symmetric_uncertainty_f64(log_mode, sigma_log);
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
        let sumz = ((sum == 0.0) as u32).wrapping_neg();
        f32::from_bits((gamma_a / sum).to_bits() & sumz)
    }

    ///This uses approximate math functions so is not appropriate when high accuracy is required
    pub fn next_beta_f64(&mut self, alpha: f64, beta: f64) -> f64 {
        let gamma_a = self.next_gamma_f64(alpha);
        let gamma_b = self.next_gamma_f64(beta);
        let sum = gamma_a + gamma_b;
        let sumz = ((sum == 0.0) as u64).wrapping_neg();
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
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_rand_f32() {
        let mut rng = WyRand::new(1);
        for _ in 0..128 {
            let rv = rng.next_f32_in_range(1.0, 12.5);
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
}
