#![allow(unused)]
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
        //TODO: Replace this modulus division
        (self.next_u64() as usize) % max
    }

    // Generates a standard normal random variable (mean=0, std_dev=1)
    // using the Box-Muller transform
    pub fn next_gaussian_f32(&mut self) -> f32 {
        // Box-Muller uses rvs in (0, 1]; subtracting a rv on [0, 1) from 1 gives an rv in (0, 1]
        let u1 = 1.0 - self.next_f32();
        let u2 = 1.0 - self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        //r * (2.0 * std::f32::consts::PI * u1).sin()
        r * (2.0 * std::f32::consts::PI * u2).cos()
    }

    // Generates a standard normal random variable (mean=0, std_dev=1)
    // using the Box-Muller transform
    pub fn next_gaussian_f64(&mut self) -> f64 {
        // Box-Muller uses rvs in (0, 1]; subtracting a rv on [0, 1) from 1 gives an rv in (0, 1]
        let u1 = 1.0 - self.next_f64();
        let u2 = 1.0 - self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        //r * (2.0 * std::f64::consts::PI * u1).sin()
        r * (2.0 * std::f64::consts::PI * u2).cos()
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
    pub fn next_asymmetric_uncertainty_f32(&mut self, mode: f32, sigma_low_mag: f32, sigma_high_mag: f32) -> f32 {
        let z = self.next_gaussian_f32();
        if z < 0.0 {
            z.mul_add(sigma_low_mag, mode)
        } else {
            z.mul_add(sigma_high_mag, mode)
        }
    }

    pub fn next_asymmetric_uncertainty_f64(&mut self, mode: f64, sigma_low_mag: f64, sigma_high_mag: f64) -> f64 {
        let z = self.next_gaussian_f64();
        if z < 0.0 {
            z.mul_add(sigma_low_mag, mode)
        } else {
            z.mul_add(sigma_high_mag, mode)
        }
    }

    // Clamped Gaussian
    pub fn next_clamped_symmetric_uncertainty_f32(&mut self, mode: f32, sigma: f32, limit: f32) -> f32 {
        let z = self.next_gaussian_f32().clamp(-limit, limit);
        z.mul_add(sigma, mode)
    }

    pub fn next_clamped_symmetric_uncertainty_f64(&mut self, mode: f64, sigma: f64, limit: f64) -> f64 {
        let z = self.next_gaussian_f64().clamp(-limit, limit);
        z.mul_add(sigma, mode)
    }

    pub fn next_ln_symmetric_f32(&mut self, ln_mode: f32, sigma_ln: f32) -> f32 {
        let exponent = self.next_symmetric_uncertainty_f32(ln_mode, sigma_ln);
        exponent.exp()
    }

    pub fn next_ln_symmetric_f64(&mut self, ln_mode: f64, sigma_ln: f64) -> f64 {
        let exponent = self.next_symmetric_uncertainty_f64(ln_mode, sigma_ln);
        exponent.exp()
    }
    
    
    pub fn next_log_symmetric_f32(&mut self, log_mode: f32, sigma_log: f32) -> f32 {
        let exponent = self.next_symmetric_uncertainty_f32(log_mode, sigma_log);
        10.0_f32.powf(exponent)
    }

    pub fn next_log_symmetric_f64(&mut self, log_mode: f64, sigma_log: f64) -> f64 {
        let exponent = self.next_symmetric_uncertainty_f64(log_mode, sigma_log);
        10.0_f64.powf(exponent)
    }

}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_rand_f32() {
        println!("Running rand f32 test");
        let mut rng = WyRand::new(1);
        for _ in 0..128 {
            let rv = rng.next_f32_in_range(1.0, 12.5);
            println!("{}", rv);
        }
    }
    #[test]
    fn test_rand_f64() {
        println!("Running rand f64 test");
        let mut rng = WyRand::new(1);
        for _ in 0..128 {
            let rv = rng.next_f64();
            println!("{}", rv);
        }
    }
}
