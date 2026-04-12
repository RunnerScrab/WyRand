use crate::WyRand;
use crate::traits::ParamSource;
use fptricks::*;

impl WyRand {
    #[inline(always)]
    pub fn next_power_law(&mut self, min: f64, max: f64, k: f64) -> f64 {
        let roll = self.next_uniform_f64();
        let kp1 = k + 1.0;
        let kp1lt: u64 = ((kp1.abs() < 1e-9) as u64).wrapping_neg();
        f64::from_bits(((min * (max/min).approx_powf(roll)).to_bits() & kp1lt) | (!kp1lt & 
        {
            let min_pow = min.approx_powf(kp1);
            let max_pow = max.approx_powf(kp1);
            (roll * (max_pow - min_pow) + min_pow).approx_powf(1.0 / kp1)
        }.to_bits()))
    }

    #[inline(always)]
    pub fn next_rayleigh_f32(&mut self, sigma: f32) -> f32 {
        let u = 1.0 - self.next_uniform_f32();
        let r = (-u.approx_ln().fast_mul2()).approx_sqrt();
        r * sigma
    }

    #[inline(always)]
    pub fn next_rayleigh_f64(&mut self, sigma: f64) -> f64 {
        let u = 1.0 - self.next_uniform_f64();
        let r = (-u.approx_ln().fast_mul2()).approx_sqrt();
        r * sigma
    }

    #[inline(always)]
    pub fn next_gamma_f32(&mut self, alpha: f32) -> f32 {
        if alpha <= 0.0 {
            return 0.0;
        }
        if alpha < 1.0 {
            let u1 = 1.0 - self.next_uniform_f32();
            let gamma = self.next_gamma_f32(alpha + 1.0);
            return gamma * (u1.approx_ln() / alpha).approx_exp();
        }
        const ONE_THIRD: f32 = 1.0/3.0;
        let d = alpha - ONE_THIRD;
        let c = 1.0 / (9.0 * d).approx_sqrt();
        loop {
            let z = self.next_std_normal_f32();
            let v = 1.0 + c * z;
            if v <= 0.0 {
                continue;
            }
            let v = v.approx_powi(3);

            let u = 1.0 - self.next_uniform_f32();
            let z_sq = z * z;

            if u < 1.0 - 0.0331 * z_sq * z_sq {
                return d * v;
            }
            if u.approx_ln() < 0.5 * z_sq + d * (1.0 - v + v.approx_ln()) {
                return d * v;
            }
        }
    }

    #[inline(always)]
    pub fn next_gamma_f64(&mut self, alpha: f64) -> f64 {
        if alpha <= 0.0 {
            return 0.0;
        }
        if alpha < 1.0 {
            let u1 = 1.0 - self.next_uniform_f64();
            let gamma = self.next_gamma_f64(alpha + 1.0);
            return gamma * (u1.approx_ln() / alpha).approx_exp();
        }

        const ONE_THIRD: f64 = 1.0 / 3.0;
        let d = alpha - ONE_THIRD;
        let c = 1.0 / (9.0 * d).approx_sqrt();
        loop {
            let z = self.next_std_normal_f64();
            let v = 1.0 + c * z;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;

            let u = 1.0 - self.next_uniform_f64();
            let z_sq = z * z;

            if u < 1.0 - 0.0331 * z_sq * z_sq {
                return d * v;
            }
            if u.approx_ln() < 0.5 * z_sq + d * (1.0 - v + v.approx_ln()) {
                return d * v;
            }
        }
    }

    #[inline(always)]
    pub fn next_beta_f32(&mut self, alpha: f32, beta: f32) -> f32 {
        let gamma_a = self.next_gamma_f32(alpha);
        let gamma_b = self.next_gamma_f32(beta);
        let sum = gamma_a + gamma_b;
        let sumz = ((sum != 0.0) as u32).wrapping_neg();
        f32::from_bits((gamma_a / sum).to_bits() & sumz)
    }

    #[inline(always)]
    pub fn next_beta_f64(&mut self, alpha: f64, beta: f64) -> f64 {
        let gamma_a = self.next_gamma_f64(alpha);
        let gamma_b = self.next_gamma_f64(beta);
        let sum = gamma_a + gamma_b;
        let sumz = ((sum != 0.0) as u64).wrapping_neg();
        f64::from_bits((gamma_a / sum).to_bits() & sumz)
    }

    #[inline(always)]
    pub fn next_chi_squared_f32(&mut self, k: f32) -> f32 {
        self.next_gamma_f32(k * 0.5).fast_mul2()
    }

    #[inline(always)]
    pub fn next_chi_squared_f64(&mut self, k: f64) -> f64 {
        self.next_gamma_f64(k * 0.5).fast_mul2()
    }

    #[inline(always)]
    pub fn next_poisson_u32(&mut self, lambda: f32) -> u32 {
        if lambda <= 0.0 { return 0; }
        let l = (-lambda).approx_exp();
        let mut k = 0u32;
        let mut p = 1.0_f32;
        loop {
            k += 1;
            p *= self.next_uniform_f32();
            if p <= l {
                break;
            }
        }
        k - 1
    }

    #[inline(always)]
    pub fn next_poisson_f64_u32(&mut self, lambda: f64) -> u32 {
        if lambda <= 0.0 { return 0; }
        let l = (-lambda).approx_exp();
        let mut k = 0u32;
        let mut p = 1.0_f64;
        loop {
            k += 1;
            p *= self.next_uniform_f64();
            if p <= l {
                break;
            }
        }
        k - 1
    }

    #[inline]
    pub fn fill_rayleigh_f32<S>(&mut self, buf: &mut [f32], sigma: S)
    where
        S: ParamSource<f32>,
    {
        let limit = buf.len().min(sigma.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let mut u = [0.0; 8];
            for j in 0..8 {
                u[j] = 1.0 - self.next_uniform_f32();
            }
            let batch_ln = fptricks::batch_approx_ln_f32(u);
            let r_input = fptricks::batch_fmadd_f32(batch_ln, -2.0, 0.0);
            let r = fptricks::batch_approx_sqrt_f32(r_input);
            
            let s_chunk = sigma.chunk::<8>(offset);
            let res = fptricks::batch_fmadd_cols_f32(r, s_chunk, [0.0; 8]);
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_rayleigh_f32(sigma.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_rayleigh_f64<S>(&mut self, buf: &mut [f64], sigma: S)
    where
        S: ParamSource<f64>,
    {
        let limit = buf.len().min(sigma.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let mut u = [0.0; 4];
            for j in 0..4 {
                u[j] = 1.0 - self.next_uniform_f64();
            }
            let batch_ln = fptricks::batch_approx_ln_f64(u);
            let r_input = fptricks::batch_fmadd_f64(batch_ln, -2.0, 0.0);
            let r = fptricks::batch_approx_sqrt_f64(r_input);
            
            let s_chunk = sigma.chunk::<4>(offset);
            let res = fptricks::batch_fmadd_cols_f64(r, s_chunk, [0.0; 4]);
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_rayleigh_f64(sigma.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_gamma_f32<A>(&mut self, buf: &mut [f32], alpha: A)
    where
        A: ParamSource<f32>,
    {
        let limit = buf.len().min(alpha.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let a_arr = alpha.chunk::<8>(offset);
            for j in 0..8 {
                chunk[j] = self.next_gamma_f32(a_arr[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_gamma_f32(alpha.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_gamma_f64<A>(&mut self, buf: &mut [f64], alpha: A)
    where
        A: ParamSource<f64>,
    {
        let limit = buf.len().min(alpha.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let a_arr = alpha.chunk::<4>(offset);
            for j in 0..4 {
                chunk[j] = self.next_gamma_f64(a_arr[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_gamma_f64(alpha.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_poisson_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where
        L: ParamSource<f32>,
    {
        let limit = buf.len().min(lambda.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let l_arr = lambda.chunk::<8>(offset);
            let neg_lambda = [
                -l_arr[0], -l_arr[1], -l_arr[2], -l_arr[3],
                -l_arr[4], -l_arr[5], -l_arr[6], -l_arr[7]
            ];
            let thresholds = fptricks::batch_approx_exp_f32(neg_lambda);

            for j in 0..8 {
                if l_arr[j] <= 0.0 {
                    chunk[j] = 0;
                    continue;
                }
                let l = thresholds[j];
                let mut k = 0u32;
                let mut p = 1.0_f32;
                loop {
                    k += 1;
                    p *= self.next_uniform_f32();
                    if p <= l { break; }
                }
                chunk[j] = k - 1;
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_poisson_u32(lambda.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_poisson_f64_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where
        L: ParamSource<f64>,
    {
        let limit = buf.len().min(lambda.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let l_arr = lambda.chunk::<4>(offset);
            let neg_lambda = [-l_arr[0], -l_arr[1], -l_arr[2], -l_arr[3]];
            let thresholds = fptricks::batch_approx_exp_f64(neg_lambda);

            for j in 0..4 {
                if l_arr[j] <= 0.0 {
                    chunk[j] = 0;
                    continue;
                }
                let l = thresholds[j];
                let mut k = 0u32;
                let mut p = 1.0_f64;
                loop {
                    k += 1;
                    p *= self.next_uniform_f64();
                    if p <= l { break; }
                }
                chunk[j] = k - 1;
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_poisson_f64_u32(lambda.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_poisson_collecting_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where
        L: ParamSource<f32>,
    {
        let limit = buf.len().min(lambda.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let l_arr = lambda.chunk::<8>(offset);
            let neg_lambda = [
                -l_arr[0], -l_arr[1], -l_arr[2], -l_arr[3],
                -l_arr[4], -l_arr[5], -l_arr[6], -l_arr[7]
            ];
            let thresholds = fptricks::batch_approx_exp_f32(neg_lambda);

            let mut counts = [0u32; 8];
            let mut p = self.next_f32_8();
            
            let mut mask = 0u8;
            for j in 0..8 {
                if l_arr[j] > 0.0 && p[j] > thresholds[j] {
                    mask |= 1 << j;
                }
            }

            while mask != 0 {
                let mut next_mask = 0u8;
                let u = self.next_f32_8();
                for j in 0..8 {
                    if (mask >> j) & 1 != 0 {
                        counts[j] += 1;
                        p[j] *= u[j];
                        if p[j] > thresholds[j] {
                            next_mask |= 1 << j;
                        }
                    }
                }
                mask = next_mask;
            }
            chunk.copy_from_slice(&counts);
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_poisson_u32(lambda.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_poisson_collecting_f64_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where
        L: ParamSource<f64>,
    {
        let limit = buf.len().min(lambda.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let l_arr = lambda.chunk::<4>(offset);
            let neg_lambda = [-l_arr[0], -l_arr[1], -l_arr[2], -l_arr[3]];
            let thresholds = fptricks::batch_approx_exp_f64(neg_lambda);

            let mut counts = [0u32; 4];
            let mut p = self.next_f64_4();
            
            let mut mask = 0u8;
            for j in 0..4 {
                if l_arr[j] > 0.0 && p[j] > thresholds[j] {
                    mask |= 1 << j;
                }
            }

            while mask != 0 {
                let mut next_mask = 0u8;
                let u = self.next_f64_4();
                for j in 0..4 {
                    if (mask >> j) & 1 != 0 {
                        counts[j] += 1;
                        p[j] *= u[j];
                        if p[j] > thresholds[j] {
                            next_mask |= 1 << j;
                        }
                    }
                }
                mask = next_mask;
            }
            chunk.copy_from_slice(&counts);
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_poisson_f64_u32(lambda.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_beta_f32<A, B>(&mut self, buf: &mut [f32], alpha: A, beta: B)
    where
        A: ParamSource<f32>,
        B: ParamSource<f32>,
    {
        let limit = buf.len().min(alpha.len()).min(beta.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let a_arr = alpha.chunk::<8>(offset);
            let b_arr = beta.chunk::<8>(offset);
            for j in 0..8 {
                chunk[j] = self.next_beta_f32(a_arr[j], b_arr[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_beta_f32(alpha.get(offset + i), beta.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_beta_f64<A, B>(&mut self, buf: &mut [f64], alpha: A, beta: B)
    where
        A: ParamSource<f64>,
        B: ParamSource<f64>,
    {
        let limit = buf.len().min(alpha.len()).min(beta.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let a_arr = alpha.chunk::<4>(offset);
            let b_arr = beta.chunk::<4>(offset);
            for j in 0..4 {
                chunk[j] = self.next_beta_f64(a_arr[j], b_arr[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_beta_f64(alpha.get(offset + i), beta.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_chi_squared_f32<K>(&mut self, buf: &mut [f32], k: K)
    where
        K: ParamSource<f32>,
    {
        let limit = buf.len().min(k.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let k_arr = k.chunk::<8>(offset);
            for j in 0..8 {
                chunk[j] = self.next_chi_squared_f32(k_arr[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_chi_squared_f32(k.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_chi_squared_f64<K>(&mut self, buf: &mut [f64], k: K)
    where
        K: ParamSource<f64>,
    {
        let limit = buf.len().min(k.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let k_arr = k.chunk::<4>(offset);
            for j in 0..4 {
                chunk[j] = self.next_chi_squared_f64(k_arr[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_chi_squared_f64(k.get(offset + i));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_gamma_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let alphas = [0.5, 1.0, 2.0, 5.0];
        for &alpha in &alphas {
            let (mean, var) = calculate_stats(|| rng.next_gamma_f32(alpha as f32) as f64, n);
            assert!((mean - alpha).abs() < alpha * 0.15);
            assert!((var - alpha).abs() < alpha * 0.15);
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
            assert!((mean - expected_mean).abs() < 0.1);
            assert!((var - expected_var).abs() < 0.1);
        }
    }

    #[test]
    fn test_rayleigh_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let sigma = 2.0;
        const HALFPI: f64 = std::f64::consts::PI / 2.0;
        let expected_mean = sigma * (HALFPI).sqrt();
        let (mean, _) = calculate_stats(|| rng.next_rayleigh_f32(sigma as f32) as f64, n);
        assert!((mean - expected_mean).abs() < 0.15);
    }

    #[test]
    fn test_poisson_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let lambdas = [0.5, 1.0, 5.0, 10.0];
        for &lambda in &lambdas {
            let (mean, var) = calculate_stats(|| rng.next_poisson_u32(lambda as f32) as f64, n);
            assert!((mean - lambda).abs() < lambda * 0.15 + 0.05);
            assert!((var - lambda).abs() < lambda * 0.15 + 0.1);
        }
    }

    #[test]
    fn test_poisson_collecting_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let lambdas = [0.5, 1.0, 5.0, 10.0];
        let mut buf = vec![0u32; n];
        for &lambda in &lambdas {
            rng.fill_poisson_collecting_u32(&mut buf, lambda);
            
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for &val in &buf {
                let v = val as f64;
                sum += v;
                sum_sq += v * v;
            }
            let mean = sum / n as f64;
            let var = sum_sq / n as f64 - mean * mean;

            let lamb = lambda as f64;
            assert!((mean - lamb).abs() < lamb * 0.15 + 0.05);
            assert!((var - lamb).abs() < lamb * 0.15 + 0.1);
        }
    }
}
