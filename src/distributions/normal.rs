
use crate::WyRand;
use crate::traits::ParamSource;
use fptricks::*;

impl WyRand {
    pub(crate) const TWO_PI_F32: f32 = 2.0 * std::f32::consts::PI;
    pub(crate) const TWO_PI_F64: f64 = 2.0 * std::f64::consts::PI;

    // Generates a standard normal random variable (mean=0, std_dev=1)
    // using the Box-Muller transform
    #[inline(always)]
    pub fn next_std_normal_f32(&mut self) -> f32 {
        let u1 = 1.0 - self.next_uniform_f32();
        let u2 = 1.0 - self.next_uniform_f32();
        let r = (-u1.approx_ln().fast_mul2()).approx_sqrt();
        r * (Self::TWO_PI_F32 * u2).approx_cos()
    }

    #[inline(always)]
    pub fn next_std_normal_f64(&mut self) -> f64 {
        let u1 = 1.0 - self.next_uniform_f64();
        let u2 = 1.0 - self.next_uniform_f64();
        let r = (-u1.approx_ln().fast_mul2()).approx_sqrt();
        r * (Self::TWO_PI_F64 * u2).approx_cos()
    }

    #[inline(always)]
    pub fn next_std_normal_pair_f32(&mut self) -> (f32, f32) {
        let u1 = 1.0 - self.next_uniform_f32();
        let u2 = 1.0 - self.next_uniform_f32();
        let r = (-u1.approx_ln().fast_mul2()).approx_sqrt();
        let (s, c) = (Self::TWO_PI_F32 * u2).approx_sin_cos();
        (r * s, r * c)
    }

    #[inline(always)]
    pub fn next_std_normal_pair_f64(&mut self) -> (f64, f64) {
        let u1 = 1.0 - self.next_uniform_f64();
        let u2 = 1.0 - self.next_uniform_f64();
        let r = (-u1.approx_ln().fast_mul2()).approx_sqrt();
        let (s, c) = (Self::TWO_PI_F64 * u2).approx_sin_cos();
        (r * s, r * c)
    }

    #[inline(always)]
    pub fn next_normal_f32(&mut self, mode: f32, sigma: f32) -> f32 {
        self.next_std_normal_f32().mul_add(sigma, mode)
    }

    #[inline(always)]
    pub fn next_normal_f64(&mut self, mode: f64, sigma: f64) -> f64 {
        self.next_std_normal_f64().mul_add(sigma, mode)
    }

    #[inline(always)]
    pub fn next_split_normal_f32(
        &mut self,
        mode: f32,
        sigma_low_mag: f32,
        sigma_high_mag: f32,
    ) -> f32 {
        let z = self.next_std_normal_f32();
        let zlt_mask: u32 = ((z < 0.0) as u32).wrapping_neg();
        let zgeq_mask: u32 = ((z >= 0.0) as u32).wrapping_neg();

        let sigma = (sigma_low_mag.to_bits() & zlt_mask) | (sigma_high_mag.to_bits() & zgeq_mask);
        z.mul_add(f32::from_bits(sigma), mode)
    }

    #[inline(always)]
    pub fn next_split_normal_f64(
        &mut self,
        mode: f64,
        sigma_low_mag: f64,
        sigma_high_mag: f64,
    ) -> f64 {
        let z = self.next_std_normal_f64();
        let zlt_mask: u64 = ((z < 0.0) as u64).wrapping_neg();
        let zgeq_mask: u64 = ((z >= 0.0) as u64).wrapping_neg();

        let sigma = (sigma_low_mag.to_bits() & zlt_mask) | (sigma_high_mag.to_bits() & zgeq_mask);
        z.mul_add(f64::from_bits(sigma), mode)
    }

    #[inline(always)]
    pub fn next_normal_w_clamped_err_f32(
        &mut self,
        mode: f32,
        sigma: f32,
        limit: f32,
    ) -> f32 {
        let z = self.next_std_normal_f32().clamp(-limit, limit);
        z.mul_add(sigma, mode)
    }

    #[inline(always)]
    pub fn next_normal_w_clamped_err_f64(
        &mut self,
        mode: f64,
        sigma: f64,
        limit: f64,
    ) -> f64 {
        let z = self.next_std_normal_f64().clamp(-limit, limit);
        z.mul_add(sigma, mode)
    }

    #[inline(always)]
    pub fn next_ln_normal_f32(&mut self, ln_mode: f32, sigma_ln: f32) -> f32 {
        let exponent = self.next_normal_f32(ln_mode, sigma_ln);
        exponent.approx_exp()
    }

    #[inline(always)]
    pub fn next_ln_normal_f64(&mut self, ln_mode: f64, sigma_ln: f64) -> f64 {
        let exponent = self.next_normal_f64(ln_mode, sigma_ln);
        exponent.approx_exp()
    }

    #[inline(always)]
    pub fn next_log10_normal_f32(&mut self, log_mode: f32, sigma_log: f32) -> f32 {
        let exponent = self.next_normal_f32(log_mode, sigma_log);
        10.0_f32.approx_powf(exponent)
    }

    #[inline(always)]
    pub fn next_log10_normal_f64(&mut self, log_mode: f64, sigma_log: f64) -> f64 {
        let exponent = self.next_normal_f64(log_mode, sigma_log);
        10.0_f64.approx_powf(exponent)
    }

    ///Samples from the isotropic distribution, which is useful specifically
    ///for generating polar angles in a spherical coordinate system. Simply
    ///drawing from a uniform distribution on [0, 2π] for θ would cause a bias towards
    ///the poles, because a change in azimuthal angle dφ of a vector would make
    ///that vector move faster when θ pointed it closer to the equator; in other words,
    ///the same dθ maps to less and less 3D surface area the closer θ gets to 0 or π.
    ///Look at the expression for differential area: dA = (r^2) sin(θ) dθdφ

    ///(Similarly, generating 3 uniformly distributed random values for a Cartesian
    ///coordinate vector would result in bias towards corners of a cube; in that case,
    ///Gaussian RVs can be used.)
    #[inline(always)]
    pub fn next_isotropic_polar_angle_f32(&mut self) -> f32 {

        let u = self.next_uniform_f32();
        (1.0 - u.fast_mul2()).approx_acos()
    }

    #[inline(always)]
    pub fn next_isotropic_polar_angle_f64(&mut self) -> f64 {
        let u = self.next_uniform_f64();
        (1.0 - u.fast_mul2()).approx_acos()
    }

    #[inline]
    pub fn fill_std_normal_f32(&mut self, buf: &mut [f32]) {
        let mut iter = buf.chunks_exact_mut(16);
        for chunk in iter.by_ref() {
            let u1_8 = self.next_f32_8();
            let u2_8 = self.next_f32_8();
            let mut u1 = [0.0; 8];
            let mut u2 = [0.0; 8];
            for j in 0..8 {
                u1[j] = 1.0 - u1_8[j];
                u2[j] = 1.0 - u2_8[j];
            }
            let batch_ln = fptricks::batch_approx_ln_f32(u1);
            let r_input = fptricks::batch_fmadd_f32(batch_ln, -2.0, 0.0);
            let r = fptricks::batch_approx_sqrt_f32(r_input);

            let u2_scaled = fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0);
            let (s, c) = fptricks::batch_approx_sin_cos_f32(u2_scaled);

            for j in 0..8 {
                chunk[j << 1] = r[j] * s[j];
                chunk[(j << 1) + 1] = r[j] * c[j];
            }
        }
        
        let remainder = iter.into_remainder();
        let mut i = 0;
        while i + 1 < remainder.len() {
            let (s, c) = self.next_std_normal_pair_f32();
            remainder[i] = s;
            remainder[i + 1] = c;
            i += 2;
        }
        if i < remainder.len() {
            remainder[i] = self.next_std_normal_f32();
        }
    }

    #[inline]
    pub fn fill_std_normal_f64(&mut self, buf: &mut [f64]) {
        let mut iter = buf.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let u1_4 = self.next_f64_4();
            let u2_4 = self.next_f64_4();
            let mut u1 = [0.0; 4];
            let mut u2 = [0.0; 4];
            for j in 0..4 {
                u1[j] = 1.0 - u1_4[j];
                u2[j] = 1.0 - u2_4[j];
            }
            let batch_ln = fptricks::batch_approx_ln_f64(u1);
            let r_input = fptricks::batch_fmadd_f64(batch_ln, -2.0, 0.0);
            let r = fptricks::batch_approx_sqrt_f64(r_input);

            let u2_scaled = fptricks::batch_fmadd_f64(u2, Self::TWO_PI_F64, 0.0);
            let (s, c) = fptricks::batch_approx_sin_cos_f64(u2_scaled);

            for j in 0..4 {
                chunk[j << 1] = r[j] * s[j];
                chunk[(j << 1) + 1] = r[j] * c[j];
            }
        }
        
        let remainder = iter.into_remainder();
        let mut i = 0;
        while i + 1 < remainder.len() {
            let (s, c) = self.next_std_normal_pair_f64();
            remainder[i] = s;
            remainder[i + 1] = c;
            i += 2;
        }
        if i < remainder.len() {
            remainder[i] = self.next_std_normal_f64();
        }
    }

    #[inline]
    pub fn fill_normal_f32<M, S>(&mut self, buf: &mut [f32], mode: M, sigma: S)
    where
        M: ParamSource<f32>,
        S: ParamSource<f32>,
    {
        let limit = buf.len().min(mode.len()).min(sigma.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(16);
        
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i * 16;
            let u1_8 = self.next_f32_8();
            let u2_8 = self.next_f32_8();
            
            let mut u1 = [0.0; 8];
            let mut u2 = [0.0; 8];
            for j in 0..8 {
                u1[j] = 1.0 - u1_8[j];
                u2[j] = 1.0 - u2_8[j];
            }
            let batch_ln = fptricks::batch_approx_ln_f32(u1);
            let r_input = fptricks::batch_fmadd_f32(batch_ln, -2.0, 0.0);
            let r = fptricks::batch_approx_sqrt_f32(r_input);
            let u2_scaled = fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0);
            let (s, c) = fptricks::batch_approx_sin_cos_f32(u2_scaled);

            let m1 = mode.chunk::<8>(offset);
            let m2 = mode.chunk::<8>(offset + 8);
            let s1 = sigma.chunk::<8>(offset);
            let s2 = sigma.chunk::<8>(offset + 8);

            for j in 0..8 {
                chunk[j << 1] = m1[j] + r[j] * s[j] * s1[j];
                chunk[(j << 1) + 1] = m2[j] + r[j] * c[j] * s2[j];
            }
        }
        
        let rem = iter.into_remainder();
        let offset = limit & !15;
        let mut i = 0;
        while i + 1 < rem.len() {
            let (s, c) = self.next_std_normal_pair_f32();
            rem[i] = mode.get(offset + i) + s * sigma.get(offset + i);
            rem[i + 1] = mode.get(offset + i + 1) + c * sigma.get(offset + i + 1);
            i += 2;
        }
        if i < rem.len() {
            rem[i] = self.next_std_normal_f32().mul_add(sigma.get(offset + i), mode.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_normal_f64<M, S>(&mut self, buf: &mut [f64], mode: M, sigma: S)
    where
        M: ParamSource<f64>,
        S: ParamSource<f64>,
    {
        let limit = buf.len().min(mode.len()).min(sigma.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let u1_4 = self.next_f64_4();
            let u2_4 = self.next_f64_4();
            
            let mut u1 = [0.0; 4];
            let mut u2 = [0.0; 4];
            for j in 0..4 {
                u1[j] = 1.0 - u1_4[j];
                u2[j] = 1.0 - u2_4[j];
            }
            let batch_ln = fptricks::batch_approx_ln_f64(u1);
            let r_input = fptricks::batch_fmadd_f64(batch_ln, -2.0, 0.0);
            let r = fptricks::batch_approx_sqrt_f64(r_input);
            let u2_scaled = fptricks::batch_fmadd_f64(u2, Self::TWO_PI_F64, 0.0);
            let (s, c) = fptricks::batch_approx_sin_cos_f64(u2_scaled);

            let m1 = mode.chunk::<4>(offset);
            let m2 = mode.chunk::<4>(offset + 4);
            let s1 = sigma.chunk::<4>(offset);
            let s2 = sigma.chunk::<4>(offset + 4);

            for j in 0..4 {
                chunk[j << 1] = m1[j] + r[j] * s[j] * s1[j];
                chunk[(j << 1) + 1] = m2[j] + r[j] * c[j] * s2[j];
            }
        }
        
        let rem = iter.into_remainder();
        let offset = limit & !7;
        let mut i = 0;
        while i + 1 < rem.len() {
            let (s, c) = self.next_std_normal_pair_f64();
            rem[i] = mode.get(offset + i) + s * sigma.get(offset + i);
            rem[i + 1] = mode.get(offset + i + 1) + c * sigma.get(offset + i + 1);
            i += 2;
        }
        if i < rem.len() {
            rem[i] = self.next_std_normal_f64().mul_add(sigma.get(offset + i), mode.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_normal_w_clamped_err_f32<M, S, L>(&mut self, buf: &mut [f32], mode: M, sigma: S, limit: L)
    where
        M: ParamSource<f32>,
        S: ParamSource<f32>,
        L: ParamSource<f32>,
    {
        let shared_limit = buf.len().min(mode.len()).min(sigma.len()).min(limit.len());
        let (active, _) = buf.split_at_mut(shared_limit);
        self.fill_std_normal_f32(active);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let lim = limit.chunk::<8>(offset);
            let mut arr = [0.0; 8];
            arr.copy_from_slice(chunk);
            for j in 0..8 {
                arr[j] = arr[j].clamp(-lim[j], lim[j]);
            }
            let res = fptricks::batch_fmadd_cols_f32(arr, sigma.chunk::<8>(offset), mode.chunk::<8>(offset));
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = shared_limit & !7;
        for (i, val) in rem.iter_mut().enumerate() {
            let lim = limit.get(offset + i);
            *val = val.clamp(-lim, lim).mul_add(sigma.get(offset + i), mode.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_normal_w_clamped_err_f64<M, S, L>(&mut self, buf: &mut [f64], mode: M, sigma: S, limit: L)
    where
        M: ParamSource<f64>,
        S: ParamSource<f64>,
        L: ParamSource<f64>,
    {
        let shared_limit = buf.len().min(mode.len()).min(sigma.len()).min(limit.len());
        let (active, _) = buf.split_at_mut(shared_limit);
        self.fill_std_normal_f64(active);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let lim = limit.chunk::<4>(offset);
            let mut arr = [0.0; 4];
            arr.copy_from_slice(chunk);
            for j in 0..4 {
                arr[j] = arr[j].clamp(-lim[j], lim[j]);
            }
            let res = fptricks::batch_fmadd_cols_f64(arr, sigma.chunk::<4>(offset), mode.chunk::<4>(offset));
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = shared_limit & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            let lim = limit.get(offset + i);
            *val = val.clamp(-lim, lim).mul_add(sigma.get(offset + i), mode.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_ln_normal_f32<M, S>(&mut self, buf: &mut [f32], ln_mode: M, sigma_ln: S)
    where
        M: ParamSource<f32>,
        S: ParamSource<f32>,
    {
        let limit = buf.len().min(ln_mode.len()).min(sigma_ln.len());
        let (active, _) = buf.split_at_mut(limit);
        self.fill_normal_f32(active, ln_mode, sigma_ln);
        let mut iter = active.chunks_exact_mut(8);
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

    #[inline]
    pub fn fill_ln_normal_f64<M, S>(&mut self, buf: &mut [f64], ln_mode: M, sigma_ln: S)
    where
        M: ParamSource<f64>,
        S: ParamSource<f64>,
    {
        let limit = buf.len().min(ln_mode.len()).min(sigma_ln.len());
        let (active, _) = buf.split_at_mut(limit);
        self.fill_normal_f64(active, ln_mode, sigma_ln);
        let mut iter = active.chunks_exact_mut(4);
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

    #[inline]
    pub fn fill_log10_normal_f32<M, S>(&mut self, buf: &mut [f32], log_mode: M, sigma_log: S)
    where
        M: ParamSource<f32>,
        S: ParamSource<f32>,
    {
        let limit = buf.len().min(log_mode.len()).min(sigma_log.len());
        let (active, _) = buf.split_at_mut(limit);
        self.fill_normal_f32(active, log_mode, sigma_log);
        let mut iter = active.chunks_exact_mut(8);
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

    #[inline]
    pub fn fill_log10_normal_f64<M, S>(&mut self, buf: &mut [f64], log_mode: M, sigma_log: S)
    where
        M: ParamSource<f64>,
        S: ParamSource<f64>,
    {
        let limit = buf.len().min(log_mode.len()).min(sigma_log.len());
        let (active, _) = buf.split_at_mut(limit);
        self.fill_normal_f64(active, log_mode, sigma_log);
        let mut iter = active.chunks_exact_mut(4);
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

    #[inline]
    pub fn fill_split_normal_f32<M, SLO, SHI>(&mut self, buf: &mut [f32], mode: M, sigma_lo: SLO, sigma_hi: SHI)
    where
        M: ParamSource<f32>,
        SLO: ParamSource<f32>,
        SHI: ParamSource<f32>,
    {
        let limit = buf.len().min(mode.len()).min(sigma_lo.len()).min(sigma_hi.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(16);
        
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i * 16;
            let u1_8 = self.next_f32_8();
            let u2_8 = self.next_f32_8();
            
            let mut u1 = [0.0; 8];
            let mut u2 = [0.0; 8];
            for j in 0..8 {
                u1[j] = 1.0 - u1_8[j];
                u2[j] = 1.0 - u2_8[j];
            }
            let bl = fptricks::batch_approx_ln_f32(u1);
            let ri = fptricks::batch_fmadd_f32(bl, -2.0, 0.0);
            let r = fptricks::batch_approx_sqrt_f32(ri);
            let u2s = fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0);
            let (s, c) = fptricks::batch_approx_sin_cos_f32(u2s);

            let m1 = mode.chunk::<8>(offset);
            let m2 = mode.chunk::<8>(offset + 8);
            let sl1 = sigma_lo.chunk::<8>(offset);
            let sl2 = sigma_lo.chunk::<8>(offset + 8);
            let sh1 = sigma_hi.chunk::<8>(offset);
            let sh2 = sigma_hi.chunk::<8>(offset + 8);

            for j in 0..8 {
                let x = r[j] * s[j];
                let m = m1[j];
                chunk[j << 1] = if x < 0.0 { m + x * sl1[j] } else { m + x * sh1[j] };
                
                let x2 = r[j] * c[j];
                let m2v = m2[j];
                chunk[(j << 1) + 1] = if x2 < 0.0 { m2v + x2 * sl2[j] } else { m2v + x2 * sh2[j] };
            }
        }
        
        let rem = iter.into_remainder();
        let offset = limit & !15;
        for (i, val) in rem.iter_mut().enumerate() {
            let x = self.next_std_normal_f32();
            let m = mode.get(offset + i);
            let s_lo = sigma_lo.get(offset + i);
            let s_hi = sigma_hi.get(offset + i);
            *val = if x < 0.0 { m + x * s_lo } else { m + x * s_hi };
        }
    }

    #[inline]
    pub fn fill_split_normal_f64<M, SLO, SHI>(&mut self, buf: &mut [f64], mode: M, sigma_lo: SLO, sigma_hi: SHI)
    where
        M: ParamSource<f64>,
        SLO: ParamSource<f64>,
        SHI: ParamSource<f64>,
    {
        let limit = buf.len().min(mode.len()).min(sigma_lo.len()).min(sigma_hi.len());
        let (active, _) = buf.split_at_mut(limit);
        self.fill_std_normal_f64(active);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let mut arr = [0.0; 4];
            arr.copy_from_slice(chunk);
            let res = fptricks::batch_asymmetric_fma_cols_f64(arr, mode.chunk::<4>(offset), 
                sigma_lo.chunk::<4>(offset), sigma_hi.chunk::<4>(offset));
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
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

    #[inline]
    pub fn fill_split_normal_stuple_f32<M, T>(&mut self, buf: &mut [f32], mode: M, table: T)
    where
        M: ParamSource<f32>,
        T: ParamSource<(f32, f32)>,
    {
        let limit = buf.len().min(mode.len()).min(table.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(16);
        
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 4;
            let u1_8 = self.next_f32_8();
            let u2_8 = self.next_f32_8();
            
            let mut u1 = [0.0; 8];
            let mut u2 = [0.0; 8];
            for j in 0..8 {
                u1[j] = 1.0 - u1_8[j];
                u2[j] = 1.0 - u2_8[j];
            }
            let bl = fptricks::batch_approx_ln_f32(u1);
            let ri = fptricks::batch_fmadd_f32(bl, -2.0, 0.0);
            let r = fptricks::batch_approx_sqrt_f32(ri);
            let u2s = fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0);
            let (s, c) = fptricks::batch_approx_sin_cos_f32(u2s);

            let m1 = mode.chunk::<8>(offset);
            let m2 = mode.chunk::<8>(offset + 8);
            let sigs1 = table.chunk::<8>(offset);
            let sigs2 = table.chunk::<8>(offset + 8);

            for j in 0..8 {
                let x = r[j] * s[j];
                let (sl1, sh1) = sigs1[j];
                chunk[j << 1] = if x < 0.0 { m1[j] + x * sl1 } else { m1[j] + x * sh1 };
                
                let x2 = r[j] * c[j];
                let (sl2, sh2) = sigs2[j];
                chunk[(j << 1) + 1] = if x2 < 0.0 { m2[j] + x2 * sl2 } else { m2[j] + x2 * sh2 };
            }
        }
        
        let rem = iter.into_remainder();
        let offset = limit & !15;
        for (i, val) in rem.iter_mut().enumerate() {
            let x = self.next_std_normal_f32();
            let m = mode.get(offset + i);
            let (s_lo, s_hi) = table.get(offset + i);
            *val = if x < 0.0 { m + x * s_lo } else { m + x * s_hi };
        }
    }

    #[inline]
    pub fn fill_split_normal_stuple_f64<M, T>(&mut self, buf: &mut [f64], mode: M, table: T)
    where
        M: ParamSource<f64>,
        T: ParamSource<(f64, f64)>,
    {
        let limit = buf.len().min(mode.len()).min(table.len());
        let (active, _) = buf.split_at_mut(limit);
        self.fill_std_normal_f64(active);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let mut arr = [0.0; 4];
            arr.copy_from_slice(chunk);
            
            let modes = mode.chunk::<4>(offset);
            let sigs = table.chunk::<4>(offset);
            
            let mut sl = [0.0; 4];
            let mut sh = [0.0; 4];
            for j in 0..4 {
                sl[j] = sigs[j].0;
                sh[j] = sigs[j].1;
            }

            let res = fptricks::batch_asymmetric_fma_cols_f64(arr, modes, sl, sh);
            chunk.copy_from_slice(&res);
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            let (sig_lo, sig_hi) = table.get(offset + i);
            let sig_lo_bits = sig_lo.to_bits();
            let sig_hi_bits = sig_hi.to_bits();
            let zlt_mask: u64 = ((*val < 0.0) as u64).wrapping_neg();
            let zgeq_mask: u64 = ((*val >= 0.0) as u64).wrapping_neg();
            let sigma = (sig_lo_bits & zlt_mask) | (sig_hi_bits & zgeq_mask);
            *val = val.mul_add(f64::from_bits(sigma), mode.get(offset + i));
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
    fn test_normal_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let (mean, var) = calculate_stats(|| rng.next_std_normal_f32() as f64, n);
        assert!(mean.abs() < 0.1);
        assert!((var - 1.0).abs() < 0.15);

        let (mean, var) = calculate_stats(|| rng.next_std_normal_f64(), n);
        assert!(mean.abs() < 0.1);
        assert!((var - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_normal_pair_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let mut sum_s = 0.0;
        let mut sum_c = 0.0;
        let mut sum_sq_s = 0.0;
        let mut sum_sq_c = 0.0;

        for _ in 0..n {
            let (s, c) = rng.next_std_normal_pair_f32();
            sum_s += s as f64;
            sum_c += c as f64;
            sum_sq_s += (s * s) as f64;
            sum_sq_c += (c * c) as f64;
        }

        let mean_s = sum_s / n as f64;
        let var_s = sum_sq_s / n as f64 - mean_s * mean_s;
        let mean_c = sum_c / n as f64;
        let var_c = sum_sq_c / n as f64 - mean_c * mean_c;

        assert!(mean_s.abs() < 0.1);
        assert!((var_s - 1.0).abs() < 0.15);
        assert!(mean_c.abs() < 0.1);
        assert!((var_c - 1.0).abs() < 0.15);
    }

    #[test]
    fn test_batch_fill_normal() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let mut buf = vec![0.0f32; n];
        rng.fill_std_normal_f32(&mut buf);
        
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for &val in &buf {
            sum += val as f64;
            sum_sq += (val * val) as f64;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.1);
        assert!((var - 1.0).abs() < 0.15);
    }

    #[test]
    fn test_split_normal_table_correctness() {
        let mut rng = WyRand::new(1);
        let n = 1024;
        let mut buf = vec![0.0f32; n];
        let mode = 0.0;
        let table = vec![(1.0, 2.0); n]; // lo=1.0, hi=2.0
        
        rng.fill_split_normal_stuple_f32(&mut buf, mode, &table);
        
        for &val in &buf {
            // Standard normal x. If x < 0, val = m + x * lo. If x >= 0, val = m + x * hi.
            // Since mode=0, val = x * lo or x * hi. 
            // So x = val/lo if val < 0, x = val/hi if val >= 0.
            // We can't perfectly reconstruct x, but we can verify bounds/stats if we had a lot,
            // or just check that it's using the parameters.
            if val < 0.0 {
                // val should be ~ x * 1.0 (mean 0, var 1)
            } else {
                // val should be ~ x * 2.0 (mean 0, var 4)
            }
        }
        
        // Simple statistical check
        let mut sum_lo = 0.0;
        let mut count_lo = 0;
        let mut sum_hi = 0.0;
        let mut count_hi = 0;
        for &val in &buf {
            if val < 0.0 {
                sum_lo += val;
                count_lo += 1;
            } else {
                sum_hi += val;
                count_hi += 1;
            }
        }
        
        // Expected mean of half-normal |N(0,1)| is sqrt(2/pi) approx 0.8
        // For lo=1.0, expected mean of negative half is -0.8
        // For hi=2.0, expected mean of positive half is 1.6
        if count_lo > 0 {
            let mean_lo = sum_lo / count_lo as f32;
            assert!(mean_lo < -0.5 && mean_lo > -1.1);
        }
        if count_hi > 0 {
            let mean_hi = sum_hi / count_hi as f32;
            assert!(mean_hi > 1.2 && mean_hi < 2.0);
        }
    }

    #[test]
    fn test_radian_sampling_stats() {
        let mut rng = WyRand::new(42);
        let n = 100_000;
        
        // Test: Full sphere using specialized function
        // Inclination angle for a uniform sphere: cos(i) is uniform in [-1, 1]
        let mut sum_cos = 0.0;
        for _ in 0..n {
            let i = rng.next_isotropic_polar_angle_f32();
            sum_cos += i.cos() as f64; 
        }
        let mean_cos = sum_cos / n as f64;
        assert!(mean_cos.abs() < 0.05, "Full sphere f32 mean_cos: {}", mean_cos);

        let mut sum_cos = 0.0;
        for _ in 0..n {
            let i = rng.next_isotropic_polar_angle_f64();
            sum_cos += i.cos() as f64; 
        }
        let mean_cos = sum_cos / n as f64;
        assert!(mean_cos.abs() < 0.05, "Full sphere f64 mean_cos: {}", mean_cos);
    }


}
