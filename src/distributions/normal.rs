use std::mem::MaybeUninit;
use crate::WyRand;
use crate::traits::ParamSource;
use fptricks::*;

impl WyRand {
    pub(crate) const TWO_PI_F32: f32 = 2.0 * std::f32::consts::PI;
    pub(crate) const TWO_PI_F64: f64 = 2.0 * std::f64::consts::PI;

    #[inline(always)]
    pub fn next_std_normal_f32(&mut self) -> f32 {
        // Clamp u1 away from 0.0: approx_ln(0) is implementation-defined in
        // fptricks and may not return -∞, producing NaN that poisons the
        // Marsaglia-Tsang gamma rejection loop (all NaN comparisons are false).
        let u1 = (1.0 - self.next_uniform_f32()).max(f32::MIN_POSITIVE);
        let u2 = 1.0 - self.next_uniform_f32();
        let r = (-u1.approx_ln().fast_mul2()).approx_sqrt();
        r * (Self::TWO_PI_F32 * u2).approx_cos()
    }

    #[inline(always)]
    pub fn next_std_normal_f64(&mut self) -> f64 {
        // Same guard as the f32 variant — see comment above.
        let u1 = (1.0 - self.next_uniform_f64()).max(f64::MIN_POSITIVE);
        let u2 = 1.0 - self.next_uniform_f64();
        let r = (-u1.approx_ln().fast_mul2()).approx_sqrt();
        r * (Self::TWO_PI_F64 * u2).approx_cos()
    }

    #[inline(always)]
    pub fn next_std_normal_pair_f32(&mut self) -> (f32, f32) {
        let u1 = (1.0 - self.next_uniform_f32()).max(f32::MIN_POSITIVE);
        let u2 = 1.0 - self.next_uniform_f32();
        let r = (-u1.approx_ln().fast_mul2()).approx_sqrt();
        let (s, c) = (Self::TWO_PI_F32 * u2).approx_sin_cos();
        (r * s, r * c)
    }

    #[inline(always)]
    pub fn next_std_normal_pair_f64(&mut self) -> (f64, f64) {
        let u1 = (1.0 - self.next_uniform_f64()).max(f64::MIN_POSITIVE);
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
    pub fn next_split_normal_f32(&mut self, mode: f32, sigma_low_mag: f32, sigma_high_mag: f32) -> f32 {
        let z = self.next_std_normal_f32();
        let zlt_mask: u32 = ((z < 0.0) as u32).wrapping_neg();
        let zgeq_mask: u32 = ((z >= 0.0) as u32).wrapping_neg();
        let sigma = (sigma_low_mag.to_bits() & zlt_mask) | (sigma_high_mag.to_bits() & zgeq_mask);
        z.mul_add(f32::from_bits(sigma), mode)
    }

    #[inline(always)]
    pub fn next_split_normal_f64(&mut self, mode: f64, sigma_low_mag: f64, sigma_high_mag: f64) -> f64 {
        let z = self.next_std_normal_f64();
        let zlt_mask: u64 = ((z < 0.0) as u64).wrapping_neg();
        let zgeq_mask: u64 = ((z >= 0.0) as u64).wrapping_neg();
        let sigma = (sigma_low_mag.to_bits() & zlt_mask) | (sigma_high_mag.to_bits() & zgeq_mask);
        z.mul_add(f64::from_bits(sigma), mode)
    }

    #[inline(always)]
    pub fn next_normal_w_clamped_err_f32(&mut self, mode: f32, sigma: f32, limit: f32) -> f32 {
        self.next_std_normal_f32().clamp(-limit, limit).mul_add(sigma, mode)
    }

    #[inline(always)]
    pub fn next_normal_w_clamped_err_f64(&mut self, mode: f64, sigma: f64, limit: f64) -> f64 {
        self.next_std_normal_f64().clamp(-limit, limit).mul_add(sigma, mode)
    }

    #[inline(always)]
    pub fn next_ln_normal_f32(&mut self, ln_mode: f32, sigma_ln: f32) -> f32 {
        self.next_normal_f32(ln_mode, sigma_ln).approx_exp()
    }

    #[inline(always)]
    pub fn next_ln_normal_f64(&mut self, ln_mode: f64, sigma_ln: f64) -> f64 {
        self.next_normal_f64(ln_mode, sigma_ln).approx_exp()
    }

    #[inline(always)]
    pub fn next_log10_normal_f32(&mut self, log_mode: f32, sigma_log: f32) -> f32 {
        10.0_f32.approx_powf(self.next_normal_f32(log_mode, sigma_log))
    }

    #[inline(always)]
    pub fn next_log10_normal_f64(&mut self, log_mode: f64, sigma_log: f64) -> f64 {
        10.0_f64.approx_powf(self.next_normal_f64(log_mode, sigma_log))
    }

    #[inline(always)]
    pub fn next_isotropic_polar_angle_f32(&mut self) -> f32 {
        (1.0 - self.next_uniform_f32().fast_mul2()).approx_acos()
    }

    #[inline(always)]
    pub fn next_isotropic_polar_angle_f64(&mut self) -> f64 {
        (1.0 - self.next_uniform_f64().fast_mul2()).approx_acos()
    }

    // -------------------------------------------------------------------------
    // fill_* — write into caller-owned slice (heap-friendly, runtime length)
    // -------------------------------------------------------------------------

    #[inline]
    pub fn fill_std_normal_f32(&mut self, buf: &mut [f32]) {
        let mut iter = buf.chunks_exact_mut(16);
        for chunk in iter.by_ref() {
            let u1_8 = self.next_f32_8();
            let u2_8 = self.next_f32_8();
            let mut u1 = [0.0f32; 8];
            let mut u2 = [0.0f32; 8];
            for j in 0..8 {
                u1[j] = (1.0 - u1_8[j]).max(f32::MIN_POSITIVE);
                u2[j] = 1.0 - u2_8[j];
            }
            let r = fptricks::batch_approx_sqrt_f32(fptricks::batch_fmadd_f32(fptricks::batch_approx_ln_f32(u1), -2.0, 0.0));
            let (s, c) = fptricks::batch_approx_sin_cos_f32(fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0));
            let ps = fptricks::batch_mul_cols_f32(&r, &s);
            let pc = fptricks::batch_mul_cols_f32(&r, &c);

            chunk[0] = ps[0]; chunk[1] = pc[0];
            chunk[2] = ps[1]; chunk[3] = pc[1];
            chunk[4] = ps[2]; chunk[5] = pc[2];
            chunk[6] = ps[3]; chunk[7] = pc[3];
            chunk[8] = ps[4]; chunk[9] = pc[4];
            chunk[10] = ps[5]; chunk[11] = pc[5];
            chunk[12] = ps[6]; chunk[13] = pc[6];
            chunk[14] = ps[7]; chunk[15] = pc[7];
        }
        let rem = iter.into_remainder();
        let mut i = 0;
        while i + 1 < rem.len() {
            let (s, c) = self.next_std_normal_pair_f32();
            rem[i] = s; rem[i + 1] = c; i += 2;
        }
        if i < rem.len() { rem[i] = self.next_std_normal_f32(); }
    }

    #[inline]
    pub fn fill_std_normal_f64(&mut self, buf: &mut [f64]) {
        let mut iter = buf.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let u1_4 = self.next_f64_4();
            let u2_4 = self.next_f64_4();
            let mut u1 = [0.0f64; 4];
            let mut u2 = [0.0f64; 4];
            for j in 0..4 {
                u1[j] = (1.0 - u1_4[j]).max(f64::MIN_POSITIVE);
                u2[j] = 1.0 - u2_4[j];
            }
            let r = fptricks::batch_approx_sqrt_f64(fptricks::batch_fmadd_f64(fptricks::batch_approx_ln_f64(u1), -2.0, 0.0));
            let (s, c) = fptricks::batch_approx_sin_cos_f64(fptricks::batch_fmadd_f64(u2, Self::TWO_PI_F64, 0.0));
            let ps = fptricks::batch_mul_cols_f64(&r, &s);
            let pc = fptricks::batch_mul_cols_f64(&r, &c);

            chunk[0] = ps[0]; chunk[1] = pc[0];
            chunk[2] = ps[1]; chunk[3] = pc[1];
            chunk[4] = ps[2]; chunk[5] = pc[2];
            chunk[6] = ps[3]; chunk[7] = pc[3];
        }
        let rem = iter.into_remainder();
        let mut i = 0;
        while i + 1 < rem.len() {
            let (s, c) = self.next_std_normal_pair_f64();
            rem[i] = s; rem[i + 1] = c; i += 2;
        }
        if i < rem.len() { rem[i] = self.next_std_normal_f64(); }
    }

    #[inline]
    pub fn fill_normal_f32<M, S>(&mut self, buf: &mut [f32], mode: M, sigma: S)
    where M: ParamSource<f32>, S: ParamSource<f32>,
    {
        let limit = buf.len().min(mode.len()).min(sigma.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(16);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 4;
            let u1_8 = self.next_f32_8(); let u2_8 = self.next_f32_8();
            let mut u1 = [0.0f32; 8]; let mut u2 = [0.0f32; 8];
            for j in 0..8 {
                u1[j] = (1.0 - u1_8[j]).max(f32::MIN_POSITIVE);
                u2[j] = 1.0 - u2_8[j];
            }
            let r = fptricks::batch_approx_sqrt_f32(fptricks::batch_fmadd_f32(fptricks::batch_approx_ln_f32(u1), -2.0, 0.0));
            let (s, c) = fptricks::batch_approx_sin_cos_f32(fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0));
            let m1 = mode.chunk::<8>(offset); let m2 = mode.chunk::<8>(offset + 8);
            let s1 = sigma.chunk::<8>(offset); let s2 = sigma.chunk::<8>(offset + 8);
            let res1 = fptricks::batch_add_cols_f32(&m1, &fptricks::batch_mul_3_cols_f32(&r, &s, &s1));
            let res2 = fptricks::batch_add_cols_f32(&m2, &fptricks::batch_mul_3_cols_f32(&r, &c, &s2));
            
            chunk[0] = res1[0]; chunk[1] = res2[0];
            chunk[2] = res1[1]; chunk[3] = res2[1];
            chunk[4] = res1[2]; chunk[5] = res2[2];
            chunk[6] = res1[3]; chunk[7] = res2[3];
            chunk[8] = res1[4]; chunk[9] = res2[4];
            chunk[10] = res1[5]; chunk[11] = res2[5];
            chunk[12] = res1[6]; chunk[13] = res2[6];
            chunk[14] = res1[7]; chunk[15] = res2[7];
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
    where M: ParamSource<f64>, S: ParamSource<f64>,
    {
        let limit = buf.len().min(mode.len()).min(sigma.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let u1_4 = self.next_f64_4(); let u2_4 = self.next_f64_4();
            let mut u1 = [0.0f64; 4]; let mut u2 = [0.0f64; 4];
            for j in 0..4 {
                u1[j] = (1.0 - u1_4[j]).max(f64::MIN_POSITIVE);
                u2[j] = 1.0 - u2_4[j];
            }
            let r = fptricks::batch_approx_sqrt_f64(fptricks::batch_fmadd_f64(fptricks::batch_approx_ln_f64(u1), -2.0, 0.0));
            let (s, c) = fptricks::batch_approx_sin_cos_f64(fptricks::batch_fmadd_f64(u2, Self::TWO_PI_F64, 0.0));
            let m1 = mode.chunk::<4>(offset); let m2 = mode.chunk::<4>(offset + 4);
            let s1 = sigma.chunk::<4>(offset); let s2 = sigma.chunk::<4>(offset + 4);
            let res1 = fptricks::batch_add_cols_f64(&m1, &fptricks::batch_mul_3_cols_f64(&r, &s, &s1));
            let res2 = fptricks::batch_add_cols_f64(&m2, &fptricks::batch_mul_3_cols_f64(&r, &c, &s2));
            
            chunk[0] = res1[0]; chunk[1] = res2[0];
            chunk[2] = res1[1]; chunk[3] = res2[1];
            chunk[4] = res1[2]; chunk[5] = res2[2];
            chunk[6] = res1[3]; chunk[7] = res2[3];
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
    where M: ParamSource<f32>, S: ParamSource<f32>, L: ParamSource<f32>,
    {
        let shared = buf.len().min(mode.len()).min(sigma.len()).min(limit.len());
        let (active, _) = buf.split_at_mut(shared);
        self.fill_std_normal_f32(active);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let lim = limit.chunk::<8>(offset);
            let mut arr = [0.0f32; 8]; arr.copy_from_slice(chunk);
            for j in 0..8 { arr[j] = arr[j].clamp(-lim[j], lim[j]); }
            chunk.copy_from_slice(&fptricks::batch_fmadd_cols_f32(arr, sigma.chunk::<8>(offset), mode.chunk::<8>(offset)));
        }
        let rem = iter.into_remainder();
        let offset = shared & !7;
        for (i, val) in rem.iter_mut().enumerate() {
            let lim = limit.get(offset + i);
            *val = val.clamp(-lim, lim).mul_add(sigma.get(offset + i), mode.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_normal_w_clamped_err_f64<M, S, L>(&mut self, buf: &mut [f64], mode: M, sigma: S, limit: L)
    where M: ParamSource<f64>, S: ParamSource<f64>, L: ParamSource<f64>,
    {
        let shared = buf.len().min(mode.len()).min(sigma.len()).min(limit.len());
        let (active, _) = buf.split_at_mut(shared);
        self.fill_std_normal_f64(active);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let lim = limit.chunk::<4>(offset);
            let mut arr = [0.0f64; 4]; arr.copy_from_slice(chunk);
            for j in 0..4 { arr[j] = arr[j].clamp(-lim[j], lim[j]); }
            chunk.copy_from_slice(&fptricks::batch_fmadd_cols_f64(arr, sigma.chunk::<4>(offset), mode.chunk::<4>(offset)));
        }
        let rem = iter.into_remainder();
        let offset = shared & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            let lim = limit.get(offset + i);
            *val = val.clamp(-lim, lim).mul_add(sigma.get(offset + i), mode.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_ln_normal_f32<M, S>(&mut self, buf: &mut [f32], ln_mode: M, sigma_ln: S)
    where M: ParamSource<f32>, S: ParamSource<f32>,
    {
        let limit = buf.len().min(ln_mode.len()).min(sigma_ln.len());
        let (active, _) = buf.split_at_mut(limit);
        self.fill_normal_f32(active, ln_mode, sigma_ln);
        let mut iter = active.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let mut tmp = [0.0f32; 8]; tmp.copy_from_slice(chunk);
            chunk.copy_from_slice(&fptricks::batch_approx_exp_f32(tmp));
        }
        for val in iter.into_remainder() { *val = val.approx_exp(); }
    }

    #[inline]
    pub fn fill_ln_normal_f64<M, S>(&mut self, buf: &mut [f64], ln_mode: M, sigma_ln: S)
    where M: ParamSource<f64>, S: ParamSource<f64>,
    {
        let limit = buf.len().min(ln_mode.len()).min(sigma_ln.len());
        let (active, _) = buf.split_at_mut(limit);
        self.fill_normal_f64(active, ln_mode, sigma_ln);
        let mut iter = active.chunks_exact_mut(4);
        for chunk in iter.by_ref() {
            let mut tmp = [0.0f64; 4]; tmp.copy_from_slice(chunk);
            chunk.copy_from_slice(&fptricks::batch_approx_exp_f64(tmp));
        }
        for val in iter.into_remainder() { *val = val.approx_exp(); }
    }

    #[inline]
    pub fn fill_log10_normal_f32<M, S>(&mut self, buf: &mut [f32], log_mode: M, sigma_log: S)
    where M: ParamSource<f32>, S: ParamSource<f32>,
    {
        let limit = buf.len().min(log_mode.len()).min(sigma_log.len());
        let (active, _) = buf.split_at_mut(limit);
        self.fill_normal_f32(active, log_mode, sigma_log);
        let mut iter = active.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let mut tmp = [0.0f32; 8]; tmp.copy_from_slice(chunk);
            chunk.copy_from_slice(&fptricks::batch_approx_powf_f32(10.0, tmp));
        }
        for val in iter.into_remainder() { *val = 10.0_f32.powf(*val); }
    }

    #[inline]
    pub fn fill_log10_normal_f64<M, S>(&mut self, buf: &mut [f64], log_mode: M, sigma_log: S)
    where M: ParamSource<f64>, S: ParamSource<f64>,
    {
        let limit = buf.len().min(log_mode.len()).min(sigma_log.len());
        let (active, _) = buf.split_at_mut(limit);
        self.fill_normal_f64(active, log_mode, sigma_log);
        let mut iter = active.chunks_exact_mut(4);
        for chunk in iter.by_ref() {
            let mut tmp = [0.0f64; 4]; tmp.copy_from_slice(chunk);
            chunk.copy_from_slice(&fptricks::batch_approx_powf_f64(10.0, tmp));
        }
        for val in iter.into_remainder() { *val = 10.0_f64.powf(*val); }
    }

    #[inline]
    pub fn fill_split_normal_f32<M, SLO, SHI>(&mut self, buf: &mut [f32], mode: M, sigma_lo: SLO, sigma_hi: SHI)
    where M: ParamSource<f32>, SLO: ParamSource<f32>, SHI: ParamSource<f32>,
    {
        let limit = buf.len().min(mode.len()).min(sigma_lo.len()).min(sigma_hi.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(16);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 4;
            let u1_8 = self.next_f32_8(); let u2_8 = self.next_f32_8();
            let mut u1 = [0.0f32; 8]; let mut u2 = [0.0f32; 8];
            for j in 0..8 {
                u1[j] = (1.0 - u1_8[j]).max(f32::MIN_POSITIVE);
                u2[j] = 1.0 - u2_8[j];
            }
            let r = fptricks::batch_approx_sqrt_f32(fptricks::batch_fmadd_f32(fptricks::batch_approx_ln_f32(u1), -2.0, 0.0));
            let (s, c) = fptricks::batch_approx_sin_cos_f32(fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0));
            let m1 = mode.chunk::<8>(offset); let m2 = mode.chunk::<8>(offset + 8);
            let sl1 = sigma_lo.chunk::<8>(offset); let sl2 = sigma_lo.chunk::<8>(offset + 8);
            let sh1 = sigma_hi.chunk::<8>(offset); let sh2 = sigma_hi.chunk::<8>(offset + 8);
            for j in 0..8 {
                let x = r[j] * s[j];
                let m = m1[j];
                let xltz: u32 = ((x < 0.0) as u32).wrapping_neg();
                chunk[j << 1] = f32::from_bits((xltz & (m + x * sl1[j]).to_bits()) | (!xltz & (m + x * sh1[j]).to_bits()));
                let x2 = r[j] * c[j];
                let m2v = m2[j];
                let x2ltz: u32 = ((x2 < 0.0) as u32).wrapping_neg();
                chunk[(j << 1) + 1] = f32::from_bits((x2ltz & (m2v + x2 * sl2[j]).to_bits()) | (!x2ltz & (m2v + x2 * sh2[j]).to_bits()));
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !15;
        for (i, val) in rem.iter_mut().enumerate() {
            let x = self.next_std_normal_f32();
            let m = mode.get(offset + i);
            let s_lo = sigma_lo.get(offset + i); let s_hi = sigma_hi.get(offset + i);
            let xltz: u32 = ((x < 0.0) as u32).wrapping_neg();
            *val = f32::from_bits((xltz & (m + x * s_lo).to_bits()) | (!xltz & (m + x * s_hi).to_bits()));
        }
    }

    #[inline]
    pub fn fill_split_normal_f64<M, SLO, SHI>(&mut self, buf: &mut [f64], mode: M, sigma_lo: SLO, sigma_hi: SHI)
    where M: ParamSource<f64>, SLO: ParamSource<f64>, SHI: ParamSource<f64>,
    {
        let limit = buf.len().min(mode.len()).min(sigma_lo.len()).min(sigma_hi.len());
        let (active, _) = buf.split_at_mut(limit);
        self.fill_std_normal_f64(active);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let mut tmp = [0.0f64; 4]; tmp.copy_from_slice(chunk);
            chunk.copy_from_slice(&fptricks::batch_asymmetric_fma_cols_f64(
                tmp, mode.chunk::<4>(offset), sigma_lo.chunk::<4>(offset), sigma_hi.chunk::<4>(offset)));
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            let zlt: u64 = ((*val < 0.0) as u64).wrapping_neg();
            let zgeq: u64 = ((*val >= 0.0) as u64).wrapping_neg();
            let sigma = (sigma_lo.get(offset + i).to_bits() & zlt) | (sigma_hi.get(offset + i).to_bits() & zgeq);
            *val = val.mul_add(f64::from_bits(sigma), mode.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_split_normal_stuple_f32<M, T>(&mut self, buf: &mut [f32], mode: M, table: T)
    where M: ParamSource<f32>, T: ParamSource<(f32, f32)>,
    {
        let limit = buf.len().min(mode.len()).min(table.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(16);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 4;
            let u1_8 = self.next_f32_8(); let u2_8 = self.next_f32_8();
            let mut u1 = [0.0f32; 8]; let mut u2 = [0.0f32; 8];
            for j in 0..8 {
                u1[j] = (1.0 - u1_8[j]).max(f32::MIN_POSITIVE);
                u2[j] = 1.0 - u2_8[j];
            }
            let r = fptricks::batch_approx_sqrt_f32(fptricks::batch_fmadd_f32(fptricks::batch_approx_ln_f32(u1), -2.0, 0.0));
            let (s, c) = fptricks::batch_approx_sin_cos_f32(fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0));
            let m1 = mode.chunk::<8>(offset); let m2 = mode.chunk::<8>(offset + 8);
            let sigs1 = table.chunk::<8>(offset); let sigs2 = table.chunk::<8>(offset + 8);
            for j in 0..8 {
                let x = r[j] * s[j]; let (sl1, sh1) = sigs1[j];
                let xltz: u32 = ((x < 0.0) as u32).wrapping_neg();
                chunk[j << 1] = f32::from_bits((xltz & (m1[j] + x * sl1).to_bits()) | (!xltz & (m1[j] + x * sh1).to_bits()));
                let x2 = r[j] * c[j]; let (sl2, sh2) = sigs2[j];
                let x2ltz: u32 = ((x2 < 0.0) as u32).wrapping_neg();
                chunk[(j << 1) + 1] = f32::from_bits((x2ltz & (m2[j] + x2 * sl2).to_bits()) | (!x2ltz & (m2[j] + x2 * sh2).to_bits()));
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !15;
        for (i, val) in rem.iter_mut().enumerate() {
            let x = self.next_std_normal_f32();
            let m = mode.get(offset + i); let (s_lo, s_hi) = table.get(offset + i);
            let xltz: u32 = ((x < 0.0) as u32).wrapping_neg();
            *val = f32::from_bits((xltz & (m + x * s_lo).to_bits()) | (!xltz & (m + x * s_hi).to_bits()));
        }
    }

    #[inline]
    pub fn fill_split_normal_stuple_f64<M, T>(&mut self, buf: &mut [f64], mode: M, table: T)
    where M: ParamSource<f64>, T: ParamSource<(f64, f64)>,
    {
        let limit = buf.len().min(mode.len()).min(table.len());
        let (active, _) = buf.split_at_mut(limit);
        self.fill_std_normal_f64(active);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let mut tmp = [0.0f64; 4]; tmp.copy_from_slice(chunk);
            let sigs = table.chunk::<4>(offset);
            let mut sl = [0.0f64; 4]; let mut sh = [0.0f64; 4];
            for j in 0..4 { sl[j] = sigs[j].0; sh[j] = sigs[j].1; }
            chunk.copy_from_slice(&fptricks::batch_asymmetric_fma_cols_f64(tmp, mode.chunk::<4>(offset), sl, sh));
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            let (sig_lo, sig_hi) = table.get(offset + i);
            let zlt: u64 = ((*val < 0.0) as u64).wrapping_neg();
            let zgeq: u64 = ((*val >= 0.0) as u64).wrapping_neg();
            let sigma = (sig_lo.to_bits() & zlt) | (sig_hi.to_bits() & zgeq);
            *val = val.mul_add(f64::from_bits(sigma), mode.get(offset + i));
        }
    }

    // -------------------------------------------------------------------------
    // make_filled_* — allocate, fill, and return a [T; N] array (stack)
    // -------------------------------------------------------------------------

    #[inline]
    pub fn make_filled_std_normal_f32<const N: usize>(&mut self) -> [f32; N] {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let mut iter = slice.chunks_exact_mut(16);
        for chunk in iter.by_ref() {
            let u1_8 = self.next_f32_8(); let u2_8 = self.next_f32_8();
            let mut u1 = [0.0f32; 8]; let mut u2 = [0.0f32; 8];
            for j in 0..8 {
                u1[j] = (1.0 - u1_8[j]).max(f32::MIN_POSITIVE);
                u2[j] = 1.0 - u2_8[j];
            }
            let r = fptricks::batch_approx_sqrt_f32(fptricks::batch_fmadd_f32(fptricks::batch_approx_ln_f32(u1), -2.0, 0.0));
            let (s, c) = fptricks::batch_approx_sin_cos_f32(fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0));
            let ps = fptricks::batch_mul_cols_f32(&r, &s);
            let pc = fptricks::batch_mul_cols_f32(&r, &c);

            chunk[0].write(ps[0]); chunk[1].write(pc[0]);
            chunk[2].write(ps[1]); chunk[3].write(pc[1]);
            chunk[4].write(ps[2]); chunk[5].write(pc[2]);
            chunk[6].write(ps[3]); chunk[7].write(pc[3]);
            chunk[8].write(ps[4]); chunk[9].write(pc[4]);
            chunk[10].write(ps[5]); chunk[11].write(pc[5]);
            chunk[12].write(ps[6]); chunk[13].write(pc[6]);
            chunk[14].write(ps[7]); chunk[15].write(pc[7]);
        }
        let rem = iter.into_remainder();
        let mut i = 0;
        while i + 1 < rem.len() {
            let (s, c) = self.next_std_normal_pair_f32();
            rem[i].write(s); rem[i + 1].write(c); i += 2;
        }
        if i < rem.len() { rem[i].write(self.next_std_normal_f32()); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_std_normal_f64<const N: usize>(&mut self) -> [f64; N] {
        let mut buf = MaybeUninit::<[f64; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f64>, N) };
        let mut iter = slice.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let u1_4 = self.next_f64_4(); let u2_4 = self.next_f64_4();
            let mut u1 = [0.0f64; 4]; let mut u2 = [0.0f64; 4];
            for j in 0..4 {
                u1[j] = (1.0 - u1_4[j]).max(f64::MIN_POSITIVE);
                u2[j] = 1.0 - u2_4[j];
            }
            let r = fptricks::batch_approx_sqrt_f64(fptricks::batch_fmadd_f64(fptricks::batch_approx_ln_f64(u1), -2.0, 0.0));
            let (s, c) = fptricks::batch_approx_sin_cos_f64(fptricks::batch_fmadd_f64(u2, Self::TWO_PI_F64, 0.0));
            let ps = fptricks::batch_mul_cols_f64(&r, &s);
            let pc = fptricks::batch_mul_cols_f64(&r, &c);

            chunk[0].write(ps[0]); chunk[1].write(pc[0]);
            chunk[2].write(ps[1]); chunk[3].write(pc[1]);
            chunk[4].write(ps[2]); chunk[5].write(pc[2]);
            chunk[6].write(ps[3]); chunk[7].write(pc[3]);
        }
        let rem = iter.into_remainder();
        let mut i = 0;
        while i + 1 < rem.len() {
            let (s, c) = self.next_std_normal_pair_f64();
            rem[i].write(s); rem[i + 1].write(c); i += 2;
        }
        if i < rem.len() { rem[i].write(self.next_std_normal_f64()); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_normal_f32<M, S, const N: usize>(&mut self, mode: M, sigma: S) -> [f32; N]
    where M: ParamSource<f32>, S: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let limit = slice.len().min(mode.len()).min(sigma.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(16);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 4;
            let u1_8 = self.next_f32_8(); let u2_8 = self.next_f32_8();
            let mut u1 = [0.0f32; 8]; let mut u2 = [0.0f32; 8];
            for j in 0..8 {
                u1[j] = (1.0 - u1_8[j]).max(f32::MIN_POSITIVE);
                u2[j] = 1.0 - u2_8[j];
            }
            let r = fptricks::batch_approx_sqrt_f32(fptricks::batch_fmadd_f32(fptricks::batch_approx_ln_f32(u1), -2.0, 0.0));
            let (s, c) = fptricks::batch_approx_sin_cos_f32(fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0));
            let m1 = mode.chunk::<8>(offset); let m2 = mode.chunk::<8>(offset + 8);
            let s1 = sigma.chunk::<8>(offset); let s2 = sigma.chunk::<8>(offset + 8);
            let res1 = fptricks::batch_add_cols_f32(&m1, &fptricks::batch_mul_3_cols_f32(&r, &s, &s1));
            let res2 = fptricks::batch_add_cols_f32(&m2, &fptricks::batch_mul_3_cols_f32(&r, &c, &s2));
            
            chunk[0].write(res1[0]); chunk[1].write(res2[0]);
            chunk[2].write(res1[1]); chunk[3].write(res2[1]);
            chunk[4].write(res1[2]); chunk[5].write(res2[2]);
            chunk[6].write(res1[3]); chunk[7].write(res2[3]);
            chunk[8].write(res1[4]); chunk[9].write(res2[4]);
            chunk[10].write(res1[5]); chunk[11].write(res2[5]);
            chunk[12].write(res1[6]); chunk[13].write(res2[6]);
            chunk[14].write(res1[7]); chunk[15].write(res2[7]);
        }
        let rem = iter.into_remainder();
        let offset = limit & !15;
        let mut i = 0;
        while i + 1 < rem.len() {
            let (s, c) = self.next_std_normal_pair_f32();
            rem[i].write(mode.get(offset + i) + s * sigma.get(offset + i));
            rem[i + 1].write(mode.get(offset + i + 1) + c * sigma.get(offset + i + 1));
            i += 2;
        }
        if i < rem.len() {
            rem[i].write(self.next_std_normal_f32().mul_add(sigma.get(offset + i), mode.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_normal_f64<M, S, const N: usize>(&mut self, mode: M, sigma: S) -> [f64; N]
    where M: ParamSource<f64>, S: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[f64; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f64>, N) };
        let limit = slice.len().min(mode.len()).min(sigma.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let u1_4 = self.next_f64_4(); let u2_4 = self.next_f64_4();
            let mut u1 = [0.0f64; 4]; let mut u2 = [0.0f64; 4];
            for j in 0..4 {
                u1[j] = (1.0 - u1_4[j]).max(f64::MIN_POSITIVE);
                u2[j] = 1.0 - u2_4[j];
            }
            let r = fptricks::batch_approx_sqrt_f64(fptricks::batch_fmadd_f64(fptricks::batch_approx_ln_f64(u1), -2.0, 0.0));
            let (s, c) = fptricks::batch_approx_sin_cos_f64(fptricks::batch_fmadd_f64(u2, Self::TWO_PI_F64, 0.0));
            let m1 = mode.chunk::<4>(offset); let m2 = mode.chunk::<4>(offset + 4);
            let s1 = sigma.chunk::<4>(offset); let s2 = sigma.chunk::<4>(offset + 4);
            let res1 = fptricks::batch_add_cols_f64(&m1, &fptricks::batch_mul_3_cols_f64(&r, &s, &s1));
            let res2 = fptricks::batch_add_cols_f64(&m2, &fptricks::batch_mul_3_cols_f64(&r, &c, &s2));
            
            chunk[0].write(res1[0]); chunk[1].write(res2[0]);
            chunk[2].write(res1[1]); chunk[3].write(res2[1]);
            chunk[4].write(res1[2]); chunk[5].write(res2[2]);
            chunk[6].write(res1[3]); chunk[7].write(res2[3]);
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        let mut i = 0;
        while i + 1 < rem.len() {
            let (s, c) = self.next_std_normal_pair_f64();
            rem[i].write(mode.get(offset + i) + s * sigma.get(offset + i));
            rem[i + 1].write(mode.get(offset + i + 1) + c * sigma.get(offset + i + 1));
            i += 2;
        }
        if i < rem.len() {
            rem[i].write(self.next_std_normal_f64().mul_add(sigma.get(offset + i), mode.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_normal_w_clamped_err_f32<M, S, L, const N: usize>(&mut self, mode: M, sigma: S, limit: L) -> [f32; N]
    where M: ParamSource<f32>, S: ParamSource<f32>, L: ParamSource<f32>,
    {
        let mut z: [f32; N] = self.make_filled_std_normal_f32();
        let shared = N.min(mode.len()).min(sigma.len()).min(limit.len());
        let mut iter = z[..shared].chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let lim = limit.chunk::<8>(offset);
            let mut arr = [0.0f32; 8]; arr.copy_from_slice(chunk);
            for j in 0..8 { arr[j] = arr[j].clamp(-lim[j], lim[j]); }
            chunk.copy_from_slice(&fptricks::batch_fmadd_cols_f32(arr, sigma.chunk::<8>(offset), mode.chunk::<8>(offset)));
        }
        let rem = iter.into_remainder();
        let offset = shared & !7;
        for (i, val) in rem.iter_mut().enumerate() {
            let lim = limit.get(offset + i);
            *val = val.clamp(-lim, lim).mul_add(sigma.get(offset + i), mode.get(offset + i));
        }
        z
    }

    #[inline]
    pub fn make_filled_normal_w_clamped_err_f64<M, S, L, const N: usize>(&mut self, mode: M, sigma: S, limit: L) -> [f64; N]
    where M: ParamSource<f64>, S: ParamSource<f64>, L: ParamSource<f64>,
    {
        let mut z: [f64; N] = self.make_filled_std_normal_f64();
        let shared = N.min(mode.len()).min(sigma.len()).min(limit.len());
        let mut iter = z[..shared].chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let lim = limit.chunk::<4>(offset);
            let mut arr = [0.0f64; 4]; arr.copy_from_slice(chunk);
            for j in 0..4 { arr[j] = arr[j].clamp(-lim[j], lim[j]); }
            chunk.copy_from_slice(&fptricks::batch_fmadd_cols_f64(arr, sigma.chunk::<4>(offset), mode.chunk::<4>(offset)));
        }
        let rem = iter.into_remainder();
        let offset = shared & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            let lim = limit.get(offset + i);
            *val = val.clamp(-lim, lim).mul_add(sigma.get(offset + i), mode.get(offset + i));
        }
        z
    }

    #[inline]
    pub fn make_filled_ln_normal_f32<M, S, const N: usize>(&mut self, ln_mode: M, sigma_ln: S) -> [f32; N]
    where M: ParamSource<f32>, S: ParamSource<f32>,
    {
        let mut arr: [f32; N] = self.make_filled_normal_f32(ln_mode, sigma_ln);
        let mut iter = arr.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let mut tmp = [0.0f32; 8]; tmp.copy_from_slice(chunk);
            chunk.copy_from_slice(&fptricks::batch_approx_exp_f32(tmp));
        }
        for val in iter.into_remainder() { *val = val.approx_exp(); }
        arr
    }

    #[inline]
    pub fn make_filled_ln_normal_f64<M, S, const N: usize>(&mut self, ln_mode: M, sigma_ln: S) -> [f64; N]
    where M: ParamSource<f64>, S: ParamSource<f64>,
    {
        let mut arr: [f64; N] = self.make_filled_normal_f64(ln_mode, sigma_ln);
        let mut iter = arr.chunks_exact_mut(4);
        for chunk in iter.by_ref() {
            let mut tmp = [0.0f64; 4]; tmp.copy_from_slice(chunk);
            chunk.copy_from_slice(&fptricks::batch_approx_exp_f64(tmp));
        }
        for val in iter.into_remainder() { *val = val.approx_exp(); }
        arr
    }

    #[inline]
    pub fn make_filled_log10_normal_f32<M, S, const N: usize>(&mut self, log_mode: M, sigma_log: S) -> [f32; N]
    where M: ParamSource<f32>, S: ParamSource<f32>,
    {
        let mut arr: [f32; N] = self.make_filled_normal_f32(log_mode, sigma_log);
        let mut iter = arr.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let mut tmp = [0.0f32; 8]; tmp.copy_from_slice(chunk);
            chunk.copy_from_slice(&fptricks::batch_approx_powf_f32(10.0, tmp));
        }
        for val in iter.into_remainder() { *val = 10.0_f32.powf(*val); }
        arr
    }

    #[inline]
    pub fn make_filled_log10_normal_f64<M, S, const N: usize>(&mut self, log_mode: M, sigma_log: S) -> [f64; N]
    where M: ParamSource<f64>, S: ParamSource<f64>,
    {
        let mut arr: [f64; N] = self.make_filled_normal_f64(log_mode, sigma_log);
        let mut iter = arr.chunks_exact_mut(4);
        for chunk in iter.by_ref() {
            let mut tmp = [0.0f64; 4]; tmp.copy_from_slice(chunk);
            chunk.copy_from_slice(&fptricks::batch_approx_powf_f64(10.0, tmp));
        }
        for val in iter.into_remainder() { *val = 10.0_f64.powf(*val); }
        arr
    }

    #[inline]
    pub fn make_filled_split_normal_f32<M, SLO, SHI, const N: usize>(&mut self, mode: M, sigma_lo: SLO, sigma_hi: SHI) -> [f32; N]
    where M: ParamSource<f32>, SLO: ParamSource<f32>, SHI: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let limit = slice.len().min(mode.len()).min(sigma_lo.len()).min(sigma_hi.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(16);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 4;
            let u1_8 = self.next_f32_8(); let u2_8 = self.next_f32_8();
            let mut u1 = [0.0f32; 8]; let mut u2 = [0.0f32; 8];
            for j in 0..8 {
                u1[j] = (1.0 - u1_8[j]).max(f32::MIN_POSITIVE);
                u2[j] = 1.0 - u2_8[j];
            }
            let r = fptricks::batch_approx_sqrt_f32(fptricks::batch_fmadd_f32(fptricks::batch_approx_ln_f32(u1), -2.0, 0.0));
            let (s, c) = fptricks::batch_approx_sin_cos_f32(fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0));
            let m1 = mode.chunk::<8>(offset); let m2 = mode.chunk::<8>(offset + 8);
            let sl1 = sigma_lo.chunk::<8>(offset); let sl2 = sigma_lo.chunk::<8>(offset + 8);
            let sh1 = sigma_hi.chunk::<8>(offset); let sh2 = sigma_hi.chunk::<8>(offset + 8);
            for j in 0..8 {
                let x = r[j] * s[j]; let m = m1[j];
                let xltz: u32 = ((x < 0.0) as u32).wrapping_neg();
                chunk[j << 1].write(f32::from_bits((xltz & (m + x * sl1[j]).to_bits()) | (!xltz & (m + x * sh1[j]).to_bits())));
                let x2 = r[j] * c[j]; let m2v = m2[j];
                let x2ltz: u32 = ((x2 < 0.0) as u32).wrapping_neg();
                chunk[(j << 1) + 1].write(f32::from_bits((x2ltz & (m2v + x2 * sl2[j]).to_bits()) | (!x2ltz & (m2v + x2 * sh2[j]).to_bits())));
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !15;
        for (i, slot) in rem.iter_mut().enumerate() {
            let x = self.next_std_normal_f32();
            let m = mode.get(offset + i);
            let s_lo = sigma_lo.get(offset + i); let s_hi = sigma_hi.get(offset + i);
            let xltz: u32 = ((x < 0.0) as u32).wrapping_neg();
            slot.write(f32::from_bits((xltz & (m + x * s_lo).to_bits()) | (!xltz & (m + x * s_hi).to_bits())));
        }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_split_normal_f64<M, SLO, SHI, const N: usize>(&mut self, mode: M, sigma_lo: SLO, sigma_hi: SHI) -> [f64; N]
    where M: ParamSource<f64>, SLO: ParamSource<f64>, SHI: ParamSource<f64>,
    {
        let mut arr: [f64; N] = self.make_filled_std_normal_f64();
        let limit = N.min(mode.len()).min(sigma_lo.len()).min(sigma_hi.len());
        let mut iter = arr[..limit].chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let mut tmp = [0.0f64; 4]; tmp.copy_from_slice(chunk);
            chunk.copy_from_slice(&fptricks::batch_asymmetric_fma_cols_f64(
                tmp, mode.chunk::<4>(offset), sigma_lo.chunk::<4>(offset), sigma_hi.chunk::<4>(offset)));
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            let zlt: u64 = ((*val < 0.0) as u64).wrapping_neg();
            let zgeq: u64 = ((*val >= 0.0) as u64).wrapping_neg();
            let sigma = (sigma_lo.get(offset + i).to_bits() & zlt) | (sigma_hi.get(offset + i).to_bits() & zgeq);
            *val = val.mul_add(f64::from_bits(sigma), mode.get(offset + i));
        }
        arr
    }

    #[inline]
    pub fn make_filled_split_normal_stuple_f32<M, T, const N: usize>(&mut self, mode: M, table: T) -> [f32; N]
    where M: ParamSource<f32>, T: ParamSource<(f32, f32)>,
    {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let limit = slice.len().min(mode.len()).min(table.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(16);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 4;
            let u1_8 = self.next_f32_8(); let u2_8 = self.next_f32_8();
            let mut u1 = [0.0f32; 8]; let mut u2 = [0.0f32; 8];
            for j in 0..8 {
                u1[j] = (1.0 - u1_8[j]).max(f32::MIN_POSITIVE);
                u2[j] = 1.0 - u2_8[j];
            }
            let r = fptricks::batch_approx_sqrt_f32(fptricks::batch_fmadd_f32(fptricks::batch_approx_ln_f32(u1), -2.0, 0.0));
            let (s, c) = fptricks::batch_approx_sin_cos_f32(fptricks::batch_fmadd_f32(u2, Self::TWO_PI_F32, 0.0));
            let m1 = mode.chunk::<8>(offset); let m2 = mode.chunk::<8>(offset + 8);
            let sigs1 = table.chunk::<8>(offset); let sigs2 = table.chunk::<8>(offset + 8);
            for j in 0..8 {
                let x = r[j] * s[j]; let (sl1, sh1) = sigs1[j];
                let xltz: u32 = ((x < 0.0) as u32).wrapping_neg();
                chunk[j << 1].write(f32::from_bits((xltz & (m1[j] + x * sl1).to_bits()) | (!xltz & (m1[j] + x * sh1).to_bits())));
                let x2 = r[j] * c[j]; let (sl2, sh2) = sigs2[j];
                let x2ltz: u32 = ((x2 < 0.0) as u32).wrapping_neg();
                chunk[(j << 1) + 1].write(f32::from_bits((x2ltz & (m2[j] + x2 * sl2).to_bits()) | (!x2ltz & (m2[j] + x2 * sh2).to_bits())));
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !15;
        for (i, slot) in rem.iter_mut().enumerate() {
            let x = self.next_std_normal_f32();
            let m = mode.get(offset + i); let (s_lo, s_hi) = table.get(offset + i);
            let xltz: u32 = ((x < 0.0) as u32).wrapping_neg();
            slot.write(f32::from_bits((xltz & (m + x * s_lo).to_bits()) | (!xltz & (m + x * s_hi).to_bits())));
        }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_split_normal_stuple_f64<M, T, const N: usize>(&mut self, mode: M, table: T) -> [f64; N]
    where M: ParamSource<f64>, T: ParamSource<(f64, f64)>,
    {
        let mut arr: [f64; N] = self.make_filled_std_normal_f64();
        let limit = N.min(mode.len()).min(table.len());
        let mut iter = arr[..limit].chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let mut tmp = [0.0f64; 4]; tmp.copy_from_slice(chunk);
            let sigs = table.chunk::<4>(offset);
            let mut sl = [0.0f64; 4]; let mut sh = [0.0f64; 4];
            for j in 0..4 { sl[j] = sigs[j].0; sh[j] = sigs[j].1; }
            chunk.copy_from_slice(&fptricks::batch_asymmetric_fma_cols_f64(tmp, mode.chunk::<4>(offset), sl, sh));
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, val) in rem.iter_mut().enumerate() {
            let (sig_lo, sig_hi) = table.get(offset + i);
            let zlt: u64 = ((*val < 0.0) as u64).wrapping_neg();
            let zgeq: u64 = ((*val >= 0.0) as u64).wrapping_neg();
            let sigma = (sig_lo.to_bits() & zlt) | (sig_hi.to_bits() & zgeq);
            *val = val.mul_add(f64::from_bits(sigma), mode.get(offset + i));
        }
        arr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calculate_stats<F>(mut generator: F, n: usize) -> (f64, f64)
    where F: FnMut() -> f64,
    {
        let mut sum = 0.0; let mut sum_sq = 0.0;
        for _ in 0..n { let val = generator(); sum += val; sum_sq += val * val; }
        let mean = sum / n as f64;
        (mean, sum_sq / n as f64 - mean * mean)
    }

    #[test]
    fn test_normal_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let (mean, var) = calculate_stats(|| rng.next_std_normal_f32() as f64, n);
        assert!(mean.abs() < 0.1); assert!((var - 1.0).abs() < 0.15);
        let (mean, var) = calculate_stats(|| rng.next_std_normal_f64(), n);
        assert!(mean.abs() < 0.1); assert!((var - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_normal_pair_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let mut sum_s = 0.0; let mut sum_c = 0.0;
        let mut sum_sq_s = 0.0; let mut sum_sq_c = 0.0;
        for _ in 0..n {
            let (s, c) = rng.next_std_normal_pair_f32();
            sum_s += s as f64; sum_c += c as f64;
            sum_sq_s += (s * s) as f64; sum_sq_c += (c * c) as f64;
        }
        let mean_s = sum_s / n as f64; let var_s = sum_sq_s / n as f64 - mean_s * mean_s;
        let mean_c = sum_c / n as f64; let var_c = sum_sq_c / n as f64 - mean_c * mean_c;
        assert!(mean_s.abs() < 0.1); assert!((var_s - 1.0).abs() < 0.15);
        assert!(mean_c.abs() < 0.1); assert!((var_c - 1.0).abs() < 0.15);
    }

    #[test]
    fn test_fill_std_normal_f32() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let mut buf = vec![0.0f32; n];
        rng.fill_std_normal_f32(&mut buf);
        let mut sum = 0.0; let mut sum_sq = 0.0;
        for &v in &buf { sum += v as f64; sum_sq += (v * v) as f64; }
        let mean = sum / n as f64; let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.1); assert!((var - 1.0).abs() < 0.15);
    }

    #[test]
    fn test_make_filled_std_normal_f32() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let buf: [f32; 100_000] = rng.make_filled_std_normal_f32();
        let mut sum = 0.0; let mut sum_sq = 0.0;
        for &v in &buf { sum += v as f64; sum_sq += (v * v) as f64; }
        let mean = sum / n as f64; let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.1); assert!((var - 1.0).abs() < 0.15);
    }

    #[test]
    fn test_split_normal_table_correctness() {
        let mut rng = WyRand::new(1);
        let n = 1024usize;
        let mode = 0.0f32;
        let table = vec![(1.0f32, 2.0f32); n];
        let buf: [f32; 1024] = rng.make_filled_split_normal_stuple_f32(mode, &table);
        let mut sum_lo = 0.0f32; let mut count_lo = 0;
        let mut sum_hi = 0.0f32; let mut count_hi = 0;
        for &v in &buf {
            if v < 0.0 { sum_lo += v; count_lo += 1; }
            else { sum_hi += v; count_hi += 1; }
        }
        if count_lo > 0 { let m = sum_lo / count_lo as f32; assert!(m < -0.5 && m > -1.1); }
        if count_hi > 0 { let m = sum_hi / count_hi as f32; assert!(m > 1.2 && m < 2.0); }
    }

    #[test]
    fn test_radian_sampling_stats() {
        let mut rng = WyRand::new(42);
        let n = 100_000;
        let mut sum_cos = 0.0;
        for _ in 0..n { let i = rng.next_isotropic_polar_angle_f32(); sum_cos += i.cos() as f64; }
        assert!((sum_cos / n as f64).abs() < 0.05);
        let mut sum_cos = 0.0;
        for _ in 0..n { let i = rng.next_isotropic_polar_angle_f64(); sum_cos += i.cos() as f64; }
        assert!((sum_cos / n as f64).abs() < 0.05);
    }

    #[test]
    fn test_normal_nan_stability() {
        let mut rng = WyRand::new(12345);
        // Stress test millions of draws to verify clamping and NaN-free operation
        for _ in 0..10_000_000 {
            let x = rng.next_std_normal_f32();
            assert!(!x.is_nan() && !x.is_infinite(), "NaN or Inf detected in scalar StdNormal");
        }
        
        let mut buf = [0.0f32; 1024];
        for _ in 0..10_000 {
            rng.fill_std_normal_f32(&mut buf);
            for &v in &buf {
                assert!(!v.is_nan() && !v.is_infinite(), "NaN or Inf detected in batch StdNormal");
            }
        }
    }
}
