use std::mem::MaybeUninit;
use crate::WyRand;
use crate::traits::ParamSource;
use fptricks::*;

impl WyRand {
    #[inline(always)]
    pub fn next_power_law(&mut self, min: f64, max: f64, k: f64) -> f64 {
        let roll = self.next_uniform_f64();
        let kp1 = k + 1.0;
        let kp1lt: u64 = ((kp1.abs() < 1e-9) as u64).wrapping_neg();
        f64::from_bits(((min * (max/min).approx_powf(roll)).to_bits() & kp1lt) | (!kp1lt & {
            let min_pow = min.approx_powf(kp1);
            let max_pow = max.approx_powf(kp1);
            (roll * (max_pow - min_pow) + min_pow).approx_powf(1.0 / kp1)
        }.to_bits()))
    }

    #[inline(always)]
    pub fn next_rayleigh_f32(&mut self, sigma: f32) -> f32 {
        (-( 1.0 - self.next_uniform_f32()).approx_ln().fast_mul2()).approx_sqrt() * sigma
    }

    #[inline(always)]
    pub fn next_rayleigh_f64(&mut self, sigma: f64) -> f64 {
        (-(1.0 - self.next_uniform_f64()).approx_ln().fast_mul2()).approx_sqrt() * sigma
    }

    #[inline(always)]
    pub fn next_gamma_f32(&mut self, alpha: f32) -> f32 {
        if alpha <= 0.0 { return 0.0; }
        if alpha < 1.0 {
            let u1 = 1.0 - self.next_uniform_f32();
            return self.next_gamma_f32(alpha + 1.0) * (u1.approx_ln() / alpha).approx_exp();
        }
        const ONE_THIRD: f32 = 1.0 / 3.0;
        let d = alpha - ONE_THIRD;
        let c = 1.0 / (9.0 * d).approx_sqrt();
        loop {
            let z = self.next_std_normal_f32();
            let v = 1.0 + c * z;
            if v <= 0.0 { continue; }
            let v = v.approx_powi(3);
            let u = 1.0 - self.next_uniform_f32();
            let z_sq = z * z;
            if u < 1.0 - 0.0331 * z_sq * z_sq { return d * v; }
            if u.approx_ln() < 0.5 * z_sq + d * (1.0 - v + v.approx_ln()) { return d * v; }
        }
    }

    #[inline(always)]
    pub fn next_gamma_f64(&mut self, alpha: f64) -> f64 {
        if alpha <= 0.0 { return 0.0; }
        if alpha < 1.0 {
            let u1 = 1.0 - self.next_uniform_f64();
            return self.next_gamma_f64(alpha + 1.0) * (u1.approx_ln() / alpha).approx_exp();
        }
        const ONE_THIRD: f64 = 1.0 / 3.0;
        let d = alpha - ONE_THIRD;
        let c = 1.0 / (9.0 * d).approx_sqrt();
        loop {
            let z = self.next_std_normal_f64();
            let v = 1.0 + c * z;
            if v <= 0.0 { continue; }
            let v = v * v * v;
            let u = 1.0 - self.next_uniform_f64();
            let z_sq = z * z;
            if u < 1.0 - 0.0331 * z_sq * z_sq { return d * v; }
            if u.approx_ln() < 0.5 * z_sq + d * (1.0 - v + v.approx_ln()) { return d * v; }
        }
    }

    #[inline(always)]
    pub fn next_beta_f32(&mut self, alpha: f32, beta: f32) -> f32 {
        let a = self.next_gamma_f32(alpha); let b = self.next_gamma_f32(beta);
        let sum = a + b;
        f32::from_bits((a / sum).to_bits() & ((sum != 0.0) as u32).wrapping_neg())
    }

    #[inline(always)]
    pub fn next_beta_f64(&mut self, alpha: f64, beta: f64) -> f64 {
        let a = self.next_gamma_f64(alpha); let b = self.next_gamma_f64(beta);
        let sum = a + b;
        f64::from_bits((a / sum).to_bits() & ((sum != 0.0) as u64).wrapping_neg())
    }

    #[inline(always)]
    pub fn next_chi_squared_f32(&mut self, k: f32) -> f32 { self.next_gamma_f32(k * 0.5).fast_mul2() }

    #[inline(always)]
    pub fn next_chi_squared_f64(&mut self, k: f64) -> f64 { self.next_gamma_f64(k * 0.5).fast_mul2() }

    #[inline(always)]
    pub fn next_poisson_u32(&mut self, lambda: f32) -> u32 {
        if lambda <= 0.0 { return 0; }
        let l = (-lambda).approx_exp();
        let mut k = 0u32; let mut p = 1.0_f32;
        loop { k += 1; p *= self.next_uniform_f32(); if p <= l { break; } }
        k - 1
    }

    #[inline(always)]
    pub fn next_poisson_f64_u32(&mut self, lambda: f64) -> u32 {
        if lambda <= 0.0 { return 0; }
        let l = (-lambda).approx_exp();
        let mut k = 0u32; let mut p = 1.0_f64;
        loop { k += 1; p *= self.next_uniform_f64(); if p <= l { break; } }
        k - 1
    }

    // -------------------------------------------------------------------------
    // fill_* — write into caller-owned slice (heap-friendly, runtime length)
    // -------------------------------------------------------------------------

    #[inline]
    pub fn fill_rayleigh_f32<S>(&mut self, buf: &mut [f32], sigma: S)
    where S: ParamSource<f32>,
    {
        let limit = buf.len().min(sigma.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let mut u = [0.0f32; 8];
            for j in 0..8 { u[j] = 1.0 - self.next_uniform_f32(); }
            let r = fptricks::batch_approx_sqrt_f32(fptricks::batch_fmadd_f32(fptricks::batch_approx_ln_f32(u), -2.0, 0.0));
            let s_chunk = sigma.chunk::<8>(offset);
            for j in 0..8 { chunk[j] = r[j] * s_chunk[j]; }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() { *slot = self.next_rayleigh_f32(sigma.get(offset + i)); }
    }

    #[inline]
    pub fn fill_rayleigh_f64<S>(&mut self, buf: &mut [f64], sigma: S)
    where S: ParamSource<f64>,
    {
        let limit = buf.len().min(sigma.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let mut u = [0.0f64; 4];
            for j in 0..4 { u[j] = 1.0 - self.next_uniform_f64(); }
            let r = fptricks::batch_approx_sqrt_f64(fptricks::batch_fmadd_f64(fptricks::batch_approx_ln_f64(u), -2.0, 0.0));
            let s_chunk = sigma.chunk::<4>(offset);
            for j in 0..4 { chunk[j] = r[j] * s_chunk[j]; }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() { *slot = self.next_rayleigh_f64(sigma.get(offset + i)); }
    }

    #[inline]
    pub fn fill_gamma_f32<A>(&mut self, buf: &mut [f32], alpha: A)
    where A: ParamSource<f32>,
    {
        let limit = buf.len().min(alpha.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<8>(i << 3);
            for j in 0..8 { chunk[j] = self.next_gamma_f32(a[j]); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() { *slot = self.next_gamma_f32(alpha.get(offset + i)); }
    }

    #[inline]
    pub fn fill_gamma_f64<A>(&mut self, buf: &mut [f64], alpha: A)
    where A: ParamSource<f64>,
    {
        let limit = buf.len().min(alpha.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<4>(i << 2);
            for j in 0..4 { chunk[j] = self.next_gamma_f64(a[j]); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() { *slot = self.next_gamma_f64(alpha.get(offset + i)); }
    }

    #[inline]
    pub fn fill_poisson_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where L: ParamSource<f32>,
    {
        let limit = buf.len().min(lambda.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<8>(i << 3);
            let thresholds = fptricks::batch_approx_exp_f32([-l_arr[0],-l_arr[1],-l_arr[2],-l_arr[3],-l_arr[4],-l_arr[5],-l_arr[6],-l_arr[7]]);
            for j in 0..8 {
                if l_arr[j] <= 0.0 { chunk[j] = 0; continue; }
                let l = thresholds[j]; let mut k = 0u32; let mut p = 1.0_f32;
                loop { k += 1; p *= self.next_uniform_f32(); if p <= l { break; } }
                chunk[j] = k - 1;
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() { *slot = self.next_poisson_u32(lambda.get(offset + i)); }
    }

    #[inline]
    pub fn fill_poisson_f64_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where L: ParamSource<f64>,
    {
        let limit = buf.len().min(lambda.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<4>(i << 2);
            let thresholds = fptricks::batch_approx_exp_f64([-l_arr[0],-l_arr[1],-l_arr[2],-l_arr[3]]);
            for j in 0..4 {
                if l_arr[j] <= 0.0 { chunk[j] = 0; continue; }
                let l = thresholds[j]; let mut k = 0u32; let mut p = 1.0_f64;
                loop { k += 1; p *= self.next_uniform_f64(); if p <= l { break; } }
                chunk[j] = k - 1;
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() { *slot = self.next_poisson_f64_u32(lambda.get(offset + i)); }
    }

    #[inline]
    pub fn fill_poisson_collecting_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where L: ParamSource<f32>,
    {
        let limit = buf.len().min(lambda.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<8>(i << 3);
            let thresholds = fptricks::batch_approx_exp_f32([-l_arr[0],-l_arr[1],-l_arr[2],-l_arr[3],-l_arr[4],-l_arr[5],-l_arr[6],-l_arr[7]]);
            let mut counts = [0u32; 8]; let mut p = self.next_f32_8();
            let mut mask = 0u8;
            for j in 0..8 {
                mask |= ((l_arr[j] > 0.0) as u8 & (p[j] > thresholds[j]) as u8) << j;
            }
            while mask != 0 {
                let mut next_mask = 0u8; let u = self.next_f32_8();
                for j in 0..8 {
                    let active = (mask >> j) & 1;
                    let am: u32 = (active as u32).wrapping_neg();
                    counts[j] += active as u32;
                    p[j] *= f32::from_bits((u[j].to_bits() & am) | (1.0f32.to_bits() & !am));
                    next_mask |= (still_gt(p[j], thresholds[j]) & active) << j;
                }
                mask = next_mask;
            }
            chunk.copy_from_slice(&counts);
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() { *slot = self.next_poisson_u32(lambda.get(offset + i)); }
    }

    #[inline]
    pub fn fill_poisson_collecting_f64_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where L: ParamSource<f64>,
    {
        let limit = buf.len().min(lambda.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<4>(i << 2);
            let thresholds = fptricks::batch_approx_exp_f64([-l_arr[0],-l_arr[1],-l_arr[2],-l_arr[3]]);
            let mut counts = [0u32; 4]; let mut p = self.next_f64_4();
            let mut mask = 0u8;
            for j in 0..4 {
                mask |= ((l_arr[j] > 0.0) as u8 & (p[j] > thresholds[j]) as u8) << j;
            }
            while mask != 0 {
                let mut next_mask = 0u8; let u = self.next_f64_4();
                for j in 0..4 {
                    let active = (mask >> j) & 1;
                    let am: u64 = (active as u64).wrapping_neg();
                    counts[j] += active as u32;
                    p[j] *= f64::from_bits((u[j].to_bits() & am) | (1.0f64.to_bits() & !am));
                    next_mask |= (still_gt_f64(p[j], thresholds[j]) & active) << j;
                }
                mask = next_mask;
            }
            chunk.copy_from_slice(&counts);
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() { *slot = self.next_poisson_f64_u32(lambda.get(offset + i)); }
    }

    #[inline]
    pub fn fill_beta_f32<A, B>(&mut self, buf: &mut [f32], alpha: A, beta: B)
    where A: ParamSource<f32>, B: ParamSource<f32>,
    {
        let limit = buf.len().min(alpha.len()).min(beta.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<8>(i << 3); let b = beta.chunk::<8>(i << 3);
            for j in 0..8 { chunk[j] = self.next_beta_f32(a[j], b[j]); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() { *slot = self.next_beta_f32(alpha.get(offset + i), beta.get(offset + i)); }
    }

    #[inline]
    pub fn fill_beta_f64<A, B>(&mut self, buf: &mut [f64], alpha: A, beta: B)
    where A: ParamSource<f64>, B: ParamSource<f64>,
    {
        let limit = buf.len().min(alpha.len()).min(beta.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<4>(i << 2); let b = beta.chunk::<4>(i << 2);
            for j in 0..4 { chunk[j] = self.next_beta_f64(a[j], b[j]); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() { *slot = self.next_beta_f64(alpha.get(offset + i), beta.get(offset + i)); }
    }

    #[inline]
    pub fn fill_chi_squared_f32<K>(&mut self, buf: &mut [f32], k: K)
    where K: ParamSource<f32>,
    {
        let limit = buf.len().min(k.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let k_arr = k.chunk::<8>(i << 3);
            for j in 0..8 { chunk[j] = self.next_chi_squared_f32(k_arr[j]); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() { *slot = self.next_chi_squared_f32(k.get(offset + i)); }
    }

    #[inline]
    pub fn fill_chi_squared_f64<K>(&mut self, buf: &mut [f64], k: K)
    where K: ParamSource<f64>,
    {
        let limit = buf.len().min(k.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let k_arr = k.chunk::<4>(i << 2);
            for j in 0..4 { chunk[j] = self.next_chi_squared_f64(k_arr[j]); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() { *slot = self.next_chi_squared_f64(k.get(offset + i)); }
    }

    // -------------------------------------------------------------------------
    // make_filled_* — allocate, fill, and return a [T; N] array (stack)
    // -------------------------------------------------------------------------

    #[inline]
    pub fn make_filled_rayleigh_f32<S, const N: usize>(&mut self, sigma: S) -> [f32; N]
    where S: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let limit = slice.len().min(sigma.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let mut u = [0.0f32; 8];
            for j in 0..8 { u[j] = 1.0 - self.next_uniform_f32(); }
            let r = fptricks::batch_approx_sqrt_f32(fptricks::batch_fmadd_f32(fptricks::batch_approx_ln_f32(u), -2.0, 0.0));
            let s_chunk = sigma.chunk::<8>(offset);
            for j in 0..8 { chunk[j].write(r[j] * s_chunk[j]); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() { slot.write(self.next_rayleigh_f32(sigma.get(offset + i))); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_rayleigh_f64<S, const N: usize>(&mut self, sigma: S) -> [f64; N]
    where S: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[f64; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f64>, N) };
        let limit = slice.len().min(sigma.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 2;
            let mut u = [0.0f64; 4];
            for j in 0..4 { u[j] = 1.0 - self.next_uniform_f64(); }
            let r = fptricks::batch_approx_sqrt_f64(fptricks::batch_fmadd_f64(fptricks::batch_approx_ln_f64(u), -2.0, 0.0));
            let s_chunk = sigma.chunk::<4>(offset);
            for j in 0..4 { chunk[j].write(r[j] * s_chunk[j]); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() { slot.write(self.next_rayleigh_f64(sigma.get(offset + i))); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_gamma_f32<A, const N: usize>(&mut self, alpha: A) -> [f32; N]
    where A: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let limit = slice.len().min(alpha.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<8>(i << 3);
            for j in 0..8 { chunk[j].write(self.next_gamma_f32(a[j])); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() { slot.write(self.next_gamma_f32(alpha.get(offset + i))); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_gamma_f64<A, const N: usize>(&mut self, alpha: A) -> [f64; N]
    where A: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[f64; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f64>, N) };
        let limit = slice.len().min(alpha.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<4>(i << 2);
            for j in 0..4 { chunk[j].write(self.next_gamma_f64(a[j])); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() { slot.write(self.next_gamma_f64(alpha.get(offset + i))); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_poisson_u32<L, const N: usize>(&mut self, lambda: L) -> [u32; N]
    where L: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[u32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<u32>, N) };
        let limit = slice.len().min(lambda.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<8>(i << 3);
            let thresholds = fptricks::batch_approx_exp_f32([-l_arr[0],-l_arr[1],-l_arr[2],-l_arr[3],-l_arr[4],-l_arr[5],-l_arr[6],-l_arr[7]]);
            for j in 0..8 {
                if l_arr[j] <= 0.0 { chunk[j].write(0); continue; }
                let l = thresholds[j]; let mut k = 0u32; let mut p = 1.0_f32;
                loop { k += 1; p *= self.next_uniform_f32(); if p <= l { break; } }
                chunk[j].write(k - 1);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() { slot.write(self.next_poisson_u32(lambda.get(offset + i))); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_poisson_f64_u32<L, const N: usize>(&mut self, lambda: L) -> [u32; N]
    where L: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[u32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<u32>, N) };
        let limit = slice.len().min(lambda.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<4>(i << 2);
            let thresholds = fptricks::batch_approx_exp_f64([-l_arr[0],-l_arr[1],-l_arr[2],-l_arr[3]]);
            for j in 0..4 {
                if l_arr[j] <= 0.0 { chunk[j].write(0); continue; }
                let l = thresholds[j]; let mut k = 0u32; let mut p = 1.0_f64;
                loop { k += 1; p *= self.next_uniform_f64(); if p <= l { break; } }
                chunk[j].write(k - 1);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() { slot.write(self.next_poisson_f64_u32(lambda.get(offset + i))); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_poisson_collecting_u32<L, const N: usize>(&mut self, lambda: L) -> [u32; N]
    where L: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[u32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<u32>, N) };
        let limit = slice.len().min(lambda.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<8>(i << 3);
            let thresholds = fptricks::batch_approx_exp_f32([-l_arr[0],-l_arr[1],-l_arr[2],-l_arr[3],-l_arr[4],-l_arr[5],-l_arr[6],-l_arr[7]]);
            let mut counts = [0u32; 8]; let mut p = self.next_f32_8();
            let mut mask = 0u8;
            for j in 0..8 { mask |= ((l_arr[j] > 0.0) as u8 & (p[j] > thresholds[j]) as u8) << j; }
            while mask != 0 {
                let mut next_mask = 0u8; let u = self.next_f32_8();
                for j in 0..8 {
                    let act = (mask >> j) & 1; let am: u32 = (act as u32).wrapping_neg();
                    counts[j] += act as u32;
                    p[j] *= f32::from_bits((u[j].to_bits() & am) | (1.0f32.to_bits() & !am));
                    next_mask |= ((p[j] > thresholds[j]) as u8 & act) << j;
                }
                mask = next_mask;
            }
            for j in 0..8 { chunk[j].write(counts[j]); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() { slot.write(self.next_poisson_u32(lambda.get(offset + i))); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_poisson_collecting_f64_u32<L, const N: usize>(&mut self, lambda: L) -> [u32; N]
    where L: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[u32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<u32>, N) };
        let limit = slice.len().min(lambda.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<4>(i << 2);
            let thresholds = fptricks::batch_approx_exp_f64([-l_arr[0],-l_arr[1],-l_arr[2],-l_arr[3]]);
            let mut counts = [0u32; 4]; let mut p = self.next_f64_4();
            let mut mask = 0u8;
            for j in 0..4 { mask |= ((l_arr[j] > 0.0) as u8 & (p[j] > thresholds[j]) as u8) << j; }
            while mask != 0 {
                let mut next_mask = 0u8; let u = self.next_f64_4();
                for j in 0..4 {
                    let act = (mask >> j) & 1; let am: u64 = (act as u64).wrapping_neg();
                    counts[j] += act as u32;
                    p[j] *= f64::from_bits((u[j].to_bits() & am) | (1.0f64.to_bits() & !am));
                    next_mask |= ((p[j] > thresholds[j]) as u8 & act) << j;
                }
                mask = next_mask;
            }
            for j in 0..4 { chunk[j].write(counts[j]); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() { slot.write(self.next_poisson_f64_u32(lambda.get(offset + i))); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_beta_f32<A, B, const N: usize>(&mut self, alpha: A, beta: B) -> [f32; N]
    where A: ParamSource<f32>, B: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let limit = slice.len().min(alpha.len()).min(beta.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<8>(i << 3); let b = beta.chunk::<8>(i << 3);
            for j in 0..8 { chunk[j].write(self.next_beta_f32(a[j], b[j])); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() { slot.write(self.next_beta_f32(alpha.get(offset + i), beta.get(offset + i))); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_beta_f64<A, B, const N: usize>(&mut self, alpha: A, beta: B) -> [f64; N]
    where A: ParamSource<f64>, B: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[f64; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f64>, N) };
        let limit = slice.len().min(alpha.len()).min(beta.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<4>(i << 2); let b = beta.chunk::<4>(i << 2);
            for j in 0..4 { chunk[j].write(self.next_beta_f64(a[j], b[j])); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() { slot.write(self.next_beta_f64(alpha.get(offset + i), beta.get(offset + i))); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_chi_squared_f32<K, const N: usize>(&mut self, k: K) -> [f32; N]
    where K: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let limit = slice.len().min(k.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let k_arr = k.chunk::<8>(i << 3);
            for j in 0..8 { chunk[j].write(self.next_chi_squared_f32(k_arr[j])); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() { slot.write(self.next_chi_squared_f32(k.get(offset + i))); }
        unsafe { buf.assume_init() }
    }

    #[inline]
    pub fn make_filled_chi_squared_f64<K, const N: usize>(&mut self, k: K) -> [f64; N]
    where K: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[f64; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f64>, N) };
        let limit = slice.len().min(k.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let k_arr = k.chunk::<4>(i << 2);
            for j in 0..4 { chunk[j].write(self.next_chi_squared_f64(k_arr[j])); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() { slot.write(self.next_chi_squared_f64(k.get(offset + i))); }
        unsafe { buf.assume_init() }
    }
}

// Helper shims used in the collecting Poisson variants
#[inline(always)]
fn still_gt(a: f32, b: f32) -> u8 { (a > b) as u8 }
#[inline(always)]
fn still_gt_f64(a: f64, b: f64) -> u8 { (a > b) as u8 }

#[cfg(test)]
mod tests {
    use super::*;

    fn calculate_stats<F>(mut f: F, n: usize) -> (f64, f64)
    where F: FnMut() -> f64,
    {
        let mut sum = 0.0; let mut sq = 0.0;
        for _ in 0..n { let v = f(); sum += v; sq += v * v; }
        let m = sum / n as f64; (m, sq / n as f64 - m * m)
    }

    #[test]
    fn test_gamma_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        for &alpha in &[0.5f64, 1.0, 2.0, 5.0] {
            let (mean, var) = calculate_stats(|| rng.next_gamma_f32(alpha as f32) as f64, n);
            assert!((mean - alpha).abs() < alpha * 0.15);
            assert!((var - alpha).abs() < alpha * 0.15);
        }
    }

    #[test]
    fn test_beta_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        for &(a, b) in &[(0.5f64, 0.5), (1.0, 3.0), (2.0, 2.0)] {
            let em = a / (a + b);
            let ev = (a * b) / ((a + b) * (a + b) * (a + b + 1.0));
            let (mean, var) = calculate_stats(|| rng.next_beta_f32(a as f32, b as f32) as f64, n);
            assert!((mean - em).abs() < 0.1);
            assert!((var - ev).abs() < 0.1);
        }
    }

    #[test]
    fn test_rayleigh_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let sigma = 2.0f64;
        let expected_mean = sigma * (std::f64::consts::PI / 2.0).sqrt();
        let (mean, _) = calculate_stats(|| rng.next_rayleigh_f32(sigma as f32) as f64, n);
        assert!((mean - expected_mean).abs() < 0.15);
    }

    #[test]
    fn test_poisson_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        for &lambda in &[0.5f64, 1.0, 5.0, 10.0] {
            let (mean, var) = calculate_stats(|| rng.next_poisson_u32(lambda as f32) as f64, n);
            assert!((mean - lambda).abs() < lambda * 0.15 + 0.05);
            assert!((var - lambda).abs() < lambda * 0.15 + 0.1);
        }
    }

    #[test]
    fn test_poisson_collecting_stats() {
        let mut rng = WyRand::new(1);
        let n = 100_000;
        let lambdas = [0.5f32, 1.0, 5.0, 10.0];
        for &lambda in &lambdas {
            let mut buf = vec![0u32; n];
            rng.fill_poisson_collecting_u32(&mut buf, lambda);
            let mut sum = 0.0; let mut sq = 0.0;
            for &v in &buf { let f = v as f64; sum += f; sq += f * f; }
            let mean = sum / n as f64; let var = sq / n as f64 - mean * mean;
            let l = lambda as f64;
            assert!((mean - l).abs() < l * 0.15 + 0.05);
            assert!((var - l).abs() < l * 0.15 + 0.1);
        }
    }
}
