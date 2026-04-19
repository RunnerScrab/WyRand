use crate::WyRand;
use crate::traits::ParamSource;
use fptricks::*;
use std::mem::MaybeUninit;

impl WyRand {
    #[inline(always)]
    pub fn next_power_law(&mut self, min: f64, max: f64, k: f64) -> f64 {
        let roll = self.next_uniform_f64();
        let kp1 = k + 1.0;
        let kp1lt: u64 = ((kp1.abs() < 1e-9) as u64).wrapping_neg();
        f64::from_bits(
            ((min * (max / min).approx_powf(roll)).to_bits() & kp1lt)
                | (!kp1lt & {
                    let min_pow = min.approx_powf(kp1);
                    let max_pow = max.approx_powf(kp1);
                    (roll * (max_pow - min_pow) + min_pow).approx_powf(1.0 / kp1)
                }
                .to_bits()),
        )
    }

    #[inline(always)]
    pub fn next_rayleigh_f32(&mut self, sigma: f32) -> f32 {
        (-(1.0 - self.next_uniform_f32()).approx_ln() * 2.0).approx_sqrt() * sigma
    }

    #[inline(always)]
    pub fn next_rayleigh_f64(&mut self, sigma: f64) -> f64 {
        (-(1.0 - self.next_uniform_f64()).approx_ln() * 2.0).approx_sqrt() * sigma
    }

    #[inline(always)]
    pub fn next_gamma_f32(&mut self, alpha: f32) -> f32 {
        if !(alpha > 0.0) {
            return 0.0;
        }
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
        if !(alpha > 0.0) {
            return 0.0;
        }
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
        let a = self.next_gamma_f32(alpha);
        let b = self.next_gamma_f32(beta);
        let sum = a + b;
        f32::from_bits((a / sum).to_bits() & ((sum != 0.0) as u32).wrapping_neg())
    }

    #[inline(always)]
    pub fn next_beta_f64(&mut self, alpha: f64, beta: f64) -> f64 {
        let a = self.next_gamma_f64(alpha);
        let b = self.next_gamma_f64(beta);
        let sum = a + b;
        f64::from_bits((a / sum).to_bits() & ((sum != 0.0) as u64).wrapping_neg())
    }

    #[inline(always)]
    pub fn next_chi_squared_f32(&mut self, k: f32) -> f32 {
        self.next_gamma_f32(k * 0.5) * 2.0
    }

    #[inline(always)]
    pub fn next_chi_squared_f64(&mut self, k: f64) -> f64 {
        self.next_gamma_f64(k * 0.5) * 2.0
    }

    #[inline(always)]
    pub fn next_poisson_u32(&mut self, lambda: f32) -> u32 {
        //This is not equivalent to lambda <= 0.0 whatever the linter says because of NaN behavior
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        match lambda.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => {
                if lambda > 30.0 {
                    let z = self.next_std_normal_f32();
                    return (z * lambda.approx_sqrt() + lambda).max(0.0) as u32;
                }
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
            _ => 0,
        }
    }

    #[inline(always)]
    pub fn next_poisson_f64_u32(&mut self, lambda: f64) -> u32 {
        match lambda.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => {
                if lambda > 30.0 {
                    let z = self.next_std_normal_f64();
                    return (z * lambda.sqrt() + lambda).max(0.0) as u32;
                }
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
            _ => 0,
        }
    }

    // -------------------------------------------------------------------------
    // fill_* — write into caller-owned slice (heap-friendly, runtime length)
    // -------------------------------------------------------------------------

    #[inline(always)]
    pub fn fill_rayleigh_f32<S>(&mut self, buf: &mut [f32], sigma: S)
    where
        S: ParamSource<f32>,
    {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ))]
        {
            unsafe {
                use core::arch::x86_64::*;
                let limit = buf.len().min(sigma.len());
                let (active, _) = buf.split_at_mut(limit);
                let mut iter = active.chunks_exact_mut(8);
                let ntwos: __m256 = _mm256_set1_ps(-2.0);
                for (i, chunk) in iter.by_ref().enumerate() {
                    let mut u = self.make_filled_uniform_f32::<8>();
                    (0..8).for_each(|j| {
                        u[j] = 1.0 - u[j];
                    });
                    let x: __m256 = core::mem::transmute(u);
                    let r = _mm256_sqrt_ps(_mm256_mul_ps(fptricks::raw_batch_ln_f32(x), ntwos));
                    let s_arr = sigma.chunk::<8>(i << 3);
                    let s: __m256 = core::mem::transmute(s_arr);
                    let c: __m256 = _mm256_mul_ps(r, s);
                    _mm256_storeu_ps(chunk.as_mut_ptr(), c);
                }
                let rem = iter.into_remainder();
                let offset = limit & !7;
                for (i, slot) in rem.iter_mut().enumerate() {
                    *slot = self.next_rayleigh_f32(sigma.get(offset + i));
                }
            }
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )))]
        {
            let limit = buf.len().min(sigma.len());
            let (active, _) = buf.split_at_mut(limit);
            let mut iter = active.chunks_exact_mut(8);
            for (i, chunk) in iter.by_ref().enumerate() {
                let offset = i << 3;
                let mut u: [f32; 8] = self.make_filled_uniform_f32();
                (0..8).for_each(|j| {
                    u[j] = 1.0 - u[j];
                });
                let r = fptricks::batch_approx_sqrt_f32(fptricks::batch_fmadd_f32(
                    fptricks::batch_approx_ln_f32(u),
                    -2.0,
                    0.0,
                ));
                let s_chunk = sigma.chunk::<8>(offset);
                chunk.copy_from_slice(&fptricks::batch_mul_cols_f32(&r, &s_chunk));
            }
            let rem = iter.into_remainder();
            let offset = limit & !7;
            for (i, slot) in rem.iter_mut().enumerate() {
                *slot = self.next_rayleigh_f32(sigma.get(offset + i));
            }
        }
    }

    #[inline(always)]
    pub fn fill_rayleigh_f64<S>(&mut self, buf: &mut [f64], sigma: S)
    where
        S: ParamSource<f64>,
    {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ))]
        {
            unsafe {
                use std::arch::x86_64::*;
                let limit = buf.len().min(sigma.len());
                let (active, _) = buf.split_at_mut(limit);
                let mut iter = active.chunks_exact_mut(4);
                let ntwos: __m256d = _mm256_set1_pd(-2.0);
                for (i, chunk) in iter.by_ref().enumerate() {
                    let mut u: [f64; 4] = self.make_filled_uniform_f64();
                    (0..4).for_each(|idx| {
                        u[idx] = 1.0 - u[idx];
                    });
                    let x: __m256d = core::mem::transmute(u);
                    let r = _mm256_sqrt_pd(_mm256_mul_pd(fptricks::raw_batch_ln_f64(x), ntwos));
                    let s_arr = sigma.chunk::<4>(i << 2);
                    let s: __m256d = core::mem::transmute(s_arr);
                    let c: __m256d = _mm256_mul_pd(r, s);
                    _mm256_storeu_pd(chunk.as_mut_ptr(), c);
                }
                let rem = iter.into_remainder();
                let offset = limit & !3;
                for (i, slot) in rem.iter_mut().enumerate() {
                    *slot = self.next_rayleigh_f64(sigma.get(offset + i));
                }
            }
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )))]
        {
            let limit = buf.len().min(sigma.len());
            let (active, _) = buf.split_at_mut(limit);
            let mut iter = active.chunks_exact_mut(4);
            for (i, chunk) in iter.by_ref().enumerate() {
                let offset = i << 2;
                let mut u: [f64; 4] = self.make_filled_uniform_f64();

                (0..4).for_each(|idx| {
                    u[idx] = 1.0 - u[idx];
                });

                let r = fptricks::batch_approx_sqrt_f64(fptricks::batch_fmadd_f64(
                    fptricks::batch_approx_ln_f64(u),
                    -2.0,
                    0.0,
                ));
                let s_chunk = sigma.chunk::<4>(offset);
                chunk.copy_from_slice(&fptricks::batch_mul_cols_f64(&r, &s_chunk));
            }
            let rem = iter.into_remainder();
            let offset = limit & !3;
            for (i, slot) in rem.iter_mut().enumerate() {
                *slot = self.next_rayleigh_f64(sigma.get(offset + i));
            }
        }
    }

    #[inline(always)]
    pub fn fill_gamma_f32<A>(&mut self, buf: &mut [f32], alpha: A)
    where
        A: ParamSource<f32>,
    {
        let limit = buf.len().min(alpha.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<8>(i << 3);
            for j in 0..8 {
                chunk[j] = self.next_gamma_f32(a[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() {
            *slot = self.next_gamma_f32(alpha.get(offset + i));
        }
    }

    #[inline(always)]
    pub fn fill_gamma_f64<A>(&mut self, buf: &mut [f64], alpha: A)
    where
        A: ParamSource<f64>,
    {
        let limit = buf.len().min(alpha.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<4>(i << 2);
            for j in 0..4 {
                chunk[j] = self.next_gamma_f64(a[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() {
            *slot = self.next_gamma_f64(alpha.get(offset + i));
        }
    }

    #[inline(always)]
    pub fn fill_poisson_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where
        L: ParamSource<f32>,
    {
        let limit = buf.len().min(lambda.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<8>(i << 3);
            let mut l_eff = [0.0f32; 8];
            for j in 0..8 {
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(l_arr[j] > 0.0) {
                    chunk[j] = 0;
                } else if l_arr[j] > 30.0 {
                    chunk[j] = (self.next_std_normal_f32() * l_arr[j].approx_sqrt() + l_arr[j])
                        .max(0.0) as u32;
                } else {
                    l_eff[j] = l_arr[j];
                }
            }
            let thresholds = fptricks::batch_approx_exp_f32([
                -l_eff[0], -l_eff[1], -l_eff[2], -l_eff[3], -l_eff[4], -l_eff[5], -l_eff[6],
                -l_eff[7],
            ]);
            for j in 0..8 {
                if l_eff[j] <= 0.0 {
                    continue;
                }
                let l = thresholds[j];
                let mut k = 0u32;
                let mut p = 1.0_f32;
                loop {
                    k += 1;
                    p *= self.next_uniform_f32();
                    if p <= l {
                        break;
                    }
                }
                chunk[j] = k - 1;
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() {
            *slot = self.next_poisson_u32(lambda.get(offset + i));
        }
    }

    #[inline(always)]
    pub fn fill_poisson_f64_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where
        L: ParamSource<f64>,
    {
        let limit = buf.len().min(lambda.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<4>(i << 2);
            let mut l_eff = [0.0f64; 4];
            for j in 0..4 {
                if let Some(std::cmp::Ordering::Greater) = l_arr[j].partial_cmp(&0.0) {
                    if l_arr[j] > 30.0 {
                        chunk[j] = (self.next_std_normal_f64() * l_arr[j].approx_sqrt() + l_arr[j])
                            .max(0.0) as u32;
                    } else {
                        l_eff[j] = l_arr[j];
                    }
                }
            }
            let thresholds =
                fptricks::batch_approx_exp_f64([-l_eff[0], -l_eff[1], -l_eff[2], -l_eff[3]]);
            for j in 0..4 {
                if l_eff[j] <= 0.0 {
                    continue;
                }
                let l = thresholds[j];
                let mut k = 0u32;
                let mut p = 1.0_f64;
                loop {
                    k += 1;
                    p *= self.next_uniform_f64();
                    if p <= l {
                        break;
                    }
                }
                chunk[j] = k - 1;
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() {
            *slot = self.next_poisson_f64_u32(lambda.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_poisson_collecting_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where
        L: ParamSource<f32>,
    {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ))]
        {
            use core::arch::x86_64::*;

            let limit = buf.len().min(lambda.len());
            let (active, _) = buf.split_at_mut(limit);
            let mut iter = active.chunks_exact_mut(8);

            for (i, chunk) in iter.by_ref().enumerate() {
                let l_arr = lambda.chunk::<8>(i << 3);

                unsafe {
                    // 1. Load lambda parameters and prepare constants
                    let l_vec = _mm256_loadu_ps(l_arr.as_ptr());
                    let zero = _mm256_setzero_ps();
                    let thirty = _mm256_set1_ps(30.0);

                    // 2. Compute Segment Masks
                    let gt_zero = _mm256_cmp_ps(l_vec, zero, _CMP_GT_OQ);
                    let gt_thirty = _mm256_cmp_ps(l_vec, thirty, _CMP_GT_OQ);

                    let normal_mask = _mm256_and_ps(gt_zero, gt_thirty);
                    let poisson_mask = _mm256_andnot_ps(gt_thirty, gt_zero);

                    let mut normal_counts = _mm256_setzero_si256();

                    // 3. Normal Approximation Branch (lambda > 30.0)
                    if _mm256_movemask_ps(normal_mask) != 0 {
                        let norms: [f32; 8] = self.make_filled_std_normal_f32();
                        let norms_vec = _mm256_loadu_ps(norms.as_ptr());

                        let sqrt_l = _mm256_sqrt_ps(l_vec);

                        // --- FMA UPDATE ---
                        // Calculate: (norms_vec * sqrt_l) + l_vec in a single instruction
                        let vals = _mm256_fmadd_ps(norms_vec, sqrt_l, l_vec);

                        // .max(0.0) and truncate to 32-bit integer
                        let vals_max = _mm256_max_ps(vals, zero);
                        normal_counts = _mm256_cvttps_epi32(vals_max);
                    }

                    let mut poisson_counts = _mm256_setzero_si256();

                    // 4. Exact Poisson Branch (0.0 < lambda <= 30.0)
                    if _mm256_movemask_ps(poisson_mask) != 0 {
                        let l_eff_vec = _mm256_and_ps(l_vec, poisson_mask);

                        // Negate l_eff_vec inline to construct [-l_eff[0], ...]
                        let neg_l_eff_vec = _mm256_sub_ps(zero, l_eff_vec);
                        let mut neg_l_eff_arr = [0.0f32; 8];
                        _mm256_storeu_ps(neg_l_eff_arr.as_mut_ptr(), neg_l_eff_vec);

                        // Compute batch thresholds
                        let thresholds_arr = fptricks::batch_approx_exp_f32(neg_l_eff_arr);
                        let thresholds = _mm256_loadu_ps(thresholds_arr.as_ptr());

                        let p_arr = self.next_f32_8();
                        let mut p = _mm256_loadu_ps(p_arr.as_ptr());

                        // Initial active mask: l_eff > 0.0 & p > thresholds
                        let p_gt_thresh = _mm256_cmp_ps(p, thresholds, _CMP_GT_OQ);
                        let mut active_mask = _mm256_and_ps(poisson_mask, p_gt_thresh);

                        let one = _mm256_set1_ps(1.0);

                        // AVX2 Loop over active bits
                        while _mm256_movemask_ps(active_mask) != 0 {
                            // Trick: active_mask evaluates to 0xFFFFFFFF for true.
                            // Treating it as a signed int and subtracting effectively adds 1.
                            poisson_counts =
                                _mm256_sub_epi32(poisson_counts, _mm256_castps_si256(active_mask));

                            let u_arr = self.next_f32_8();
                            let u = _mm256_loadu_ps(u_arr.as_ptr());

                            // Multiply p by u only if lane is active; else multiply by 1.0
                            let u_eff = _mm256_blendv_ps(one, u, active_mask);
                            p = _mm256_mul_ps(p, u_eff);

                            // Update mask: p > thresholds & still active
                            let still_gt = _mm256_cmp_ps(p, thresholds, _CMP_GT_OQ);
                            active_mask = _mm256_and_ps(active_mask, still_gt);
                        }
                    }

                    // 5. Combine and Store
                    // Select normal_counts where normal_mask is true, else fallback to poisson_counts
                    let combined_counts = _mm256_blendv_epi8(
                        poisson_counts,
                        normal_counts,
                        _mm256_castps_si256(normal_mask),
                    );

                    // Store directly into our chunk's memory footprint
                    _mm256_storeu_si256(chunk.as_mut_ptr() as *mut __m256i, combined_counts);
                }
            }

            // 6. Serial Remainder Handling
            let rem = iter.into_remainder();
            let offset = limit & !7;
            for (i, slot) in rem.iter_mut().enumerate() {
                *slot = self.next_poisson_u32(lambda.get(offset + i));
            }
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )))]
        {
            self.scalar_fill_poisson_collecting_u32(buf, lambda)
        }
    }

    #[inline(always)]
    fn scalar_fill_poisson_collecting_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where
        L: ParamSource<f32>,
    {
        let limit = buf.len().min(lambda.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<8>(i << 3);
            let mut l_eff = [0.0f32; 8];
            let mut counts: [u32; 8] = [0; 8];
            for j in 0..8 {
                if let Some(std::cmp::Ordering::Greater) = l_arr[j].partial_cmp(&0.0) {
                    if l_arr[j] > 30.0 {
                        counts[j] = (self.next_std_normal_f32() * l_arr[j].approx_sqrt() + l_arr[j])
                            .max(0.0) as u32;
                    } else {
                        l_eff[j] = l_arr[j];
                    }
                }
            }
            let thresholds = fptricks::batch_approx_exp_f32([
                -l_eff[0], -l_eff[1], -l_eff[2], -l_eff[3], -l_eff[4], -l_eff[5], -l_eff[6],
                -l_eff[7],
            ]);
            let mut p = self.next_f32_8();
            let mut mask = 0u8;
            for j in 0..8 {
                mask |= ((l_eff[j] > 0.0) as u8 & (p[j] > thresholds[j]) as u8) << j;
            }
            while mask != 0 {
                let mut next_mask = 0u8;
                let u = self.next_f32_8();
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
        for (i, slot) in rem.iter_mut().enumerate() {
            *slot = self.next_poisson_u32(lambda.get(offset + i));
        }
    }
    #[inline]
    pub fn fill_poisson_collecting_f64_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where
        L: ParamSource<f64>,
    {
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ))]
        {
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::*;

            let limit = buf.len().min(lambda.len());
            let (active, _) = buf.split_at_mut(limit);

            // Chunking by 4 because 4 f64s fit in a 256-bit register
            let mut iter = active.chunks_exact_mut(4);

            for (i, chunk) in iter.by_ref().enumerate() {
                let l_arr = lambda.chunk::<4>(i << 2);

                unsafe {
                    // 1. Load lambda parameters (double precision)
                    let l_vec = _mm256_loadu_pd(l_arr.as_ptr());
                    let zero = _mm256_setzero_pd();
                    let thirty = _mm256_set1_pd(30.0);

                    // 2. Compute Segment Masks
                    let gt_zero = _mm256_cmp_pd(l_vec, zero, _CMP_GT_OQ);
                    let gt_thirty = _mm256_cmp_pd(l_vec, thirty, _CMP_GT_OQ);

                    let normal_mask = _mm256_and_pd(gt_zero, gt_thirty);
                    let poisson_mask = _mm256_andnot_pd(gt_thirty, gt_zero);

                    // _mm256_cvtpd_epi32 creates a 128-bit vector of 4x 32-bit integers
                    let mut normal_counts_32 = _mm_setzero_si128();

                    // 3. Normal Approximation Branch (lambda > 30.0)
                    if _mm256_movemask_pd(normal_mask) != 0 {
                        let norms: [f64; 4] = self.make_filled_std_normal_f64();
                        let norms_vec = _mm256_loadu_pd(norms.as_ptr());

                        let sqrt_l = _mm256_sqrt_pd(l_vec);
                        let vals = _mm256_fmadd_pd(norms_vec, sqrt_l, l_vec);

                        let vals_max = _mm256_max_pd(vals, zero);
                        // Directly truncate 4x f64s into 4x i32s (occupies lower 128 bits)
                        normal_counts_32 = _mm256_cvtpd_epi32(vals_max);
                    }

                    let mut poisson_counts_64 = _mm256_setzero_si256();

                    // 4. Exact Poisson Branch (0.0 < lambda <= 30.0)
                    if _mm256_movemask_pd(poisson_mask) != 0 {
                        let l_eff_vec = _mm256_and_pd(l_vec, poisson_mask);

                        let neg_l_eff_vec = _mm256_sub_pd(zero, l_eff_vec);
                        let mut neg_l_eff_arr = [0.0f64; 4];
                        _mm256_storeu_pd(neg_l_eff_arr.as_mut_ptr(), neg_l_eff_vec);

                        let thresholds_arr = fptricks::batch_approx_exp_f64(neg_l_eff_arr);
                        let thresholds = _mm256_loadu_pd(thresholds_arr.as_ptr());

                        let p_arr = self.next_f64_4();
                        let mut p = _mm256_loadu_pd(p_arr.as_ptr());

                        let p_gt_thresh = _mm256_cmp_pd(p, thresholds, _CMP_GT_OQ);
                        let mut active_mask = _mm256_and_pd(poisson_mask, p_gt_thresh);

                        let one = _mm256_set1_pd(1.0);

                        while _mm256_movemask_pd(active_mask) != 0 {
                            // Subtracting a 64-bit mask lane (-1 if true) adds 1 to the count
                            poisson_counts_64 = _mm256_sub_epi64(
                                poisson_counts_64,
                                _mm256_castpd_si256(active_mask),
                            );

                            let u_arr = self.next_f64_4();
                            let u = _mm256_loadu_pd(u_arr.as_ptr());

                            let u_eff = _mm256_blendv_pd(one, u, active_mask);
                            p = _mm256_mul_pd(p, u_eff);

                            let still_gt = _mm256_cmp_pd(p, thresholds, _CMP_GT_OQ);
                            active_mask = _mm256_and_pd(active_mask, still_gt);
                        }
                    }

                    // 5. Blending & Packing
                    // Zero-extend our 4x 32-bit normal counts to 4x 64-bit counts so we can blend
                    let normal_counts_64 = _mm256_cvtepi32_epi64(normal_counts_32);

                    let combined_counts_64 = _mm256_blendv_epi8(
                        poisson_counts_64,
                        normal_counts_64,
                        _mm256_castpd_si256(normal_mask),
                    );

                    // We now have four 64-bit integers. We need to pack them into four 32-bit ints.
                    // Step 5a: Split the 256-bit register into two 128-bit halves
                    let lo = _mm256_castsi256_si128(combined_counts_64); // [C0_64, C1_64]
                    let hi = _mm256_extracti128_si256(combined_counts_64, 1); // [C2_64, C3_64]

                    // Step 5b: Shuffle the 32-bit blocks inside each half.
                    // 0b10_00_10_00 moves the lower 32-bit of each 64-bit int next to each other.
                    let lo_shuf = _mm_shuffle_epi32(lo, 0b10_00_10_00); // -> [C0_32, C1_32, _, _]
                    let hi_shuf = _mm_shuffle_epi32(hi, 0b10_00_10_00); // -> [C2_32, C3_32, _, _]

                    // Step 5c: Interleave the two halves together into a single 128-bit register
                    let packed_32 = _mm_unpacklo_epi64(lo_shuf, hi_shuf); // -> [C0_32, C1_32, C2_32, C3_32]

                    // 6. Store directly to the u32 slice
                    _mm_storeu_si128(chunk.as_mut_ptr() as *mut __m128i, packed_32);
                }
            }

            // 7. Serial Remainder Handling
            let rem = iter.into_remainder();
            let offset = limit & !3;
            for (i, slot) in rem.iter_mut().enumerate() {
                *slot = self.next_poisson_f64_u32(lambda.get(offset + i));
            }
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )))]
        {
            self.scalar_fill_poisson_collecting_f64_u32(buf, lambda);
        }
    }
    #[inline(always)]
    pub fn scalar_fill_poisson_collecting_f64_u32<L>(&mut self, buf: &mut [u32], lambda: L)
    where
        L: ParamSource<f64>,
    {
        let limit = buf.len().min(lambda.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<4>(i << 2);
            let mut l_eff = [0.0f64; 4];
            let mut counts = [0u32; 4];
            for j in 0..4 {
                if let Some(std::cmp::Ordering::Greater) = l_arr[j].partial_cmp(&0.0) {
                    if l_arr[j] > 30.0 {
                        counts[j] = (self.next_std_normal_f64() * l_arr[j].approx_sqrt() + l_arr[j])
                            .max(0.0) as u32;
                    } else {
                        l_eff[j] = l_arr[j];
                    }
                }
            }
            let thresholds =
                fptricks::batch_approx_exp_f64([-l_eff[0], -l_eff[1], -l_eff[2], -l_eff[3]]);
            let mut p = self.next_f64_4();
            let mut mask = 0u8;
            for j in 0..4 {
                mask |= ((l_eff[j] > 0.0) as u8 & (p[j] > thresholds[j]) as u8) << j;
            }
            while mask != 0 {
                let mut next_mask = 0u8;
                let u = self.next_f64_4();
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
        for (i, slot) in rem.iter_mut().enumerate() {
            *slot = self.next_poisson_f64_u32(lambda.get(offset + i));
        }
    }

    #[inline(always)]
    pub fn fill_beta_f32<A, B>(&mut self, buf: &mut [f32], alpha: A, beta: B)
    where
        A: ParamSource<f32>,
        B: ParamSource<f32>,
    {
        let limit = buf.len().min(alpha.len()).min(beta.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<8>(i << 3);
            let b = beta.chunk::<8>(i << 3);
            for j in 0..8 {
                chunk[j] = self.next_beta_f32(a[j], b[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() {
            *slot = self.next_beta_f32(alpha.get(offset + i), beta.get(offset + i));
        }
    }

    #[inline(always)]
    pub fn fill_beta_f64<A, B>(&mut self, buf: &mut [f64], alpha: A, beta: B)
    where
        A: ParamSource<f64>,
        B: ParamSource<f64>,
    {
        let limit = buf.len().min(alpha.len()).min(beta.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<4>(i << 2);
            let b = beta.chunk::<4>(i << 2);
            for j in 0..4 {
                chunk[j] = self.next_beta_f64(a[j], b[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() {
            *slot = self.next_beta_f64(alpha.get(offset + i), beta.get(offset + i));
        }
    }

    #[inline(always)]
    pub fn fill_chi_squared_f32<K>(&mut self, buf: &mut [f32], k: K)
    where
        K: ParamSource<f32>,
    {
        let limit = buf.len().min(k.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let k_arr = k.chunk::<8>(i << 3);
            for j in 0..8 {
                chunk[j] = self.next_chi_squared_f32(k_arr[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() {
            *slot = self.next_chi_squared_f32(k.get(offset + i));
        }
    }

    #[inline(always)]
    pub fn fill_chi_squared_f64<K>(&mut self, buf: &mut [f64], k: K)
    where
        K: ParamSource<f64>,
    {
        let limit = buf.len().min(k.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let k_arr = k.chunk::<4>(i << 2);
            for j in 0..4 {
                chunk[j] = self.next_chi_squared_f64(k_arr[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() {
            *slot = self.next_chi_squared_f64(k.get(offset + i));
        }
    }

    // -------------------------------------------------------------------------
    // make_filled_* — allocate, fill, and return a [T; N] array (stack)
    // -------------------------------------------------------------------------

    #[inline(always)]
    pub fn make_filled_rayleigh_f32<S, const N: usize>(&mut self, sigma: S) -> [f32; N]
    where
        S: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let limit = slice.len().min(sigma.len());
        let (active, _) = slice.split_at_mut(limit);

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ))]
        {
            unsafe {
                use core::arch::x86_64::*;
                let mut iter = active.chunks_exact_mut(8);
                let ntwos: __m256 = _mm256_set1_ps(-2.0);
                for (i, chunk) in iter.by_ref().enumerate() {
                    let mut u = self.make_filled_uniform_f32::<8>();
                    (0..8).for_each(|j| {
                        u[j] = 1.0 - u[j];
                    });
                    let x: __m256 = core::mem::transmute(u);
                    let r = _mm256_sqrt_ps(_mm256_mul_ps(fptricks::raw_batch_ln_f32(x), ntwos));
                    let s_arr = sigma.chunk::<8>(i << 3);
                    let s: __m256 = core::mem::transmute(s_arr);
                    let c: __m256 = _mm256_mul_ps(r, s);
                    _mm256_storeu_ps(chunk.as_mut_ptr() as *mut f32, c);
                }
                let rem = iter.into_remainder();
                let offset = limit & !7;
                for (i, slot) in rem.iter_mut().enumerate() {
                    slot.write(self.next_rayleigh_f32(sigma.get(offset + i)));
                }
            }
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )))]
        {
            let mut iter = active.chunks_exact_mut(8);
            for (i, chunk) in iter.by_ref().enumerate() {
                let offset = i << 3;
                let mut u: [f32; 8] = self.make_filled_uniform_f32();
                (0..8).for_each(|idx| {
                    u[idx] = 1.0 - u[idx];
                });
                let r = fptricks::batch_approx_sqrt_f32(fptricks::batch_fmadd_f32(
                    fptricks::batch_approx_ln_f32(u),
                    -2.0,
                    0.0,
                ));
                let s_chunk = sigma.chunk::<8>(offset);
                let res = fptricks::batch_mul_cols_f32(&r, &s_chunk);
                for j in 0..8 {
                    chunk[j].write(res[j]);
                }
            }
            let rem = iter.into_remainder();
            let offset = limit & !7;
            for (i, slot) in rem.iter_mut().enumerate() {
                slot.write(self.next_rayleigh_f32(sigma.get(offset + i)));
            }
        }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_rayleigh_f64<S, const N: usize>(&mut self, sigma: S) -> [f64; N]
    where
        S: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[f64; N]>::uninit();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f64>, N) };
        let limit = slice.len().min(sigma.len());
        let (active, _) = slice.split_at_mut(limit);

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        ))]
        {
            unsafe {
                use core::arch::x86_64::*;
                let mut iter = active.chunks_exact_mut(4);
                let ntwos: __m256d = _mm256_set1_pd(-2.0);
                for (i, chunk) in iter.by_ref().enumerate() {
                    let mut u: [f64; 4] = self.make_filled_uniform_f64();
                    (0..4).for_each(|idx| {
                        u[idx] = 1.0 - u[idx];
                    });
                    let x: __m256d = core::mem::transmute(u);
                    let r = _mm256_sqrt_pd(_mm256_mul_pd(fptricks::raw_batch_ln_f64(x), ntwos));
                    let s_arr = sigma.chunk::<4>(i << 2);
                    let s: __m256d = core::mem::transmute(s_arr);
                    let c: __m256d = _mm256_mul_pd(r, s);
                    _mm256_storeu_pd(chunk.as_mut_ptr() as *mut f64, c);
                }
                let rem = iter.into_remainder();
                let offset = limit & !3;
                for (i, slot) in rem.iter_mut().enumerate() {
                    slot.write(self.next_rayleigh_f64(sigma.get(offset + i)));
                }
            }
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )))]
        {
            let mut iter = active.chunks_exact_mut(4);
            for (i, chunk) in iter.by_ref().enumerate() {
                let offset = i << 2;
                let mut u: [f64; 4] = self.make_filled_uniform_f64();
                (0..4).for_each(|idx| {
                    u[idx] = 1.0 - u[idx];
                });
                let r = fptricks::batch_approx_sqrt_f64(fptricks::batch_fmadd_f64(
                    fptricks::batch_approx_ln_f64(u),
                    -2.0,
                    0.0,
                ));
                let s_chunk = sigma.chunk::<4>(offset);
                let res = fptricks::batch_mul_cols_f64(&r, &s_chunk);
                for j in 0..4 {
                    chunk[j].write(res[j]);
                }
            }
            let rem = iter.into_remainder();
            let offset = limit & !3;
            for (i, slot) in rem.iter_mut().enumerate() {
                slot.write(self.next_rayleigh_f64(sigma.get(offset + i)));
            }
        }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_gamma_f32<A, const N: usize>(&mut self, alpha: A) -> [f32; N]
    where
        A: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let limit = slice.len().min(alpha.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<8>(i << 3);
            for j in 0..8 {
                chunk[j].write(self.next_gamma_f32(a[j]));
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() {
            slot.write(self.next_gamma_f32(alpha.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_gamma_f64<A, const N: usize>(&mut self, alpha: A) -> [f64; N]
    where
        A: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[f64; N]>::uninit();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f64>, N) };
        let limit = slice.len().min(alpha.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<4>(i << 2);
            for j in 0..4 {
                chunk[j].write(self.next_gamma_f64(a[j]));
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() {
            slot.write(self.next_gamma_f64(alpha.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_poisson_u32<L, const N: usize>(&mut self, lambda: L) -> [u32; N]
    where
        L: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[u32; N]>::uninit();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<u32>, N) };
        let limit = slice.len().min(lambda.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<8>(i << 3);
            let mut l_eff = [0.0f32; 8];
            for j in 0..8 {
                if let Some(std::cmp::Ordering::Greater) = l_arr[j].partial_cmp(&0.0) {
                    if l_arr[j] > 30.0 {
                        chunk[j].write(
                            (self.next_std_normal_f32() * l_arr[j].approx_sqrt() + l_arr[j])
                                .max(0.0) as u32,
                        );
                        l_eff[j] = 0.0;
                    } else {
                        l_eff[j] = l_arr[j];
                    }
                } else {
                    chunk[j].write(0);
                }
            }
            let thresholds = fptricks::batch_approx_exp_f32([
                -l_eff[0], -l_eff[1], -l_eff[2], -l_eff[3], -l_eff[4], -l_eff[5], -l_eff[6],
                -l_eff[7],
            ]);
            for j in 0..8 {
                if l_eff[j] <= 0.0 {
                    continue;
                }
                let l = thresholds[j];
                let mut k = 0u32;
                let mut p = 1.0_f32;
                loop {
                    k += 1;
                    p *= self.next_uniform_f32();
                    if p <= l {
                        break;
                    }
                }
                chunk[j].write(k - 1);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() {
            slot.write(self.next_poisson_u32(lambda.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_poisson_f64_u32<L, const N: usize>(&mut self, lambda: L) -> [u32; N]
    where
        L: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[u32; N]>::uninit();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<u32>, N) };
        let limit = slice.len().min(lambda.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<4>(i << 2);
            let mut l_eff = [0.0f64; 4];
            for j in 0..4 {
                if let Some(std::cmp::Ordering::Greater) = l_arr[j].partial_cmp(&0.0) {
                    if l_arr[j] > 30.0 {
                        chunk[j].write(
                            (self.next_std_normal_f64() * l_arr[j].sqrt() + l_arr[j]).max(0.0)
                                as u32,
                        );
                        l_eff[j] = 0.0;
                    } else {
                        l_eff[j] = l_arr[j];
                    }
                } else {
                    chunk[j].write(0);
                    l_eff[j] = 0.0;
                }
            }
            let thresholds =
                fptricks::batch_approx_exp_f64([-l_eff[0], -l_eff[1], -l_eff[2], -l_eff[3]]);
            for j in 0..4 {
                if l_eff[j] <= 0.0 {
                    continue;
                }
                let l = thresholds[j];
                let mut k = 0u32;
                let mut p = 1.0_f64;
                loop {
                    k += 1;
                    p *= self.next_uniform_f64();
                    if p <= l {
                        break;
                    }
                }
                chunk[j].write(k - 1);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() {
            slot.write(self.next_poisson_f64_u32(lambda.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_poisson_collecting_u32<L, const N: usize>(&mut self, lambda: L) -> [u32; N]
    where
        L: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[u32; N]>::uninit();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<u32>, N) };
        let limit = slice.len().min(lambda.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<8>(i << 3);
            let mut l_eff = [0.0f32; 8];
            let mut counts = [0u32; 8];
            for j in 0..8 {
                if let Some(std::cmp::Ordering::Greater) = l_arr[j].partial_cmp(&0.0) {
                    if l_arr[j] > 30.0 {
                        counts[j] = (self.next_std_normal_f32() * l_arr[j].approx_sqrt() + l_arr[j])
                            .max(0.0) as u32;
                        l_eff[j] = 0.0;
                    } else {
                        l_eff[j] = l_arr[j];
                    }
                } else {
                    counts[j] = 0;
                    l_eff[j] = 0.0;
                }
            }
            let thresholds = fptricks::batch_approx_exp_f32([
                -l_eff[0], -l_eff[1], -l_eff[2], -l_eff[3], -l_eff[4], -l_eff[5], -l_eff[6],
                -l_eff[7],
            ]);
            let mut p = self.next_f32_8();
            let mut mask = 0u8;
            for j in 0..8 {
                mask |= ((l_eff[j] > 0.0) as u8 & (p[j] > thresholds[j]) as u8) << j;
            }
            while mask != 0 {
                let mut next_mask = 0u8;
                let u = self.next_f32_8();
                for j in 0..8 {
                    let act = (mask >> j) & 1;
                    let am: u32 = (act as u32).wrapping_neg();
                    counts[j] += act as u32;
                    p[j] *= f32::from_bits((u[j].to_bits() & am) | (1.0f32.to_bits() & !am));
                    next_mask |= ((p[j] > thresholds[j]) as u8 & act) << j;
                }
                mask = next_mask;
            }
            for j in 0..8 {
                chunk[j].write(counts[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() {
            slot.write(self.next_poisson_u32(lambda.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_poisson_collecting_f64_u32<L, const N: usize>(
        &mut self,
        lambda: L,
    ) -> [u32; N]
    where
        L: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[u32; N]>::uninit();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<u32>, N) };
        let limit = slice.len().min(lambda.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let l_arr = lambda.chunk::<4>(i << 2);
            let mut l_eff = [0.0f64; 4];
            let mut counts = [0u32; 4];
            for j in 0..4 {
                if let Some(std::cmp::Ordering::Greater) = l_arr[j].partial_cmp(&0.0) {
                    if l_arr[j] > 30.0 {
                        counts[j] = (self.next_std_normal_f64() * l_arr[j].approx_sqrt() + l_arr[j])
                            .max(0.0) as u32;
                        l_eff[j] = 0.0;
                    } else {
                        l_eff[j] = l_arr[j];
                    }
                }
            }
            let thresholds =
                fptricks::batch_approx_exp_f64([-l_eff[0], -l_eff[1], -l_eff[2], -l_eff[3]]);
            let mut p = self.next_f64_4();
            let mut mask = 0u8;
            for j in 0..4 {
                mask |= ((l_eff[j] > 0.0) as u8 & (p[j] > thresholds[j]) as u8) << j;
            }
            while mask != 0 {
                let mut next_mask = 0u8;
                let u = self.next_f64_4();
                for j in 0..4 {
                    let act = (mask >> j) & 1;
                    let am: u64 = (act as u64).wrapping_neg();
                    counts[j] += act as u32;
                    p[j] *= f64::from_bits((u[j].to_bits() & am) | (1.0f64.to_bits() & !am));
                    next_mask |= ((p[j] > thresholds[j]) as u8 & act) << j;
                }
                mask = next_mask;
            }
            for j in 0..4 {
                chunk[j].write(counts[j]);
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() {
            slot.write(self.next_poisson_f64_u32(lambda.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_beta_f32<A, B, const N: usize>(&mut self, alpha: A, beta: B) -> [f32; N]
    where
        A: ParamSource<f32>,
        B: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let limit = slice.len().min(alpha.len()).min(beta.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<8>(i << 3);
            let b = beta.chunk::<8>(i << 3);
            for j in 0..8 {
                chunk[j].write(self.next_beta_f32(a[j], b[j]));
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() {
            slot.write(self.next_beta_f32(alpha.get(offset + i), beta.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_beta_f64<A, B, const N: usize>(&mut self, alpha: A, beta: B) -> [f64; N]
    where
        A: ParamSource<f64>,
        B: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[f64; N]>::uninit();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f64>, N) };
        let limit = slice.len().min(alpha.len()).min(beta.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let a = alpha.chunk::<4>(i << 2);
            let b = beta.chunk::<4>(i << 2);
            for j in 0..4 {
                chunk[j].write(self.next_beta_f64(a[j], b[j]));
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() {
            slot.write(self.next_beta_f64(alpha.get(offset + i), beta.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_chi_squared_f32<K, const N: usize>(&mut self, k: K) -> [f32; N]
    where
        K: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let limit = slice.len().min(k.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let k_arr = k.chunk::<8>(i << 3);
            for j in 0..8 {
                chunk[j].write(self.next_chi_squared_f32(k_arr[j]));
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() {
            slot.write(self.next_chi_squared_f32(k.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_chi_squared_f64<K, const N: usize>(&mut self, k: K) -> [f64; N]
    where
        K: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[f64; N]>::uninit();
        let slice =
            unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f64>, N) };
        let limit = slice.len().min(k.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(4);
        for (i, chunk) in iter.by_ref().enumerate() {
            let k_arr = k.chunk::<4>(i << 2);
            for j in 0..4 {
                chunk[j].write(self.next_chi_squared_f64(k_arr[j]));
            }
        }
        let rem = iter.into_remainder();
        let offset = limit & !3;
        for (i, slot) in rem.iter_mut().enumerate() {
            slot.write(self.next_chi_squared_f64(k.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }
}

// Helper shims used in the collecting Poisson variants
#[inline(always)]
fn still_gt(a: f32, b: f32) -> u8 {
    (a > b) as u8
}
#[inline(always)]
fn still_gt_f64(a: f64, b: f64) -> u8 {
    (a > b) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calculate_stats<F>(mut f: F, n: usize) -> (f64, f64)
    where
        F: FnMut() -> f64,
    {
        let mut sum = 0.0;
        let mut sq = 0.0;
        for _ in 0..n {
            let v = f();
            sum += v;
            sq += v * v;
        }
        let m = sum / n as f64;
        (m, sq / n as f64 - m * m)
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
            let mut sum = 0.0;
            let mut sq = 0.0;
            for &v in &buf {
                let f = v as f64;
                sum += f;
                sq += f * f;
            }
            let mean = sum / n as f64;
            let var = sq / n as f64 - mean * mean;
            let l = lambda as f64;
            assert!((mean - l).abs() < l * 0.15 + 0.05);
            assert!((var - l).abs() < l * 0.15 + 0.1);
        }
    }

    #[test]
    fn test_poisson_stability() {
        let mut rng = WyRand::new(42);
        // Robustness against NaN (IEEE-754 comparison safety)
        assert_eq!(rng.next_poisson_u32(f32::NAN), 0);
        assert_eq!(rng.next_poisson_f64_u32(f64::NAN), 0);
        // Robustness against negative and large Infinity
        assert_eq!(rng.next_poisson_u32(-1.0), 0);
        let k = rng.next_poisson_u32(f32::INFINITY);
        assert!(k > 0);
    }

    #[test]
    fn test_gamma_stability() {
        let mut rng = WyRand::new(42);
        // Robustness against NaN
        assert_eq!(rng.next_gamma_f32(f32::NAN), 0.0);
        assert_eq!(rng.next_gamma_f64(f64::NAN), 0.0);
        // Robustness against non-positive
        assert_eq!(rng.next_gamma_f32(0.0), 0.0);
        assert_eq!(rng.next_gamma_f32(-1.0), 0.0);
    }

    #[test]
    fn test_large_lambda_poisson() {
        let mut rng = WyRand::new(42);
        // Verify normal approximation works for high lambda
        let k = rng.next_poisson_u32(100.0);
        assert!(k > 50 && k < 150);

        let mut buf = [0u32; 16];
        rng.fill_poisson_u32(&mut buf, 100.0);
        for &k in &buf {
            assert!(k > 0);
        }
    }

    #[test]
    fn test_bulk_poisson_stability() {
        let mut rng = WyRand::new(42);
        let mut buf = [0u32; 10];
        let nan_f32 = [f32::NAN; 10];
        let nan_f64 = [f64::NAN; 10];

        rng.fill_poisson_collecting_u32(&mut buf, &nan_f32);
        for &k in &buf {
            assert_eq!(k, 0);
        }

        rng.fill_poisson_collecting_f64_u32(&mut buf, &nan_f64);
        for &k in &buf {
            assert_eq!(k, 0);
        }

        let res: [u32; 10] = rng.make_filled_poisson_collecting_u32(&nan_f32);
        for &k in &res {
            assert_eq!(k, 0);
        }

        let res: [u32; 10] = rng.make_filled_poisson_collecting_f64_u32(&nan_f64);
        for &k in &res {
            assert_eq!(k, 0);
        }
    }
}
