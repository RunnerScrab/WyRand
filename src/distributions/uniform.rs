use std::mem::MaybeUninit;
use crate::WyRand;
use crate::traits::ParamSource;

impl WyRand {
    #[inline(always)]
    pub fn next_uniform_f64(&mut self) -> f64 {
        let rv = self.next_u64();
        f64::from_bits((rv >> 12) | 0x3FF0_0000_0000_0000) - 1.0
    }

    #[inline(always)]
    pub fn next_uniform_f32(&mut self) -> f32 {
        let rv = self.next_u64();
        f32::from_bits(((rv & Self::FLOAT32_MASK) | 0x3F80_0000) as u32) - 1.0
    }

    #[inline(always)]
    pub fn next_uniform_in_range_f32(&mut self, min: f32, max: f32) -> f32 {
        self.next_uniform_f32().mul_add(max - min, min)
    }

    #[inline(always)]
    pub fn next_uniform_in_range_f64(&mut self, min: f64, max: f64) -> f64 {
        self.next_uniform_f64().mul_add(max - min, min)
    }

    #[inline(always)]
    pub(crate) fn next_f32_16(&mut self) -> [f32; 16] {
        let mut u = [0.0f32; 16];
        let mut s = self.0;
        for j in 0..16 {
            s = s.wrapping_add(Self::INC);
            let tmp = (s as u128).wrapping_mul(0xa3b195354a39b70d);
            let m1 = ((tmp >> 64) as u64) ^ (tmp as u64);
            let tmp2 = (m1 as u128).wrapping_mul(0x1b03738712fad5c9);
            let rv = ((tmp2 >> 64) as u64) ^ (tmp2 as u64);
            u[j] = f32::from_bits(((rv & Self::FLOAT32_MASK) | 0x3F80_0000) as u32) - 1.0;
        }
        self.0 = self.0.wrapping_add(Self::INC << 4);
        u
    }

    #[inline(always)]
    pub(crate) fn next_f32_8(&mut self) -> [f32; 8] {
        let mut u = [0.0f32; 8];
        let mut s = self.0;
        for j in 0..8 {
            s = s.wrapping_add(Self::INC);
            let tmp = (s as u128).wrapping_mul(0xa3b195354a39b70d);
            let m1 = ((tmp >> 64) as u64) ^ (tmp as u64);
            let tmp2 = (m1 as u128).wrapping_mul(0x1b03738712fad5c9);
            let rv = ((tmp2 >> 64) as u64) ^ (tmp2 as u64);
            u[j] = f32::from_bits(((rv & Self::FLOAT32_MASK) | 0x3F80_0000) as u32) - 1.0;
        }
        self.0 = self.0.wrapping_add(Self::INC << 3);
        u
    }

    #[inline(always)]
    pub(crate) fn next_f64_8(&mut self) -> [f64; 8] {
        let mut u = [0.0f64; 8];
        let mut s = self.0;
        for j in 0..8 {
            s = s.wrapping_add(Self::INC);
            let tmp = (s as u128).wrapping_mul(0xa3b195354a39b70d);
            let m1 = ((tmp >> 64) as u64) ^ (tmp as u64);
            let tmp2 = (m1 as u128).wrapping_mul(0x1b03738712fad5c9);
            let rv = ((tmp2 >> 64) as u64) ^ (tmp2 as u64);
            u[j] = f64::from_bits((rv >> 12) | 0x3FF0_0000_0000_0000) - 1.0;
        }
        self.0 = self.0.wrapping_add(Self::INC << 3);
        u
    }

    #[inline(always)]
    pub(crate) fn next_f64_4(&mut self) -> [f64; 4] {
        let mut u = [0.0f64; 4];
        let mut s = self.0;
        for j in 0..4 {
            s = s.wrapping_add(Self::INC);
            let tmp = (s as u128).wrapping_mul(0xa3b195354a39b70d);
            let m1 = ((tmp >> 64) as u64) ^ (tmp as u64);
            let tmp2 = (m1 as u128).wrapping_mul(0x1b03738712fad5c9);
            let rv = ((tmp2 >> 64) as u64) ^ (tmp2 as u64);
            u[j] = f64::from_bits((rv >> 12) | 0x3FF0_0000_0000_0000) - 1.0;
        }
        self.0 = self.0.wrapping_add(Self::INC << 2);
        u
    }

    #[inline(always)]
    pub fn next_u64_in_range(&mut self, min: u64, max: u64) -> u64 {
        let rv: u128 = self.next_u64() as u128;
        min + ((rv * (max - min) as u128) >> 64) as u64
    }

    #[inline(always)]
    pub fn next_usize_in_range(&mut self, max: usize) -> usize {
        self.next_u64_in_range(0, max as u64) as usize
    }

    // -------------------------------------------------------------------------
    // fill_* — write into caller-owned slice (heap-friendly, runtime length)
    // -------------------------------------------------------------------------

    #[inline(always)]
    pub fn fill_uniform_f32(&mut self, buf: &mut [f32]) {
        let mut iter = buf.chunks_exact_mut(16);
        for chunk in iter.by_ref() {
            let u = self.next_f32_16();
            chunk.copy_from_slice(&u);
        }
        for slot in iter.into_remainder() { *slot = self.next_uniform_f32(); }
    }

    #[inline(always)]
    pub fn fill_uniform_f64(&mut self, buf: &mut [f64]) {
        let mut iter = buf.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let u = self.next_f64_8();
            chunk.copy_from_slice(&u);
        }
        for slot in iter.into_remainder() { *slot = self.next_uniform_f64(); }
    }

    #[inline(always)]
    pub fn fill_uniform_in_range_f32<MIN, MAX>(&mut self, buf: &mut [f32], min: MIN, max: MAX)
    where MIN: ParamSource<f32>, MAX: ParamSource<f32>,
    {
        let limit = buf.len().min(min.len()).min(max.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(16);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 4;
            let u = self.next_f32_16();
            let mi = min.chunk::<16>(offset);
            let ma = max.chunk::<16>(offset);
            for j in 0..16 { chunk[j] = mi[j] + u[j] * (ma[j] - mi[j]); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !15;
        for (i, slot) in rem.iter_mut().enumerate() {
            *slot = self.next_uniform_in_range_f32(min.get(offset + i), max.get(offset + i));
        }
    }

    #[inline(always)]
    pub fn fill_uniform_in_range_f64<MIN, MAX>(&mut self, buf: &mut [f64], min: MIN, max: MAX)
    where MIN: ParamSource<f64>, MAX: ParamSource<f64>,
    {
        let limit = buf.len().min(min.len()).min(max.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let u = self.next_f64_8();
            let mi = min.chunk::<8>(offset);
            let ma = max.chunk::<8>(offset);
            for j in 0..8 { chunk[j] = mi[j] + u[j] * (ma[j] - mi[j]); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() {
            *slot = self.next_uniform_in_range_f64(min.get(offset + i), max.get(offset + i));
        }
    }

    // -------------------------------------------------------------------------
    // make_filled_* — allocate, fill, and return a [T; N] array (stack)
    // -------------------------------------------------------------------------

    #[inline(always)]
    pub fn make_filled_uniform_f32<const N: usize>(&mut self) -> [f32; N] {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let mut iter = slice.chunks_exact_mut(16);
        for chunk in iter.by_ref() {
            let u = self.next_f32_16();
            for j in 0..16 { chunk[j].write(u[j]); }
        }
        for slot in iter.into_remainder() { slot.write(self.next_uniform_f32()); }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_uniform_f64<const N: usize>(&mut self) -> [f64; N] {
        let mut buf = MaybeUninit::<[f64; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f64>, N) };
        let mut iter = slice.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let u = self.next_f64_8();
            for j in 0..8 { chunk[j].write(u[j]); }
        }
        for slot in iter.into_remainder() { slot.write(self.next_uniform_f64()); }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_uniform_in_range_f32<MIN, MAX, const N: usize>(&mut self, min: MIN, max: MAX) -> [f32; N]
    where MIN: ParamSource<f32>, MAX: ParamSource<f32>,
    {
        let mut buf = MaybeUninit::<[f32; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f32>, N) };
        let limit = slice.len().min(min.len()).min(max.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(16);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 4;
            let u = self.next_f32_16();
            let mi = min.chunk::<16>(offset);
            let ma = max.chunk::<16>(offset);
            for j in 0..16 { chunk[j].write(mi[j] + u[j] * (ma[j] - mi[j])); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !15;
        for (i, slot) in rem.iter_mut().enumerate() {
            slot.write(self.next_uniform_in_range_f32(min.get(offset + i), max.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }

    #[inline(always)]
    pub fn make_filled_uniform_in_range_f64<MIN, MAX, const N: usize>(&mut self, min: MIN, max: MAX) -> [f64; N]
    where MIN: ParamSource<f64>, MAX: ParamSource<f64>,
    {
        let mut buf = MaybeUninit::<[f64; N]>::uninit();
        let slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut MaybeUninit<f64>, N) };
        let limit = slice.len().min(min.len()).min(max.len());
        let (active, _) = slice.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let u = self.next_f64_8();
            let mi = min.chunk::<8>(offset);
            let ma = max.chunk::<8>(offset);
            for j in 0..8 { chunk[j].write(mi[j] + u[j] * (ma[j] - mi[j])); }
        }
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, slot) in rem.iter_mut().enumerate() {
            slot.write(self.next_uniform_in_range_f64(min.get(offset + i), max.get(offset + i)));
        }
        unsafe { buf.assume_init() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand_f32() {
        let mut rng = WyRand::new(1);
        for _ in 0..128 {
            let rv = rng.next_uniform_in_range_f32(1.0, 12.5);
            assert!(rv >= 1.0 && rv <= 12.5);
        }
    }

    #[test]
    fn test_rand_f64() {
        let mut rng = WyRand::new(1);
        for _ in 0..128 {
            let rv = rng.next_uniform_f64();
            assert!(rv >= 0.0 && rv <= 1.0);
        }
    }

    #[test]
    fn test_fill_vs_make_filled_uniform_f32() {
        let mut rng_a = WyRand::new(42);
        let mut rng_b = WyRand::new(42);
        let n = 1024;
        let mut buf = vec![0.0f32; n];
        rng_a.fill_uniform_f32(&mut buf);
        let arr: [f32; 1024] = rng_b.make_filled_uniform_f32();
        assert_eq!(&buf[..], &arr[..]);
    }
}
