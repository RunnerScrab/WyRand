use crate::WyRand;
use crate::traits::ParamSource;

impl WyRand {
    #[inline(always)]
    pub fn next_uniform_f64(&mut self) -> f64 {
        let rv = self.next_u64();
        let bits = (rv >> 12) | 0x3FF0_0000_0000_0000;
        f64::from_bits(bits) - 1.0
    }

    #[inline(always)]
    pub fn next_uniform_f32(&mut self) -> f32 {
        let rv = self.next_u64();
        f32::from_bits(((rv & Self::FLOAT32_MASK) | 0x3F80_0000) as u32) - 1.0
    }

    /// Generates a uniform f32 in the range [min, max].
    #[inline(always)]
    pub fn next_uniform_in_range_f32(&mut self, min: f32, max: f32) -> f32 {
        let rv = self.next_uniform_f32();
        rv.mul_add(max - min, min)
    }

    /// Generates a uniform f64 in the range [min, max].
    #[inline(always)]
    pub fn next_uniform_in_range_f64(&mut self, min: f64, max: f64) -> f64 {
        let rv = self.next_uniform_f64();
        rv.mul_add(max - min, min)
    }

    #[inline(always)]
    pub(crate) fn next_f32_16(&mut self) -> [f32; 16] {
        let mut u = [0.0; 16];
        let mut current_s = self.0;
        for j in 0..16 {
            current_s = current_s.wrapping_add(Self::INC);
            let tmp = (current_s as u128).wrapping_mul(0xa3b195354a39b70d);
            let m1 = ((tmp.wrapping_shr(64)) as u64) ^ (tmp as u64);
            let tmp2 = (m1 as u128).wrapping_mul(0x1b03738712fad5c9);
            let rv = ((tmp2.wrapping_shr(64)) as u64) ^ (tmp2 as u64);
            u[j] = f32::from_bits(((rv & Self::FLOAT32_MASK) | 0x3F80_0000) as u32) - 1.0;
        }
        self.0 = self.0.wrapping_add(Self::INC << 4);
        u
    }

    #[inline(always)]
    pub(crate) fn next_f32_8(&mut self) -> [f32; 8] {
        let mut u = [0.0; 8];
        let mut current_s = self.0;
        for j in 0..8 {
            current_s = current_s.wrapping_add(Self::INC);
            let tmp = (current_s as u128).wrapping_mul(0xa3b195354a39b70d);
            let m1 = ((tmp.wrapping_shr(64)) as u64) ^ (tmp as u64);
            let tmp2 = (m1 as u128).wrapping_mul(0x1b03738712fad5c9);
            let rv = ((tmp2.wrapping_shr(64)) as u64) ^ (tmp2 as u64);
            u[j] = f32::from_bits(((rv & Self::FLOAT32_MASK) | 0x3F80_0000) as u32) - 1.0;
        }
        self.0 = self.0.wrapping_add(Self::INC << 3);
        u
    }

    #[inline(always)]
    pub(crate) fn next_f64_8(&mut self) -> [f64; 8] {
        let mut u = [0.0; 8];
        let mut current_s = self.0;
        for j in 0..8 {
            current_s = current_s.wrapping_add(Self::INC);
            let tmp = (current_s as u128).wrapping_mul(0xa3b195354a39b70d);
            let m1 = ((tmp.wrapping_shr(64)) as u64) ^ (tmp as u64);
            let tmp2 = (m1 as u128).wrapping_mul(0x1b03738712fad5c9);
            let rv = ((tmp2.wrapping_shr(64)) as u64) ^ (tmp2 as u64);
            let bits = (rv >> 12) | 0x3FF0_0000_0000_0000;
            u[j] = f64::from_bits(bits) - 1.0;
        }
        self.0 = self.0.wrapping_add(Self::INC << 3);
        u
    }

    #[inline(always)]
    pub(crate) fn next_f64_4(&mut self) -> [f64; 4] {
        let mut u = [0.0; 4];
        let mut current_s = self.0;
        for j in 0..4 {
            current_s = current_s.wrapping_add(Self::INC);
            let tmp = (current_s as u128).wrapping_mul(0xa3b195354a39b70d);
            let m1 = ((tmp.wrapping_shr(64)) as u64) ^ (tmp as u64);
            let tmp2 = (m1 as u128).wrapping_mul(0x1b03738712fad5c9);
            let rv = ((tmp2.wrapping_shr(64)) as u64) ^ (tmp2 as u64);
            let bits = (rv >> 12) | 0x3FF0_0000_0000_0000;
            u[j] = f64::from_bits(bits) - 1.0;
        }
        self.0 = self.0.wrapping_add(Self::INC << 2);
        u
    }

    #[inline(always)]
    pub fn next_u64_in_range(&mut self, min: u64, max: u64) -> u64 {
        let rv: u128 = self.next_u64() as u128;
        let span: u128 = (max - min) as u128;
        min + (rv * span).wrapping_shr(64) as u64
    }

    #[inline(always)]
    pub fn next_usize_in_range(&mut self, max: usize) -> usize {
        self.next_u64_in_range(0, max as u64) as usize
    }

    #[inline(always)]
    pub fn fill_uniform_f32(&mut self, buf: &mut [f32]) {
        let mut iter = buf.chunks_exact_mut(16);
        for chunk in iter.by_ref() {
            let u = self.next_f32_16();
            chunk.copy_from_slice(&u);
        }
        for val in iter.into_remainder() {
            *val = self.next_uniform_f32();
        }
    }

    #[inline(always)]
    pub fn fill_uniform_f64(&mut self, buf: &mut [f64]) {
        let mut iter = buf.chunks_exact_mut(8);
        for chunk in iter.by_ref() {
            let u = self.next_f64_8();
            chunk.copy_from_slice(&u);
        }
        for val in iter.into_remainder() {
            *val = self.next_uniform_f64();
        }
    }

    /// Parameters `min` and `max` can be either scalars (broadcasting to the 
    /// entire buffer) or slices/vectors (providing per-element values).
    #[inline]
    pub fn fill_uniform_in_range_f32<MIN, MAX>(&mut self, buf: &mut [f32], min: MIN, max: MAX)
    where
        MIN: ParamSource<f32>,
        MAX: ParamSource<f32>,
    {
        let limit = buf.len().min(min.len()).min(max.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(16);
        
        assert!(limit <= min.len());
        assert!(limit <= max.len());

        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 4;
            let u = self.next_f32_16();
            
            let mi1 = min.chunk::<16>(offset);
            let ma1 = max.chunk::<16>(offset);

            for j in 0..16 {
                chunk[j] = mi1[j] + (1.0 - u[j]) * (ma1[j] - mi1[j]);
            }
        }
        
        let rem = iter.into_remainder();
        let offset = limit & !15;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_uniform_in_range_f32(min.get(offset + i), max.get(offset + i));
        }
    }

    #[inline]
    pub fn fill_uniform_in_range_f64<MIN, MAX>(&mut self, buf: &mut [f64], min: MIN, max: MAX)
    where
        MIN: ParamSource<f64>,
        MAX: ParamSource<f64>,
    {
        let limit = buf.len().min(min.len()).min(max.len());
        let (active, _) = buf.split_at_mut(limit);
        let mut iter = active.chunks_exact_mut(8);
        
        assert!(limit <= min.len());
        assert!(limit <= max.len());

        for (i, chunk) in iter.by_ref().enumerate() {
            let offset = i << 3;
            let u = self.next_f64_8();
            
            let mi8 = min.chunk::<8>(offset);
            let ma8 = max.chunk::<8>(offset);

            for j in 0..8 {
                chunk[j] = mi8[j] + (1.0 - u[j]) * (ma8[j] - mi8[j]);
            }
        }
        
        let rem = iter.into_remainder();
        let offset = limit & !7;
        for (i, val) in rem.iter_mut().enumerate() {
            *val = self.next_uniform_in_range_f64(min.get(offset + i), max.get(offset + i));
        }
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
}
