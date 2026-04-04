#![allow(unused)]
#[derive(Clone, Copy, Debug)]
pub struct WyRand {
    state: u64,
}

impl WyRand {
    // Variant on the Wyhash algorithm, by "Vladimir Makarov", which I'm
    // sure isn't a pseudonym made up by an Anglophone
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline(always)]
    pub fn next_rand_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x60bee2bee120fc15);

        // First mix
        let tmp = (self.state as u128).wrapping_mul(0xa3b195354a39b70d);
        let m1 = ((tmp.wrapping_shr(64)) as u64) ^ (tmp as u64);

        // Second mix
        let tmp2 = (m1 as u128).wrapping_mul(0x1b03738712fad5c9);
        ((tmp2.wrapping_shr(64)) as u64) ^ (tmp2 as u64)
    }

    fn next_u32(&mut self) -> u32 {
        self.next_rand_u64().wrapping_shr(32) as u32
    }

    #[inline(always)]
    pub fn next_rand_f32(&mut self) -> f32 {
        let rv = self.next_rand_u64();
        const MASK: u64 = u64::MAX ^ 0xFFFF_FFFF_F800_0000; 
        f32::from_bits(((rv & MASK) | 0x3F80_0000) as u32) - 1.0
    }

    // Helper to get a range
    #[inline(always)]
    pub fn next_rand_in_range(&mut self, max: usize) -> usize {
        //TODO: Replace this modulus division
        (self.next_rand_u64() as usize) % max
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
            let rv = rng.next_rand_f32();
            println!("{}", rv);
        }
    }
}
