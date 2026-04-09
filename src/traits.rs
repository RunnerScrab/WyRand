/// A trait for types that can provide SIMD chunks of parameters.
/// 
/// This trait is implemented for scalars (providing a broadcasted chunk) and
/// slices/vectors/arrays (providing a direct memory load).
pub trait ParamSource<T: Copy>: Copy {
    /// Returns the length of the source. Scalars return usize::MAX.
    fn len(&self) -> usize;
    /// Returns a chunk of N elements starting at the given offset.
    /// If the chunk extends beyond the length, it is padded with the last element or zeros.
    fn chunk<const N: usize>(&self, offset: usize) -> [T; N];
    /// Returns a single element at the given index.
    fn get(&self, idx: usize) -> T;
}

impl ParamSource<f32> for f32 {
    #[inline(always)]
    fn len(&self) -> usize { usize::MAX }
    #[inline(always)]
    fn chunk<const N: usize>(&self, _: usize) -> [f32; N] { [*self; N] }
    #[inline(always)]
    fn get(&self, _: usize) -> f32 { *self }
}

impl<'a> ParamSource<f32> for &'a [f32] {
    #[inline(always)]
    fn len(&self) -> usize { (self as &[f32]).len() }
    #[inline(always)]
    fn chunk<const N: usize>(&self, offset: usize) -> [f32; N] {
        let mut arr = [0.0; N];
        let len = self.len();
        if offset + N <= len {
            arr.copy_from_slice(&self[offset..offset + N]);
        } else if offset < len {
            let src = &self[offset..];
            let take = N.min(src.len());
            arr[..take].copy_from_slice(&src[..take]);
        }
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> f32 { self[idx] }
}

impl<'a> ParamSource<f32> for &'a Vec<f32> {
    #[inline(always)]
    fn len(&self) -> usize { (**self).len() }
    #[inline(always)]
    fn chunk<const N: usize>(&self, offset: usize) -> [f32; N] {
        let mut arr = [0.0; N];
        let slice = self.as_slice();
        let len = slice.len();
        if offset + N <= len {
            arr.copy_from_slice(&slice[offset..offset + N]);
        } else if offset < len {
            let src = &slice[offset..];
            let take = N.min(src.len());
            arr[..take].copy_from_slice(&src[..take]);
        }
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> f32 { self[idx] }
}

impl<'a, const LANES: usize> ParamSource<f32> for &'a [f32; LANES] {
    #[inline(always)]
    fn len(&self) -> usize { LANES }
    #[inline(always)]
    fn chunk<const N: usize>(&self, offset: usize) -> [f32; N] {
        let mut arr = [0.0; N];
        if offset + N <= LANES {
            arr.copy_from_slice(&self[offset..offset + N]);
        } else if offset < LANES {
            let src = &self[offset..];
            let take = N.min(src.len());
            arr[..take].copy_from_slice(&src[..take]);
        }
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> f32 { self[idx] }
}

impl ParamSource<f64> for f64 {
    #[inline(always)]
    fn len(&self) -> usize { usize::MAX }
    #[inline(always)]
    fn chunk<const N: usize>(&self, _: usize) -> [f64; N] { [*self; N] }
    #[inline(always)]
    fn get(&self, _: usize) -> f64 { *self }
}

impl<'a> ParamSource<f64> for &'a [f64] {
    #[inline(always)]
    fn len(&self) -> usize { (self as &[f64]).len() }
    #[inline(always)]
    fn chunk<const N: usize>(&self, offset: usize) -> [f64; N] {
        let mut arr = [0.0; N];
        let len = self.len();
        if offset + N <= len {
            arr.copy_from_slice(&self[offset..offset + N]);
        } else if offset < len {
            let src = &self[offset..];
            let take = N.min(src.len());
            arr[..take].copy_from_slice(&src[..take]);
        }
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> f64 { self[idx] }
}

impl<'a> ParamSource<f64> for &'a Vec<f64> {
    #[inline(always)]
    fn len(&self) -> usize { (**self).len() }
    #[inline(always)]
    fn chunk<const N: usize>(&self, offset: usize) -> [f64; N] {
        let mut arr = [0.0; N];
        let slice = self.as_slice();
        let len = slice.len();
        if offset + N <= len {
            arr.copy_from_slice(&slice[offset..offset + N]);
        } else if offset < len {
            let src = &slice[offset..];
            let take = N.min(src.len());
            arr[..take].copy_from_slice(&src[..take]);
        }
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> f64 { self[idx] }
}

impl<'a, const LANES: usize> ParamSource<f64> for &'a [f64; LANES] {
    #[inline(always)]
    fn len(&self) -> usize { LANES }
    #[inline(always)]
    fn chunk<const N: usize>(&self, offset: usize) -> [f64; N] {
        let mut arr = [0.0; N];
        if offset + N <= LANES {
            arr.copy_from_slice(&self[offset..offset + N]);
        } else if offset < LANES {
            let src = &self[offset..];
            let take = N.min(src.len());
            arr[..take].copy_from_slice(&src[..take]);
        }
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> f64 { self[idx] }
}

impl ParamSource<(f32, f32)> for (f32, f32) {
    #[inline(always)]
    fn len(&self) -> usize { usize::MAX }
    #[inline(always)]
    fn chunk<const N: usize>(&self, _: usize) -> [(f32, f32); N] { [*self; N] }
    #[inline(always)]
    fn get(&self, _: usize) -> (f32, f32) { *self }
}

impl<'a> ParamSource<(f32, f32)> for &'a [(f32, f32)] {
    #[inline(always)]
    fn len(&self) -> usize { (self as &[(f32, f32)]).len() }
    #[inline(always)]
    fn chunk<const N: usize>(&self, offset: usize) -> [(f32, f32); N] {
        let mut arr = [(0.0, 0.0); N];
        let len = self.len();
        if offset + N <= len {
            arr.copy_from_slice(&self[offset..offset + N]);
        } else if offset < len {
            let src = &self[offset..];
            let take = N.min(src.len());
            arr[..take].copy_from_slice(&src[..take]);
        }
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> (f32, f32) { self[idx] }
}

impl<'a> ParamSource<(f32, f32)> for &'a Vec<(f32, f32)> {
    #[inline(always)]
    fn len(&self) -> usize { (**self).len() }
    #[inline(always)]
    fn chunk<const N: usize>(&self, offset: usize) -> [(f32, f32); N] {
        let mut arr = [(0.0, 0.0); N];
        let slice = self.as_slice();
        let len = slice.len();
        if offset + N <= len {
            arr.copy_from_slice(&slice[offset..offset + N]);
        } else if offset < len {
            let src = &slice[offset..];
            let take = N.min(src.len());
            arr[..take].copy_from_slice(&src[..take]);
        }
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> (f32, f32) { self[idx] }
}

impl ParamSource<(f64, f64)> for (f64, f64) {
    #[inline(always)]
    fn len(&self) -> usize { usize::MAX }
    #[inline(always)]
    fn chunk<const N: usize>(&self, _: usize) -> [(f64, f64); N] { [*self; N] }
    #[inline(always)]
    fn get(&self, _: usize) -> (f64, f64) { *self }
}

impl<'a> ParamSource<(f64, f64)> for &'a [(f64, f64)] {
    #[inline(always)]
    fn len(&self) -> usize { (self as &[(f64, f64)]).len() }
    #[inline(always)]
    fn chunk<const N: usize>(&self, offset: usize) -> [(f64, f64); N] {
        let mut arr = [(0.0, 0.0); N];
        let len = self.len();
        if offset + N <= len {
            arr.copy_from_slice(&self[offset..offset + N]);
        } else if offset < len {
            let src = &self[offset..];
            let take = N.min(src.len());
            arr[..take].copy_from_slice(&src[..take]);
        }
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> (f64, f64) { self[idx] }
}

impl<'a> ParamSource<(f64, f64)> for &'a Vec<(f64, f64)> {
    #[inline(always)]
    fn len(&self) -> usize { (**self).len() }
    #[inline(always)]
    fn chunk<const N: usize>(&self, offset: usize) -> [(f64, f64); N] {
        let mut arr = [(0.0, 0.0); N];
        let slice = self.as_slice();
        let len = slice.len();
        if offset + N <= len {
            arr.copy_from_slice(&slice[offset..offset + N]);
        } else if offset < len {
            let src = &slice[offset..];
            let take = N.min(src.len());
            arr[..take].copy_from_slice(&src[..take]);
        }
        arr
    }
    #[inline(always)]
    fn get(&self, idx: usize) -> (f64, f64) { self[idx] }
}
