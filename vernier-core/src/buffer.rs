use crate::complex::Complex32;
use crate::error::{Result, VernierError};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BufferLayout {
    pub width: usize,
    pub height: usize,
    pub row_stride: usize,
}

impl BufferLayout {
    #[inline]
    pub const fn packed(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            row_stride: width,
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.width * self.height
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    #[inline]
    pub const fn is_contiguous(&self) -> bool {
        self.row_stride == self.width
    }

    /// Checks that `len` elements fill this layout as one contiguous, non-empty
    /// buffer, which is what every backend's `upload` needs.
    pub fn check_upload(&self, len: usize) -> Result<()> {
        if !self.is_contiguous() {
            return Err(VernierError::NonContiguous { stride: self.row_stride, width: self.width });
        }
        if self.is_empty() {
            return Err(VernierError::UnsupportedSize(self.width, self.height));
        }
        if len != self.len() {
            return Err(VernierError::LengthMismatch { expected: self.len(), actual: len });
        }
        Ok(())
    }

    #[inline]
    pub fn flat_index(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.height && col < self.width, "index out of bounds");
        row * self.row_stride + col
    }
}

pub trait Buffer2D {
    fn layout(&self) -> BufferLayout;

    #[inline]
    fn width(&self) -> usize {
        self.layout().width
    }

    #[inline]
    fn height(&self) -> usize {
        self.layout().height
    }
}

pub trait BufferElement: bytemuck::Pod {}
impl BufferElement for Complex32 {}
