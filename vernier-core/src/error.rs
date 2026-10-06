use crate::buffer::BufferLayout;

pub type Result<T> = core::result::Result<T, VernierError>;

#[derive(Debug, thiserror::Error)]
pub enum VernierError {
    #[error("non-contiguous buffer: row_stride {stride} != width {width}")]
    NonContiguous { stride: usize, width: usize },

    #[error("shape mismatch: {lhs:?} vs {rhs:?}")]
    ShapeMismatch {
        lhs: BufferLayout,
        rhs: BufferLayout,
    },

    #[error("buffer holds {actual} elements, layout needs {expected}")]
    LengthMismatch { expected: usize, actual: usize },

    #[error("unsupported transform size: {0}x{1}")]
    UnsupportedSize(usize, usize),

    #[error("backend error: {0}")]
    Backend(String),

    /// A general library-level error carrying a human-readable message
    /// (used e.g. by the object/factory parity layer in `vernier-detector`).
    #[error("{0}")]
    Message(String),
}
