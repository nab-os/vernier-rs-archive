//! Maximal-length LFSR sequences — the absolute position code (André et al.
//! 2020, 2021).
//!
//! A maximal LFSR of order `n` produces `2ⁿ − 1` bits where every window of `n`
//! consecutive bits is unique. That's what makes the position absolute: any
//! local view of `n` bits tells you where you are in the sequence. Since a
//! maximal LFSR never hits the all-zero state, its windows exclude all-zeros;
//! every other `n`-bit word, all-ones included, appears exactly once.

/// A maximal-length LFSR of the given order, holding the `2ⁿ − 1` bit sequence
/// used as the absolute code along one axis.
#[derive(Clone, Debug)]
pub struct Lfsr {
    /// Register order (bits per unique window).
    pub order: u32,
    /// The generated bit sequence, length `2^order - 1`.
    bits: Vec<u8>,
}

impl Lfsr {
    /// Builds a maximal-length sequence of the given `order`, or `None` if the
    /// order is outside 4..=12 — the range C++ `MegarenaBitSequence::generate`
    /// supports. Restricting to the same range guarantees any sequence produced
    /// here matches a physically printed C++-generated pattern; a differing
    /// generator would render and decode self-consistently in Rust but fail in
    /// the field against real patterns.
    pub fn maximal(order: u32) -> Option<Self> {
        if !(4..=12).contains(&order) {
            return None;
        }
        // Fibonacci left-shift LFSR matching C++ MegarenaBitSequence::generate().
        let code_max = 1u32 << order;
        let code_count = code_max - 1;
        let mut bits = Vec::with_capacity(code_count as usize);
        let mut code = code_count; // start from the all-ones state
        bits.push(1u8);
        for _ in 1..code_count {
            let nb = cpp_next_bit(order, code);
            code = (code * 2) % code_max + nb as u32;
            bits.push(nb);
        }
        Some(Self { order, bits })
    }

    /// The full bit sequence (length `2^order - 1`).
    pub fn bits(&self) -> &[u8] {
        &self.bits
    }

    /// Sequence length, `2^order - 1`.
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// The bit at code position `k`, wrapping around the sequence.
    pub fn bit_at(&self, k: usize) -> u8 {
        self.bits[k % self.bits.len()]
    }

    /// Finds a window of `order` consecutive bits and returns its starting
    /// position, or `None` if it doesn't occur. `window` must have length
    /// `order`. Scans the sequence once; for repeated lookups use
    /// [`WindowIndex`] instead.
    pub fn locate(&self, window: &[u8]) -> Option<usize> {
        let n = self.order as usize;
        if window.len() != n {
            return None;
        }
        let len = self.bits.len();
        for start in 0..len {
            let matches = (0..n).all(|j| self.bits[(start + j) % len] == window[j]);
            if matches {
                return Some(start);
            }
        }
        None
    }

    /// Builds a precomputed window→position index for fast repeated localization.
    pub fn window_index(&self) -> WindowIndex {
        let n = self.order as usize;
        let len = self.bits.len();
        let mut map = std::collections::HashMap::with_capacity(len);
        for start in 0..len {
            let mut word = 0u32;
            for j in 0..n {
                word = (word << 1) | self.bits[(start + j) % len] as u32;
            }
            map.insert(word, start);
        }
        WindowIndex {
            order: self.order,
            map,
        }
    }
}

/// A precomputed map from `order`-bit window (packed MSB-first) to its position
/// in the sequence, for O(1) lookups.
#[derive(Clone, Debug)]
pub struct WindowIndex {
    order: u32,
    map: std::collections::HashMap<u32, usize>,
}

impl WindowIndex {
    /// Locates a decoded bit window. `window` must have length `order`; returns
    /// `None` on a wrong length or a window that never occurs (e.g. from a
    /// decode error).
    pub fn locate(&self, window: &[u8]) -> Option<usize> {
        let n = self.order as usize;
        if window.len() != n {
            return None;
        }
        let mut word = 0u32;
        for &b in window {
            word = (word << 1) | (b & 1) as u32;
        }
        self.map.get(&word).copied()
    }
}

/// Feedback bit for the Fibonacci left-shift LFSR, matching C++
/// MegarenaBitSequence::nextBit().
fn cpp_next_bit(order: u32, state: u32) -> u8 {
    let b = |pos: u32| ((state >> pos) & 1) as u8;
    match order {
        4 => b(0) ^ b(3),
        5 => b(1) ^ b(4),
        6 => b(0) ^ b(5),
        7 => b(2) ^ b(6),
        8 => b(0) ^ b(1) ^ b(6) ^ b(7),
        9 => b(3) ^ b(8),
        10 => b(6) ^ b(9),
        11 => b(1) ^ b(10),
        12 => b(0) ^ b(1) ^ b(7) ^ b(11),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximal_length_is_correct() {
        for order in 4..=12 {
            let l = Lfsr::maximal(order).unwrap();
            assert_eq!(l.len(), (1usize << order) - 1, "order {order}");
        }
    }

    #[test]
    fn rejects_orders_outside_cpp_range() {
        assert!(Lfsr::maximal(3).is_none());
        assert!(Lfsr::maximal(13).is_none());
    }

    #[test]
    fn every_n_window_is_unique() {
        // The defining property: every n-bit window is distinct.
        let order = 6u32;
        let l = Lfsr::maximal(order).unwrap();
        let bits = l.bits();
        let n = order as usize;
        let mut seen = std::collections::HashSet::new();
        for start in 0..bits.len() {
            let mut word = 0u32;
            for j in 0..n {
                word = (word << 1) | bits[(start + j) % bits.len()] as u32;
            }
            assert!(seen.insert(word), "duplicate window at {start}");
        }
        assert_eq!(seen.len(), (1usize << n) - 1);
    }

    #[test]
    fn sequence_excludes_all_zero_window() {
        // A maximal LFSR never hits the all-zero state, so no window is all zeros.
        let order = 5u32;
        let l = Lfsr::maximal(order).unwrap();
        let bits = l.bits();
        let n = order as usize;
        for start in 0..bits.len() {
            let all_zero = (0..n).all(|j| bits[(start + j) % bits.len()] == 0);
            assert!(!all_zero, "found all-zero window at {start}");
        }
    }

    #[test]
    fn locate_inverts_bit_at() {
        // Read a window at any position and `locate` recovers that position —
        // how the megarena decoder turns decoded bits into an absolute order k.
        let order = 8u32;
        let l = Lfsr::maximal(order).unwrap();
        let n = order as usize;
        for start in 0..l.len() {
            let window: Vec<u8> = (0..n).map(|j| l.bit_at(start + j)).collect();
            assert_eq!(l.locate(&window), Some(start), "scan locate at {start}");
        }
    }

    #[test]
    fn window_index_matches_scan() {
        let order = 9u32;
        let l = Lfsr::maximal(order).unwrap();
        let idx = l.window_index();
        let n = order as usize;
        for start in 0..l.len() {
            let window: Vec<u8> = (0..n).map(|j| l.bit_at(start + j)).collect();
            assert_eq!(idx.locate(&window), Some(start), "index locate at {start}");
        }
    }

    #[test]
    fn locate_rejects_invalid_window() {
        let l = Lfsr::maximal(6).unwrap();
        // All-zero window never occurs in a maximal sequence.
        assert_eq!(l.locate(&[0, 0, 0, 0, 0, 0]), None);
        // Wrong length.
        assert_eq!(l.locate(&[1, 0, 1]), None);
    }
}
