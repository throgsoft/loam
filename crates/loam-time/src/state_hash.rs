// FNV-1a 64 (Fowler/Noll/Vo 1991, <http://www.isthe.com/chongo/tech/comp/fnv/>).
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over unframed little-endian words; callers own the sampled state schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateHash(u64);

impl Default for StateHash {
    fn default() -> Self {
        Self::new()
    }
}

impl StateHash {
    pub fn new() -> Self {
        Self(FNV_OFFSET_BASIS)
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    pub fn write_u32(&mut self, word: u32) {
        self.write_bytes(&word.to_le_bytes());
    }

    pub fn write_u32s(&mut self, words: &[u32]) {
        for &word in words {
            self.write_u32(word);
        }
    }

    pub fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    /// Preserves signed zero and NaN payload bits.
    pub fn write_f32(&mut self, value: f32) {
        self.write_u32(value.to_bits());
    }

    pub fn finish(&self) -> u64 {
        self.0
    }
}
