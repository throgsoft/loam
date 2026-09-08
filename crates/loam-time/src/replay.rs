use std::fmt;

use thiserror::Error;

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

pub const TAPE_MAGIC: [u8; 8] = *b"LOAMTAPE";

/// Version of the tape container, independent of the game's input and state schemas.
pub const TAPE_FORMAT_VERSION: u32 = 1;

// Tape v1 header: magic, version, hz, seed, words_per_tick, ticks, checkpoint count.
const HEADER_LEN: usize = 8 + 4 + 4 + 8 + 4 + 8 + 4;
const CHECKPOINT_LEN: usize = 16;

/// A state hash observed after `tick` completed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    pub tick: u64,
    pub state_hash: u64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TapeError {
    #[error("not a loam tape: leading bytes are {found:02x?}")]
    BadMagic { found: [u8; 8] },
    #[error("tape format version {found}, this build reads {TAPE_FORMAT_VERSION}")]
    UnsupportedVersion { found: u32 },
    #[error("tape declares {expected} bytes, got {found}")]
    LengthMismatch { expected: u64, found: usize },
    #[error("checkpoint ticks are not strictly ascending at index {index}")]
    CheckpointOrder { index: usize },
}

/// `words_per_tick` may be zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tape {
    tick_hz: u32,
    seed: u64,
    words_per_tick: u32,
    ticks: u64,
    inputs: Vec<u32>,
    checkpoints: Vec<Checkpoint>,
}

impl Tape {
    pub fn new(tick_hz: u32, seed: u64, words_per_tick: u32) -> Self {
        Self {
            tick_hz,
            seed,
            words_per_tick,
            ticks: 0,
            inputs: Vec::new(),
            checkpoints: Vec::new(),
        }
    }

    pub fn tick_hz(&self) -> u32 {
        self.tick_hz
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn words_per_tick(&self) -> u32 {
        self.words_per_tick
    }

    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    /// Panics if `input.len()` is not `words_per_tick`.
    pub fn push_tick(&mut self, input: &[u32]) {
        assert_eq!(
            input.len(),
            self.words_per_tick as usize,
            "tape frame is {} words",
            self.words_per_tick,
        );
        self.inputs.extend_from_slice(input);
        self.ticks += 1;
    }

    pub fn input(&self, tick: u64) -> Option<&[u32]> {
        if tick >= self.ticks {
            return None;
        }
        let width = self.words_per_tick as usize;
        let start = tick as usize * width;
        Some(&self.inputs[start..start + width])
    }

    /// Panics unless `tick` is past the last checkpoint.
    pub fn checkpoint(&mut self, tick: u64, state_hash: u64) {
        if let Some(last) = self.checkpoints.last() {
            assert!(
                tick > last.tick,
                "checkpoint ticks must ascend: {tick} after {}",
                last.tick,
            );
        }
        self.checkpoints.push(Checkpoint { tick, state_hash });
    }

    pub fn checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            HEADER_LEN + self.inputs.len() * 4 + self.checkpoints.len() * CHECKPOINT_LEN,
        );
        bytes.extend_from_slice(&TAPE_MAGIC);
        bytes.extend_from_slice(&TAPE_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.tick_hz.to_le_bytes());
        bytes.extend_from_slice(&self.seed.to_le_bytes());
        bytes.extend_from_slice(&self.words_per_tick.to_le_bytes());
        bytes.extend_from_slice(&self.ticks.to_le_bytes());
        bytes.extend_from_slice(&(self.checkpoints.len() as u32).to_le_bytes());
        for &word in &self.inputs {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for checkpoint in &self.checkpoints {
            bytes.extend_from_slice(&checkpoint.tick.to_le_bytes());
            bytes.extend_from_slice(&checkpoint.state_hash.to_le_bytes());
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, TapeError> {
        let mut reader = Reader::new(bytes);
        let magic = reader.take::<8>().ok_or(TapeError::LengthMismatch {
            expected: HEADER_LEN as u64,
            found: bytes.len(),
        })?;
        if magic != TAPE_MAGIC {
            return Err(TapeError::BadMagic { found: magic });
        }
        // Version before length: another version has another layout.
        let version = reader.u32().ok_or(TapeError::LengthMismatch {
            expected: HEADER_LEN as u64,
            found: bytes.len(),
        })?;
        if version != TAPE_FORMAT_VERSION {
            return Err(TapeError::UnsupportedVersion { found: version });
        }
        let (tick_hz, seed, words_per_tick, ticks, checkpoint_count) =
            reader.header_tail().ok_or(TapeError::LengthMismatch {
                expected: HEADER_LEN as u64,
                found: bytes.len(),
            })?;

        let input_bytes = u64::from(words_per_tick)
            .saturating_mul(ticks)
            .saturating_mul(4);
        let expected = (HEADER_LEN as u64)
            .saturating_add(input_bytes)
            .saturating_add(u64::from(checkpoint_count) * CHECKPOINT_LEN as u64);
        if expected != bytes.len() as u64 {
            return Err(TapeError::LengthMismatch {
                expected,
                found: bytes.len(),
            });
        }

        let word_count = (input_bytes / 4) as usize;
        let mut inputs = Vec::with_capacity(word_count);
        for _ in 0..word_count {
            inputs.push(reader.u32().ok_or(TapeError::LengthMismatch {
                expected,
                found: bytes.len(),
            })?);
        }
        let mut checkpoints = Vec::with_capacity(checkpoint_count as usize);
        for index in 0..checkpoint_count as usize {
            let tick = reader.u64().ok_or(TapeError::LengthMismatch {
                expected,
                found: bytes.len(),
            })?;
            let state_hash = reader.u64().ok_or(TapeError::LengthMismatch {
                expected,
                found: bytes.len(),
            })?;
            if let Some(last) = checkpoints.last() {
                let Checkpoint { tick: prev, .. } = *last;
                if tick <= prev {
                    return Err(TapeError::CheckpointOrder { index });
                }
            }
            checkpoints.push(Checkpoint { tick, state_hash });
        }

        Ok(Self {
            tick_hz,
            seed,
            words_per_tick,
            ticks,
            inputs,
            checkpoints,
        })
    }
}

impl fmt::Display for Tape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tape v{TAPE_FORMAT_VERSION}: {} ticks at {}Hz, seed {:#018x}, \
             {} input words/tick, {} checkpoints",
            self.ticks,
            self.tick_hz,
            self.seed,
            self.words_per_tick,
            self.checkpoints.len(),
        )
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (value, remaining) = self.bytes.split_first_chunk::<N>()?;
        self.bytes = remaining;
        Some(*value)
    }

    fn u32(&mut self) -> Option<u32> {
        self.take::<4>().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Option<u64> {
        self.take::<8>().map(u64::from_le_bytes)
    }

    fn header_tail(&mut self) -> Option<(u32, u64, u32, u64, u32)> {
        Some((
            self.u32()?,
            self.u64()?,
            self.u32()?,
            self.u64()?,
            self.u32()?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recorded() -> Tape {
        let mut tape = Tape::new(60, 0x1234_5678_9abc_def0, 3);
        for tick in 0..8u64 {
            let t = tick as u32;
            tape.push_tick(&[t, t.wrapping_mul(7), (1.5f32 * t as f32).to_bits()]);
        }
        tape.checkpoint(3, 0xdead_beef_0000_0001);
        tape.checkpoint(7, 0xdead_beef_0000_0002);
        tape
    }

    #[test]
    fn hash_detects_word_order_and_bit_changes() {
        let mut ab = StateHash::new();
        ab.write_u32(1);
        ab.write_u32(2);
        let mut ba = StateHash::new();
        ba.write_u32(2);
        ba.write_u32(1);
        assert_ne!(ab.finish(), ba.finish(), "swapped operands must be visible");

        let mut low = StateHash::new();
        low.write_u32s(&[7, 7, 7, 7]);
        let mut flipped = StateHash::new();
        flipped.write_u32s(&[7, 7, 7 ^ 1, 7]);
        assert_ne!(low.finish(), flipped.finish());
    }

    #[test]
    fn hash_uses_little_endian_words() {
        let mut split = StateHash::new();
        split.write_u32(0);
        split.write_u32(1);
        let mut whole = StateHash::new();
        whole.write_u64(1 << 32);
        assert_eq!(split.finish(), whole.finish());
    }

    #[test]
    fn hash_preserves_signed_zero_and_nan_payloads() {
        for (a, b) in [(0x0000_0000, 0x8000_0000), (0x7fc0_0001, 0x7fc0_0002)] {
            let mut first = StateHash::new();
            first.write_f32(f32::from_bits(a));
            let mut second = StateHash::new();
            second.write_f32(f32::from_bits(b));
            assert_ne!(first.finish(), second.finish());
        }
    }

    #[test]
    fn encoded_tape_preserves_fields() {
        let tape = recorded();
        let decoded = Tape::decode(&tape.encode()).expect("own encoding decodes");
        assert_eq!(decoded, tape);
    }

    #[test]
    fn input_ends_at_tick_count() {
        let tape = recorded();
        assert_eq!(tape.ticks(), 8);
        assert_eq!(tape.input(0), Some(&[0u32, 0, 0.0f32.to_bits()][..]));
        assert_eq!(tape.input(5).map(|w| w[1]), Some(35));
        assert_eq!(tape.input(8), None, "one past the last tick has no input");
    }

    #[test]
    fn zero_width_tape_preserves_ticks() {
        let mut tape = Tape::new(120, 7, 0);
        for _ in 0..4 {
            tape.push_tick(&[]);
        }
        tape.checkpoint(3, 0xabc);
        let decoded = Tape::decode(&tape.encode()).expect("width-zero tape decodes");
        assert_eq!(decoded.ticks(), 4);
        assert_eq!(decoded.input(0), Some(&[][..]));
        assert_eq!(decoded.checkpoints(), tape.checkpoints());
    }

    #[test]
    fn future_container_version_is_rejected() {
        let mut bytes = recorded().encode();
        let bumped = TAPE_FORMAT_VERSION + 1;
        bytes[8..12].copy_from_slice(&bumped.to_le_bytes());
        assert_eq!(
            Tape::decode(&bytes),
            Err(TapeError::UnsupportedVersion { found: bumped }),
        );
    }

    #[test]
    fn bad_magic_and_empty_tapes_are_rejected() {
        let mut bytes = recorded().encode();
        bytes[0] = b'X';
        let mut found = TAPE_MAGIC;
        found[0] = b'X';
        assert_eq!(Tape::decode(&bytes), Err(TapeError::BadMagic { found }));
        assert!(matches!(
            Tape::decode(&[]),
            Err(TapeError::LengthMismatch { .. }),
        ));
    }

    #[test]
    fn truncated_and_padded_payloads_are_rejected() {
        let full = recorded().encode();
        let expected = full.len() as u64;
        assert_eq!(
            Tape::decode(&full[..full.len() - 1]),
            Err(TapeError::LengthMismatch {
                expected,
                found: full.len() - 1,
            }),
        );

        let mut padded = full.clone();
        padded.push(0);
        assert_eq!(
            Tape::decode(&padded),
            Err(TapeError::LengthMismatch {
                expected,
                found: padded.len(),
            }),
            "trailing bytes mean the writer and the reader disagree",
        );
    }

    #[test]
    fn oversized_input_width_is_rejected() {
        let mut bytes = recorded().encode();
        bytes[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            Tape::decode(&bytes),
            Err(TapeError::LengthMismatch { .. }),
        ));
    }

    #[test]
    fn overflowing_tick_count_is_rejected() {
        let mut bytes = recorded().encode();
        bytes[28..36].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            Tape::decode(&bytes),
            Err(TapeError::LengthMismatch { .. }),
        ));
    }

    #[test]
    fn decoded_checkpoints_must_ascend() {
        let mut tape = recorded();
        tape.checkpoints[1].tick = 3;
        assert_eq!(
            Tape::decode(&tape.encode()),
            Err(TapeError::CheckpointOrder { index: 1 }),
        );
    }

    #[test]
    #[should_panic(expected = "tape frame is 3 words")]
    fn short_input_cannot_shift_tick_boundaries() {
        let mut tape = Tape::new(60, 0, 3);
        tape.push_tick(&[1, 2]);
    }

    #[test]
    #[should_panic(expected = "checkpoint ticks must ascend")]
    fn writer_rejects_duplicate_checkpoint_ticks() {
        let mut tape = Tape::new(60, 0, 0);
        tape.checkpoint(5, 1);
        tape.checkpoint(5, 2);
    }
}
