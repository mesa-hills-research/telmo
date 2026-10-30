//! SHA-256 (FIPS 180-4), for checking downloaded model files.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// A SHA-256 digest computed over data fed in any number of pieces.
pub struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    /// Bytes waiting in `block`.
    pending: usize,
    /// Bytes fed in all.
    total: u64,
}

impl Sha256 {
    pub fn new() -> Sha256 {
        Sha256 {
            state: INITIAL,
            block: [0; 64],
            pending: 0,
            total: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if self.pending > 0 {
            let take = data.len().min(64 - self.pending);
            self.block[self.pending..self.pending + take].copy_from_slice(&data[..take]);
            self.pending += take;
            data = &data[take..];
            if self.pending < 64 {
                return;
            }
            compress(&mut self.state, &self.block);
            self.pending = 0;
        }
        let mut blocks = data.chunks_exact(64);
        for block in &mut blocks {
            compress(&mut self.state, block.try_into().unwrap());
        }
        let rest = blocks.remainder();
        self.block[..rest.len()].copy_from_slice(rest);
        self.pending = rest.len();
    }

    pub fn finish(mut self) -> [u8; 32] {
        let bits = self.total * 8;
        self.update(&[0x80]);
        while self.pending != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());
        let mut digest = [0; 32];
        for (out, word) in digest.chunks_exact_mut(4).zip(self.state) {
            out.copy_from_slice(&word.to_be_bytes());
        }
        digest
    }
}

fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (w, bytes) in w.iter_mut().zip(block.chunks_exact(4)) {
        *w = u32::from_be_bytes(bytes.try_into().unwrap());
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *s = s.wrapping_add(v);
    }
}

/// The digest as lowercase hex, the way checksums are usually written.
pub fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// The hex digest of a file, read a megabyte at a time.
pub fn file_hex(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut sha = Sha256::new();
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            return Ok(hex(&sha.finish()));
        }
        sha.update(&buf[..n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(data: &[u8]) -> String {
        let mut sha = Sha256::new();
        sha.update(data);
        hex(&sha.finish())
    }

    #[test]
    fn digests_match_the_fips_180_4_examples() {
        assert_eq!(
            digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            digest(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            digest(b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
        assert_eq!(
            digest(&vec![b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn feeding_data_in_pieces_gives_the_same_digest() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 + i / 13) as u8).collect();
        let whole = digest(&data);
        for size in [1, 3, 55, 56, 63, 64, 65, 128, 999] {
            let mut sha = Sha256::new();
            for piece in data.chunks(size) {
                sha.update(piece);
            }
            assert_eq!(hex(&sha.finish()), whole, "pieces of {size} bytes");
        }
    }

    #[test]
    fn a_file_hashes_like_its_bytes() {
        let dir = crate::tests::temp_dir("sha256");
        let path = dir.join("data");
        let data = vec![0x5a; (1 << 20) + 17];
        std::fs::write(&path, &data).unwrap();
        assert_eq!(file_hex(&path).unwrap(), digest(&data));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
