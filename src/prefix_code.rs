#![macro_escape]

use std::mem;
use std::intrinsics::ctlz32;
use std::cmp::max;
use std::iter::range_step;

static MAX_SUPPORTED_SYMS: u32 = 1024;
static MAX_EVER_CODE_SIZE: u32 = 34;
static MAX_EXPECTED_CODE_SIZE: uint = 16;

pub struct OrdFreq {
	f: u16,
	s: u16
}

impl OrdFreq {
	pub fn new(sym: u32, freq: u32) -> OrdFreq {
		OrdFreq { s: sym as u16, f: freq as u16 }
	}

	pub fn freq(self) -> u32 {
		self.f as u32
	}

	pub fn sym(self) -> u16 {
		self.s
	}
}

pub fn sort_symbols2<'a>(mut first: &'a mut [OrdFreq], mut second: &'a mut [OrdFreq]) -> &'a mut [OrdFreq] {
	let mut hist = [0u32, ..256 * 2];

	for &s in first.iter() {
		let f = s.freq();
		hist[       (f       & 0xff) as uint] += 1;
		hist[256 + ((f >> 8) & 0xff) as uint] += 1;
	}

	let num_syms = first.len();

	// If all radix-1 digits are zero, we only need one pass
	let passes = if hist[256] == num_syms as u32 { 1 } else { 2 };

	for pass in range(0, passes) {
		let c = &mut first[0] as *mut _;
		let n = &mut second[0] as *mut _;

		let histp = &mut hist[pass << 8] as *mut _;

		let mut offsets: [u32, ..256] = unsafe { mem::uninitialized() };
		let mut cur_ofs = 0;

		for i in range_step(0u, 256, 2) {
			offsets[i] = cur_ofs;
			cur_ofs += unsafe { *histp.offset(i as int) };

			offsets[i + 1] = cur_ofs;
			cur_ofs += unsafe { *histp.offset(i as int + 1) };
		}

		let pass_shift = pass << 3;

		let mut p = c;
		let endp = unsafe { c.offset(num_syms as int) };

		while p != endp {
			let mut f = unsafe { *p }.freq();

			f = (f >> pass_shift) & 0xff;

			let dst_offset = offsets[f as uint];
			offsets[f as uint] += 1;
			unsafe {
				*n.offset(dst_offset as int) = *p;
				p = p.offset(1);
			}

		}

		mem::swap(&mut first, &mut second);
	}

	let mut prev = 0;
	for i in range(0, num_syms) {
		assert!(first[i].freq() >= prev);
		prev = first[i].freq();
	}

	first
}

#[deriving(Clone)]
pub struct PrefixCode(u32);

impl PrefixCode {
	#[inline]
	pub fn new(code: u32, size: u8) -> PrefixCode {
		PrefixCode(code + (size as u32 << 16))
	}

	pub fn code(self) -> u32 {
		let PrefixCode(v) = self;
		v & 0xffff
	}

	pub fn size(self) -> u32 {
		let PrefixCode(v) = self;
		v >> 16
	}
}

#[inline]
pub fn reverse_u16(mut v: u32) -> u32 {
	v = (v & 0xff00) >> 8 | (v & 0x00ff) << 8;
	v = (v & 0xf0f0) >> 4 | (v & 0x0f0f) << 4;
	v = (v & 0xcccc) >> 2 | (v & 0x3333) << 2;
	v = (v & 0xaaaa) >> 1 | (v & 0x5555) << 1;
	v
}

pub fn generate_codes(sizes: &[u8], codes: &mut [PrefixCode]) -> bool {
	let mut num_codes: [u32, ..MAX_EXPECTED_CODE_SIZE + 1] = [0, ..MAX_EXPECTED_CODE_SIZE + 1];
	let mut next_code: [u32, ..MAX_EXPECTED_CODE_SIZE + 1] = [0, ..MAX_EXPECTED_CODE_SIZE + 1];

	for &s in sizes.iter() {
		num_codes[s as uint] += 1;
	}

	let mut code = 0u32;

	for i in range(1, MAX_EXPECTED_CODE_SIZE + 1) {
		next_code[i] = code;
		code += num_codes[i];
		code <<= 1;
	}

	if code != (1 << (MAX_EXPECTED_CODE_SIZE + 1)) {
		let mut t = 0u32;
		for i in range(1, MAX_EXPECTED_CODE_SIZE + 1) {
			t += num_codes[i];
			if t > 1 {
				//return false; // Error, sizes don't add up
				fail!("Code sizes don't add up");
			}
		}
	}

	for i in range(0, sizes.len()) {
		let c = sizes[i];
		let code = next_code[c as uint];
		next_code[c as uint] += 1;

		let rev_code = reverse_u16(code) >> (16 - c as uint);
		codes[i] = PrefixCode::new(rev_code, c);
	} 

	true
}

pub fn generate_codes_for_decode(
	sizes: &[u8],
	codes: &mut [PrefixCode],
	dec_first_offset: &mut [u16, ..17],
	dec_max_code: &mut [u32, ..18],
	dec_offset_to_sym: &mut [u16],
	decoder_table: &mut [u16],
	max_code_size: u32) -> bool {

	let mut num_codes: [u32, ..MAX_EXPECTED_CODE_SIZE + 1] = [0, ..MAX_EXPECTED_CODE_SIZE + 1];
	let mut next_code: [u32, ..MAX_EXPECTED_CODE_SIZE + 1] = [0, ..MAX_EXPECTED_CODE_SIZE + 1];

	for &s in sizes.iter() {
		num_codes[s as uint] += 1;
	}

	let mut code = 0u32;
	let mut offset = 0u32;

	for i in range(1, MAX_EXPECTED_CODE_SIZE + 1) {
		next_code[i] = code;
		dec_first_offset[i] = offset as u16 - code as u16;
		code += num_codes[i];
		dec_max_code[i] = code << (16 - i);
		code <<= 1;
		offset += num_codes[i];
	}
	dec_max_code[17] = 0x10000;

	if code != (1 << (MAX_EXPECTED_CODE_SIZE + 1)) {
		let mut t = 0u32;
		for i in range(1, MAX_EXPECTED_CODE_SIZE + 1) {
			t += num_codes[i];
			if t > 1 {
				//return false; // Error, sizes don't add up
				fail!("Code sizes don't add up");
			}
		}
	}

	for p in decoder_table.mut_iter() {
		*p = 0xffff;
	}

	for i in range(0, sizes.len()) {
		let s = sizes[i] as uint;

		let code = next_code[s];
		next_code[s] += 1;

		let offset = (code as u16 + dec_first_offset[s]) as uint;
		dec_offset_to_sym[offset] = i as u16;

		let rev_code = reverse_u16(code) >> (16 - s);
		codes[i] = PrefixCode::new(rev_code, s as u8);

		if s as u32 <= max_code_size {
			let step = 1 << s;
			let code = rev_code;

			for p in range_step(code, 1 << max_code_size as uint, step) {
				decoder_table[p as uint] = i as u16;
			}
		}
	} 

	true
}

pub fn generate_decoder_table(codes: &[PrefixCode], decoder_table: &mut [u16], max_code_size: u32) {
	assert!(decoder_table.len() == (1 << max_code_size as uint));

	for p in decoder_table.mut_iter() {
		*p = 0xffff;
	}

	for i in range(0, codes.len()) {
		if codes[i].size() as u32 <= max_code_size {
			assert!(codes[i].size() > 0);

			let step = 1 << codes[i].size() as uint;
			let code = codes[i].code();

			for p in range_step(code, 1 << max_code_size as uint, step) {
				decoder_table[p as uint] = i as u16;
			}
		}
	}
}

static POLAR_MAX_SYMBOLS: u32 = 256;

pub fn polar_code_lengths(symbols: &[OrdFreq], sizes: &mut [u8]) -> u32 {
	unsafe {
		let mut tmp_freq: [u32, ..POLAR_MAX_SYMBOLS] = mem::uninitialized();
		let mut orig_total_freq = 0;
		let mut cur_total = 0;
		let mut start_index = 0;
		let mut max_code_size = 0;
		let num_syms = symbols.len() as u32;

		for i in range(0, symbols.len()) {
			let sym_freq = symbols[symbols.len() - 1 - i].freq();
			//let sym_freq = symbols[i].freq();
			let sym_len = 31 - ctlz32(sym_freq);
			let adjusted_sym_freq = 1 << sym_len as uint;

			orig_total_freq += sym_freq;
			tmp_freq[i] = adjusted_sym_freq;
			cur_total += adjusted_sym_freq;
		}

		let mut tree_total = 1 << (31 - ctlz32(orig_total_freq)) as uint;
		if tree_total < orig_total_freq {
			tree_total <<= 1;
		}

		while cur_total < tree_total && start_index < num_syms {
			let mut i = start_index;
			while i < num_syms {
				let freq = tmp_freq[i as uint];
				if cur_total + freq <= tree_total {
					tmp_freq[i as uint] += freq;
					cur_total += freq;
					if cur_total == tree_total {
						break;
					}
				} else {
					start_index = i + 1;
				}
				i += 1;
			}
		}

		assert_eq!(cur_total, tree_total);

		let tree_total_bits = 32 - ctlz32(tree_total);

		for i in range(0, symbols.len()) {
			let codesize = tree_total_bits - (32 - ctlz32(tmp_freq[i]));
			max_code_size = max(max_code_size, codesize);
			sizes[symbols[symbols.len() - 1 - i].sym() as uint] = codesize as u8;
			//sizes[symbols[i].sym() as uint] = codesize as u8;
		}

		max_code_size
	}
	
}

pub trait PrefixModel {
	fn incr(&mut self, sym: u32);
	fn update(&mut self, for_encoding: bool);
	fn write<BW: ::prefix_code::BitWriter>(&mut self, bw: &mut BW, sym: u32);
	fn read<BR: ::prefix_code::BitReader>(&mut self, br: &mut BR) -> u32;
}

pub trait BitWriter {
	fn push_bits_uni(&mut self, bits: u32, count: u32);
}

pub trait BitReader {
	fn pull_bits_uni(&mut self, count: u32) -> u32;
	fn peek_bits_uni16(&self) -> u16;
	fn skip_bits_uni(&mut self, count: u32);
}

#[deriving(Copy)]
pub struct Foo {
    f: [u32, ..256]
}

impl Clone for Foo {
	fn clone(&self) -> Foo {
		Foo {
			f: self.f
		}
	}
}

macro_rules! define_polar_model(
	($name: ident, $symbol_count: expr) => {
		//#[deriving(Clone)]
		pub struct $name {
		    freq: [u32, ..$symbol_count],
		    codes: [::prefix_code::PrefixCode, ..$symbol_count],
		    decoder_table: [u16, ..(1 << 9)],
		    sum: u32,
		    next_rebuild: u32,

		    dec_max_code: [u32, ..18],
			dec_first_offset: [u16, ..17],
			dec_offset_to_sym: [u16, ..$symbol_count]
		}

		impl Clone for $name {
			fn clone(&self) -> $name {
				$name {
					freq: self.freq,
					codes: self.codes,
					decoder_table: self.decoder_table,
					sum: self.sum,
					next_rebuild: self.next_rebuild,

					dec_max_code: self.dec_max_code,
					dec_first_offset: self.dec_first_offset,
					dec_offset_to_sym: self.dec_offset_to_sym
				}
			}
		}

		impl $name {
			pub fn new() -> $name {
				$name {
					freq: [1u32, ..$symbol_count],
					codes: [::prefix_code::PrefixCode::new(0, 0), ..$symbol_count],
					decoder_table: unsafe { ::std::mem::uninitialized() },
					sum: $symbol_count,
					next_rebuild: $symbol_count,

					dec_max_code: unsafe { ::std::mem::uninitialized() },
					dec_first_offset: unsafe { ::std::mem::uninitialized() },
					dec_offset_to_sym: unsafe { ::std::mem::uninitialized() }
				}
			}

			pub fn print_codes(&self) {
				for i in range(0, self.codes.len()) {
					let c = self.codes[i];
					print!("{} ->", i);
					for b in range(0, c.size() as uint) {
						print!("{}", (c.code() >> b) & 1);
					}
					println!("");
				}

				for p in range(0u, 256) {
					let i = self.decoder_table[p];
					for b in range(0u, 16).rev() {
						print!("{}", (p >> b) & 1);
					}
					println!(" -> {}", i);
				}
			}
		}

		impl ::prefix_code::PrefixModel for $name {
			fn incr(&mut self, sym: u32) {
				self.freq[sym as uint] += 1;
				self.sum += 1;
			}

			fn update(&mut self, for_encoding: bool) {
				if self.sum >= self.next_rebuild {
					//println!("Rebuilding at {}", self.sum);

					let mut lengths = [0u8, ..$symbol_count];
					let mut symbols: [::prefix_code::OrdFreq, ..$symbol_count] = unsafe { ::std::mem::uninitialized() };
					let mut symbols2: [::prefix_code::OrdFreq, ..$symbol_count] = unsafe { ::std::mem::uninitialized() };

					let shift = unsafe { (32 - ::std::intrinsics::ctlz32(self.sum >> 16)) as uint };
					let offset = (1 << shift) - 1;

					for i in range(0u, $symbol_count) {
						symbols[i] = ::prefix_code::OrdFreq::new(
							i as u32,
							(self.freq[i] + offset) >> shift);
					}

					let sorted_symbols = ::prefix_code::sort_symbols2(symbols, symbols2);
					::prefix_code::polar_code_lengths(sorted_symbols, lengths);
					if !for_encoding {
						::prefix_code::generate_codes_for_decode(
							lengths,
							self.codes,
							&mut self.dec_first_offset,
							&mut self.dec_max_code,
							self.dec_offset_to_sym,
							self.decoder_table,
							9);
					} else {
						::prefix_code::generate_codes(lengths, self.codes);
					}

					//if self.sum <= 10 * ($symbol_count) {
						self.next_rebuild = self.sum * 3;
					/*
					} else {
						self.next_rebuild = self.sum + ($symbol_count) * 20;
					}*/
				}
			}

			fn write<BW: ::prefix_code::BitWriter>(&mut self, bw: &mut BW, sym: u32) {
				let c = self.codes[sym as uint];
				bw.push_bits_uni(c.code(), c.size());
			}

			fn read<BR: ::prefix_code::BitReader>(&mut self, br: &mut BR) -> u32 {

				let peek = br.peek_bits_uni16();
				let mut sym = self.decoder_table[(peek & 0x1ff) as uint] as u32;

				if sym < 0xffff {
					br.skip_bits_uni(self.codes[sym as uint].size());
					sym
				} else {
					let k = ::prefix_code::reverse_u16(peek as u32);
					let mut s = 10;
					while k >= self.dec_max_code[s] {
						s += 1;
					}

					assert!(s != 17);
					let offset = ((k >> (16 - s)) as u16 + self.dec_first_offset[s]) as uint;
					sym = self.dec_offset_to_sym[offset] as u32;

					br.skip_bits_uni(s as u32);
					sym
				}
			}
		}
	}
)

#[cfg(test)]
mod test {
	use std::intrinsics::ctlz32;
	use prefix_code::{OrdFreq, PrefixCode, PrefixModel, sort_symbols, polar_code_lengths, generate_codes};
	use std::io::{MemWriter, MemReader, BufReader, File};
	use std::path::Path;
	use hybrid_coder::{HybridWriter, HybridReader};
	use bit_models::{BitModelFast};
	use test::Bencher;

	define_polar_model!(TestModel, 10)
	define_polar_model!(ByteModel, 256)

	#[test]
	fn test_ctlz32() {
		unsafe {
			assert_eq!(5, ctlz32(0xffffffff >> 5));
		}
	}

	#[test]
	fn polar_small() {
		let mut lengths = [0u8, ..10];
		let mut codes = [PrefixCode::new(0, 0), ..10];
		let mut symbols = [OrdFreq::new(0, 0), ..10];

		for i in range(0u32, 10) {
			symbols[i as uint] = OrdFreq::new(i, (i * 2 + 1));
		}

		sort_symbols(symbols);
		polar_code_lengths(symbols, lengths);
		generate_codes(lengths, codes);

		println!("lengths: {}", lengths.as_slice());
	}

	fn number(mut x: u32) -> u32 {
		x *= 1362650787;
		let mut sum = 0;
		for i in range(3u, 12) {
			if x < (1 << i) {
				sum += 1;
			}
		}
		sum
	}

	#[test]
	fn polar_model() {
		let mut w = MemWriter::new();

		{
			let mut model = TestModel::new();
			let mut bm = BitModelFast::new();
			let mut hw = HybridWriter::new(&mut w);

			for i in range(0u32, 1000) {
				model.update(true);
				model.write(&mut hw, number(i));
				hw.push_bit_model(0, &mut bm);
				model.incr(number(i));
			}

			hw.finalize();
		}

		let mut r = MemReader::new(w.unwrap());

		{
			let mut model = TestModel::new();
			let mut bm = BitModelFast::new();
			let mut hr = HybridReader::new(&mut r);

			for i in range(0u32, 1000) {
				model.update(false);
				assert_eq!(number(i), model.read(&mut hr));
				let res = hr.pull_bit_model(&mut bm);
				if res != 0 {
					println!("at {}", i);
					fail!();
				}
				
				model.incr(number(i));
			}
		}
	}

	#[bench]
	fn bench_decode(b: &mut Bencher) {
		let mut w = MemWriter::new();

		let contents = File::open(&Path::new("/home/glip/enwik8")).unwrap().read_exact(1000000).unwrap();

		{
			let mut models = Vec::from_elem(256, ByteModel::new());

			let mut hw = HybridWriter::new(&mut w);
			let mut context = 0u8;

			for &c in contents.iter() {
				let mut m = models.get_mut(context as uint);
				m.update(true);
				m.write(&mut hw, c as u32);
				m.incr(c as u32);
				context = c;
			}

			hw.finalize();
		}

		//println!("Written {} bytes / {}", w.get_ref().len(), contents.len());

		let compressed = w.unwrap();

		b.iter(|| {
			let mut r = BufReader::new(compressed.as_slice());

			let mut models = Vec::from_elem(256, ByteModel::new());

			let mut hr = HybridReader::new(&mut r);
			let mut context = 0u8;

			for i in range(0, contents.len()) {
				let mut m = models.get_mut(context as uint);
				m.update(false);
				let read = m.read(&mut hr) as u8;
				m.incr(read as u32);
				context = read;
			}
		});
	}
}
