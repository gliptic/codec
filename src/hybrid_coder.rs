
use std::mem;
use std::io::{Writer, Reader};
use std::ptr;
use bit_models::BitModel;

static BUF_SIZE: uint = (1 << 17);

pub struct HybridWriter<'a> {
	buffer: Vec<u8>,
	next: *mut u8,
	end: *mut u8,
	reserved_range_bytes: [*mut u8, ..4],
	reserved_bit_bytes: [*mut u8, ..3],
	bits: u32, bits_written: u32,
	low: u32, high: u32,
	w: &'a mut Writer,
}

impl<'a> HybridWriter<'a> {
	pub fn new<'b>(w: &'b mut Writer) -> HybridWriter<'b> {
		unsafe {
			let mut r = HybridWriter {
				buffer: Vec::from_elem(BUF_SIZE, 0xffu8),
				next: mem::uninitialized(),
				end: mem::uninitialized(),
				reserved_range_bytes: mem::uninitialized(),
				reserved_bit_bytes: mem::uninitialized(),
				bits_written: 0, bits: 0,
				low: 0, high: 0xffffffff,
				w: w,
			};

			r.next = r.buffer.get_mut(0) as *mut _;
			r.end = r.next.offset(r.buffer.len() as int);

			for i in range(0, 4) {
				r.reserved_range_bytes[i] = r.next;
				r.next = r.next.offset(1);
			}

			for i in range(0, 3) {
				r.reserved_bit_bytes[i] = r.next;
				r.next = r.next.offset(1);
			}

			r
		}
	}

	fn flush(&mut self) {
		unsafe {
			let p = self.low | 0xffffff;
			*self.reserved_range_bytes[0] = (p >> 24) as u8;
			*self.reserved_range_bytes[1] = (p >> 16) as u8;
			*self.reserved_range_bytes[2] = (p >> 8) as u8;
			*self.reserved_bit_bytes[0] = self.bits as u8;
			*self.reserved_bit_bytes[1] = (self.bits >> 8) as u8;

			//println!("Flushing at call {}, {}, {}", self.calls, self.bcalls, self.bits_written);

			let written = (self.next as uint) - (self.buffer.get_mut(0) as *mut _) as uint;

			{
				//println!("writing {}", written);

				let arr = self.buffer.slice_to(written);

				let _ = self.w.write(arr);
			}

			self.next = self.buffer.get_mut(0) as *mut _;
			self.low = 0;
			self.high = 0xffffffff;

			for i in range(0, 4) {
				self.reserved_range_bytes[i] = self.next;
				self.next = self.next.offset(1);
			}

			self.bits = 0;
			self.bits_written = 0;

			for i in range(0, 3) {
				self.reserved_bit_bytes[i] = self.next;
				self.next = self.next.offset(1);
			}
		}
	}

	pub fn finalize(&mut self) {
		self.flush();
	}

	fn check_end(&mut self) -> bool {
		self.next == self.end
	}

	#[inline]
	fn reserve_range_byte(&mut self) {
		self.reserved_range_bytes[0] = self.reserved_range_bytes[1];
		self.reserved_range_bytes[1] = self.reserved_range_bytes[2];
		self.reserved_range_bytes[2] = self.reserved_range_bytes[3];
		if self.check_end() {
			self.flush();
		} else {
			unsafe {
				self.reserved_range_bytes[3] = self.next;
				self.next = self.next.offset(1);
			}
		}
	}

	#[inline]
	fn reserve_bit_byte(&mut self) {
		self.reserved_bit_bytes[0] = self.reserved_bit_bytes[1];
		self.reserved_bit_bytes[1] = self.reserved_bit_bytes[2];
		if self.check_end() {
			self.flush();
		} else {
			self.reserved_bit_bytes[2] = self.next;
			self.next = unsafe { self.next.offset(1) };
		}
	}

	pub fn push_bit(&mut self, y: u32, p: u32) {

		let mid = self.low + ((self.high - self.low) >> 12) * p;

		//println!("mid = {}", mid);

		if y > 0 { self.high = mid; }
		else { self.low = mid + 1; }

		//println!("low = {}, high = {}", self.low, self.high);

		while (self.low ^ self.high) < (1<<24) {
			unsafe {
				*self.reserved_range_bytes[0] = (self.high >> 24) as u8;
			}
			self.low <<= 8;
			self.high = (self.high << 8) + 0xff;
			self.reserve_range_byte();
		}
	}

	pub fn push_byte(&mut self, b: u8) {
		if self.check_end() {
			self.flush();
		}
		unsafe {
			*self.next = b;
			self.next = self.next.offset(1);
		}
	}

	#[inline]
	pub fn push_bit_model<BM: BitModel>(&mut self, y: u32, model: &mut BM) {
		self.push_bit(y, model.p());
		model.update(y);
	}
}

impl<'a> ::prefix_code::BitWriter for HybridWriter<'a> {
	fn push_bits_uni(&mut self, mut bits: u32, mut count: u32) {
		self.bits |= bits << (self.bits_written as uint);
		self.bits_written += count;

		while self.bits_written >= 8 {
			unsafe {
				*self.reserved_bit_bytes[0] = self.bits as u8;
			}
			self.bits_written -= 8;
			self.bits >>= 8;
			self.reserve_bit_byte();
		}
	}
}

pub struct HybridReader<'a> {
	buffer: Vec<u8>,
	next: *mut u8,
	end: *mut u8,
	low: u32, high: u32, x: u32,
	bits: u32, bits_left: u32,
	r: &'a mut Reader,
}

impl<'a> HybridReader<'a> {
	pub fn new<'b>(r: &'b mut Reader) -> HybridReader<'b> {
		unsafe {
			let mut ret = HybridReader {
				buffer: Vec::from_elem(BUF_SIZE, 0u8),
				next: mem::uninitialized(),
				end: mem::uninitialized(),
				low: 0, high: 0xffffffff, x: 0,
				bits: 0, bits_left: 24,
				r: r,
			};

			let bytes_read = ret.r.read(ret.buffer.as_mut_slice()).unwrap_or(0);

			//println!("reading {}", bytes_read);

			ret.next = ret.buffer.get_mut(0) as *mut _;
			ret.end = ret.next.offset(bytes_read as int);

			for _ in range(0u32, 4) {
				ret.x = (ret.x << 8) + ret.read_byte() as u32;
			}

			for i in range(0u, 3) {
				ret.bits |= (ret.read_byte() as u32) << (i * 8);
			}

			ret
		}
	}

	pub fn pull_bit(&mut self, p: u32) -> u32 {
		let mid = self.low + ((self.high - self.low) >> 12) * p;
		//println!("x = {}, mid = {}", self.x, mid);
		let y = if self.x <= mid {
			self.high = mid;
			1
		} else {
			self.low = mid + 1;
			0
		};

		while (self.low ^ self.high) < (1<<24) {
			self.low <<= 8;
			self.high = (self.high << 8) + 0xff;

			unsafe {
				if self.check_underflow() {
					self.underflow_no_read();
					break;
				}

				let b = self.unsafe_read_byte();
				self.x = (self.x << 8) + b as u32;
			}
		}

		y
	}

	#[inline]
	pub fn pull_byte(&mut self) -> u8 {
		self.read_byte()
	}

	fn flush(&mut self) {
		//println!("Flushing at call {}, {}, {}", self.calls, self.bcalls, 24 - self.bits_left);

		self.low = 0;
		self.high = 0xffffffff;
		self.x = 0;

		for _ in range(0u32, 4) {
			self.x = (self.x << 8) + self.unsafe_read_byte() as u32;
		}

		self.bits = 0;
		for i in range(0u, 3) {
			self.bits |= (self.unsafe_read_byte() as u32) << (i * 8);
		}
		self.bits_left = 24;

		
	}

	fn underflow_no_read(&mut self) {
		unsafe {
			let at_buffer_end = self.next == (self.buffer.get_mut(0) as *mut _).offset(self.buffer.len() as int);

			let bytes_read = self.r.read(self.buffer.as_mut_slice()).unwrap_or(0);

			//println!("reading {}", bytes_read);

			self.next = self.buffer.get_mut(0) as *mut _;
			self.end = self.next.offset(bytes_read as int);

			if at_buffer_end {
				self.flush();
			} else {
				fail!("Underflow should only occur at buffer end");
			}
		}
		
	}

	#[inline]
	fn check_underflow(&mut self) -> bool {
		if self.next == self.end {
			self.underflow_no_read();
			assert!(self.next != self.end);
			true
		} else {
			false
		}
	}

	#[inline]
	fn unsafe_read_byte(&mut self) -> u8 {
		unsafe {
			let b = *self.next;
			self.next = self.next.offset(1);
			b
		}
	}

	#[inline]
	fn read_byte(&mut self) -> u8 {
		self.check_underflow();
		self.unsafe_read_byte()
	}

	#[inline]
	pub fn pull_bit_model<BM: BitModel>(&mut self, model: &mut BM) -> u32 {
		let y = self.pull_bit(model.p());
		model.update(y);
		y
	}
}

impl<'a> ::prefix_code::BitReader for HybridReader<'a> {

	fn pull_bits_uni(&mut self, count: u32) -> u32 {
		let r = self.bits & ((1u32 << count as uint) - 1);

		self.skip_bits_uni(count);
		r
	}

	fn peek_bits_uni16(&self) -> u16 {
		self.bits as u16
	}

	#[inline]
	fn skip_bits_uni(&mut self, count: u32) {
		self.bits_left -= count;
		self.bits >>= count as uint;

		while self.bits_left <= 16 {
			if self.check_underflow() {
				break;
			}
			
			self.bits |= self.unsafe_read_byte() as u32 << self.bits_left as uint;
			self.bits_left += 8;
		}
	}
}

#[cfg(test)]
mod test {
	use hybrid_coder::{HybridReader, HybridWriter};
	use std::io::{MemWriter, MemReader};
	use prefix_code::{BitWriter, BitReader};

	#[test]
	fn read_write() {
		let mut w = MemWriter::new();

		let count = 10000;

		{
			let mut hw = HybridWriter::new(&mut w);

			hw.push_bits_uni(1, 5);

			for i in range(0u32, count) {
				hw.push_bit(1, 1579);
				hw.push_byte(3);
				hw.push_bits_uni(6, 7);
			}
			
			hw.push_bits_uni(2, 2);
			hw.push_byte(45);
			hw.finalize();
		}

		let mut r = MemReader::new(w.unwrap());

		{
			let mut hr = HybridReader::new(&mut r);

			assert_eq!(1, hr.pull_bits_uni(5));

			for i in range(0u32, count) {
				assert_eq!(1, hr.pull_bit(1579));
				assert_eq!(3, hr.pull_byte());
				assert_eq!(6, hr.pull_bits_uni(7));
			}

			assert_eq!(2, hr.pull_bits_uni(2));
			assert_eq!(45, hr.pull_byte());
		}
	}
}