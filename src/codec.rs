#![feature(macro_rules)]

extern crate test;
extern crate time;

use hybrid_coder::{HybridWriter, HybridReader};
use bit_models::{BitModelFast};
use std::io::{MemWriter, MemReader, BufReader, File};
use std::path::Path;

use prefix_code::{OrdFreq, PrefixCode, PrefixModel};

use time::precise_time_ns;

mod hybrid_coder;
mod prefix_code;
mod bit_models;

define_polar_model!(ByteModel, 256)


fn main() {

	/*
	let mut lengths = [0u8, ..10];
	let mut codes = [PrefixCode::new(0, 0), ..10];
	let mut symbols = [OrdFreq::new(0, 0), ..10];

	for i in range(0u32, 10) {
		symbols[i as uint] = OrdFreq(i + ((i * 2 + 1) << 16));
	}

	prefix_code::sort_symbols(symbols);
	prefix_code::polar_code_lengths(symbols, lengths);
	prefix_code::generate_codes(lengths, codes);

	println!("lengths: {}", lengths.as_slice());
	for i in range(0, codes.len()) {
		let c = codes[i];
		for b in range(0, c.size() as uint).rev() {
			print!("{}", (c.code() >> b) & 1);
		}
		println!("");
	}
	*/

/*
	let mut w = MemWriter::new();

	let contents = File::open(&Path::new("/home/glip/enwik8")).unwrap().read_to_end().unwrap();

	{
		let mut model = TestModel::new();
		let mut bm = BitModelFast::new();
		let mut hw = HybridWriter::new(&mut w);

		for &c in contents.iter() {
			model.update(true);
			model.write(&mut hw, c as u32);
			//hw.push_bit_model(0, &mut bm);
			model.incr(c as u32);
		}

		hw.finalize();
	}

	println!("Written {} bytes / {}", w.get_ref().len(), contents.len());

	let mut r = MemReader::new(w.unwrap());

	{
		let mut model = TestModel::new();
		let mut bm = BitModelFast::new();
		let mut hr = HybridReader::new(&mut r);

		for &c in contents.iter() {
			model.update(false);
			let read = model.read(&mut hr) as u8;
			assert_eq!(c, read);		
			model.incr(read as u32);
		}
	}
	*/

	let mut w = MemWriter::new();

	let contents = File::open(&Path::new("/home/glip/enwik8"))
		.unwrap()
		//.read_exact(30000000)
		.read_to_end()
		.unwrap();

	{
		let mut models = Vec::from_elem(256, ByteModel::new());
		//let mut models = [ByteModel::new(), ..256];

		let mut hw = HybridWriter::new(&mut w);
		let mut context = 0u8;

		for &c in contents.iter() {
			let m = models.get_mut(context as uint);
			m.update(true);
			m.write(&mut hw, c as u32);
			m.incr(c as u32);
			context = c;
		}

		hw.finalize();
	}

	let compressed = w.unwrap();

	println!("Written {} bytes / {}", compressed.len(), contents.len());

	let mut least_time = std::num::Bounded::max_value();

	let mut dest = Vec::with_capacity(contents.len());
	
	for _ in range(0u, 10) {
		let time = precise_time_ns();

		let mut r = BufReader::new(compressed.as_slice());

		let mut models = Vec::from_elem(256, ByteModel::new());

		let mut hr = HybridReader::new(&mut r);
		let mut context = 0u8;

		dest.clear();

		for i in range(0, contents.len()) {
			let m = models.get_mut(context as uint);
			m.update(false);
			let read = m.read(&mut hr) as u8;

/*
			if contents[i] != read {
				println!("error at {}, {} != {}", i, read, contents[i]);
				fail!();
			}*/
			m.incr(read as u32);
			dest.push(read);
			context = read;
		}

		let total_time = precise_time_ns() - time;

		least_time = std::cmp::min(least_time, total_time);
	}

	
	println!("Time: {} s", least_time as f64 / 1000000000.0);

	/*
	let mut w = MemWriter::new();

	{
		let mut hw = HybridWriter::new(&mut w);

		for i in range(0u32, 10000) {
			hw.push_bit(1, 2048);
			hw.push_byte(3);
		}
		
		hw.push_byte(45);
		hw.finalize();
	}

	let mut r = MemReader::new(w.unwrap());

	{
		let mut hr = HybridReader::new(&mut r);
		let mut y1 = 0;

		//let y1 = hr.pull_bit(2047);
		for i in range(0u32, 10000) {
			y1 = hr.pull_bit(2048);
			let _ = hr.pull_byte();
		}

		println!("Reading byte");
		let b1 = hr.pull_byte();

		println!("Written {} {}", b1, y1);
	}*/

	//println!("Written {} bytes", w.get_ref().len());
}