pub trait BitModel {
	fn p(&self) -> u32; // [0, 4095]
	fn update(&mut self, y: u32);

}

pub struct BitModelFast {
	prob: u16
}

impl BitModelFast {
	pub fn new() -> BitModelFast {
		BitModelFast { prob: 1 << 11 }
	}
}

impl BitModel for BitModelFast {
	fn p(&self) -> u32 { self.prob as u32 }

	fn update(&mut self, y: u32) {
		let mask = 0u32 - y;
		let shift1 = 5;
		let correction1 = (1u32 << shift1) - 2;
		self.prob += ((1 + ((4095 + correction1) & mask) - self.prob as u32) >> shift1) as u16;
	}
}