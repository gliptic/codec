all:
	#cargo build
	#rustc src/codec.rs -o bin/codec -O --test && ./bin/codec --bench
	rustc src/codec.rs -o bin/codec -O # && ./bin/codec