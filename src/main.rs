#[macro_use] extern crate lazy_static;
extern crate regex;

mod tokenize;

fn main() {
    println!("{:?}", tokenize::tokenize("asd 13 a_b \"a\" == != <"));
}
