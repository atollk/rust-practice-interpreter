use ::parser::{code_to_ast, run_ast};

fn main() {
    let code = std::fs::read_to_string("example2.txt").unwrap();
    let program = code_to_ast(&code).unwrap();
    let result = run_ast(&program, "1,2,4,32,7").unwrap();
    println!("{}", result);
}