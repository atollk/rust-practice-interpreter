#[macro_use]
extern crate lazy_static;
extern crate regex;

mod tokenize;
pub mod parse;
mod execute;


#[derive(Debug)]
pub struct ParseError {
    line: usize,
    column: usize,
}


pub fn code_to_ast(code: &str) -> Result<parse::parsetree::Program, ParseError> {
    let tokens = match tokenize::tokenize(code) {
        Ok(x) => x,
        Err(err) => return Err(ParseError { line: err.line, column: err.column})
    };
    let tree = match parse::parse_program(&tokens) {
        Ok(x) => x,
        Err(err) => return Err(ParseError { line: err.line, column: err.column})
    };
    Ok(tree)
}