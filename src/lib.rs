#[macro_use]
extern crate lazy_static;
extern crate regex;

mod execute;
pub mod parse;
mod tokenize;

pub use parse::parsetree::Program;

#[derive(Debug)]
pub struct Error {
    message: String,
}

pub fn code_to_ast(code: &str) -> Result<Program, Error> {
    let tokens = match tokenize::tokenize(code) {
        Ok(x) => x,
        Err(err) => {
            return Err(Error {
                message: format!("Lexer error at {}:{}", err.line, err.column),
            })
        }
    };
    let tree = match parse::parse_program(&tokens) {
        Ok(x) => x,
        Err(err) => {
            return Err(Error {
                message: format!("Parser error at {}:{}", err.line, err.column),
            })
        }
    };
    Ok(tree)
}

pub fn run_ast(program: &Program, input: &str) -> Result<String, Error> {
    match execute::execute_program(program, input) {
        Ok(output) => Ok(output),
        Err(err) => Err(Error {
            message: err.message,
        }),
    }
}
