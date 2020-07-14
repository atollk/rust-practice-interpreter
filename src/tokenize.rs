pub mod tokens {
    #[derive(Debug, std::cmp::PartialEq)]
    pub enum LiteralToken {
        Bool(bool),
        Integer(i32),
        Float(f64),
        String(String),
    }

    #[derive(Debug, std::cmp::PartialEq, std::cmp::Eq, std::hash::Hash)]
    pub enum KeywordToken {
        Fn,
        Struct,
        With,
        If,
        Else,
        While,
        Return,
        Bool,
        Int,
        Float,
        String,
    }

    #[derive(Debug, std::cmp::PartialEq, std::cmp::Eq, std::hash::Hash)]
    pub enum SymbolToken {
        LeftBrace,
        RightBrace,
        LeftParenthesis,
        RightParenthesis,
        LeftBracket,
        RightBracket,
        LeftAngle,
        RightAngle,
        RightArrow,
        Assign,
        Colon,
        Comma,
        Semicolon,
        Period,
    }

    #[derive(Debug, std::cmp::PartialEq)]
    pub enum Token {
        Literal(LiteralToken),
        Keyword(KeywordToken),
        Symbol(SymbolToken),
        Identifier(String),
        Whitespace,
    }

    #[derive(Debug, std::cmp::PartialEq)]
    pub struct PositionalToken {
        pub token: Token,
        pub line: usize,
        pub column: usize,
    }

    use std::collections::HashMap;

    lazy_static! {
        static ref KEYWORD_MAP: HashMap<KeywordToken, &'static str> = vec! {
            (KeywordToken::Fn, "fn"),
            (KeywordToken::Struct, "struct")
        }
        .into_iter()
        .collect();
    }
}

mod matcher {
    use regex::Regex;

    pub fn match_one_token(text: &str, byte_index: &mut usize) -> Option<super::tokens::Token> {
        match text[*byte_index..].chars().nth(0)? {
            ' ' | '\t' | '\n' | '\r' => {
                *byte_index += 1;
                Some(super::tokens::Token::Whitespace)
            }
            '0'..='9' => match_number_literal(text, byte_index),
            '"' => match_string_literal(text, byte_index),
            'a'..='z' | 'A'..='Z' | '_' => match_keyword_or_identifier(text, byte_index),
            _ => match_symbol(text, byte_index),
        }
    }

    fn match_number_literal(text: &str, byte_index: &mut usize) -> Option<super::tokens::Token> {
        lazy_static! {
            static ref RE: Regex = Regex::new(r"^\d+(\.\d*)?").unwrap();
        }
        let captures: regex::Captures = RE.captures(&text[*byte_index..])?;
        let is_integer = captures.get(1).is_none();
        let matched_str = captures.get(0)?.as_str();
        *byte_index += matched_str.len();

        if is_integer {
            let val: i32 = matched_str.parse().ok()?;
            Some(super::tokens::Token::Literal(
                super::tokens::LiteralToken::Integer(val),
            ))
        } else {
            let val: f64 = matched_str.parse().ok()?;
            Some(super::tokens::Token::Literal(
                super::tokens::LiteralToken::Float(val),
            ))
        }
    }

    fn match_string_literal(text: &str, byte_index: &mut usize) -> Option<super::tokens::Token> {
        lazy_static! {
            static ref RE: Regex = Regex::new(r#"^"[^"\\]*(\\\.[^\\"]*)*""#).unwrap();
        }
        let m: regex::Match = RE.find(&text[*byte_index..])?;
        let matched_str = m.as_str();
        *byte_index += matched_str.len();
        return Some(super::tokens::Token::Literal(
            super::tokens::LiteralToken::String(matched_str[1..matched_str.len() - 1].to_owned()),
        ));
    }

    fn match_keyword_or_identifier(
        text: &str,
        byte_index: &mut usize,
    ) -> Option<super::tokens::Token> {
        lazy_static! {
            static ref RE: Regex = Regex::new(r"^(_|\w)(_|\w|\d)*").unwrap();
        }
        let m: regex::Match = RE.find(&text[*byte_index..])?;
        let matched_str = m.as_str();
        *byte_index += matched_str.len();
        let token = match matched_str {
            "true" => super::tokens::Token::Literal(super::tokens::LiteralToken::Bool(true)),
            "false" => super::tokens::Token::Literal(super::tokens::LiteralToken::Bool(false)),
            "fn" => super::tokens::Token::Keyword(super::tokens::KeywordToken::Fn),
            "struct" => super::tokens::Token::Keyword(super::tokens::KeywordToken::Struct),
            "with" => super::tokens::Token::Keyword(super::tokens::KeywordToken::With),
            "if" => super::tokens::Token::Keyword(super::tokens::KeywordToken::If),
            "else" => super::tokens::Token::Keyword(super::tokens::KeywordToken::Else),
            "while" => super::tokens::Token::Keyword(super::tokens::KeywordToken::While),
            "return" => super::tokens::Token::Keyword(super::tokens::KeywordToken::Return),
            "bool" => super::tokens::Token::Keyword(super::tokens::KeywordToken::Bool),
            "int" => super::tokens::Token::Keyword(super::tokens::KeywordToken::Int),
            "float" => super::tokens::Token::Keyword(super::tokens::KeywordToken::Float),
            "string" => super::tokens::Token::Keyword(super::tokens::KeywordToken::String),
            _ => super::tokens::Token::Identifier(matched_str.to_owned()),
        };
        Some(token)
    }

    fn match_symbol(text: &str, byte_index: &mut usize) -> Option<super::tokens::Token> {
        let token = match text[*byte_index..].chars().nth(0)? {
            '{' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::LeftBrace,
            )),
            '}' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::RightBrace,
            )),
            '(' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::LeftParenthesis,
            )),
            ')' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::RightParenthesis,
            )),
            '[' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::LeftBracket,
            )),
            ']' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::RightBracket,
            )),
            '<' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::LeftAngle,
            )),
            '>' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::RightAngle,
            )),
            '-' => {
                if let Some('>') = text[*byte_index..].chars().nth(1) {
                    *byte_index += 1;
                    Some(super::tokens::Token::Symbol(
                        super::tokens::SymbolToken::RightArrow,
                    ))
                } else {
                    None
                }
            }
            '=' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::Assign,
            )),
            ':' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::Colon,
            )),
            ',' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::Comma,
            )),
            ';' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::Semicolon,
            )),
            '.' => Some(super::tokens::Token::Symbol(
                super::tokens::SymbolToken::Period,
            )),
            _ => None,
        };
        *byte_index += 1;
        token
    }
}

#[derive(Debug)]
pub struct TokenizeError {
    pub line: usize,
    pub column: usize,
}

fn index_to_lcol(text: &str, byte_index: usize) -> (usize, usize) {
    let mut line = 0;
    let mut col = 0;
    for (pos, chr) in text.char_indices() {
        if pos >= byte_index {
            return (line, col);
        }
        if chr == '\n' {
            line += 1;
            col = 0;
        } else if chr != '\r' {
            col += 1;
        }
    }
    (0, 0)
}

pub fn tokenize_once(
    input: &str,
    byte_index: &mut usize,
) -> Result<tokens::PositionalToken, TokenizeError> {
    let (line, column) = index_to_lcol(input, *byte_index);
    match matcher::match_one_token(input, byte_index) {
        Some(token) => Ok(tokens::PositionalToken {
            token,
            line,
            column,
        }),
        None => Err(TokenizeError { line, column }),
    }
}

pub fn tokenize(input: &str) -> Result<Vec<tokens::PositionalToken>, TokenizeError> {
    let mut tokens = Vec::new();
    let mut byte_index: usize = 0;
    while byte_index < input.len() {
        let token = tokenize_once(input, &mut byte_index)?;
        match token.token {
            tokens::Token::Whitespace => (),
            _ => tokens.push(token),
        };
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::tokens::*;
    use super::*;

    fn one_token(input: &str) -> Token {
        let mut index = 0;
        let token = tokenize_once(input, &mut index).unwrap();
        assert_eq!(index, input.len());
        token.token
    }

    #[test]
    fn test_tokenize_primitives() {
        let bool_literal = "false";
        let int_literal = "42";
        let float_literal = "123.456";
        let string_literal = "\"hello world\"";
        let keyword_fn = "fn";
        let keyword_struct = "struct";
        let keyword_with = "with";
        let keyword_if = "if";
        let keyword_else = "else";
        let keyword_while = "while";
        let keyword_return = "return";
        let keyword_int = "int";
        let keyword_bool = "bool";
        let keyword_float = "float";
        let keyword_string = "string";
        let symbol_lbrace = "{";
        let symbol_rbrace = "}";
        let symbol_lparant = "(";
        let symbol_rparant = ")";
        let symbol_lbracket = "[";
        let symbol_rbracket = "]";
        let symbol_langle = "<";
        let symbol_rangle = ">";
        let symbol_rightarrow = "->";
        let symbol_assign = "=";
        let symbol_colon = ":";
        let symbol_comma = ",";
        let symbol_semicolon = ";";
        let symbol_period = ".";
        let identifier = "foo";

        assert_eq!(
            one_token(bool_literal),
            Token::Literal(LiteralToken::Bool(false))
        );
        assert_eq!(
            one_token(int_literal),
            Token::Literal(LiteralToken::Integer(42))
        );
        assert_eq!(
            one_token(float_literal),
            Token::Literal(LiteralToken::Float(123.456))
        );
        assert_eq!(
            one_token(string_literal),
            Token::Literal(LiteralToken::String("hello world".to_owned()))
        );
        assert_eq!(one_token(keyword_fn), Token::Keyword(KeywordToken::Fn));
        assert_eq!(
            one_token(keyword_struct),
            Token::Keyword(KeywordToken::Struct)
        );
        assert_eq!(one_token(keyword_with), Token::Keyword(KeywordToken::With));
        assert_eq!(one_token(keyword_if), Token::Keyword(KeywordToken::If));
        assert_eq!(one_token(keyword_else), Token::Keyword(KeywordToken::Else));
        assert_eq!(
            one_token(keyword_while),
            Token::Keyword(KeywordToken::While)
        );
        assert_eq!(
            one_token(keyword_return),
            Token::Keyword(KeywordToken::Return)
        );
        assert_eq!(one_token(keyword_int), Token::Keyword(KeywordToken::Int));
        assert_eq!(one_token(keyword_bool), Token::Keyword(KeywordToken::Bool));
        assert_eq!(
            one_token(keyword_float),
            Token::Keyword(KeywordToken::Float)
        );
        assert_eq!(
            one_token(keyword_string),
            Token::Keyword(KeywordToken::String)
        );
        assert_eq!(
            one_token(symbol_lbrace),
            Token::Symbol(SymbolToken::LeftBrace)
        );
        assert_eq!(
            one_token(symbol_rbrace),
            Token::Symbol(SymbolToken::RightBrace)
        );
        assert_eq!(
            one_token(symbol_lparant),
            Token::Symbol(SymbolToken::LeftParenthesis)
        );
        assert_eq!(
            one_token(symbol_rparant),
            Token::Symbol(SymbolToken::RightParenthesis)
        );
        assert_eq!(
            one_token(symbol_lbracket),
            Token::Symbol(SymbolToken::LeftBracket)
        );
        assert_eq!(
            one_token(symbol_rbracket),
            Token::Symbol(SymbolToken::RightBracket)
        );
        assert_eq!(
            one_token(symbol_langle),
            Token::Symbol(SymbolToken::LeftAngle)
        );
        assert_eq!(
            one_token(symbol_rangle),
            Token::Symbol(SymbolToken::RightAngle)
        );
        assert_eq!(
            one_token(symbol_rightarrow),
            Token::Symbol(SymbolToken::RightArrow)
        );
        assert_eq!(one_token(symbol_assign), Token::Symbol(SymbolToken::Assign));
        assert_eq!(one_token(symbol_colon), Token::Symbol(SymbolToken::Colon));
        assert_eq!(one_token(symbol_comma), Token::Symbol(SymbolToken::Comma));
        assert_eq!(
            one_token(symbol_semicolon),
            Token::Symbol(SymbolToken::Semicolon)
        );
        assert_eq!(one_token(symbol_period), Token::Symbol(SymbolToken::Period));
        assert_eq!(one_token(identifier), Token::Identifier("foo".to_owned()));
    }

    #[test]
    fn test_difficult_identifiers_and_keywords() {
        let identifier_1 = "_fooBar123_baz";
        let identifier_2 = "fnFoo";
        let identifier_3 = "ifif";
        let identifier_4 = "_struct";
        let identifier_5 = "while1";
        let identifier_6 = "X";

        assert_eq!(
            one_token(identifier_1),
            Token::Identifier(identifier_1.to_owned())
        );
        assert_eq!(
            one_token(identifier_2),
            Token::Identifier(identifier_2.to_owned())
        );
        assert_eq!(
            one_token(identifier_3),
            Token::Identifier(identifier_3.to_owned())
        );
        assert_eq!(
            one_token(identifier_4),
            Token::Identifier(identifier_4.to_owned())
        );
        assert_eq!(
            one_token(identifier_5),
            Token::Identifier(identifier_5.to_owned())
        );
        assert_eq!(
            one_token(identifier_6),
            Token::Identifier(identifier_6.to_owned())
        );
    }

    #[test]
    fn test_complete_code() {
        let code = "
struct X {
    a: int,
    b: float,
    c: string
}

fn max(a: int, b: int) -> int {
    if less(a, b) {
        return b;
    } else {
        return a;
    }
}

fn main()
    with x: int
{
    x = max(1, 2);
    print(x);
}
";
        let tokens: Vec<_> = tokenize(code)
            .unwrap()
            .into_iter()
            .map(|pt| pt.token)
            .collect();

        let expected = vec![
            tokens::Token::Keyword(tokens::KeywordToken::Struct),
            tokens::Token::Identifier("X".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("a".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Int),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Identifier("b".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Float),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Identifier("c".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::String),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Fn),
            tokens::Token::Identifier("max".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Identifier("a".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Int),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Identifier("b".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Int),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightArrow),
            tokens::Token::Keyword(tokens::KeywordToken::Int),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Keyword(tokens::KeywordToken::If),
            tokens::Token::Identifier("less".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Identifier("a".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Identifier("b".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Return),
            tokens::Token::Identifier("b".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Else),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Return),
            tokens::Token::Identifier("a".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Fn),
            tokens::Token::Identifier("main".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Keyword(tokens::KeywordToken::With),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Int),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("max".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Literal(tokens::LiteralToken::Integer(1)),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Literal(tokens::LiteralToken::Integer(2)),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Identifier("print".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        ];

        assert_eq!(tokens, expected);
    }
}
