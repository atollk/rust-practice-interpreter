pub mod parsetree {
    use crate::tokenize::tokens;

    struct ParseError {
        line: usize,
        column: usize,
    }

    impl ParseError {
        fn from_token(token: &tokens::PositionalToken) -> ParseError {
            ParseError { line: token.line, column: token.column }
        }
    }

    fn get_postoken(tokens: &[tokens::PositionalToken], index: usize) -> Result<&tokens::PositionalToken, ParseError> {
        tokens.get(index).ok_or(ParseError::from_token(tokens.last().unwrap()))
    }

    fn get_token(tokens: &[tokens::PositionalToken], index: usize) -> Result<&tokens::Token, ParseError> {
        Ok(&get_postoken(tokens, index)?.token)
    }

    fn find_matching_brace(tokens: &[tokens::PositionalToken]) -> Option<usize> {
        // TODO
        return None;
    }

    fn parse_statement_block(tokens: &[tokens::PositionalToken]) -> Result<(Vec<Statement>, usize), ParseError> {
        let size = find_matching_brace(tokens)
            .ok_or(
                ParseError { line: tokens.last().unwrap().line, column: tokens.last().unwrap().column }
            )?;
        Ok((parse_kleene_exactly(&tokens[1..size])?, size + 1))
    }

    fn parse_kleene_exactly<T: Node<T>>(tokens: &[tokens::PositionalToken]) -> Result<Vec<T>, ParseError> {
        let mut result = Vec::new();
        let mut index = 0;
        while index < tokens.len() {
            let (node, size) = T::parse_from(tokens)?;
            result.push(node);
            index += size;
        }
        Ok(result)
    }

    trait Node<T> {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(T, usize), ParseError>;

        fn parse_from_exactly(tokens: &[tokens::PositionalToken]) -> Result<T, ParseError> {
            let (node, size) = Self::parse_from(tokens)?;
            if size == tokens.len() {
                Ok(node)
            } else {
                let token = tokens.last().unwrap();
                Err(ParseError { line: token.line, column: token.column })
            }
        }
    }

    struct Program {
        structs: Vec<Struct>,
        functions: Vec<Function>,
    }

    impl Node<Program> for Program {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(Program, usize), ParseError> {
            let mut program = Program { structs: Vec::new(), functions: Vec::new() };
            let mut token_index = 0;
            while token_index < tokens.len() {
                let token = &tokens[token_index];
                if let tokens::Token::Keyword(tokens::KeywordToken::Fn) = token.token {
                    let (f, size) = Function::parse_from(&tokens[token_index..])?;
                    program.functions.push(f);
                    token_index += size;
                } else if let tokens::Token::Keyword(tokens::KeywordToken::Struct) = token.token {
                    let (s, size) = Struct::parse_from(&tokens[token_index..])?;
                    program.structs.push(s);
                    token_index += size;
                } else {
                    return Err(ParseError::from_token(&token));
                }
            }
            return Ok((program, token_index));
        }
    }

    struct Variable {
        name: String,
        typ: Type,
        position: (usize, usize),
    }

    impl Node<Variable> for Variable {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(Variable, usize), ParseError> {
            let mut var = Variable {
                name: String::new(),
                typ: Type::Bool,
                position: (tokens[0].line, tokens[0].column),
            };

            if let tokens::Token::Identifier(name) = get_token(tokens, 0)? {
                var.name = name.clone();
            } else {
                return Err(ParseError::from_token(&tokens[0]));
            }
            if *get_token(tokens, 1)? != tokens::Token::Symbol(tokens::SymbolToken::Colon) {
                return Err(ParseError::from_token(&tokens[1]));
            }
            match get_token(tokens, 2)? {
                tokens::Token::Keyword(tokens::KeywordToken::Bool) => var.typ = Type::Bool,
                tokens::Token::Keyword(tokens::KeywordToken::Int) => var.typ = Type::Integer,
                tokens::Token::Keyword(tokens::KeywordToken::Float) => var.typ = Type::Float,
                tokens::Token::Keyword(tokens::KeywordToken::String) => var.typ = Type::String,
                _ => return Err(ParseError::from_token(&tokens[2]))
            };

            Ok((var, 3))
        }
    }


    enum Type {
        Bool,
        Integer,
        Float,
        String,
    }

    impl Type {
        fn parse_from(token: &tokens::PositionalToken) -> Result<Type, ParseError> {
            Ok(Type::Bool)
        }
    }

    struct Struct {
        name: String,
        fields: Vec<Variable>,
        position: (usize, usize),
    }

    impl Node<Struct> for Struct {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(Struct, usize), ParseError> {
            // struct name {
            let name = {
                if *get_token(tokens, 0)? != tokens::Token::Keyword(tokens::KeywordToken::Struct) {
                    return Err(ParseError::from_token(&tokens[0]));
                }
                if *get_token(tokens, 2)? != tokens::Token::Symbol(tokens::SymbolToken::LeftBrace) {
                    return Err(ParseError::from_token(&tokens[2]));
                }
                if let tokens::Token::Identifier(name) = get_token(tokens, 1)? {
                    name.clone()
                } else {
                    return Err(ParseError::from_token(&tokens[1]));
                }
            };

            // struct_field* }
            let mut index = 3;
            let fields = {
                let mut fields = Vec::new();
                while *get_token(tokens, index)? != tokens::Token::Symbol(tokens::SymbolToken::RightBrace) {
                    let (field_var, size) = Variable::parse_from(&tokens[index..])?;
                    fields.push(field_var);
                    index += size;
                }
                fields
            };

            Ok((
                Struct { name, fields, position: (tokens[0].line, tokens[0].column) },
                index + 1
            ))
        }
    }

    struct Function {
        name: String,
        parameters: Vec<Variable>,
        return_type: Option<Type>,
        local_variables: Vec<Variable>,
        body: Vec<Statement>,
        position: (usize, usize),
    }

    impl Node<Function> for Function {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(Function, usize), ParseError> {
            let mut index = 0;

            // fn name(
            let name = {
                if *get_token(tokens, 0)? != tokens::Token::Keyword(tokens::KeywordToken::Fn) {
                    return Err(ParseError::from_token(&tokens[0]));
                }
                if *get_token(tokens, 2)? != tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis) {
                    return Err(ParseError::from_token(&tokens[2]));
                }
                if let tokens::Token::Identifier(name) = get_token(tokens, 1)? {
                    index = 2;
                    name.clone()
                } else {
                    return Err(ParseError::from_token(&tokens[1]));
                }
            };

            // args*)
            let parameters = {
                let (parameters, parameters_size) =
                    Function::parse_parameters(&tokens[index..])?;
                index += parameters_size - 1;
                if *get_token(tokens, index)? != tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis) {
                    return Err(ParseError::from_token(&tokens[index]));
                }
                index += 1;
                parameters
            };

            // -> returntype
            let return_type = {
                if *get_token(tokens, index)? == tokens::Token::Symbol(tokens::SymbolToken::RightArrow) {
                    let ret = Type::parse_from(get_postoken(tokens, index + 1)?)?;
                    index += 2;
                    Some(ret)
                } else {
                    None
                }
            };

            // with localvar*
            let local_variables = {
                if *get_token(tokens, index)? == tokens::Token::Keyword(tokens::KeywordToken::With) {
                    index += 1;
                    let (local_vars, size) = Function::parse_with(&tokens[index..])?;
                    index += size;
                    local_vars
                } else {
                    Vec::new()
                }
            };

            // { body }
            let body = {
                let (body, size) = parse_statement_block(&tokens[index..])?;
                index += size;
                body
            };

            Ok((
                Function {
                    name,
                    parameters,
                    return_type,
                    local_variables,
                    body,
                    position: (tokens[0].line, tokens[0].column),
                },
                index
            )
            )
        }
    }

    impl Function {
        fn parse_parameters(tokens: &[tokens::PositionalToken]) -> Result<(Vec<Variable>, usize), ParseError> {
            // TODO
            return Ok((Vec::new(), 0));
        }

        fn parse_with(tokens: &[tokens::PositionalToken]) -> Result<(Vec<Variable>, usize), ParseError> {
            // TODO
            return Ok((Vec::new(), 0));
        }
    }

    enum Statement {
        IfElse(IfElseStatement),
        While(WhileStatement),
        Assign(AssignStatement),
        Expression(Expression),
    }

    impl Node<Statement> for Statement {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(Statement, usize), ParseError> {
            match get_token(tokens, 0)? {
                tokens::Token::Keyword(tokens::KeywordToken::If) => {
                    let (statement, size) = IfElseStatement::parse_from(tokens)?;
                    Ok((Statement::IfElse(statement), size))
                }
                tokens::Token::Keyword(tokens::KeywordToken::While) => {
                    let (statement, size) = WhileStatement::parse_from(tokens)?;
                    Ok((Statement::While(statement), size))
                }
                tokens::Token::Identifier(_) => {
                    match *get_token(tokens, 1)? {
                        tokens::Token::Symbol(tokens::SymbolToken::Assign) => {
                            let (statement, size) = AssignStatement::parse_from(tokens)?;
                            Ok((Statement::Assign(statement), size))
                        },
                        _ => {
                            let (statement, size) = Expression::parse_from(tokens)?;
                            Ok((Statement::Expression(statement), size))
                        },
                    }
                }
                _ => {
                    let (statement, size) = Expression::parse_from(tokens)?;
                    Ok((Statement::Expression(statement), size))
                }
            }
        }
    }

    struct IfElseStatement {
        condition: Expression,
        if_block: Vec<Statement>,
        else_block: Vec<Statement>,
        position: (usize, usize),
    }

    impl Node<IfElseStatement> for IfElseStatement {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(IfElseStatement, usize), ParseError> {
            let mut index = 0;

            // if cond
            if *get_token(tokens, 0)? != tokens::Token::Keyword(tokens::KeywordToken::If) {
                return Err(ParseError::from_token(&tokens[0]));
            }
            let condition = {
                let (condition, size) = Expression::parse_from(&tokens[1..])?;
                index = 1 + size;
                condition
            };

            // { .. }
            let if_block = {
                let (if_block, size) = parse_statement_block(&tokens[index..])?;
                index += size;
                if_block
            };

            // else { .. }
            let else_block = {
                if *get_token(tokens, index)? == tokens::Token::Keyword(tokens::KeywordToken::Else) {
                    index += 1;
                    let (else_block, size) = parse_statement_block(&tokens[index..])?;
                    index += size;
                    else_block
                } else {
                    Vec::new()
                }
            };

            Ok(
                (
                    IfElseStatement {
                        condition,
                        if_block,
                        else_block,
                        position: (tokens[0].line, tokens[0].column),
                    },
                    index
                )
            )
        }
    }

    struct WhileStatement {
        condition: Expression,
        body: Vec<Statement>,
        position: (usize, usize),
    }

    impl Node<WhileStatement> for WhileStatement {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(WhileStatement, usize), ParseError> {
            let mut index = 0;

            // while cond
            if *get_token(tokens, 0)? != tokens::Token::Keyword(tokens::KeywordToken::While) {
                return Err(ParseError::from_token(&tokens[0]));
            }
            let condition = {
                let (condition, size) = Expression::parse_from(&tokens[1..])?;
                index = 1 + size;
                condition
            };

            // { .. }
            let body = {
                let (body, size) = parse_statement_block(&tokens[index..])?;
                index += size;
                body
            };

            Ok(
                (
                    WhileStatement {
                        condition,
                        body,
                        position: (tokens[0].line, tokens[0].column),
                    },
                    index
                )
            )
        }
    }

    struct AssignStatement {
        target: String,
        value: Expression,
        position: (usize, usize),
    }

    impl Node<AssignStatement> for AssignStatement {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(AssignStatement, usize), ParseError> {
            // target =
            let target = if let tokens::Token::Identifier(target) = get_token(tokens, 0)? {
                target.clone()
            } else {
                return Err(ParseError::from_token(&tokens[0]));
            };
            if *get_token(tokens, 1)? != tokens::Token::Symbol(tokens::SymbolToken::Assign) {
                return Err(ParseError::from_token(&tokens[1]));
            }

            // expression ;
            let (value, value_size) = Expression::parse_from(&tokens[2..])?;
            if *get_token(tokens, 2 + value_size)? != tokens::Token::Symbol(tokens::SymbolToken::Semicolon) {
                return Err(ParseError::from_token(&tokens[2 + value_size]));
            }

            Ok(
                (
                    AssignStatement {
                        target,
                        value,
                        position: (tokens[0].line, tokens[0].column),
                    },
                    value_size + 3
                )
            )
        }
    }

    struct Expression {
        src: String
    }


    impl Node<Expression> for Expression {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(Expression, usize), ParseError> {
            // TODO
            return Ok((Expression {src: "a".to_owned()}, 1));
        }
    }
}

