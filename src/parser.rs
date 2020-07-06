pub mod parsetree {
    use crate::tokenize::tokens;

    #[derive(Debug)]
    pub struct ParseError {
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

    fn parse_join_exactly<T: Node<T>>(tokens: &[tokens::PositionalToken], separator: tokens::Token) -> Result<Vec<T>, ParseError> {
        let mut result = Vec::new();
        let mut index = 0;

        {
            let (node, size) = T::parse_from(tokens)?;
            result.push(node);
            index += size;
        }

        while index < tokens.len() {
            if *get_token(tokens, index)? != separator {
                return Err(ParseError::from_token(&tokens[index]));
            }
            index += 1;

            let (node, size) = T::parse_from(tokens)?;
            result.push(node);
            index += size;
        }
        Ok(result)
    }

    pub trait Node<T> {
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


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub struct Program {
        pub structs: Vec<Struct>,
        pub functions: Vec<Function>,
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


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub struct Variable {
        pub name: String,
        pub typ: Type,
        pub position: (usize, usize),
    }

    impl Node<Variable> for Variable {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(Variable, usize), ParseError> {
            assert!(!tokens.is_empty());

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


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub enum Type {
        Bool,
        Integer,
        Float,
        String,
    }

    impl Type {
        fn parse_from(token: &tokens::PositionalToken) -> Result<Type, ParseError> {
            match token.token {
                tokens::Token::Keyword(tokens::KeywordToken::Bool) => Ok(Type::Bool),
                tokens::Token::Keyword(tokens::KeywordToken::Int) => Ok(Type::Integer),
                tokens::Token::Keyword(tokens::KeywordToken::Float) => Ok(Type::Float),
                tokens::Token::Keyword(tokens::KeywordToken::String) => Ok(Type::String),
                _ => Err(ParseError::from_token(token))
            }
        }
    }


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub struct Struct {
        pub name: String,
        pub fields: Vec<Variable>,
        pub position: (usize, usize),
    }

    impl Node<Struct> for Struct {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(Struct, usize), ParseError> {
            assert!(!tokens.is_empty());

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


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub struct Function {
        pub name: String,
        pub parameters: Vec<Variable>,
        pub return_type: Option<Type>,
        pub local_variables: Vec<Variable>,
        pub body: Vec<Statement>,
        pub position: (usize, usize),
    }

    impl Node<Function> for Function {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(Function, usize), ParseError> {
            assert!(!tokens.is_empty());
            let mut index;

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
            let end_index = tokens
                .iter()
                .position(|token| token.token == tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis))
                .ok_or(ParseError { line: tokens.last().unwrap().line, column: tokens.last().unwrap().column })?;
            let parameters = parse_join_exactly::<Variable>(
                &tokens[..end_index],
                tokens::Token::Symbol(tokens::SymbolToken::Comma),
            )?;
            Ok((parameters, end_index))
        }

        fn parse_with(tokens: &[tokens::PositionalToken]) -> Result<(Vec<Variable>, usize), ParseError> {
            let end_index = tokens
                .iter()
                .position(|token| token.token == tokens::Token::Symbol(tokens::SymbolToken::LeftBrace))
                .ok_or(ParseError { line: tokens.last().unwrap().line, column: tokens.last().unwrap().column })?;
            let variables = parse_join_exactly::<Variable>(
                &tokens[..end_index],
                tokens::Token::Symbol(tokens::SymbolToken::Comma),
            )?;
            Ok((variables, end_index))
        }
    }


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub enum Statement {
        IfElse(IfElseStatement),
        While(WhileStatement),
        Assign(AssignStatement),
        Expression(Expression),
    }

    impl Node<Statement> for Statement {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(Statement, usize), ParseError> {
            assert!(!tokens.is_empty());

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
                        }
                        _ => {
                            let (statement, size) = Expression::parse_from(tokens)?;
                            Ok((Statement::Expression(statement), size))
                        }
                    }
                }
                _ => {
                    let (statement, size) = Expression::parse_from(tokens)?;
                    Ok((Statement::Expression(statement), size))
                }
            }
        }
    }


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub struct IfElseStatement {
        pub condition: Expression,
        pub if_block: Vec<Statement>,
        pub else_block: Vec<Statement>,
        pub position: (usize, usize),
    }

    impl Node<IfElseStatement> for IfElseStatement {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(IfElseStatement, usize), ParseError> {
            assert!(!tokens.is_empty());
            let mut index;

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


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub struct WhileStatement {
        pub condition: Expression,
        pub body: Vec<Statement>,
        pub position: (usize, usize),
    }

    impl Node<WhileStatement> for WhileStatement {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(WhileStatement, usize), ParseError> {
            assert!(!tokens.is_empty());
            let mut index;

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


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub struct AssignStatement {
        pub target: String,
        pub value: Expression,
        pub position: (usize, usize),
    }

    impl Node<AssignStatement> for AssignStatement {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(AssignStatement, usize), ParseError> {
            assert!(!tokens.is_empty());

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


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub struct Expression {
        pub value: ExpressionValue,
        pub position: (usize, usize),
    }


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub enum ExpressionValue {
        Literal(LiteralExpression),
        Atom(AtomExpression),
        Call(CallExpression),
    }


    impl Node<Expression> for Expression {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(Expression, usize), ParseError> {
            assert!(!tokens.is_empty());

            // Literal
            if let tokens::Token::Literal(_) = &get_token(tokens, 0)? {
                let (expr, size) = LiteralExpression::parse_from(tokens)?;
                return Ok(
                    (
                        Expression { value: ExpressionValue::Literal(expr), position: (tokens[0].line, tokens[0].column) },
                        size
                    )
                );
            }

            if let tokens::Token::Identifier(_) = &get_token(tokens, 0)? {
                if let Ok(tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis)) = &get_token(tokens, 1) {
                    let (expr, size) = CallExpression::parse_from(tokens)?;
                    return Ok(
                        (
                            Expression { value: ExpressionValue::Call(expr), position: (tokens[0].line, tokens[0].column) },
                            size
                        )
                    );
                } else {
                    let (expr, size) = AtomExpression::parse_from(tokens)?;
                    return Ok(
                        (
                            Expression { value: ExpressionValue::Atom(expr), position: (tokens[0].line, tokens[0].column) },
                            size
                        )
                    );
                }
            } else {
                Err(ParseError::from_token(&tokens[0]))
            }
        }
    }


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub enum LiteralExpression {
        Bool(bool),
        Int(i32),
        Float(f64),
        String(String),
    }

    impl Node<LiteralExpression> for LiteralExpression {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(LiteralExpression, usize), ParseError> {
            assert!(!tokens.is_empty());
            if let tokens::Token::Literal(lit) = get_token(tokens, 0)? {
                match lit {
                    tokens::LiteralToken::Bool(b) => Ok((LiteralExpression::Bool(*b), 1)),
                    tokens::LiteralToken::Integer(i) => Ok((LiteralExpression::Int(*i), 1)),
                    tokens::LiteralToken::Float(f) => Ok((LiteralExpression::Float(*f), 1)),
                    tokens::LiteralToken::String(s) => Ok((LiteralExpression::String(s.clone()), 1)),
                }
            } else {
                Err(ParseError::from_token(&tokens[0]))
            }
        }
    }

    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub struct AtomExpression {
        pub var_name: String,
        pub fields: Vec<String>,
    }

    impl Node<AtomExpression> for AtomExpression {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(AtomExpression, usize), ParseError> {
            assert!(!tokens.is_empty());
            let mut index = 0;

            // var
            let var_name = {
                if let tokens::Token::Identifier(name) = get_token(tokens, 0)? {
                    index += 1;
                    name.clone()
                } else {
                    return Err(ParseError::from_token(&tokens[0]));
                }
            };

            // (.field)*
            let fields = {
                let mut fields = Vec::new();
                while let Ok(tokens::Token::Symbol(tokens::SymbolToken::Period)) = get_token(tokens, index) {
                    index += 1;
                    if let tokens::Token::Identifier(name) = get_token(tokens, index)? {
                        index += 1;
                        fields.push(name.clone());
                    } else {
                        return Err(ParseError::from_token(&tokens[index]));
                    }
                }
                fields
            };

            return Ok((AtomExpression { var_name, fields }, index));
        }
    }

    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    pub struct CallExpression {
        pub func_name: String,
        pub args: Vec<Expression>,
        pub fields: Vec<String>,
    }

    impl Node<CallExpression> for CallExpression {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(CallExpression, usize), ParseError> {
            assert!(!tokens.is_empty());
            let mut index = 0;

            // fn
            let func_name = {
                if let tokens::Token::Identifier(name) = get_token(tokens, 0)? {
                    index += 1;
                    name.clone()
                } else {
                    return Err(ParseError::from_token(&tokens[0]));
                }
            };

            // (arg? (, args)*)
            if *get_token(tokens, index)? != tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis) {
                return Err(ParseError::from_token(&tokens[index]));
            }
            index += 1;
            let args = {
                let mut args = Vec::new();
                while index < tokens.len() {
                    if let Ok((arg, size)) = Expression::parse_from(&tokens[index..]) {
                        args.push(arg);
                        index += size;
                    } else {
                        break;
                    }
                    if *get_token(tokens, index)? != tokens::Token::Symbol(tokens::SymbolToken::Comma) {
                        break;
                    }
                    index += 1;
                }
                args
            };
            if *get_token(tokens, index)? != tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis) {
                return Err(ParseError::from_token(&tokens[index]));
            }
            index += 1;

            // (.field)*
            let fields = {
                let mut fields = Vec::new();
                while let Ok(tokens::Token::Symbol(tokens::SymbolToken::Period)) = get_token(tokens, index) {
                    index += 1;
                    if let tokens::Token::Identifier(name) = get_token(tokens, index)? {
                        index += 1;
                        fields.push(name.clone());
                    } else {
                        return Err(ParseError::from_token(&tokens[index]));
                    }
                }
                fields
            };

            Ok((CallExpression { func_name, args, fields }, index))
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::tokenize::tokens;
    use super::parsetree::*;

    fn give_tokens_positions(tokens: Vec<tokens::Token>) -> Vec<tokens::PositionalToken> {
        tokens
            .into_iter()
            .enumerate()
            .map(|(column, token)| tokens::PositionalToken { token, line: 0, column })
            .collect()
    }

    #[test]
    fn test_atom_expr() {
        let tokens1 = vec!(
            tokens::Token::Identifier("foo".to_owned())
        );
        let tokens2 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("bar".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("baz".to_owned())
        );
        let tokens3 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Identifier("bar".to_owned())
        );
        let tokens4 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Identifier("baz".to_owned())
        );

        let expr1 = AtomExpression {
            var_name: "foo".to_owned(),
            fields: Vec::new(),
        };
        let expr2 = AtomExpression {
            var_name: "foo".to_owned(),
            fields: vec!("bar".to_owned(), "baz".to_owned()),
        };
        let expr3 = AtomExpression {
            var_name: "foo".to_owned(),
            fields: Vec::new(),
        };

        let result1 = AtomExpression::parse_from(&give_tokens_positions(tokens1));
        let result2 = AtomExpression::parse_from(&give_tokens_positions(tokens2));
        let result3 = AtomExpression::parse_from(&give_tokens_positions(tokens3));
        let result4 = AtomExpression::parse_from(&give_tokens_positions(tokens4));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());
        assert!(result4.is_err());

        assert_eq!(result1.unwrap(), (expr1, 1));
        assert_eq!(result2.unwrap(), (expr2, 5));
        assert_eq!(result3.unwrap(), (expr3, 1));
    }

    #[test]
    fn test_call_expr() {
        let tokens1 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
        );
        let tokens2 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Identifier("bar".to_owned()),
        );
        let tokens3 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
        );
        let tokens4 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
        );

        let expr1 = CallExpression {
            func_name: "foo".to_owned(),
            args: Vec::new(),
            fields: Vec::new(),
        };
        let expr2 = CallExpression {
            func_name: "foo".to_owned(),
            args: Vec::new(),
            fields: Vec::new(),
        };

        let result1 = CallExpression::parse_from(&give_tokens_positions(tokens1));
        let result2 = CallExpression::parse_from(&give_tokens_positions(tokens2));
        let result3 = CallExpression::parse_from(&give_tokens_positions(tokens3));
        let result4 = CallExpression::parse_from(&give_tokens_positions(tokens4));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_err());
        assert!(result4.is_err());

        assert_eq!(result1.unwrap(), (expr1, 3));
        assert_eq!(result2.unwrap(), (expr2, 3));
    }

    #[test]
    fn test_literal_expr() {
        let tokens1 = vec!(
            tokens::Token::Literal(tokens::LiteralToken::Bool(true))
        );
        let tokens2 = vec!(
            tokens::Token::Literal(tokens::LiteralToken::Integer(12))
        );
        let tokens3 = vec!(
            tokens::Token::Literal(tokens::LiteralToken::Float(128.123))
        );
        let tokens4 = vec!(
            tokens::Token::Literal(tokens::LiteralToken::String("foo".to_owned()))
        );
        let tokens5 = vec!(
            tokens::Token::Literal(tokens::LiteralToken::Bool(true)),
            tokens::Token::Literal(tokens::LiteralToken::Integer(12))
        );

        let expr1 = LiteralExpression::Bool(true);
        let expr2 = LiteralExpression::Int(12);
        let expr3 = LiteralExpression::Float(128.123);
        let expr4 = LiteralExpression::String("foo".to_owned());
        let expr5 = LiteralExpression::Bool(true);

        let result1 = LiteralExpression::parse_from(&give_tokens_positions(tokens1));
        let result2 = LiteralExpression::parse_from(&give_tokens_positions(tokens2));
        let result3 = LiteralExpression::parse_from(&give_tokens_positions(tokens3));
        let result4 = LiteralExpression::parse_from(&give_tokens_positions(tokens4));
        let result5 = LiteralExpression::parse_from(&give_tokens_positions(tokens5));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());
        assert!(result4.is_ok());
        assert!(result5.is_ok());

        assert_eq!(result1.unwrap(), (expr1, 1));
        assert_eq!(result2.unwrap(), (expr2, 1));
        assert_eq!(result3.unwrap(), (expr3, 1));
        assert_eq!(result4.unwrap(), (expr4, 1));
        assert_eq!(result5.unwrap(), (expr5, 1));
    }

    #[test]
    fn test_expression() {
        let tokens1 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("bar".to_owned())
        );
        let tokens2 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Identifier("bar".to_owned()),
        );
        let tokens3 = vec!(
            tokens::Token::Literal(tokens::LiteralToken::Float(128.123))
        );
        let tokens4 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Identifier("a".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Identifier("b".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Literal(tokens::LiteralToken::Integer(2)),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Identifier("c".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("d".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("e".to_owned()),
        );

        let expr1 = Expression {
            value: ExpressionValue::Atom(
                AtomExpression {
                    var_name: "foo".to_owned(),
                    fields: vec!("bar".to_owned()),
                }
            ),
            position: (0, 0),
        };
        let expr2 = Expression {
            value: ExpressionValue::Call(
                CallExpression {
                    func_name: "foo".to_owned(),
                    args: Vec::new(),
                    fields: Vec::new(),
                }
            ),
            position: (0, 0),
        };
        let expr3 = Expression {
            value: ExpressionValue::Literal(LiteralExpression::Float(128.123)),
            position: (0, 0),
        };
        let expr4 = Expression {
            value: ExpressionValue::Call(
                CallExpression {
                    func_name: "foo".to_owned(),
                    args: vec!(
                        Expression {
                            value: ExpressionValue::Atom(
                                AtomExpression {
                                    var_name: "a".to_owned(),
                                    fields: Vec::new(),
                                }
                            ),
                            position: (0, 2)
                        },
                        Expression {
                            value: ExpressionValue::Call(
                                CallExpression {
                                    func_name: "b".to_owned(),
                                    args: Vec::new(),
                                    fields: Vec::new(),
                                }
                            ),
                            position: (0, 4)
                        },
                        Expression {
                            value: ExpressionValue::Literal(
                                LiteralExpression::Int(2)
                            ),
                            position: (0, 8)
                        },
                        Expression {
                            value: ExpressionValue::Atom(
                                AtomExpression {
                                    var_name: "c".to_owned(),
                                    fields: vec!("d".to_owned()),
                                }
                            ),
                            position: (0, 10)
                        },
                    ),
                    fields: vec!("e".to_owned())
                }
            ),
            position: (0, 0),
        };
        let result1 = Expression::parse_from(&give_tokens_positions(tokens1));
        let result2 = Expression::parse_from(&give_tokens_positions(tokens2));
        let result3 = Expression::parse_from(&give_tokens_positions(tokens3));
        let result4 = Expression::parse_from(&give_tokens_positions(tokens4));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());
        assert!(result4.is_ok());

        assert_eq!(result1.unwrap(), (expr1, 3));
        assert_eq!(result2.unwrap(), (expr2, 3));
        assert_eq!(result3.unwrap(), (expr3, 1));
        assert_eq!(result4.unwrap(), (expr4, 16));
    }
}