pub fn parse_program(tokens: &[crate::tokenize::tokens::PositionalToken]) -> Result<parsetree::Program, parsetree::ParseError> {
    use parsetree::Node;
    let (prog, size) = parsetree::Program::parse_from(tokens)?;
    if size != tokens.len() {
        Err(parsetree::ParseError { line: tokens[size].line, column: tokens[size].column })
    } else {
        Ok(prog)
    }
}


pub mod parsetree {
    use crate::tokenize::tokens;

    #[derive(Debug)]
    pub struct ParseError {
        pub line: usize,
        pub column: usize,
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

    fn parse_statement_block(tokens: &[tokens::PositionalToken]) -> Result<(Vec<Statement>, usize), ParseError> {
        assert!(!tokens.is_empty());
        if *get_token(tokens, 0)? != tokens::Token::Symbol(tokens::SymbolToken::LeftBrace) {
            return Err(ParseError::from_token(&tokens[0]));
        }

        let mut index = 1;
        let mut block = Vec::new();

        while index + 1 < tokens.len() {
            let (statement, size) = {
                if let Ok(x) = Statement::parse_from(&tokens[index..]) {
                    x
                } else {
                    break;
                }
            };
            block.push(statement);
            index += size;
        }

        if *get_token(tokens, index)? != tokens::Token::Symbol(tokens::SymbolToken::RightBrace) {
            return Err(ParseError::from_token(&tokens[0]));
        }

        Ok((block, index + 1))
    }

    fn parse_join_exactly<T: Node<T>>(tokens: &[tokens::PositionalToken], separator: tokens::Token) -> Result<Vec<T>, ParseError> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }

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

            let (node, size) = T::parse_from(&tokens[index..])?;
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
    #[derive(Clone)]
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
    #[derive(Clone)]
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
            var.typ = match get_token(tokens, 2)? {
                tokens::Token::Keyword(tokens::KeywordToken::Bool) => Type::Bool,
                tokens::Token::Keyword(tokens::KeywordToken::Int) => Type::Integer,
                tokens::Token::Keyword(tokens::KeywordToken::Float) => Type::Float,
                tokens::Token::Keyword(tokens::KeywordToken::String) => Type::String,
                tokens::Token::Identifier(x) => Type::Struct(x.clone()),
                _ => return Err(ParseError::from_token(&tokens[2]))
            };

            Ok((var, 3))
        }
    }


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    #[derive(Clone)]
    pub enum Type {
        Bool,
        Integer,
        Float,
        String,
        Struct(String)
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
    #[derive(Clone)]
    pub struct Struct {
        pub name: String,
        pub fields: Vec<Variable>,
        pub position: (usize, usize),
    }

    impl Node<Struct> for Struct {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(Struct, usize), ParseError> {
            assert!(!tokens.is_empty());

            let total_size = tokens
                .iter()
                .position(|t| t.token == tokens::Token::Symbol(tokens::SymbolToken::RightBrace))
                .ok_or(ParseError::from_token(tokens.last().unwrap()))?;

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
            let fields = parse_join_exactly(&tokens[3..total_size], tokens::Token::Symbol(tokens::SymbolToken::Comma))?;
            if *get_token(tokens, total_size)? != tokens::Token::Symbol(tokens::SymbolToken::RightBrace) {
                return Err(ParseError::from_token(&tokens[total_size]));
            }

            Ok((
                Struct { name, fields, position: (tokens[0].line, tokens[0].column) },
                total_size + 1
            ))
        }
    }


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    #[derive(Clone)]
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
                    index = 3;
                    name.clone()
                } else {
                    return Err(ParseError::from_token(&tokens[1]));
                }
            };

            // args*)
            let parameters = {
                let (parameters, parameters_size) =
                    Function::parse_parameters(&tokens[index..])?;
                index += parameters_size;
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
            ))
        }
    }

    impl Function {
        fn parse_parameters(tokens: &[tokens::PositionalToken]) -> Result<(Vec<Variable>, usize), ParseError> {
            let end_index = tokens
                .iter()
                .position(|token| token.token == tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis))
                .ok_or(ParseError { line: tokens.last().unwrap().line, column: tokens.last().unwrap().column })?;
            let parameters = parse_join_exactly(
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
    #[derive(Clone)]
    pub enum Statement {
        IfElse(IfElseStatement),
        While(WhileStatement),
        Return(ReturnStatement),
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
                tokens::Token::Keyword(tokens::KeywordToken::Return) => {
                    let (statement, size) = ReturnStatement::parse_from(tokens)?;
                    Ok((Statement::Return(statement), size))
                }
                tokens::Token::Identifier(_) => {
                    let (expr, size) = Expression::parse_from(tokens)?;
                    if let Ok(tokens::Token::Symbol(tokens::SymbolToken::Semicolon)) = get_token(tokens, size) {
                        Ok((Statement::Expression(expr), size + 1))
                    } else {
                        let (statement, size) = AssignStatement::parse_from(tokens)?;
                        Ok((Statement::Assign(statement), size))
                    }
                }
                _ => {
                    let (statement, size) = Expression::parse_from(tokens)?;
                    if *get_token(tokens, size)? != tokens::Token::Symbol(tokens::SymbolToken::Semicolon) {
                        return Err(ParseError::from_token(&tokens[size]));
                    }
                    Ok((Statement::Expression(statement), size + 1))
                }
            }
        }
    }

    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    #[derive(Clone)]
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
                if let Ok(tokens::Token::Keyword(tokens::KeywordToken::Else)) = get_token(tokens, index) {
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
    #[derive(Clone)]
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
    #[derive(Clone)]
    pub struct ReturnStatement {
        pub expression: Option<Expression>,
        pub position: (usize, usize),
    }

    impl Node<ReturnStatement> for ReturnStatement {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(ReturnStatement, usize), ParseError> {
            assert!(!tokens.is_empty());

            if *get_token(tokens, 0)? != tokens::Token::Keyword(tokens::KeywordToken::Return) {
                return Err(ParseError::from_token(&tokens[0]));
            }

            let (expr, size) = if *get_token(tokens, 1)? == tokens::Token::Symbol(tokens::SymbolToken::Semicolon) {
                (None, 2)
            } else {
                let (expr, size) = Expression::parse_from(&tokens[1..])?;
                if *get_token(tokens, size + 1)? != tokens::Token::Symbol(tokens::SymbolToken::Semicolon) {
                    return Err(ParseError::from_token(&tokens[size + 1]));
                }
                (Some(expr), size + 2)
            };

            Ok((ReturnStatement { expression: expr, position: (tokens[0].line, tokens[0].column) }, size))
        }
    }

    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    #[derive(Clone)]
    pub struct AssignStatement {
        pub target: AtomExpression,
        pub value: Expression,
        pub position: (usize, usize),
    }

    impl Node<AssignStatement> for AssignStatement {
        fn parse_from(tokens: &[tokens::PositionalToken]) -> Result<(AssignStatement, usize), ParseError> {
            assert!(!tokens.is_empty());
            let mut index = 0;

            // target(.field)* =
            let target = {
                let (target, size) = AtomExpression::parse_from(tokens)?;
                index += size;
                target
            };
            if *get_token(tokens, index)? != tokens::Token::Symbol(tokens::SymbolToken::Assign) {
                return Err(ParseError::from_token(&tokens[1]));
            }
            index += 1;


            // expression ;
            if index >= tokens.len() {
                return Err(ParseError::from_token(tokens.last().unwrap()));
            }
            let value = {
                let (value, size) = Expression::parse_from(&tokens[index..])?;
                index += size;
                value
            };
            if *get_token(tokens, index)? != tokens::Token::Symbol(tokens::SymbolToken::Semicolon) {
                return Err(ParseError::from_token(&tokens[index]));
            }

            Ok(
                (
                    AssignStatement {
                        target,
                        value,
                        position: (tokens[0].line, tokens[0].column),
                    },
                    index + 1
                )
            )
        }
    }


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    #[derive(Clone)]
    pub struct Expression {
        pub value: ExpressionValue,
        pub position: (usize, usize),
    }


    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    #[derive(Clone)]
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
    #[derive(Clone)]
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
                Err(ParseError::from_token(
                    &tokens[0]))
            }
        }
    }

    #[derive(Debug)]
    #[derive(std::cmp::PartialEq)]
    #[derive(Clone)]
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
    #[derive(Clone)]
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
        // foo
        let tokens1 = vec!(
            tokens::Token::Identifier("foo".to_owned())
        );
        // foo.bar.baz
        let tokens2 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("bar".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("baz".to_owned())
        );
        // foo bar
        let tokens3 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Identifier("bar".to_owned())
        );
        // foo.(baz
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
        // foo()
        let tokens1 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
        );
        // foo() bar
        let tokens2 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Identifier("bar".to_owned()),
        );
        // foo(
        let tokens3 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
        );
        // foo(,)
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
        // true
        let tokens1 = vec!(
            tokens::Token::Literal(tokens::LiteralToken::Bool(true))
        );
        // 12
        let tokens2 = vec!(
            tokens::Token::Literal(tokens::LiteralToken::Integer(12))
        );
        // 128.123
        let tokens3 = vec!(
            tokens::Token::Literal(tokens::LiteralToken::Float(128.123))
        );
        // "foo"
        let tokens4 = vec!(
            tokens::Token::Literal(tokens::LiteralToken::String("foo".to_owned()))
        );
        // true 12
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
        // foo.bar
        let tokens1 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("bar".to_owned())
        );
        // foo() bar
        let tokens2 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Identifier("bar".to_owned()),
        );
        // 128.123
        let tokens3 = vec!(
            tokens::Token::Literal(tokens::LiteralToken::Float(128.123))
        );
        // foo(a, b(), 2, c.d).e
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
                            position: (0, 2),
                        },
                        Expression {
                            value: ExpressionValue::Call(
                                CallExpression {
                                    func_name: "b".to_owned(),
                                    args: Vec::new(),
                                    fields: Vec::new(),
                                }
                            ),
                            position: (0, 4),
                        },
                        Expression {
                            value: ExpressionValue::Literal(
                                LiteralExpression::Int(2)
                            ),
                            position: (0, 8),
                        },
                        Expression {
                            value: ExpressionValue::Atom(
                                AtomExpression {
                                    var_name: "c".to_owned(),
                                    fields: vec!("d".to_owned()),
                                }
                            ),
                            position: (0, 10),
                        },
                    ),
                    fields: vec!("e".to_owned()),
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

    #[test]
    fn test_assign_stmt() {
        // foo = bar;
        let tokens1 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("bar".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
        );
        // foo.bar.baz = a.b; c
        let tokens2 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("bar".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("baz".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("a".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("b".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Identifier("c".to_owned()),
        );
        // foo = bar
        let tokens3 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("bar".to_owned()),
        );
        // foo = bar(;
        let tokens4 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("bar".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
        );
        // foo =
        let tokens5 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
        );

        let stmt1 = AssignStatement {
            target: AtomExpression {
                var_name: "foo".to_owned(),
                fields: Vec::new(),
            },
            value: Expression {
                value: ExpressionValue::Atom(
                    AtomExpression {
                        var_name: "bar".to_owned(),
                        fields: Vec::new(),
                    }
                ),
                position: (0, 2),
            },
            position: (0, 0),
        };
        let stmt2 = AssignStatement {
            target: AtomExpression {
                var_name: "foo".to_owned(),
                fields: vec!("bar".to_owned(), "baz".to_owned()),
            },
            value: Expression {
                value: ExpressionValue::Atom(
                    AtomExpression {
                        var_name: "a".to_owned(),
                        fields: vec!("b".to_owned()),
                    }
                ),
                position: (0, 6),
            },
            position: (0, 0),
        };


        let result1 = AssignStatement::parse_from(&give_tokens_positions(tokens1));
        let result2 = AssignStatement::parse_from(&give_tokens_positions(tokens2));
        let result3 = AssignStatement::parse_from(&give_tokens_positions(tokens3));
        let result4 = AssignStatement::parse_from(&give_tokens_positions(tokens4));
        let result5 = AssignStatement::parse_from(&give_tokens_positions(tokens5));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_err());
        assert!(result4.is_err());
        assert!(result5.is_err());

        assert_eq!(result1.unwrap(), (stmt1, 4));
        assert_eq!(result2.unwrap(), (stmt2, 10));
    }

    #[test]
    fn test_ifelse_stmt() {
        // if true {}
        let tokens1 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::If),
            tokens::Token::Literal(tokens::LiteralToken::Bool(true)),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // if test(1) {} else {} 1
        let tokens2 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::If),
            tokens::Token::Identifier("test".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Literal(tokens::LiteralToken::Integer(1)),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Else),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Literal(tokens::LiteralToken::Integer(1)),
        );
        // if true { "foo"; } else { x = y; }
        let tokens3 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::If),
            tokens::Token::Literal(tokens::LiteralToken::Bool(true)),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Literal(tokens::LiteralToken::String("foo".to_owned())),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Else),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // if {}
        let tokens4 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::If),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // if true { 1; } else { 2 }
        let tokens5 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::If),
            tokens::Token::Literal(tokens::LiteralToken::Bool(true)),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Literal(tokens::LiteralToken::Integer(1)),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Else),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Literal(tokens::LiteralToken::Integer(2)),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );

        let stmt1 = IfElseStatement {
            condition: Expression {
                value: ExpressionValue::Literal(LiteralExpression::Bool(true)),
                position: (0, 1),
            },
            if_block: Vec::new(),
            else_block: Vec::new(),
            position: (0, 0),
        };
        let stmt2 = IfElseStatement {
            condition: Expression {
                value: ExpressionValue::Call(
                    CallExpression {
                        func_name: "test".to_owned(),
                        args: vec!(
                            Expression {
                                value: ExpressionValue::Literal(LiteralExpression::Int(1)),
                                position: (0, 3),
                            }
                        ),
                        fields: Vec::new(),
                    }
                ),
                position: (0, 1),
            },
            if_block: Vec::new(),
            else_block: Vec::new(),
            position: (0, 0),
        };
        let stmt3 = IfElseStatement {
            condition: Expression {
                value: ExpressionValue::Literal(LiteralExpression::Bool(true)),
                position: (0, 1),
            },
            if_block: vec!(
                Statement::Expression(
                    Expression {
                        value: ExpressionValue::Literal(LiteralExpression::String("foo".to_owned())),
                        position: (0, 3),
                    }
                )
            ),
            else_block: vec!(
                Statement::Assign(
                    AssignStatement {
                        target: AtomExpression {
                            var_name: "x".to_owned(),
                            fields: Vec::new(),
                        },
                        value: Expression {
                            value: ExpressionValue::Atom(
                                AtomExpression {
                                    var_name: "y".to_owned(),
                                    fields: Vec::new(),
                                }
                            ),
                            position: (0, 10),
                        },
                        position: (0, 8),
                    }
                )
            ),
            position: (0, 0),
        };

        let result1 = IfElseStatement::parse_from(&give_tokens_positions(tokens1));
        let result2 = IfElseStatement::parse_from(&give_tokens_positions(tokens2));
        let result3 = IfElseStatement::parse_from(&give_tokens_positions(tokens3));
        let result4 = IfElseStatement::parse_from(&give_tokens_positions(tokens4));
        let result5 = IfElseStatement::parse_from(&give_tokens_positions(tokens5));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());
        assert!(result4.is_err());
        assert!(result5.is_err());

        assert_eq!(result1.unwrap(), (stmt1, 4));
        assert_eq!(result2.unwrap(), (stmt2, 10));
        assert_eq!(result3.unwrap(), (stmt3, 13));
    }

    #[test]
    fn test_while_stmt() {
        // while true {}
        let tokens1 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::While),
            tokens::Token::Literal(tokens::LiteralToken::Bool(true)),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // while test(1) { "foo"; x = y; } 1
        let tokens2 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::While),
            tokens::Token::Identifier("test".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Literal(tokens::LiteralToken::Integer(1)),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Literal(tokens::LiteralToken::String("foo".to_owned())),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Literal(tokens::LiteralToken::Integer(1)),
        );
        // while {}
        let tokens3 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::While),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // while true { 1 }
        let tokens4 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::While),
            tokens::Token::Literal(tokens::LiteralToken::Bool(true)),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Literal(tokens::LiteralToken::Integer(1)),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );

        let stmt1 = WhileStatement {
            condition: Expression {
                value: ExpressionValue::Literal(LiteralExpression::Bool(true)),
                position: (0, 1),
            },
            body: Vec::new(),
            position: (0, 0),
        };
        let stmt2 = WhileStatement {
            condition: Expression {
                value: ExpressionValue::Call(
                    CallExpression {
                        func_name: "test".to_owned(),
                        args: vec!(
                            Expression {
                                value: ExpressionValue::Literal(LiteralExpression::Int(1)),
                                position: (0, 3),
                            }
                        ),
                        fields: Vec::new(),
                    }
                ),
                position: (0, 1),
            },
            body: vec!(
                Statement::Expression(
                    Expression {
                        value: ExpressionValue::Literal(LiteralExpression::String("foo".to_owned())),
                        position: (0, 6),
                    }
                ),
                Statement::Assign(
                    AssignStatement {
                        target: AtomExpression {
                            var_name: "x".to_owned(),
                            fields: Vec::new(),
                        },
                        value: Expression {
                            value: ExpressionValue::Atom(
                                AtomExpression {
                                    var_name: "y".to_owned(),
                                    fields: Vec::new(),
                                }
                            ),
                            position: (0, 10),
                        },
                        position: (0, 8),
                    }
                )
            ),
            position: (0, 0),
        };

        let result1 = WhileStatement::parse_from(&give_tokens_positions(tokens1));
        let result2 = WhileStatement::parse_from(&give_tokens_positions(tokens2));
        let result3 = WhileStatement::parse_from(&give_tokens_positions(tokens3));
        let result4 = WhileStatement::parse_from(&give_tokens_positions(tokens4));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_err());
        assert!(result4.is_err());

        assert_eq!(result1.unwrap(), (stmt1, 4));
        assert_eq!(result2.unwrap(), (stmt2, 13));
    }

    #[test]
    fn test_return_stmt() {
        // return foo.bar;
        let tokens1 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Return),
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("bar".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon)
        );
        // return;
        let tokens2 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Return),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon)
        );
        // return x = y;
        let tokens3 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Return),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon)
        );
        // return foo.bar
        let tokens4 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Return),
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Identifier("bar".to_owned()),
        );

        let stmt1 = ReturnStatement {
            expression: Some(
                Expression {
                    value: ExpressionValue::Atom(
                        AtomExpression {
                            var_name: "foo".to_owned(),
                            fields: vec!("bar".to_owned()),
                        }
                    ),
                    position: (0, 1),
                }
            ),
            position: (0, 0),
        };
        let stmt2 = ReturnStatement {
            expression: None,
            position: (0, 0),
        };

        let result1 = ReturnStatement::parse_from(&give_tokens_positions(tokens1));
        let result2 = ReturnStatement::parse_from(&give_tokens_positions(tokens2));
        let result3 = ReturnStatement::parse_from(&give_tokens_positions(tokens3));
        let result4 = ReturnStatement::parse_from(&give_tokens_positions(tokens4));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_err());
        assert!(result4.is_err());

        assert_eq!(result1.unwrap(), (stmt1, 5));
        assert_eq!(result2.unwrap(), (stmt2, 2));
    }

    #[test]
    fn test_statement() {
        // foo.bar;
        let tokens1 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("bar".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
        );
        // foo.bar.baz = a.b; c
        let tokens2 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("bar".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("baz".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("a".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("b".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Identifier("c".to_owned()),
        );
        // if true { "foo"; } else { x = y; }
        let tokens3 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::If),
            tokens::Token::Literal(tokens::LiteralToken::Bool(true)),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Literal(tokens::LiteralToken::String("foo".to_owned())),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Else),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // while test(1) { "foo"; x = y; }
        let tokens4 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::While),
            tokens::Token::Identifier("test".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Literal(tokens::LiteralToken::Integer(1)),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Literal(tokens::LiteralToken::String("foo".to_owned())),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // return 1;
        let tokens5 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Return),
            tokens::Token::Literal(tokens::LiteralToken::Integer(1)),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
        );
        // if foo { if bar {x=y;} else {y=x;} } else { while baz {1; 2; 3;} }
        let tokens6 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::If),
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Keyword(tokens::KeywordToken::If),
            tokens::Token::Identifier("bar".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Else),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Else),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Keyword(tokens::KeywordToken::While),
            tokens::Token::Identifier("baz".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Literal(tokens::LiteralToken::Integer(1)),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Literal(tokens::LiteralToken::Integer(2)),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Literal(tokens::LiteralToken::Integer(3)),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // if foo { bar {x=y;} else {y=x;} } else { while baz {1; 2; 3;} }
        let tokens7 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::If),
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("bar".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Else),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Keyword(tokens::KeywordToken::Else),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Keyword(tokens::KeywordToken::While),
            tokens::Token::Identifier("baz".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Literal(tokens::LiteralToken::Integer(1)),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Literal(tokens::LiteralToken::Integer(2)),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Literal(tokens::LiteralToken::Integer(3)),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // foo.bar
        let tokens8 = vec!(
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Period),
            tokens::Token::Identifier("bar".to_owned()),
        );
        // 1
        let tokens9 = vec!(
            tokens::Token::Literal(tokens::LiteralToken::Integer(1)),
        );

        let stmt1 = Statement::Expression(
            Expression {
                value: ExpressionValue::Atom(
                    AtomExpression {
                        var_name: "foo".to_owned(),
                        fields: vec!("bar".to_owned()),
                    }
                ),
                position: (0, 0),
            }
        );
        let stmt2 = Statement::Assign(
            AssignStatement {
                target: AtomExpression {
                    var_name: "foo".to_owned(),
                    fields: vec!("bar".to_owned(), "baz".to_owned()),
                },
                value: Expression {
                    value: ExpressionValue::Atom(
                        AtomExpression {
                            var_name: "a".to_owned(),
                            fields: vec!("b".to_owned()),
                        }
                    ),
                    position: (0, 6),
                },
                position: (0, 0),
            }
        );
        let stmt3 = Statement::IfElse(
            IfElseStatement {
                condition: Expression {
                    value: ExpressionValue::Literal(LiteralExpression::Bool(true)),
                    position: (0, 1),
                },
                if_block: vec!(
                    Statement::Expression(
                        Expression {
                            value: ExpressionValue::Literal(LiteralExpression::String("foo".to_owned())),
                            position: (0, 3),
                        }
                    )
                ),
                else_block: vec!(
                    Statement::Assign(
                        AssignStatement {
                            target: AtomExpression {
                                var_name: "x".to_owned(),
                                fields: Vec::new(),
                            },
                            value: Expression {
                                value: ExpressionValue::Atom(
                                    AtomExpression {
                                        var_name: "y".to_owned(),
                                        fields: Vec::new(),
                                    }
                                ),
                                position: (0, 10),
                            },
                            position: (0, 8),
                        }
                    )
                ),
                position: (0, 0),
            }
        );
        let stmt4 = Statement::While(
            WhileStatement {
                condition: Expression {
                    value: ExpressionValue::Call(
                        CallExpression {
                            func_name: "test".to_owned(),
                            args: vec!(
                                Expression {
                                    value: ExpressionValue::Literal(LiteralExpression::Int(1)),
                                    position: (0, 3),
                                }
                            ),
                            fields: Vec::new(),
                        }
                    ),
                    position: (0, 1),
                },
                body: vec!(
                    Statement::Expression(
                        Expression {
                            value: ExpressionValue::Literal(LiteralExpression::String("foo".to_owned())),
                            position: (0, 6),
                        }
                    ),
                    Statement::Assign(
                        AssignStatement {
                            target: AtomExpression {
                                var_name: "x".to_owned(),
                                fields: Vec::new(),
                            },
                            value: Expression {
                                value: ExpressionValue::Atom(
                                    AtomExpression {
                                        var_name: "y".to_owned(),
                                        fields: Vec::new(),
                                    }
                                ),
                                position: (0, 10),
                            },
                            position: (0, 8),
                        }
                    )
                ),
                position: (0, 0),
            }
        );
        let stmt5 = Statement::Return(
            ReturnStatement {
                expression: Some(
                    Expression {
                        value: ExpressionValue::Literal(
                            LiteralExpression::Int(1)
                        ),
                        position: (0, 1),
                    }
                ),
                position: (0, 0),
            }
        );


        let stmt6 = Statement::IfElse(
            IfElseStatement {
                condition: Expression {
                    value: ExpressionValue::Atom(
                        AtomExpression {
                            var_name: "foo".to_owned(),
                            fields: Vec::new(),
                        }
                    ),
                    position: (0, 1),
                },
                if_block: vec!(
                    Statement::IfElse(
                        IfElseStatement {
                            condition: Expression {
                                value: ExpressionValue::Atom(
                                    AtomExpression {
                                        var_name: "bar".to_owned(),
                                        fields: Vec::new(),
                                    }
                                ),
                                position: (0, 4),
                            },
                            if_block: vec!(
                                Statement::Assign(
                                    AssignStatement {
                                        target: AtomExpression {
                                            var_name: "x".to_owned(),
                                            fields: Vec::new(),
                                        },
                                        value: Expression {
                                            value: ExpressionValue::Atom(
                                                AtomExpression {
                                                    var_name: "y".to_owned(),
                                                    fields: Vec::new(),
                                                }
                                            ),
                                            position: (0, 8),
                                        },
                                        position: (0, 6),
                                    }
                                )
                            ),
                            else_block: vec!(
                                Statement::Assign(
                                    AssignStatement {
                                        target: AtomExpression {
                                            var_name: "y".to_owned(),
                                            fields: Vec::new(),
                                        },
                                        value: Expression {
                                            value: ExpressionValue::Atom(
                                                AtomExpression {
                                                    var_name: "x".to_owned(),
                                                    fields: Vec::new(),
                                                }
                                            ),
                                            position: (0, 15),
                                        },
                                        position: (0, 13),
                                    }
                                )
                            ),
                            position: (0, 3),
                        }
                    )
                ),
                else_block: vec!(
                    Statement::While(
                        WhileStatement {
                            condition: Expression {
                                value: ExpressionValue::Atom(
                                    AtomExpression {
                                        var_name: "baz".to_owned(),
                                        fields: Vec::new(),
                                    }
                                ),
                                position: (0, 22),
                            },
                            body: vec!(
                                Statement::Expression(
                                    Expression {
                                        value: ExpressionValue::Literal(LiteralExpression::Int(1)),
                                        position: (0, 24),
                                    }
                                ),
                                Statement::Expression(
                                    Expression {
                                        value: ExpressionValue::Literal(LiteralExpression::Int(2)),
                                        position: (0, 26),
                                    }
                                ),
                                Statement::Expression(
                                    Expression {
                                        value: ExpressionValue::Literal(LiteralExpression::Int(3)),
                                        position: (0, 28),
                                    }
                                )
                            ),
                            position: (0, 21),
                        }
                    )
                ),
                position: (0, 0),
            }
        );

        let result1 = Statement::parse_from(&give_tokens_positions(tokens1));
        let result2 = Statement::parse_from(&give_tokens_positions(tokens2));
        let result3 = Statement::parse_from(&give_tokens_positions(tokens3));
        let result4 = Statement::parse_from(&give_tokens_positions(tokens4));
        let result5 = Statement::parse_from(&give_tokens_positions(tokens5));
        let result6 = Statement::parse_from(&give_tokens_positions(tokens6));
        let result7 = Statement::parse_from(&give_tokens_positions(tokens7));
        let result8 = Statement::parse_from(&give_tokens_positions(tokens8));
        let result9 = Statement::parse_from(&give_tokens_positions(tokens9));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());
        assert!(result4.is_ok());
        assert!(result5.is_ok());
        assert!(result6.is_ok());
        assert!(result7.is_err());
        assert!(result8.is_err());
        assert!(result9.is_err());

        assert_eq!(result1.unwrap(), (stmt1, 4));
        assert_eq!(result2.unwrap(), (stmt2, 10));
        assert_eq!(result3.unwrap(), (stmt3, 13));
        assert_eq!(result4.unwrap(), (stmt4, 13));
        assert_eq!(result5.unwrap(), (stmt5, 3));
        assert_eq!(result6.unwrap(), (stmt6, 32));
    }

    #[test]
    fn test_variable() {
        // x: int
        let tokens1 = vec!(
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Int),
        );
        // x: 123
        let tokens2 = vec!(
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Literal(tokens::LiteralToken::Integer(123))
        );

        let var1 = Variable {
            name: "x".to_owned(),
            typ: Type::Integer,
            position: (0, 0),
        };

        let result1 = Variable::parse_from(&give_tokens_positions(tokens1));
        let result2 = Variable::parse_from(&give_tokens_positions(tokens2));

        assert!(result1.is_ok());
        assert!(result2.is_err());

        assert_eq!(result1.unwrap(), (var1, 3));
    }

    #[test]
    fn test_function() {
        // fn foo() {}
        let tokens1 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Fn),
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // fn foo(x: int, y: float) -> string {x = y; return "";}
        let tokens2 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Fn),
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Int),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Float),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightArrow),
            tokens::Token::Keyword(tokens::KeywordToken::String),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Keyword(tokens::KeywordToken::Return),
            tokens::Token::Literal(tokens::LiteralToken::String("".to_owned())),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // fn foo() with a: int {}
        let tokens3 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Fn),
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Keyword(tokens::KeywordToken::With),
            tokens::Token::Identifier("a".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Int),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // fn foo(x: int, y: float) -> {return "";}
        let tokens4 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Fn),
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Int),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Float),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::RightArrow),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Assign),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Keyword(tokens::KeywordToken::Return),
            tokens::Token::Literal(tokens::LiteralToken::String("".to_owned())),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // fn foo(x: int, y: float);
        let tokens5 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Fn),
            tokens::Token::Identifier("foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftParenthesis),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Int),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Float),
            tokens::Token::Symbol(tokens::SymbolToken::RightParenthesis),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
        );

        let fn1 = Function {
            name: "foo".to_owned(),
            parameters: Vec::new(),
            return_type: None,
            local_variables: Vec::new(),
            body: Vec::new(),
            position: (0, 0),
        };
        let fn2 = Function {
            name: "foo".to_owned(),
            parameters: vec!(
                Variable {
                    name: "x".to_owned(),
                    typ: Type::Integer,
                    position: (0, 3),
                },
                Variable {
                    name: "y".to_owned(),
                    typ: Type::Float,
                    position: (0, 7),
                }
            ),
            return_type: Some(Type::String),
            local_variables: Vec::new(),
            body: vec!(
                Statement::Assign(
                    AssignStatement {
                        target: AtomExpression {
                            var_name: "x".to_owned(),
                            fields: Vec::new(),
                        },
                        value: Expression {
                            value: ExpressionValue::Atom(
                                AtomExpression {
                                    var_name: "y".to_owned(),
                                    fields: Vec::new(),
                                }
                            ),
                            position: (0, 16),
                        },
                        position: (0, 14),
                    }
                ),
                Statement::Return(
                    ReturnStatement {
                        expression: Some(
                            Expression {
                                value: ExpressionValue::Literal(LiteralExpression::String("".to_owned())),
                                position: (0, 19),
                            }
                        ),
                        position: (0, 18),
                    }
                )
            ),
            position: (0, 0),
        };
        let fn3 = Function {
            name: "foo".to_owned(),
            parameters: Vec::new(),
            return_type: None,
            local_variables: vec!(
                Variable {
                    name: "a".to_owned(),
                    typ: Type::Integer,
                    position: (0, 5),
                }
            ),
            body: Vec::new(),
            position: (0, 0),
        };

        let result1 = Function::parse_from(&give_tokens_positions(tokens1));
        let result2 = Function::parse_from(&give_tokens_positions(tokens2));
        let result3 = Function::parse_from(&give_tokens_positions(tokens3));
        let result4 = Function::parse_from(&give_tokens_positions(tokens4));
        let result5 = Function::parse_from(&give_tokens_positions(tokens5));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());
        assert!(result4.is_err());
        assert!(result5.is_err());

        assert_eq!(result1.unwrap(), (fn1, 6));
        assert_eq!(result2.unwrap(), (fn2, 22));
        assert_eq!(result3.unwrap(), (fn3, 10));
    }

    #[test]
    fn test_struct() {
        // struct Foo {}
        let tokens1 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Struct),
            tokens::Token::Identifier("Foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // struct Foo {x: int, y: float}
        let tokens2 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Struct),
            tokens::Token::Identifier("Foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Int),
            tokens::Token::Symbol(tokens::SymbolToken::Comma),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Float),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // struct Foo {x: Bar}
        let tokens3 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Struct),
            tokens::Token::Identifier("Foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Identifier("Bar".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // struct Foo {x: int; y: float}
        let tokens4 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Struct),
            tokens::Token::Identifier("Foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Int),
            tokens::Token::Symbol(tokens::SymbolToken::Semicolon),
            tokens::Token::Identifier("y".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::Colon),
            tokens::Token::Keyword(tokens::KeywordToken::Float),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );
        // struct Foo {x}
        let tokens5 = vec!(
            tokens::Token::Keyword(tokens::KeywordToken::Struct),
            tokens::Token::Identifier("Foo".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::LeftBrace),
            tokens::Token::Identifier("x".to_owned()),
            tokens::Token::Symbol(tokens::SymbolToken::RightBrace),
        );

        let str1 = Struct {
            name: "Foo".to_owned(),
            fields: Vec::new(),
            position: (0, 0),
        };
        let str2 = Struct {
            name: "Foo".to_owned(),
            fields: vec!(
                Variable {
                    name: "x".to_owned(),
                    typ: Type::Integer,
                    position: (0, 3),
                },
                Variable {
                    name: "y".to_owned(),
                    typ: Type::Float,
                    position: (0, 7),
                }
            ),
            position: (0, 0),
        };
        let str3 = Struct {
            name: "Foo".to_owned(),
            fields: vec!(
                Variable {
                    name: "x".to_owned(),
                    typ: Type::Struct("Bar".to_owned()),
                    position: (0, 3),
                },
            ),
            position: (0, 0),
        };

        let result1 = Struct::parse_from(&give_tokens_positions(tokens1));
        let result2 = Struct::parse_from(&give_tokens_positions(tokens2));
        let result3 = Struct::parse_from(&give_tokens_positions(tokens3));
        let result4 = Struct::parse_from(&give_tokens_positions(tokens4));
        let result5 = Struct::parse_from(&give_tokens_positions(tokens5));

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert!(result3.is_ok());
        assert!(result4.is_err());
        assert!(result5.is_err());

        assert_eq!(result1.unwrap(), (str1, 4));
        assert_eq!(result2.unwrap(), (str2, 11));
        assert_eq!(result3.unwrap(), (str3, 7));
    }

    #[test]
    fn test_program() {
        /*
        struct X {
            a: int,
            b: float,
            c: string
        }

        fn max(a: int, b: int) -> int {
            if a < b {
                return b;
            } else {
                return a;
            }
        }

        fn main()
            with x: int
        {
            x := max(1, 2);
            print(a)
        }
        */
        let tokens = vec!(
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
        );

        /*
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
        */
        let program = Program {
            structs: vec!(
                Struct {
                    name: "X".to_owned(),
                    fields: vec!(
                        Variable {
                            name: "a".to_owned(),
                            typ: Type::Integer,
                            position: (0, 3),
                        },
                        Variable {
                            name: "b".to_owned(),
                            typ: Type::Float,
                            position: (0, 7),
                        },
                        Variable {
                            name: "c".to_owned(),
                            typ: Type::String,
                            position: (0, 11),
                        }
                    ),
                    position: (0, 0),
                }
            ),
            functions: vec!(
                Function {
                    name: "max".to_owned(),
                    parameters: vec!(
                        Variable {
                            name: "a".to_owned(),
                            typ: Type::Integer,
                            position: (0, 18),
                        },
                        Variable {
                            name: "b".to_owned(),
                            typ: Type::Integer,
                            position: (0, 22),
                        }
                    ),
                    return_type: Some(Type::Integer),
                    local_variables: Vec::new(),
                    body: vec!(
                        Statement::IfElse(
                            IfElseStatement {
                                condition: Expression {
                                    value: ExpressionValue::Call(
                                        CallExpression {
                                            func_name: "less".to_owned(),
                                            args: vec!(
                                                Expression {
                                                    value: ExpressionValue::Atom(
                                                        AtomExpression {
                                                            var_name: "a".to_owned(),
                                                            fields: Vec::new(),
                                                        }
                                                    ),
                                                    position: (0, 32),
                                                },
                                                Expression {
                                                    value: ExpressionValue::Atom(
                                                        AtomExpression {
                                                            var_name: "b".to_owned(),
                                                            fields: Vec::new(),
                                                        }
                                                    ),
                                                    position: (0, 34),
                                                }
                                            ),
                                            fields: Vec::new(),
                                        }
                                    ),
                                    position: (0, 30),
                                },
                                if_block: vec!(
                                    Statement::Return(
                                        ReturnStatement {
                                            expression: Some(
                                                Expression {
                                                    value: ExpressionValue::Atom(
                                                        AtomExpression {
                                                            var_name: "b".to_owned(),
                                                            fields: Vec::new(),
                                                        }
                                                    ),
                                                    position: (0, 38),
                                                }
                                            ),
                                            position: (0, 37),
                                        }
                                    )
                                ),
                                else_block: vec!(
                                    Statement::Return(
                                        ReturnStatement {
                                            expression: Some(
                                                Expression {
                                                    value: ExpressionValue::Atom(
                                                        AtomExpression {
                                                            var_name: "a".to_owned(),
                                                            fields: Vec::new(),
                                                        }
                                                    ),
                                                    position: (0, 44),
                                                }
                                            ),
                                            position: (0, 43),
                                        }
                                    )
                                ),
                                position: (0, 29),
                            }
                        )
                    ),
                    position: (0, 15),
                },
                Function {
                    name: "main".to_owned(),
                    parameters: Vec::new(),
                    return_type: None,
                    local_variables: vec!(
                        Variable {
                            name: "x".to_owned(),
                            typ: Type::Integer,
                            position: (0, 53),
                        }
                    ),
                    body: vec!(
                        Statement::Assign(
                            AssignStatement {
                                target: AtomExpression {
                                    var_name: "x".to_owned(),
                                    fields: Vec::new(),
                                },
                                value: Expression {
                                    value: ExpressionValue::Call(
                                        CallExpression {
                                            func_name: "max".to_owned(),
                                            args: vec!(
                                                Expression {
                                                    value: ExpressionValue::Literal(
                                                        LiteralExpression::Int(1)
                                                    ),
                                                    position: (0, 61),
                                                },
                                                Expression {
                                                    value: ExpressionValue::Literal(
                                                        LiteralExpression::Int(2)
                                                    ),
                                                    position: (0, 63),
                                                }
                                            ),
                                            fields: Vec::new(),
                                        }
                                    ),
                                    position: (0, 59),
                                },
                                position: (0, 57),
                            }
                        ),
                        Statement::Expression(
                            Expression {
                                value: ExpressionValue::Call(
                                    CallExpression {
                                        func_name: "print".to_owned(),
                                        args: vec!(
                                            Expression {
                                                value: ExpressionValue::Atom(
                                                    AtomExpression {
                                                        var_name: "x".to_owned(),
                                                        fields: Vec::new(),
                                                    }
                                                ),
                                                position: (0, 68),
                                            }
                                        ),
                                        fields: Vec::new(),
                                    }
                                ),
                                position: (0, 66),
                            }
                        )
                    ),
                    position: (0, 48),
                }
            ),
        };

        let result = Program::parse_from(&give_tokens_positions(tokens));

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), (program, 72));
    }
}