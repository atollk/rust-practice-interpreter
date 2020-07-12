use parser::parse::parsetree::*;

#[test]
fn test_tokenize_and_parse_program() {
    let src =
"struct X {
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
    let expected = Program {
        structs: vec!(
            Struct {
                name: "X".to_owned(),
                fields: vec!(
                    Variable {
                        name: "a".to_owned(),
                        typ: Type::Integer,
                        position: (1, 4),
                    },
                    Variable {
                        name: "b".to_owned(),
                        typ: Type::Float,
                        position: (2, 4),
                    },
                    Variable {
                        name: "c".to_owned(),
                        typ: Type::String,
                        position: (3, 4),
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
                        position: (6, 7),
                    },
                    Variable {
                        name: "b".to_owned(),
                        typ: Type::Integer,
                        position: (6, 15),
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
                                                position: (7, 12),
                                            },
                                            Expression {
                                                value: ExpressionValue::Atom(
                                                    AtomExpression {
                                                        var_name: "b".to_owned(),
                                                        fields: Vec::new(),
                                                    }
                                                ),
                                                position: (7, 15),
                                            }
                                        ),
                                        fields: Vec::new(),
                                    }
                                ),
                                position: (7, 7),
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
                                                position: (8, 15),
                                            }
                                        ),
                                        position: (8, 8),
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
                                                position: (10, 15),
                                            }
                                        ),
                                        position: (10, 8),
                                    }
                                )
                            ),
                            position: (7, 4),
                        }
                    )
                ),
                position: (6, 0),
            },
            Function {
                name: "main".to_owned(),
                parameters: Vec::new(),
                return_type: None,
                local_variables: vec!(
                    Variable {
                        name: "x".to_owned(),
                        typ: Type::Integer,
                        position: (15, 9),
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
                                                position: (17, 12),
                                            },
                                            Expression {
                                                value: ExpressionValue::Literal(
                                                    LiteralExpression::Int(2)
                                                ),
                                                position: (17, 15),
                                            }
                                        ),
                                        fields: Vec::new(),
                                    }
                                ),
                                position: (17, 8),
                            },
                            position: (17, 4),
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
                                            position: (18, 10),
                                        }
                                    ),
                                    fields: Vec::new(),
                                }
                            ),
                            position: (18, 4),
                        }
                    )
                ),
                position: (14, 0),
            }
        ),
    };

    let parsed = parser::code_to_ast(src).unwrap();

    assert_eq!(parsed, expected);
}