use crate::tokenize::tokens::Token;

pub mod parsetree {
    struct Program {
        structs: Vec<Struct>
    }


    struct Variable {
        name: String,
        typ: Type,
    }

    enum Type {
        Bool,
        Integer,
        Float,
        String,
    }

    struct Struct {
        name: String,
        fields: Vec<Variable>,
    }

    struct Function {
        name: String,
        parameters: Vec<Variable>,
        local_variables: Vec<Variable>,
        body: Vec<Statement>,
    }

    enum Statement {
        IfElse(IfElseStatement),
        While(WhileStatement),
        Assign(AssignStatement),
        Expression(Expression),
    }

    struct IfElseStatement {
        if_block: IfStatement,
        elifs: Vec<IfStatement>,
        else_block: Vec<Statement>,
    }

    struct IfStatement {
        condition: Expression,
        body: Vec<Statement>,
    }

    struct WhileStatement {
        condition: Expression,
        body: Vec<Statement>,
    }

    struct AssignStatement {
        target: String,
        value: Expression,
    }

    struct Expression {
        src: String
    }
}

