use std::collections::HashMap;
use crate::parse::parsetree;
use std::iter;

#[derive(Debug)]
#[derive(Clone)]
pub struct ExecutionError {
    pub message: String
}

pub fn execute_program(program: &parsetree::Program) -> Result<String, ExecutionError> {
    let definitions = definitions_from_program(program);
    let mut context = Context::new();
    let main_function = definitions
        .function_definitions
        .get("main")
        .ok_or(ExecutionError { message: "No main function found.".to_owned() })?;
    main_function.execute(&definitions, &mut context)?;
    Ok(context.output)
}

struct Definitions {
    struct_definitions: HashMap<String, StructDefinition>,
    function_definitions: HashMap<String, FunctionDefinition>,
}

#[derive(Debug)]
#[derive(Clone)]
struct Context {
    variables: HashMap<String, Value>,
    arguments: Vec<Value>,
    return_value: Option<Option<Value>>,
    output: String,
}

impl Context {
    fn new() -> Context {
        Context {
            variables: HashMap::new(),
            arguments: Vec::new(),
            return_value: None,
            output: String::new(),
        }
    }
}

trait Executable {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError>;
}

struct StructDefinition {
    name: String,
    fields: HashMap<String, ValueType>,
}

impl StructDefinition {
    fn from_node(node: &parsetree::Struct) -> StructDefinition {
        return StructDefinition {
            name: node.name.clone(),
            fields: node.fields.iter().map(|var| (var.name.clone(), ValueType::from_node(&var.typ))).collect(),
        };
    }

    fn new_value(&self, definitions: &Definitions) -> Result<StructValue, ExecutionError> {
        let fields = self.fields
            .iter()
            .map(|(name, typ)| -> Result<(String, Value), ExecutionError> {
                Ok((name.clone(), typ.new_value(definitions)?))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(StructValue {
            struct_name: self.name.clone(),
            fields: fields.into_iter().collect(),
        })
    }

    fn constructor(&self, definitions: &Definitions) -> impl Fn(&Definitions, &mut Context) -> Result<(), ExecutionError> {
        let value = self.new_value(definitions);
        move |_, context: &mut Context| {
            match &value {
                Ok(stru_val) => {
                    context.return_value = Some(Some(Value::Struct(stru_val.clone())));
                    Ok(())
                }
                Err(e) => Err((*e).clone())
            }
        }
    }
}

enum FunctionDefinition {
    Predefined(Box<Fn(&Definitions, &mut Context) -> Result<(), ExecutionError>>),
    Userdefined(parsetree::Function),
}

impl Executable for FunctionDefinition {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        match self {
            FunctionDefinition::Predefined(f) => (*f)(definitions, context),
            FunctionDefinition::Userdefined(f) => f.execute(definitions, context)
        }
    }
}

#[derive(Debug)]
#[derive(Clone)]
enum Value {
    Bool(bool),
    Integer(i32),
    Float(f64),
    String(String),
    Struct(StructValue),
}

impl Value {
    fn new(typ: &ValueType, definitions: &Definitions) -> Result<Value, ExecutionError> {
        match typ {
            ValueType::Bool => Ok(Value::Bool(false)),
            ValueType::Integer => Ok(Value::Integer(0)),
            ValueType::Float => Ok(Value::Float(0.0)),
            ValueType::String => Ok(Value::String(String::new())),
            ValueType::Struct(s) => {
                let sdef = definitions.struct_definitions.get(s).ok_or(ExecutionError { message: format!("Invalid type: {}", s) })?;
                sdef.new_value(definitions).map(|sv| Value::Struct(sv))
            }
        }
    }

    fn access_fields(&self, field_names: &[String]) -> Result<&Value, ExecutionError> {
        if field_names.is_empty() {
            return Ok(self);
        }

        match self {
            Value::Struct(x) => {
                let child = x.fields
                    .get(&field_names[0])
                    .ok_or(ExecutionError { message: format!("No field with name {}.", field_names[0]) })?;
                child.access_fields(&field_names[1..])
            }
            _ => Err(ExecutionError { message: "Tried to access fields of non-struct value.".to_owned() })
        }
    }

    fn access_fields_mut(&mut self, field_names: &[String]) -> Result<&mut Value, ExecutionError> {
        if field_names.is_empty() {
            return Ok(self);
        }

        match self {
            Value::Struct(x) => {
                let child = x.fields
                    .get_mut(&field_names[0])
                    .ok_or(ExecutionError { message: format!("No field with name {}.", field_names[0]) })?;
                child.access_fields_mut(&field_names[1..])
            }
            _ => Err(ExecutionError { message: "Tried to access fields of non-struct value.".to_owned() })
        }
    }
}

#[derive(Debug)]
#[derive(Clone)]
enum ValueType {
    Bool,
    Integer,
    Float,
    String,
    Struct(String),
}

impl ValueType {
    fn from_node(typ: &parsetree::Type) -> ValueType {
        match typ {
            parsetree::Type::Bool => ValueType::Bool,
            parsetree::Type::Integer => ValueType::Integer,
            parsetree::Type::Float => ValueType::Float,
            parsetree::Type::String => ValueType::String,
            parsetree::Type::Struct(x) => ValueType::Struct(x.clone())
        }
    }

    fn new_value(&self, definitions: &Definitions) -> Result<Value, ExecutionError> {
        match self {
            ValueType::Bool => Ok(Value::Bool(false)),
            ValueType::Integer => Ok(Value::Integer(0)),
            ValueType::Float => Ok(Value::Float(0.0)),
            ValueType::String => Ok(Value::String(String::new())),
            ValueType::Struct(stru_name) => {
                match definitions.struct_definitions.get(stru_name) {
                    Some(stru_def) => Ok(Value::Struct(stru_def.new_value(definitions)?)),
                    None => Err(ExecutionError { message: format!("No struct by name {}.", stru_name) })
                }
            }
        }
    }
}

#[derive(Debug)]
#[derive(Clone)]
struct StructValue {
    struct_name: String,
    fields: HashMap<String, Value>,
}


fn definitions_from_program(prog: &parsetree::Program) -> Definitions {
    let mut def = Definitions { struct_definitions: HashMap::new(), function_definitions: HashMap::new() };
    for stru in prog.structs.iter() {
        let stru_def = StructDefinition::from_node(&stru);
        def.struct_definitions.insert(stru_def.name.clone(), stru_def);
    }
    for (stru_name, stru_def) in def.struct_definitions.iter() {
        def.function_definitions.insert(
            stru_name.clone(),
            FunctionDefinition::Predefined(Box::new(stru_def.constructor(&def))),
        );
    }
    for func in prog.functions.iter() {
        def.function_definitions.insert(func.name.clone(), FunctionDefinition::Userdefined(func.clone()));
    }
    add_predefined_functions(&mut def.function_definitions);
    def
}

fn add_predefined_functions(function_definitions: &mut HashMap<String, FunctionDefinition>) {
    fn fn_print(definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        if context.arguments.len() != 1 {
            return Err(ExecutionError { message: "print requires exactly one argument.".to_owned() });
        }
        context.output += &match &context.arguments[0] {
            Value::Bool(b) => b.to_string(),
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::String(s) => s.clone(),
            Value::Struct(sval) => return Err(ExecutionError { message: "Cannot print struct".to_owned() })
        };
        Ok(())
    }
    fn fn_less(definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        if context.arguments.len() != 2 {
            return Err(ExecutionError { message: "less requires exactly two arguments.".to_owned() });
        }
        let result = match (&context.arguments[0], &context.arguments[1]) {
            (Value::Integer(x), Value::Integer(y)) => x < y,
            (Value::Float(x), Value::Float(y)) => x < y,
            _ => return Err(ExecutionError { message: "Cannot compare non-numeric values.".to_owned() })
        };
        context.return_value = Some(Some(Value::Bool(result)));
        Ok(())
    }

    function_definitions.insert("print".to_owned(), FunctionDefinition::Predefined(Box::new(fn_print)));
    function_definitions.insert("less".to_owned(), FunctionDefinition::Predefined(Box::new(fn_less)));
}


impl Executable for parsetree::Function {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        // Setup context.
        if self.parameters.len() != context.arguments.len() {
            return Err(ExecutionError { message: format!("Expected {} arguments but found {} when calling {}.", self.parameters.len(), context.arguments.len(), self.name) });
        }
        let args = std::mem::take(&mut context.arguments);
        for (param, arg) in self.parameters.iter().zip(args.into_iter()) {
            context.variables.insert(param.name.clone(), arg);
        }
        for lvar in self.local_variables.iter() {
            if context.variables.contains_key(&lvar.name) {
                return Err(ExecutionError { message: format!("Variable {} is both local and a parameter.", lvar.name) });
            } else {
                context.variables.insert(
                    lvar.name.clone(),
                    Value::new(&ValueType::from_node(&lvar.typ), definitions)?,
                );
            }
        }

        // Run function.
        for stmt in self.body.iter() {
            stmt.execute(definitions, context)?;
            if context.return_value.is_some() {
                break;
            }
        }
        Ok(())
    }
}


impl Executable for parsetree::Statement {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        match self {
            parsetree::Statement::Expression(expr) => expr.execute(definitions, context),
            parsetree::Statement::Assign(ass) => ass.execute(definitions, context),
            parsetree::Statement::Return(ret) => ret.execute(definitions, context),
            parsetree::Statement::IfElse(ifelse) => ifelse.execute(definitions, context),
            parsetree::Statement::While(whil) => whil.execute(definitions, context),
        }
    }
}

impl Executable for parsetree::Expression {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        self.evaluate(definitions, context)?;
        Ok(())
    }
}

impl Executable for parsetree::AssignStatement {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        let value = self.value
            .evaluate(definitions, context)?
            .ok_or(ExecutionError { message: "Expression in assign statement did not return value.".to_owned() })?;
        let mut target = context.variables
            .get_mut(&self.target.var_name)
            .ok_or(ExecutionError { message: format!("Variable {} not found.", self.target.var_name) })?;
        let target = target.access_fields_mut(&self.target.fields)?;
        *target = value;
        Ok(())
    }
}

impl Executable for parsetree::ReturnStatement {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        let value = match &self.expression {
            Some(expr) => expr.evaluate(definitions, context)?,
            None => None
        };
        context.return_value = Some(value);
        Ok(())
    }
}

impl Executable for parsetree::IfElseStatement {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        let condition = self.condition.evaluate(definitions, context)?;
        if let Some(Value::Bool(b)) = condition {
            let block = if b { &self.if_block } else { &self.else_block };
            for stmt in block.iter() {
                stmt.execute(definitions, context)?;
                if context.return_value.is_some() {
                    break;
                }
            }
            Ok(())
        } else {
            Err(ExecutionError { message: "Invalid if-condition.".to_owned() })
        }
    }
}

impl Executable for parsetree::WhileStatement {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        while let condition = self.condition.evaluate(definitions, context)?
            {
                if let Some(Value::Bool(b)) = condition {
                    if !b {
                        break;
                    }
                } else {
                    return Err(ExecutionError { message: "Invalid while-condition.".to_owned() });
                }
                for stmt in self.body.iter() {
                    stmt.execute(definitions, context)?;
                    if context.return_value.is_some() {
                        break;
                    }
                }
            }
        Ok(())
    }
}

impl parsetree::Expression {
    fn evaluate(&self, definitions: &Definitions, context: &mut Context) -> Result<Option<Value>, ExecutionError> {
        match &self.value {
            parsetree::ExpressionValue::Literal(l) => l.evaluate(definitions, context),
            parsetree::ExpressionValue::Atom(a) => a.evaluate(definitions, context),
            parsetree::ExpressionValue::Call(c) => c.evaluate(definitions, context)
        }
    }
}

impl parsetree::LiteralExpression {
    fn evaluate(&self, definitions: &Definitions, context: &mut Context) -> Result<Option<Value>, ExecutionError> {
        match self {
            parsetree::LiteralExpression::Bool(b) => Ok(Some(Value::Bool(*b))),
            parsetree::LiteralExpression::Int(i) => Ok(Some(Value::Integer(*i))),
            parsetree::LiteralExpression::Float(f) => Ok(Some(Value::Float(*f))),
            parsetree::LiteralExpression::String(s) => Ok(Some(Value::String(s.clone())))
        }
    }
}

impl parsetree::AtomExpression {
    fn evaluate(&self, definitions: &Definitions, context: &mut Context) -> Result<Option<Value>, ExecutionError> {
        let value = context.variables
            .get(&self.var_name)
            .ok_or(ExecutionError { message: format!("Variable {} not found.", self.var_name) })?;
        let value = value.access_fields(&self.fields)?;
        Ok(Some(value.clone()))
    }
}

impl parsetree::CallExpression {
    fn evaluate(&self, definitions: &Definitions, context: &mut Context) -> Result<Option<Value>, ExecutionError> {
        let args: Vec<Option<Value>> = self.args
            .iter()
            .map(|arg| arg.evaluate(definitions, context))
            .collect::<Result<Vec<_>, _>>()?;
        let args: Vec<Value> = args
            .into_iter()
            .map(|arg| arg.ok_or(ExecutionError { message: "Invalid function argument.".to_owned() }))
            .collect::<Result<Vec<_>, _>>()?;
        let fn_context = parsetree::CallExpression::call_function(&self.func_name, args, definitions)?;
        let fn_return = match fn_context.return_value {
            Some(x) => x,
            None => None
        };
        let fn_output = fn_context.output;

        context.output += &fn_output;
        match fn_return {
            Some(x) => Ok(Some(x.access_fields(&self.fields)?.clone())),
            None => if self.fields.is_empty() {
                Ok(None)
            } else {
                Err(ExecutionError { message: "".to_owned() })
            }
        }
    }

    fn call_function(fn_name: &str, args: Vec<Value>, definitions: &Definitions) -> Result<Context, ExecutionError> {
        let mut context = Context::new();
        context.arguments = args;
        match definitions.function_definitions.get(fn_name) {
            Some(FunctionDefinition::Predefined(f)) => (*f)(definitions, &mut context)?,
            Some(FunctionDefinition::Userdefined(f)) => f.execute(definitions, &mut context)?,
            None => return Err(ExecutionError { message: format!("No function named {}.", fn_name) })
        };
        Ok(context)
    }
}


#[cfg(test)]
mod tests {
    use crate::parse::parsetree::*;

    #[test]
    fn execute_program() {
        let parse_tree = Program {
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

        let x = super::execute_program(&parse_tree);
        assert!(x.is_ok());
        assert_eq!(x.unwrap(), "2".to_owned());
    }
}
