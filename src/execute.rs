use std::collections::HashMap;
use crate::parse::parsetree;
use std::iter;

#[derive(Debug)]
pub struct ExecutionError {
    pub message: String
}

pub fn execute_program(program: &parsetree::Program) -> Result<String, ExecutionError> {
    let definitions = definitions_from_program(program);
    let mut context = Context::new();
    let main_function = definitions
        .function_definitions
        .get("main")
        .ok_or(ExecutionError {message: "No main function found.".to_owned()})?;
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
            output: String::new()
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

    fn new_value(&self) -> Result<Value, ExecutionError> {
        // TODO99
        Ok(Value::Bool(true))
    }
}

enum FunctionDefinition {
    Predefined(Box<Fn(&Definitions, &mut Context) -> Result<(), ExecutionError>>),
    Userdefined(parsetree::Function)
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
                let sdef = definitions.struct_definitions.get(s).ok_or(ExecutionError {message: format!("Invalid type: {}", s)})?;
                sdef.new_value()
            }
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
    for func in prog.functions.iter() {
        def.function_definitions.insert(func.name.clone(), FunctionDefinition::Userdefined(func.clone()));
    }
    def
}


impl Executable for parsetree::Function {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        // Setup context.
        if self.parameters.len() != context.arguments.len() {
            return Err(ExecutionError {message: format!("Expected {} arguments but found {} when calling {}.", self.parameters.len(), context.arguments.len(), self.name)});
        }
        let args = std::mem::take(&mut context.arguments);
        for (param, arg) in self.parameters.iter().zip(args.into_iter()) {
            context.variables.insert(param.name.clone(), arg);
        }
        for lvar in self.local_variables.iter() {
            if context.variables.contains_key(&lvar.name) {
                return Err(ExecutionError {message: format!("Variable {} is both local and a parameter.", lvar.name)});
            } else {
                context.variables.insert(
                    lvar.name.clone(),
                    Value::new(&ValueType::from_node(&lvar.typ), definitions)?
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
        // TODO
        Ok(())
    }
}

impl Executable for parsetree::ReturnStatement {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        // TODO
        Ok(())
    }
}

impl Executable for parsetree::IfElseStatement {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        // TODO
        Ok(())
    }
}

impl Executable for parsetree::WhileStatement {
    fn execute(&self, definitions: &Definitions, context: &mut Context) -> Result<(), ExecutionError> {
        // TODO
        Ok(())
    }
}

impl parsetree::Expression {
    fn evaluate(&self, definitions: &Definitions, context: &mut Context) -> Result<Value, ExecutionError> {
        // TODO
    }
}