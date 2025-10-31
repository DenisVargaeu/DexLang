import argparse
import ast
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class DexCompileError(Exception):
    def __init__(self, message: str, lineno: Optional[int] = None) -> None:
        super().__init__(message)
        self.lineno = lineno


class Logger:
    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.launch_timestamp = datetime.now().strftime("%H:%M:%S")

    def log(self, tag: str, message: str) -> None:
        print(f"[{tag}] {message}")

    def line(self, lineno: int, text: str) -> None:
        self.log(f"line {lineno}", f"parsed: {text}")

    def done(self, exit_code: int) -> None:
        duration = time.perf_counter() - self.start
        status = "build ok" if exit_code == 0 else "build failed"
        self.log("done", f"{status} ({duration:.2f}s)")
        self.log("done", f"exit code {exit_code}")


@dataclass
class Statement:
    lineno: int


@dataclass
class Program:
    body: List[Statement] = field(default_factory=list)


@dataclass
class LetStatement(Statement):
    name: str
    expression: str


@dataclass
class AssignStatement(Statement):
    name: str
    expression: str


@dataclass
class PrintStatement(Statement):
    expression: str


@dataclass
class IfStatement(Statement):
    condition: str
    body: List[Statement]
    orelse: List[Statement]


@dataclass
class WhileStatement(Statement):
    condition: str
    body: List[Statement]


@dataclass
class ReturnStatement(Statement):
    expression: str


@dataclass
class FunctionDefinition(Statement):
    name: str
    args: List[str]
    body: List[Statement]


@dataclass
class ExpressionStatement(Statement):
    expression: str


@dataclass
class ForStatement(Statement):
    target: str
    start: str
    end: str
    step: Optional[str]
    body: List[Statement]


@dataclass
class IncrementStatement(Statement):
    name: str
    delta: int


LineInfo = Tuple[int, str, int]


class Parser:
    def __init__(self, source: str, logger: Logger) -> None:
        self.logger = logger
        self.lines: List[LineInfo] = self._preprocess(source)
        self.pos: int = 0
        self.line_total: int = len(self.lines)

    def parse(self) -> Program:
        body = self._parse_block(expected_indent=0)
        return Program(body=body)

    def _preprocess(self, source: str) -> List[LineInfo]:
        processed: List[LineInfo] = []
        for idx, raw in enumerate(source.splitlines(), start=1):
            stripped = raw.rstrip()
            if not stripped or stripped.lstrip().startswith("#"):
                continue
            indent = len(stripped) - len(stripped.lstrip(" "))
            if indent % 4 != 0:
                raise DexCompileError("indentation must be multiples of 4 spaces", lineno=idx)
            processed.append((indent // 4, stripped.lstrip(), idx))
        return processed

    def _peek(self) -> Optional[LineInfo]:
        if self.pos >= len(self.lines):
            return None
        return self.lines[self.pos]

    def _advance(self) -> LineInfo:
        line = self._peek()
        if line is None:
            raise DexCompileError("unexpected end of file")
        self.pos += 1
        return line

    def _parse_block(self, expected_indent: int) -> List[Statement]:
        # DexLang relies on indentation depth (multiples of four spaces) to delimit blocks.
        # Whenever indentation decreases, the parser unwinds to the corresponding parent block.
        statements: List[Statement] = []
        while True:
            next_line = self._peek()
            if next_line is None:
                break
            indent, text, lineno = next_line
            if indent < expected_indent:
                break
            if indent > expected_indent:
                raise DexCompileError("unexpected indent", lineno=lineno)
            self._advance()
            statement = self._parse_statement(text, lineno, expected_indent)
            statements.append(statement)
        return statements

    def _parse_statement(self, text: str, lineno: int, current_indent: int) -> Statement:
        if text.startswith("if ") and text.endswith(":"):
            self.logger.log("if", f"detected condition @ line {lineno}")
            self.logger.line(lineno, text)
            return self._parse_if(text, lineno, current_indent)
        if text.startswith("while ") and text.endswith(":"):
            self.logger.log("loop", f"while @ line {lineno}")
            self.logger.line(lineno, text)
            return self._parse_while(text, lineno, current_indent)
        if text.startswith("fn ") and text.endswith(":"):
            self.logger.line(lineno, text)
            return self._parse_function(text, lineno, current_indent)
        if text.startswith("for ") and text.endswith(":"):
            self.logger.log("for", f"loop @ line {lineno}")
            self.logger.line(lineno, text)
            return self._parse_for(text, lineno, current_indent)
        if text.startswith("++") or text.startswith("--"):
            statement = self._parse_increment(text, lineno)
            self.logger.line(lineno, text)
            return statement
        if text.startswith("let "):
            statement = self._parse_binding(text, lineno)
        elif text.startswith("print(") and text.endswith(")"):
            expr = text[len("print(") : -1].strip()
            statement = PrintStatement(expression=expr, lineno=lineno)
        elif text.startswith("return "):
            expr = text[len("return ") :].strip()
            if not expr:
                raise DexCompileError("return requires an expression", lineno=lineno)
            statement = ReturnStatement(expression=expr, lineno=lineno)
        elif self._is_call_expression(text):
            normalized = self._normalize_debug_call(text)
            statement = ExpressionStatement(expression=normalized, lineno=lineno)
        elif "=" in text:
            statement = self._parse_assignment(text, lineno)
        else:
            raise DexCompileError(f"unsupported statement '{text}'", lineno=lineno)
        self.logger.line(lineno, text)
        return statement

    def _parse_binding(self, text: str, lineno: int) -> LetStatement:
        try:
            _, remainder = text.split("let ", 1)
            name, expr = remainder.split("=", 1)
        except ValueError as exc:
            raise DexCompileError("invalid let binding", lineno=lineno) from exc
        name = name.strip()
        if not name.isidentifier():
            raise DexCompileError(f"invalid identifier '{name}'", lineno=lineno)
        expression = expr.strip()
        if not expression:
            raise DexCompileError("missing expression in let statement", lineno=lineno)
        return LetStatement(name=name, expression=expression, lineno=lineno)

    def _parse_assignment(self, text: str, lineno: int) -> AssignStatement:
        name, expr = text.split("=", 1)
        name = name.strip()
        if not name.isidentifier():
            raise DexCompileError(f"invalid assignment target '{name}'", lineno=lineno)
        expression = expr.strip()
        if not expression:
            raise DexCompileError("assignment requires an expression", lineno=lineno)
        return AssignStatement(name=name, expression=expression, lineno=lineno)

    def _parse_if(self, text: str, lineno: int, current_indent: int) -> IfStatement:
        condition = text[len("if ") : -1].strip()
        if_body = self._parse_block(current_indent + 1)
        elif_branches: List[Tuple[str, List[Statement], int]] = []
        else_body: List[Statement] = []
        next_line = self._peek()
        if next_line is not None:
            indent, peek_text, else_lineno = next_line
            while indent == current_indent and peek_text.startswith("elif ") and peek_text.endswith(":"):
                self._advance()
                self.logger.log("elif", f"branch @ line {else_lineno}")
                self.logger.line(else_lineno, peek_text)
                elif_condition = peek_text[len("elif ") : -1].strip()
                branch_body = self._parse_block(current_indent + 1)
                elif_branches.append((elif_condition, branch_body, else_lineno))
                next_line = self._peek()
                if next_line is None:
                    break
                indent, peek_text, else_lineno = next_line
            if next_line is not None and indent == current_indent and peek_text == "else:":
                self._advance()
                self.logger.line(else_lineno, peek_text)
                else_body = self._parse_block(current_indent + 1)
        current_else: List[Statement] = else_body
        for elif_condition, elif_body, branch_line in reversed(elif_branches):
            nested = IfStatement(condition=elif_condition, body=elif_body, orelse=current_else, lineno=branch_line)
            current_else = [nested]
        return IfStatement(condition=condition, body=if_body, orelse=current_else, lineno=lineno)

    def _parse_while(self, text: str, lineno: int, current_indent: int) -> WhileStatement:
        condition = text[len("while ") : -1].strip()
        body = self._parse_block(current_indent + 1)
        return WhileStatement(condition=condition, body=body, lineno=lineno)

    def _parse_function(self, text: str, lineno: int, current_indent: int) -> FunctionDefinition:
        header = text[len("fn ") : -1].strip()
        if "(" not in header or ")" not in header:
            raise DexCompileError("invalid function signature", lineno=lineno)
        name, args_section = header.split("(", 1)
        name = name.strip()
        if not name.isidentifier():
            raise DexCompileError(f"invalid function name '{name}'", lineno=lineno)
        args_section = args_section.rstrip(")")
        args = [arg.strip() for arg in args_section.split(",") if arg.strip()]
        for arg in args:
            if not arg.isidentifier():
                raise DexCompileError(f"invalid parameter name '{arg}'", lineno=lineno)
        self.logger.log("fn", f"def {name}() @ line {lineno}")
        body = self._parse_block(current_indent + 1)
        return FunctionDefinition(name=name, args=args, body=body, lineno=lineno)

    def _is_call_expression(self, text: str) -> bool:
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError:
            return False
        return isinstance(tree.body, ast.Call)

    def _parse_for(self, text: str, lineno: int, current_indent: int) -> ForStatement:
        header = text[len("for ") : -1].strip()
        if " in " not in header:
            raise DexCompileError("for loop must use 'in'", lineno=lineno)
        target_part, iterable = header.split(" in ", 1)
        target = target_part.strip()
        if not target.isidentifier():
            raise DexCompileError(f"invalid loop variable '{target}'", lineno=lineno)
        iterable = iterable.strip()
        if not iterable.startswith("range(") or not iterable.endswith(")"):
            raise DexCompileError("for loop expects range()", lineno=lineno)
        args_section = iterable[len("range(") : -1]
        if not args_section and args_section != "0":
            raise DexCompileError("range() requires arguments", lineno=lineno)
        parts = [part.strip() for part in args_section.split(",") if part.strip()]
        if not 1 <= len(parts) <= 3:
            raise DexCompileError("range() accepts 1 to 3 arguments", lineno=lineno)
        if len(parts) == 1:
            start_expr = "0"
            end_expr = parts[0]
            step_expr = None
        elif len(parts) == 2:
            start_expr, end_expr = parts
            step_expr = None
        else:
            start_expr, end_expr, step_expr = parts
        body = self._parse_block(current_indent + 1)
        return ForStatement(
            target=target,
            start=start_expr,
            end=end_expr,
            step=step_expr,
            body=body,
            lineno=lineno,
        )

    def _parse_increment(self, text: str, lineno: int) -> IncrementStatement:
        delta = 1 if text.startswith("++") else -1
        remainder = text[2:].strip()
        if not remainder.isidentifier():
            raise DexCompileError("increment/decrement requires an identifier", lineno=lineno)
        return IncrementStatement(name=remainder, delta=delta, lineno=lineno)

    def _normalize_debug_call(self, text: str) -> str:
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError:
            return text
        node = tree.body
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "debug":
            if len(node.args) > 1:
                parts: List[str] = []
                for arg in node.args:
                    segment = ast.get_source_segment(text, arg)
                    if segment is None:
                        return text
                    parts.append(segment)
                joined = " + ".join(parts)
                return f"debug({joined})"
        return text


class TypeResolver(ast.NodeVisitor):
    def __init__(self, env_stack: List[Dict[str, str]], functions: Dict[str, str]) -> None:
        self.env_stack = env_stack
        self.functions = functions

    def resolve(self, expr: str) -> str:
        tree = ast.parse(expr, mode="eval")
        return self.visit(tree.body)

    def resolve_ast(self, node: ast.AST) -> str:
        return self.visit(node)

    def visit_Name(self, node: ast.Name) -> str:
        for env in reversed(self.env_stack):
            if node.id in env:
                return env[node.id]
        return "number"

    def visit_Constant(self, node: ast.Constant) -> str:
        value = node.value
        if isinstance(value, str):
            return "string"
        return "number"

    def visit_BinOp(self, node: ast.BinOp) -> str:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if left == "string" or right == "string":
            return "string"
        return "number"

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        if isinstance(node.op, ast.Not):
            return "number"
        return self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        return "number"

    def visit_Compare(self, node: ast.Compare) -> str:
        return "number"

    def visit_Call(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in {"input", "to_string", "readfile"}:
                return "string"
            if func_name in {
                "to_number",
                "len",
                "sleep",
                "exit",
                "writefile",
                "rand",
                "abs",
                "sqrt",
                "pow",
                "system",
                "debug",
            }:
                return "number"
            if func_name in self.functions:
                return self.functions[func_name]
        return "number"

    def generic_visit(self, node: ast.AST) -> str:
        for child in ast.iter_child_nodes(node):
            result = self.visit(child)
            if result == "string":
                return "string"
        return "number"


class ExpressionEmitter(ast.NodeVisitor):
    def __init__(self) -> None:
        self._supported_binops = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "/",
            ast.Mod: "%",
            ast.Pow: "^",
        }
        self._supported_compares = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        self.resolver: Optional[TypeResolver] = None

    def bind(self, resolver: Optional[TypeResolver]) -> None:
        self.resolver = resolver

    def _node_type(self, node: ast.AST) -> str:
        if self.resolver is None:
            return "number"
        try:
            return self.resolver.resolve_ast(node)
        except DexCompileError:
            return "number"
        except Exception:
            return "number"

    def emit(self, expr: str) -> str:
        tree = ast.parse(expr, mode="eval")
        return self.visit(tree.body)

    def visit_Name(self, node: ast.Name) -> str:
        return node.id

    def visit_Constant(self, node: ast.Constant) -> str:
        value = node.value
        if isinstance(value, str):
            escaped = (
                value.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )
            return f"\"{escaped}\""
        if value is True:
            return "1"
        if value is False:
            return "0"
        return repr(value)

    def visit_BinOp(self, node: ast.BinOp) -> str:
        if isinstance(node.op, ast.Add) and self._node_type(node) == "string" and self.resolver is not None:
            string_emitter = StringEmitter(self, self.resolver)
            return string_emitter._emit_node(node)
        op_type = type(node.op)
        if op_type not in self._supported_binops:
            raise DexCompileError("unsupported binary operator")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if op_type is ast.Pow:
            return f"pow({left}, {right})"
        if op_type is ast.FloorDiv:
            return f"floor({left} / {right})"
        return f"({left} {self._supported_binops[op_type]} {right})"

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.UAdd):
            return f"(+{operand})"
        if isinstance(node.op, ast.Not):
            return f"(!{operand})"
        raise DexCompileError("unsupported unary operator")

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        op = "&&" if isinstance(node.op, ast.And) else "||"
        values = [self.visit(value) for value in node.values]
        joined = f" {op} ".join(values)
        return f"({joined})"

    def visit_Compare(self, node: ast.Compare) -> str:
        left = self.visit(node.left)
        comparisons = []
        current_left = left
        for op, comparator in zip(node.ops, node.comparators):
            c_symbol = self._supported_compares.get(type(op))
            if not c_symbol:
                raise DexCompileError("unsupported comparison operator")
            right = self.visit(comparator)
            comparisons.append(f"({current_left} {c_symbol} {right})")
            current_left = right
        return f"({' && '.join(comparisons)})" if len(comparisons) > 1 else comparisons[0]

    def visit_Call(self, node: ast.Call) -> str:
        if not isinstance(node.func, ast.Name):
            raise DexCompileError("only simple function calls are supported")
        func_name = node.func.id
        builtin_map = {
            "input": "dex_input",
            "to_string": "dex_to_string",
            "to_number": "dex_to_number",
            "len": "dex_len",
            "readfile": "dex_readfile",
            "writefile": "dex_writefile",
            "sleep": "dex_sleep",
            "exit": "dex_exit",
            "debug": "dex_debug",
            "rand": "dex_rand",
            "system": "dex_system",
            "sqrt": "sqrt",
            "pow": "pow",
            "abs": "fabs",
        }
        if func_name in builtin_map:
            func_name = builtin_map[func_name]
        args = [self.visit(arg) for arg in node.args]
        return f"{func_name}({', '.join(args)})"

    def generic_visit(self, node: ast.AST) -> str:
        raise DexCompileError("unsupported expression")


class StringEmitter:
    def __init__(self, base_emitter: ExpressionEmitter, resolver: TypeResolver) -> None:
        self.base = base_emitter
        self.resolver = resolver

    def emit(self, expr: str) -> str:
        tree = ast.parse(expr, mode="eval")
        return self._emit_node(tree.body)

    def _emit_node(self, node: ast.AST) -> str:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._emit_node(node.left)
            right = self._emit_node(node.right)
            return f"dex_concat({left}, {right})"
        cexpr = self.base.visit(node)
        node_type = self.resolver.resolve_ast(node)
        if node_type != "string":
            return f"dex_to_string({cexpr})"
        return cexpr


class CodeGenerator:
    def __init__(self, program: Program, logger: Logger) -> None:
        self.program = program
        self.logger = logger
        self.functions: Dict[str, FunctionDefinition] = {}
        self.function_returns: Dict[str, str] = {}
        self.emit = ExpressionEmitter()
        self.loop_counter = 0

    def generate(self) -> str:
        top_level: List[Statement] = []
        for stmt in self.program.body:
            if isinstance(stmt, FunctionDefinition):
                self.functions[stmt.name] = stmt
            else:
                top_level.append(stmt)
        self._infer_function_return_types()
        lines: List[str] = [
            "#include <math.h>",
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "",
            "char *dex_input(const char *prompt);",
            "void dex_print_string(const char *value);",
            "void dex_print_number(double value);",
            "char *dex_to_string(double n);",
            "double dex_to_number(const char *s);",
            "int dex_len(const char *s);",
            "char *dex_concat(const char *lhs, const char *rhs);",
            "char *dex_readfile(const char *path);",
            "double dex_writefile(const char *path, const char *text);",
            "double dex_sleep(double seconds);",
            "double dex_exit(double code);",
            "double dex_debug(const char *message);",
            "double dex_rand(double min_value, double max_value);",
            "double dex_system(const char *command);",
            "",
        ]
        for func in self.functions.values():
            self.logger.log("fn", f"compiling {func.name}()")
            lines.extend(self._generate_function(func))
            lines.append("")
        lines.append("int main(void) {")
        env_stack: List[Dict[str, str]] = [dict()]
        body_lines = self._generate_block(top_level, indent=1, env_stack=env_stack, allow_return=False)
        lines.extend(body_lines)
        lines.append("    return 0;")
        lines.append("}")
        return "\n".join(lines)

    def _infer_function_return_types(self) -> None:
        for name, func in self.functions.items():
            env_stack: List[Dict[str, str]] = [dict((arg, "number") for arg in func.args)]
            resolver = TypeResolver(env_stack, self.function_returns)
            return_types: List[str] = []
            self._collect_return_types(func.body, env_stack, resolver, return_types)
            if not return_types:
                raise DexCompileError(f"function '{name}' must return a value", lineno=func.lineno)
            first = return_types[0]
            if any(rt != first for rt in return_types):
                raise DexCompileError(f"function '{name}' has inconsistent return types", lineno=func.lineno)
            self.function_returns[name] = first

    def _collect_return_types(
        self,
        body: List[Statement],
        env_stack: List[Dict[str, str]],
        resolver: TypeResolver,
        found: List[str],
    ) -> None:
        for stmt in body:
            if isinstance(stmt, LetStatement):
                expr_type = resolver.resolve(stmt.expression)
                env_stack[-1][stmt.name] = expr_type
            elif isinstance(stmt, AssignStatement):
                target_env = None
                for env in reversed(env_stack):
                    if stmt.name in env:
                        target_env = env
                        break
                if target_env is None:
                    raise DexCompileError(f"assignment to undefined variable '{stmt.name}'", lineno=stmt.lineno)
                target_env[stmt.name] = resolver.resolve(stmt.expression)
            elif isinstance(stmt, IfStatement):
                env_stack.append(env_stack[-1].copy())
                self._collect_return_types(stmt.body, env_stack, resolver, found)
                env_stack.pop()
                if stmt.orelse:
                    env_stack.append(env_stack[-1].copy())
                    self._collect_return_types(stmt.orelse, env_stack, resolver, found)
                    env_stack.pop()
            elif isinstance(stmt, WhileStatement):
                env_stack.append(env_stack[-1].copy())
                self._collect_return_types(stmt.body, env_stack, resolver, found)
                env_stack.pop()
            elif isinstance(stmt, ReturnStatement):
                found.append(resolver.resolve(stmt.expression))
            elif isinstance(stmt, FunctionDefinition):
                continue
            elif isinstance(stmt, ForStatement):
                env_stack.append(env_stack[-1].copy())
                env_stack[-1][stmt.target] = "number"
                self._collect_return_types(stmt.body, env_stack, resolver, found)
                env_stack.pop()
            elif isinstance(stmt, IncrementStatement):
                for env in reversed(env_stack):
                    if stmt.name in env:
                        if env[stmt.name] == "string":
                            raise DexCompileError(
                                f"cannot increment/decrement string variable '{stmt.name}'",
                                lineno=stmt.lineno,
                            )
                        env[stmt.name] = "number"
                        break
                else:
                    raise DexCompileError(
                        f"increment/decrement of undefined variable '{stmt.name}'",
                        lineno=stmt.lineno,
                    )
            elif isinstance(stmt, ExpressionStatement):
                continue

    def _ctype(self, type_name: str) -> str:
        return "const char *" if type_name == "string" else "double"

    def _next_loop_id(self) -> int:
        self.loop_counter += 1
        return self.loop_counter

    def _generate_function(self, func: FunctionDefinition) -> List[str]:
        return_type = self.function_returns.get(func.name, "number")
        c_return_type = self._ctype(return_type)
        args_decl = ", ".join(f"double {arg}" for arg in func.args) or "void"
        lines = [f"{c_return_type} {func.name}({args_decl}) {{"]
        env_stack: List[Dict[str, str]] = [dict((arg, "number") for arg in func.args)]
        body_lines = self._generate_block(func.body, indent=1, env_stack=env_stack, allow_return=True)
        lines.extend(body_lines)
        lines.append("}")
        return lines

    def _generate_block(
        self,
        body: List[Statement],
        indent: int,
        env_stack: List[Dict[str, str]],
        allow_return: bool,
    ) -> List[str]:
        lines: List[str] = []
        resolver = TypeResolver(env_stack, self.function_returns)
        self.emit.bind(resolver)
        string_emitter = StringEmitter(self.emit, resolver)
        for stmt in body:
            prefix = "    " * indent
            if isinstance(stmt, LetStatement):
                expr_type = resolver.resolve(stmt.expression)
                env_stack[-1][stmt.name] = expr_type
                cexpr = (
                    string_emitter.emit(stmt.expression)
                    if expr_type == "string"
                    else self.emit.emit(stmt.expression)
                )
                lines.append(f"{prefix}{self._ctype(expr_type)} {stmt.name} = {cexpr};")
            elif isinstance(stmt, AssignStatement):
                target_env = None
                for env in reversed(env_stack):
                    if stmt.name in env:
                        target_env = env
                        break
                if target_env is None:
                    raise DexCompileError(f"assignment to undefined variable '{stmt.name}'", lineno=stmt.lineno)
                new_type = resolver.resolve(stmt.expression)
                target_env[stmt.name] = new_type
                cexpr = (
                    string_emitter.emit(stmt.expression)
                    if new_type == "string"
                    else self.emit.emit(stmt.expression)
                )
                lines.append(f"{prefix}{stmt.name} = {cexpr};")
            elif isinstance(stmt, PrintStatement):
                expr_type = resolver.resolve(stmt.expression)
                if expr_type == "string":
                    cexpr = string_emitter.emit(stmt.expression)
                    lines.append(f"{prefix}dex_print_string({cexpr});")
                else:
                    cexpr = self.emit.emit(stmt.expression)
                    lines.append(f"{prefix}dex_print_number({cexpr});")
            elif isinstance(stmt, IfStatement):
                condition = self.emit.emit(stmt.condition)
                lines.append(f"{prefix}if ({condition}) {{")
                env_stack.append(env_stack[-1].copy())
                lines.extend(
                    self._generate_block(stmt.body, indent + 1, env_stack, allow_return=allow_return)
                )
                env_stack.pop()
                lines.append(f"{prefix}}}")
                if stmt.orelse:
                    lines.append(f"{prefix}else {{")
                    env_stack.append(env_stack[-1].copy())
                    lines.extend(
                        self._generate_block(stmt.orelse, indent + 1, env_stack, allow_return=allow_return)
                    )
                    env_stack.pop()
                    lines.append(f"{prefix}}}")
            elif isinstance(stmt, WhileStatement):
                condition = self.emit.emit(stmt.condition)
                lines.append(f"{prefix}while ({condition}) {{")
                lines.append(
                    prefix
                    + "    /* DexLang uses indentation levels to delimit blocks; "
                    + "the parser counts leading spaces (multiples of four) to decide "
                    + "when to open or close braces in the generated C code. */"
                )
                env_stack.append(env_stack[-1].copy())
                lines.extend(
                    self._generate_block(stmt.body, indent + 1, env_stack, allow_return=allow_return)
                )
                env_stack.pop()
                lines.append(f"{prefix}}}")
            elif isinstance(stmt, ForStatement):
                loop_id = self._next_loop_id()
                start_expr = self.emit.emit(stmt.start)
                end_expr = self.emit.emit(stmt.end)
                step_source = stmt.step if stmt.step is not None else "1"
                step_expr = self.emit.emit(step_source)
                lines.append(f"{prefix}double __dex_step_{loop_id} = {step_expr};")
                lines.append(f"{prefix}double __dex_end_{loop_id} = {end_expr};")
                lines.append(
                    f"{prefix}for (double {stmt.target} = {start_expr}; (__dex_step_{loop_id} >= 0 ? {stmt.target} < __dex_end_{loop_id} : {stmt.target} > __dex_end_{loop_id}); {stmt.target} += __dex_step_{loop_id}) {{"
                )
                env_stack.append(env_stack[-1].copy())
                env_stack[-1][stmt.target] = "number"
                lines.extend(
                    self._generate_block(stmt.body, indent + 1, env_stack, allow_return=allow_return)
                )
                env_stack.pop()
                lines.append(f"{prefix}}}")
            elif isinstance(stmt, ReturnStatement):
                if not allow_return:
                    raise DexCompileError("return outside of function", lineno=stmt.lineno)
                return_type = resolver.resolve(stmt.expression)
                cexpr = (
                    string_emitter.emit(stmt.expression)
                    if return_type == "string"
                    else self.emit.emit(stmt.expression)
                )
                lines.append(f"{prefix}return {cexpr};")
            elif isinstance(stmt, FunctionDefinition):
                continue
            elif isinstance(stmt, ExpressionStatement):
                cexpr = self.emit.emit(stmt.expression)
                lines.append(f"{prefix}{cexpr};")
            elif isinstance(stmt, IncrementStatement):
                target_env = None
                for env in reversed(env_stack):
                    if stmt.name in env:
                        target_env = env
                        break
                if target_env is None:
                    raise DexCompileError(
                        f"increment/decrement of undefined variable '{stmt.name}'",
                        lineno=stmt.lineno,
                    )
                if target_env.get(stmt.name) == "string":
                    raise DexCompileError(
                        f"cannot increment/decrement string variable '{stmt.name}'",
                        lineno=stmt.lineno,
                    )
                target_env[stmt.name] = "number"
                op = "+=" if stmt.delta > 0 else "-="
                lines.append(f"{prefix}{stmt.name} {op} {abs(stmt.delta)};")
            else:
                raise DexCompileError("unsupported statement during code generation", lineno=stmt.lineno)
        return lines


def compile_with_gcc(logger: Logger, output_c: str, runtime_path: str, binary_path: str) -> None:
    if shutil.which("gcc") is None:
        raise DexCompileError("gcc is not installed or not on PATH")
    compile_cmd = ["gcc", output_c, runtime_path, "-o", binary_path, "-lm"]
    logger.log("gcc", f"running: {' '.join(compile_cmd)}")
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise DexCompileError("gcc invocation failed") from exc
    if result.returncode != 0:
        raise DexCompileError(result.stderr.strip() or "gcc compilation failed")
    display = binary_path if os.path.isabs(binary_path) else os.path.join(".", binary_path)
    logger.log("gcc", f"-> {display}")


def prompt_build_mode() -> str:
    print("Choose output mode:")
    print("1 - Generate raw C code only")
    print("2 - Compile to Windows .exe")
    print("3 - Compile to Linux executable")
    while True:
        choice = input("Enter choice (1/2/3): ").strip()
        if choice in {"1", "2", "3"}:
            print()
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def compile_source(source_path: str, output_c: str, logger: Logger, display_path: Optional[str] = None) -> None:
    log_path = display_path or source_path
    logger.log("parse", f"reading file: {log_path}")
    with open(source_path, "r", encoding="utf-8") as handler:
        source = handler.read()
    parser = Parser(source, logger)
    program = parser.parse()
    logger.log("parse", f"{parser.line_total} lines")
    generator = CodeGenerator(program, logger)
    c_code = generator.generate()
    logger.log("compile", f"generating {output_c}")
    with open(output_c, "w", encoding="utf-8") as handler:
        handler.write(c_code)
    logger.log("compile", f"generated {output_c}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="DexLang compiler v2")
    parser.add_argument("source", nargs="?", help="path to DexLang source file")
    parser.add_argument("--out", default="out.c", help="path for generated C source")
    parser.add_argument(
        "--exe",
        default="out.exe",
        help="path for compiled Windows executable output",
    )
    parser.add_argument(
        "--linux-exe",
        default="out",
        help="path for compiled Linux executable output",
    )
    parser.add_argument(
        "--mode",
        choices=["1", "2", "3"],
        help="optional build mode to skip interactive prompt",
    )
    args = parser.parse_args(argv)

    raw_source = args.source
    if raw_source is None:
        raw_source = input("Enter .dex file path: ").strip()
    if not raw_source:
        print("error: file not found -> (empty)")
        return 1
    expanded_source = os.path.expanduser(raw_source)
    if not os.path.exists(expanded_source):
        print(f"error: file not found -> {raw_source}")
        return 1
    display_source = os.path.abspath(expanded_source)

    logger = Logger()
    logger.log("init", "DexLang compiler v2")
    runtime_status = "found" if os.path.exists("runtime.c") else "missing"
    logger.log("load", f"runtime.c {runtime_status}")
    logger.log("dexlang", f"compiling {display_source}")

    try:
        compile_source(expanded_source, args.out, logger, display_path=display_source)
        mode = args.mode or prompt_build_mode()
        logger.log("build", f"mode {mode}")
        if mode == "2":
            compile_with_gcc(logger, args.out, "runtime.c", args.exe)
        elif mode == "3":
            compile_with_gcc(logger, args.out, "runtime.c", args.linux_exe)
    except DexCompileError as exc:
        if exc.lineno is not None:
            print(f"error(line {exc.lineno}): {exc}", file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        logger.done(exit_code=1)
        return 1
    except KeyboardInterrupt:
        print("error: build cancelled", file=sys.stderr)
        logger.done(exit_code=1)
        return 1

    logger.done(exit_code=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
