import os
import ast
import json
import re
import subprocess
import tempfile
from pathlib import Path

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from typing import Dict, List

load_dotenv()

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# ===== Utilidades mínimas =====
def java_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')

def infer_java_literal(node):
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            return f"\"{java_escape(v)}\""
        if v is None:
            return "null"
    return None

def as_identifier(id_):
    # evitamos palabras raras
    return id_

# ===== Transpilador por reglas (AST) =====
class PyToJava(ast.NodeVisitor):
    """
    Transpilador simple por reglas:
      - Soporta: def, return, if/elif/else, while, for (range), asignaciones, print, len, operaciones, comparaciones.
      - Listas: crea ArrayList "raw" (sin genéricos).
      - Variables locales: usa 'var' cuando es posible (Java >=10). Si tu profe quiere Java 8,
        puedes cambiar 'var' por tipos como 'double' o 'Object'.
    """
    def __init__(self, class_name="TranspiledExample"):
        self.class_name = class_name
        self.lines = []
        self.current_indent = 0
        self.in_method = False
        self.locals_stack = []  # por método
        self.declared_current = set()
        self.imports = set()
        self.types_stack = []
        self.current_types = {}
        self.function_return_types = {}
        self.current_function_name = None
        self.current_function_return_kind = None
        self.scanner_initialized = []

    # helpers de indent
    def emit(self, s):
        self.lines.append("    " * self.current_indent + s)

    def blank_line(self):
        if self.lines and self.lines[-1] != "":
            self.lines.append("")

    def push_locals(self):
        self.locals_stack.append(set())
        self.types_stack.append({})
        self.scanner_initialized.append(False)
        self.declared_current = self.locals_stack[-1]
        self.current_types = self.types_stack[-1]

    def pop_locals(self):
        self.locals_stack.pop()
        self.types_stack.pop()
        self.scanner_initialized.pop()
        self.declared_current = self.locals_stack[-1] if self.locals_stack else set()
        self.current_types = self.types_stack[-1] if self.types_stack else {}

    def ensure_scanner_available(self):
        if not self.scanner_initialized:
            return
        if self.scanner_initialized[-1]:
            return
        self.imports.add("java.util.Scanner")
        self.emit("Scanner scanner = new Scanner(System.in);")
        self.declared_current.add("scanner")
        self.current_types["scanner"] = "scanner"
        self.scanner_initialized[-1] = True

    def translate_input_call(self, node: ast.Call) -> str:
        self.ensure_scanner_available()
        if node.args:
            prompt_expr = self.expr(node.args[0])
            self.emit(f"System.out.println({prompt_expr});")
        return "scanner.nextLine()"

    def declare_local_if_needed(self, name, expr_java, value_node=None):
        inferred_kind = self._infer_expr_kind(value_node)
        lowered = expr_java.strip()
        if inferred_kind in (None, "object"):
            if lowered.startswith("new ArrayList"):
                if "Double" in lowered or all(self._is_numeric_literal(elt) for elt in getattr(value_node, "elts", [])):
                    inferred_kind = "list_numeric"
                else:
                    inferred_kind = "list"
        decl = self._declaration_for_kind(inferred_kind)
        if name not in self.declared_current:
            if decl is None:
                decl = "var"
            self.emit(f"{decl} {name} = {expr_java};")
            self.declared_current.add(name)
        else:
            self.emit(f"{name} = {expr_java};")
        if inferred_kind:
            self.current_types[name] = inferred_kind

    # ---- visit ----
    def visit_Module(self, node):
        body_lines = []
        prev_lines, self.lines = self.lines, body_lines

        self.emit(f"public class {self.class_name} " + "{")
        self.current_indent += 1

        has_top_level_code = any(not isinstance(stmt, ast.FunctionDef) for stmt in node.body)

        self.emit("public static void main(String[] args) {")
        self.current_indent += 1

        if has_top_level_code:
            self.in_method = True
            self.push_locals()
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef):
                    continue
                self.visit(stmt)
            self.pop_locals()
            self.in_method = False
        else:
            self.emit("// TODO: invoca tus funciones aquí si es necesario")

        self.current_indent -= 1
        self.emit("}")

        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                self.visit(stmt)

        self.current_indent -= 1
        self.emit("}")

        java_lines = self.lines
        self.lines = prev_lines

        header = []
        if self.imports:
            header.extend(sorted(f"import {imp};" for imp in self.imports))
            header.append("")
        self.lines.extend(header + java_lines)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Inferimos tipos básicos de los parámetros según su uso
        arg_types = self._infer_arg_types(node)
        args = []
        for a in node.args.args:
            name = as_identifier(a.arg)
            inferred = arg_types.get(a.arg, "object")
            if inferred == "numeric":
                decl = "double"
            elif inferred == "list_numeric":
                decl = "List<Double>"
                self.imports.add("java.util.List")
            elif inferred == "list":
                decl = "List<?>"
                self.imports.add("java.util.List")
            else:
                decl = "Object"
            args.append(f"{decl} {name}")
        args_s = ", ".join(args) if args else ""

        previous_function = self.current_function_name
        previous_return_kind = self.current_function_return_kind
        self.current_function_name = node.name
        self.current_function_return_kind = None
        self.function_return_types.setdefault(node.name, "object")

        signature_index = len(self.lines)
        signature_template = f"public static __RET__ {node.name}({args_s}) " + "{"
        self.emit(signature_template)
        self.current_indent += 1
        self.in_method = True
        self.push_locals()
        for a in node.args.args:
            name = as_identifier(a.arg)
            self.declared_current.add(name)
            self.current_types[name] = arg_types.get(a.arg, "object")

        for stmt in node.body:
            self.visit(stmt)

        # si nunca se retornó nada:
        if not self._function_has_explicit_return(node):
            self.emit("return null;")

        self.pop_locals()
        self.in_method = False
        self.current_indent -= 1
        self.emit("}")
        self.blank_line()

        resolved_kind = self.current_function_return_kind or "object"
        self.function_return_types[node.name] = resolved_kind
        return_type = self._return_type_for_kind(resolved_kind)
        self.lines[signature_index] = self.lines[signature_index].replace("__RET__", return_type)

        self.current_function_name = previous_function
        self.current_function_return_kind = previous_return_kind

    def _function_has_explicit_return(self, node: ast.FunctionDef) -> bool:
        class ReturnFinder(ast.NodeVisitor):
            def __init__(self):
                self.found = False

            def visit_Return(self, _):
                self.found = True

            def generic_visit(self, n):
                if not self.found:
                    super().generic_visit(n)

            def visit_FunctionDef(self, _):
                # no profundizamos en funciones anidadas
                pass

            def visit_AsyncFunctionDef(self, _):
                pass

            def visit_Lambda(self, _):
                pass

        finder = ReturnFinder()
        for stmt in node.body:
            if finder.found:
                break
            finder.visit(stmt)
        return finder.found

    def visit_Return(self, node: ast.Return):
        kind = self._infer_expr_kind(node.value)
        self.current_function_return_kind = self._merge_kinds(self.current_function_return_kind, kind)
        expr = self.expr(node.value)
        self.emit(f"return {expr};")

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "print":
            args = [self.expr(arg) for arg in node.value.args]
            joined = " + \" \" + ".join(args) if len(args) > 1 else (args[0] if args else "\"\"")
            self.emit(f"System.out.println({joined});")
            return
        expr_java = self.expr(node.value)
        if expr_java.startswith("/*"):
            self.emit(expr_java)
        else:
            self.emit(f"{expr_java};")

    def visit_Call(self, node: ast.Call):
        self.emit(f"// Llamada no mapeada: {ast.unparse(node)}")

    def visit_Assign(self, node: ast.Assign):
        # soporta: x = expr / a, b = ...
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = as_identifier(node.targets[0].id)
            value_java = self.expr(node.value)
            self.declare_local_if_needed(name, value_java, node.value)
        else:
            self.emit(f"// Asignación compleja no mapeada: {ast.unparse(node)}")

    def visit_If(self, node: ast.If):
        test_java = self.expr(node.test)
        self.emit(f"if ({test_java}) " + "{")
        self.current_indent += 1
        for s in node.body:
            self.visit(s)
        self.current_indent -= 1
        self.emit("}")
        if node.orelse:
            # elif es If dentro de orelse; simplificamos a else { ... }
            self.emit("else {")
            self.current_indent += 1
            for s in node.orelse:
                self.visit(s)
            self.current_indent -= 1
            self.emit("}")

    def visit_While(self, node: ast.While):
        test_java = self.expr(node.test)
        self.emit(f"while ({test_java}) " + "{")
        self.current_indent += 1
        for s in node.body:
            self.visit(s)
        self.current_indent -= 1
        self.emit("}")

    def visit_For(self, node: ast.For):
        # for i in range(a, b, step?)
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            # casos: range(stop) | range(start, stop) | range(start, stop, step)
            args = node.iter.args
            v = as_identifier(node.target.id) if isinstance(node.target, ast.Name) else "i"
            if len(args) == 1:
                stop = self.expr(args[0])
                self.emit(f"for (int {v} = 0; {v} < {stop}; {v}++) " + "{")
            elif len(args) == 2:
                start = self.expr(args[0])
                stop = self.expr(args[1])
                self.emit(f"for (int {v} = (int)({start}); {v} < {stop}; {v}++) " + "{")
            elif len(args) == 3:
                start = self.expr(args[0])
                stop = self.expr(args[1])
                step = self.expr(args[2])
                # para step negativo habría que ajustar; lo simple:
                comparator = ">" if self._is_negative_step(args[2]) else "<"
                self.emit(f"for (int {v} = (int)({start}); {v} {comparator} {stop}; {v} += {step}) " + "{")
            else:
                self.emit(f"// range con argumentos no soportados: {ast.unparse(node)}")
                return
            self.current_indent += 1
            for s in node.body:
                self.visit(s)
            self.current_indent -= 1
            self.emit("}")
        else:
            self.emit(f"// for no soportado (iteración no-range): {ast.unparse(node)}")

    # ======== Expresiones ========
    def expr(self, node):
        if node is None:
            return "null"

        # literales simples
        lit = infer_java_literal(node)
        if lit is not None:
            return lit

        if isinstance(node, ast.Name):
            return as_identifier(node.id)

        if isinstance(node, ast.BinOp):
            left = self.expr(node.left)
            right = self.expr(node.right)
            op = self.binop(node.op)
            return f"({left} {op} {right})"

        if isinstance(node, ast.UnaryOp):
            operand = self.expr(node.operand)
            op = self.unop(node.op)
            return f"({op}{operand})"

        if isinstance(node, ast.BoolOp):
            vals = [self.expr(v) for v in node.values]
            op = " && " if isinstance(node.op, ast.And) else " || "
            return "(" + op.join(vals) + ")"

        if isinstance(node, ast.Compare):
            left = self.expr(node.left)
            comps = []
            for op, comparator in zip(node.ops, node.comparators):
                right = self.expr(comparator)
                comps.append(f"{left} {self.cmpop(op)} {right}")
                left = right
            return "(" + " && ".join(comps) + ")"

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_id = node.func.id
                if func_id == "len" and len(node.args) == 1:
                    target_node = node.args[0]
                    t = self.expr(target_node)
                    if isinstance(target_node, ast.Name) and self.current_types.get(target_node.id) in {"list", "list_numeric"}:
                        return f"({t}).size()"
                    self.imports.add("java.util.List")
                    return f"((List<?>){self._wrap_for_cast(t)}).size()"
                if func_id == "input":
                    return self.translate_input_call(node)
                if func_id in {"int", "float", "str"} and len(node.args) == 1 and not node.keywords:
                    arg_node = node.args[0]
                    from_input = (
                        isinstance(arg_node, ast.Call)
                        and isinstance(arg_node.func, ast.Name)
                        and arg_node.func.id == "input"
                    )
                    if from_input:
                        input_expr = self.translate_input_call(arg_node)
                    else:
                        input_expr = self.expr(arg_node)
                    if func_id == "int":
                        return (
                            f"Integer.parseInt({input_expr})" if from_input else f"(int)({input_expr})"
                        )
                    if func_id == "float":
                        return (
                            f"Double.parseDouble({input_expr})" if from_input else f"(double)({input_expr})"
                        )
                    if func_id == "str":
                        return f"String.valueOf({input_expr})"
                if node.keywords:
                    return f"/*call*/ {java_escape(ast.unparse(node))}"
                func_name = as_identifier(func_id)
                args = ", ".join(self.expr(arg) for arg in node.args)
                return f"{func_name}({args})"
            # por defecto:
            return f"/*call*/ {java_escape(ast.unparse(node))}"

        if isinstance(node, ast.List):
            # ArrayList raw
            items = [self._prepare_list_element(elt) for elt in node.elts]
            if items:
                self.imports.update({"java.util.ArrayList", "java.util.Arrays"})
                return f"new ArrayList<>(Arrays.asList({', '.join(items)}))"
            self.imports.add("java.util.ArrayList")
            return "new ArrayList<>()"

        if isinstance(node, ast.Subscript):
            # a[i]
            target_node = node.value
            tgt = self.expr(target_node)
            idx = self.expr(node.slice)
            if isinstance(target_node, ast.Name) and self.current_types.get(target_node.id) in {"list", "list_numeric"}:
                return f"({tgt}).get((int)({idx}))"
            self.imports.add("java.util.List")
            return f"((List<?>){self._wrap_for_cast(tgt)}).get((int)({idx}))"

        if isinstance(node, ast.Attribute):
            # obj.attr -> lo dejamos como comentario (depende del tipo)
            return f"/*attr*/ {java_escape(ast.unparse(node))}"

        # por defecto
        return f"/*expr*/ {java_escape(ast.unparse(node))}"

    def binop(self, op):
        return {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.Pow: "**",  # (no mapeado directo)
            ast.FloorDiv: "/", # simplificación
        }.get(type(op), "?")

    def unop(self, op):
        return {
            ast.UAdd: "+",
            ast.USub: "-",
            ast.Not: "!",
        }.get(type(op), "")

    def cmpop(self, op):
        return {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Is: "==",
            ast.IsNot: "!=",
            ast.In: "/* in */",
            ast.NotIn: "/* not in */",
        }.get(type(op), "/* ? */")

    def _is_negative_step(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value < 0
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            val = node.operand.value
            if isinstance(val, (int, float)):
                return True
        return False

    def _infer_arg_types(self, node: ast.FunctionDef):
        tracked = {a.arg for a in node.args.args}
        visitor = _ArgUsageVisitor(tracked)
        for stmt in node.body:
            visitor.visit(stmt)
        result = {}
        for name, usage in visitor.usage.items():
            if "list_numeric" in usage:
                result[name] = "list_numeric"
            elif "list" in usage:
                result[name] = "list"
            elif "numeric" in usage:
                result[name] = "numeric"
            else:
                result[name] = "object"
        return result

    def _wrap_for_cast(self, expr_java: str) -> str:
        expr_java = expr_java.strip()
        if expr_java.startswith("(") and expr_java.endswith(")"):
            return expr_java
        return f"({expr_java})"

    def _is_numeric_literal(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            return isinstance(node.operand.value, (int, float))
        return False

    def _infer_expr_kind(self, node):
        if node is None:
            return "object"
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
            if isinstance(node.value, int):
                return "int"
            if isinstance(node.value, float):
                return "float"
            if isinstance(node.value, str):
                return "string"
            return "object"
        if isinstance(node, ast.List):
            if all(self._infer_expr_kind(elt) == "numeric" for elt in node.elts):
                return "list_numeric"
            return "list"
        if isinstance(node, ast.Name):
            return self.current_types.get(node.id, "object")
        if isinstance(node, ast.BinOp):
            left = self._infer_expr_kind(node.left)
            right = self._infer_expr_kind(node.right)
            if "string" in {left, right}:
                return "string"
            numeric_like = {"numeric", "float", "int"}
            if left in numeric_like and right in numeric_like:
                if left == right == "int":
                    return "int"
                if left == right == "float":
                    return "float"
                if "float" in {left, right}:
                    return "float"
                if "numeric" in {left, right}:
                    return "numeric"
                return "numeric"
            if left.startswith("list") or right.startswith("list"):
                return left if left.startswith("list") else right if right.startswith("list") else "object"
            return "object"
        if isinstance(node, ast.UnaryOp):
            return self._infer_expr_kind(node.operand)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == "len":
                    return "numeric"
                if node.func.id == "input":
                    return "string"
                if node.func.id == "int":
                    return "int"
                if node.func.id == "float":
                    return "float"
                if node.func.id == "str":
                    return "string"
                if node.func.id == self.current_function_name and self.current_function_return_kind:
                    return self.current_function_return_kind
                if node.func.id in self.function_return_types:
                    return self.function_return_types[node.func.id]
            return "object"
        if isinstance(node, ast.Compare):
            return "bool"
        if isinstance(node, ast.BoolOp):
            return "bool"
        if isinstance(node, ast.Subscript):
            target_kind = self._infer_expr_kind(node.value)
            if target_kind == "list_numeric":
                return "numeric"
            if target_kind == "list":
                return "object"
            return "object"
        return "object"

    def _declaration_for_kind(self, kind):
        if kind == "int":
            return "int"
        if kind in {"numeric", "float"}:
            return "double"
        if kind == "bool":
            return "boolean"
        if kind == "string":
            return "String"
        if kind == "list_numeric":
            self.imports.update({"java.util.List"})
            return "List<Double>"
        if kind == "list":
            self.imports.update({"java.util.List"})
            return "List<Object>"
        return None

    def _return_type_for_kind(self, kind):
        decl = self._declaration_for_kind(kind)
        return decl or "Object"

    def _merge_kinds(self, current, new_kind):
        if new_kind is None:
            return current or "object"
        if current is None:
            return new_kind
        if current == new_kind:
            return current
        if {current, new_kind} == {"list_numeric", "numeric"}:
            return "numeric"
        if current == "list_numeric" or new_kind == "list_numeric":
            return "object"
        numeric_like = {"numeric", "float", "int"}
        if current in numeric_like and new_kind in numeric_like:
            if "numeric" in {current, new_kind}:
                return "numeric"
            if {current, new_kind} == {"int", "float"}:
                return "float"
            if current == new_kind:
                return current
            return "numeric"
        if {current, new_kind} == {"object", "numeric"}:
            return "object"
        if {current, new_kind} == {"object", "bool"}:
            return "object"
        if {current, new_kind} == {"object", "string"}:
            return "object"
        return "object"

    def _prepare_list_element(self, node):
        lit = infer_java_literal(node)
        if lit is not None:
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                return f"{float(node.value)}"
            return lit
        if isinstance(node, ast.Name) and self.current_types.get(node.id) in {"numeric", "float", "int"}:
            return f"Double.valueOf({node.id})"
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
        ):
            return f"{float(-node.operand.value)}"
        return self.expr(node)


class _ArgUsageVisitor(ast.NodeVisitor):
    def __init__(self, tracked):
        self.tracked = set(tracked)
        self.usage = {name: set() for name in tracked}

    def mark(self, name, kind):
        if name in self.usage:
            self.usage[name].add(kind)

    def visit_BinOp(self, node: ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv)):
            for side in (node.left, node.right):
                if isinstance(side, ast.Name):
                    self.mark(side.id, "numeric")
                elif isinstance(side, ast.Subscript) and isinstance(side.value, ast.Name):
                    self.mark(side.value.id, "list_numeric")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = node.operand
            if isinstance(operand, ast.Name):
                self.mark(operand.id, "numeric")
            elif isinstance(operand, ast.Subscript) and isinstance(operand.value, ast.Name):
                self.mark(operand.value.id, "list_numeric")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        operands = [node.left, *node.comparators]
        for operand in operands:
            if isinstance(operand, ast.Name):
                self.mark(operand.id, "numeric")
            elif isinstance(operand, ast.Subscript) and isinstance(operand.value, ast.Name):
                self.mark(operand.value.id, "list_numeric")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id == "len" and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                self.mark(node.args[0].id, "list")
            if node.func.id == "range":
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        self.mark(arg.id, "numeric")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name):
            self.mark(node.value.id, "list")
        if isinstance(node.slice, ast.Name):
            self.mark(node.slice.id, "numeric")
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if isinstance(node.iter, ast.Name):
            self.mark(node.iter.id, "list")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # no profundizamos en funciones anidadas
        return

    def visit_AsyncFunctionDef(self, node):
        return


# ===== Fallback IA (opcional) =====
def ai_translate_python_to_java(py_code: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada en .env")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    prompt = f"""
Transpila el siguiente código Python a Java moderno (Java 17).
- Genera una clase ejecutable con main si corresponde.
- Tipos explícitos correctos; usa ArrayList y HashMap cuando apliquen.
- No expliques nada, solo entrega el archivo Java completo y compilable.

Código Python:
"""
    data = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "Eres un transpiler Python→Java confiable y estricto."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }
    try:
        r = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
        r.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        status = http_err.response.status_code if http_err.response else None
        if status == 401:
            raise RuntimeError(
                "La clave de OpenAI es inválida o el modelo configurado no está autorizado."
            ) from http_err
        raise RuntimeError(f"La API de OpenAI respondió con un error {status}: {http_err.response.text if http_err.response else http_err}") from http_err
    except requests.exceptions.RequestException as req_err:
        raise RuntimeError(f"No se pudo contactar la API de OpenAI: {req_err}") from req_err

    out = r.json()["choices"][0]["message"]["content"]
    # limpiamos fences si vienen
    if out.strip().startswith("```"):
        out = out.split("```", 2)[1]
        # puede venir como "java\n<code>"
        out = out.split("\n", 1)[1] if "\n" in out else out
    return out.strip()


def _summarize_ai_failure(exc: Exception) -> str:
    """Devuelve un mensaje entendible para el usuario sin códigos HTTP crudos."""

    lowered = str(exc).strip().lower()
    if not lowered:
        return "La traducción asistida por IA no está disponible en este momento."

    keywords = ("openai", "api", "clave", "unauthorized", "401", "key")
    if any(token in lowered for token in keywords):
        return "La traducción asistida por IA no está disponible: revisa la clave y el modelo configurados."

    return "La traducción asistida por IA encontró un error inesperado. Intenta nuevamente en unos minutos."


def _attempt_ai_translation(py_code: str):
    """Ejecuta la ruta IA y devuelve (java_code, analysis, warning)."""

    if not OPENAI_API_KEY:
        return None, None, None

    try:
        ai_code = ai_translate_python_to_java(py_code)
    except Exception as exc:  # noqa: BLE001 - queremos capturar y resumir cualquier fallo
        return None, None, _summarize_ai_failure(exc)

    ai_analysis = _analyze_java_output(ai_code)
    return ai_code, ai_analysis, None


def _validate_with_javac(java_code: str) -> List[str]:
    """Compila el código generado para comprobar si es válido."""

    try:
        class_name = _extract_java_class_name(java_code) or "TranspiledExample"
        with tempfile.TemporaryDirectory() as tmpdir:
            java_path = Path(tmpdir) / f"{class_name}.java"
            java_path.write_text(java_code, encoding="utf-8")
            try:
                result = subprocess.run(
                    ["javac", str(java_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except FileNotFoundError:
                return ["javac no está disponible para validar el código Java generado."]
            except subprocess.TimeoutExpired:
                return ["La validación automática con javac excedió el tiempo límite permitido."]

            if result.returncode != 0:
                return _clean_javac_errors(result.stderr, java_path, class_name)
    except Exception as exc:
        return [f"No se pudo validar el código Java con javac: {exc}"]

    return []


def _clean_javac_errors(stderr: str, java_path: Path, class_name: str) -> List[str]:
    """Formatea los errores de javac para que sean más legibles."""

    clean: List[str] = []
    replacement = f"{class_name}.java"
    path_str = str(java_path)
    for raw_line in stderr.splitlines():
        line = raw_line.strip()
        if not line or line == "^":
            continue
        line = line.replace(path_str, replacement)
        clean.append(line)
    if clean:
        clean.insert(0, "El código Java generado no compila correctamente:")
    return clean


def _extract_java_class_name(java_code: str):
    match = re.search(r"public\s+class\s+(\w+)", java_code)
    if match:
        return match.group(1)
    return None


def _analyze_java_output(java_code: str) -> Dict[str, List[str]]:
    """Detecta problemas comunes en el Java generado."""

    warnings: List[str] = []
    errors: List[str] = []

    marker_map = (
        ("/*expr*/", "Se encontraron expresiones que no pudieron convertirse automáticamente.", "error"),
        ("/*call*/", "Hay llamadas a funciones que no se reconocieron.", "error"),
        ("/*attr*/", "Hay atributos u objetos sin conversión directa.", "warning"),
        ("// Llamada no mapeada", "Existen llamadas que no fueron mapeadas a Java.", "error"),
        ("// Asignación compleja no mapeada", "Se detectaron asignaciones complejas sin conversión.", "error"),
        ("// range con argumentos no soportados", "Se usa range con argumentos que no se pudieron convertir.", "error"),
        ("// for no soportado", "Existe un bucle for que no se pudo convertir.", "error"),
    )

    for marker, message, severity in marker_map:
        if marker in java_code:
            (errors if severity == "error" else warnings).append(message)

    if "public class" not in java_code:
        errors.append("No se encontró una declaración de clase pública en el resultado.")

    if "public static void main" not in java_code:
        warnings.append("No se detectó un método main; verifica si necesitas un punto de entrada.")

    if java_code.count("{") != java_code.count("}"):
        errors.append("Las llaves de apertura y cierre no están balanceadas en el código Java generado.")

    stripped = java_code.strip()
    if not stripped or stripped.startswith("// Error"):
        errors.append("No se obtuvo código Java ejecutable a partir de la entrada proporcionada.")

    compile_feedback = _validate_with_javac(java_code)
    if compile_feedback:
        errors.extend(compile_feedback)

    return {"warnings": warnings, "errors": errors}


def _format_validation_response(java_code: str, engine: str, analysis: Dict[str, List[str]]):
    payload = {
        "javaCode": java_code,
        "engine": engine,
        "warnings": analysis.get("warnings", []),
    }
    return jsonify(payload)


@app.route("/api/transpile", methods=["POST"])
def transpile():
    """
    1) Intento por AST (rápido, local).
    2) Si falla o queda muy incompleto, y hay API key: fallback IA.
    """
    data = request.get_json(force=True)
    code = data.get("code", "") if isinstance(data, dict) else ""

    if not isinstance(code, str):
        return jsonify({"error": "Formato de solicitud inválido. Se esperaba un texto con código Python."}), 400

    if not code.strip():
        return jsonify({"error": "El código Python está vacío. Agrega contenido para continuar."}), 400

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (
            jsonify(
                {
                    "error": "El código Python contiene un error de sintaxis.",
                    "details": f"Línea {e.lineno}: {e.msg}",
                }
            ),
            400,
        )
    except Exception as e:
        return jsonify({"error": f"No se pudo analizar el código Python: {e}"}), 400

    try:
        tr = PyToJava()
        tr.visit(tree)
        java_code = "\n".join(tr.lines).strip()
        analysis = _analyze_java_output(java_code)

        if analysis["errors"]:
            ai_code, ai_analysis, ai_warning = _attempt_ai_translation(code)
            if ai_code and ai_analysis and not ai_analysis["errors"]:
                response = _format_validation_response(ai_code, "ai-fallback", ai_analysis)
                response.status_code = 200
                return response

            issues = list(analysis["errors"])
            warnings = list(analysis["warnings"])

            if ai_analysis:
                issues.extend(ai_analysis.get("errors", []))
                warnings.extend(ai_analysis.get("warnings", []))

            if ai_warning:
                warnings.append(ai_warning)

            error_message = "El código contiene construcciones que no se pudieron convertir de forma segura."
            if ai_analysis and ai_analysis.get("errors"):
                error_message = "Ninguno de los motores pudo generar Java válido."

            return jsonify({"error": error_message, "issues": issues, "warnings": warnings}), 422

        response = _format_validation_response(java_code, "ast", analysis)
        response.status_code = 200
        return response

    except Exception:
        ai_code, ai_analysis, ai_warning = _attempt_ai_translation(code)
        if ai_code and ai_analysis and not ai_analysis["errors"]:
            response = _format_validation_response(ai_code, "ai-fallback", ai_analysis)
            response.status_code = 200
            return response

        issues = ["El motor local no pudo procesar el código proporcionado."]
        warnings = []
        error_message = "No se pudo completar la transpilación automática."
        status_code = 400

        if ai_analysis:
            issues.extend(ai_analysis.get("errors", []))
            warnings.extend(ai_analysis.get("warnings", []))
            status_code = 422 if ai_analysis.get("errors") else status_code
            if ai_analysis.get("errors"):
                error_message = "Ninguno de los motores pudo generar Java válido."
            else:
                error_message = "El motor local presentó un problema, pero la IA devolvió advertencias a revisar."

        if ai_warning:
            warnings.append(ai_warning)

        return jsonify({"error": error_message, "issues": issues, "warnings": warnings}), status_code


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "ai_enabled": bool(OPENAI_API_KEY)})


if __name__ == "__main__":
    app.run(port=8080, debug=True)
