import os
import ast
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

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
            # Simplificación: números -> double, agregamos .0 si es int para mantener coherencia
            return str(float(v)) if isinstance(v, int) else str(v)
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

    # helpers de indent
    def emit(self, s):
        self.lines.append("    " * self.current_indent + s)

    def push_locals(self):
        self.locals_stack.append(set())
        self.declared_current = self.locals_stack[-1]

    def pop_locals(self):
        self.locals_stack.pop()
        self.declared_current = self.locals_stack[-1] if self.locals_stack else set()

    def declare_local_if_needed(self, name, expr_java):
        # si no está declarada, usar 'var' para declarar
        if name not in self.declared_current:
            self.emit(f"var {name} = {expr_java};")
            self.declared_current.add(name)
        else:
            self.emit(f"{name} = {expr_java};")

    # ---- visit ----
    def visit_Module(self, node):
        self.emit(f"public class {self.class_name} " + "{")
        self.current_indent += 1
        # punto de entrada (no siempre aplica, pero lo dejamos)
        self.emit("public static void main(String[] args) {")
        self.current_indent += 1
        self.emit("// TODO: invoca tus funciones aquí si es necesario")
        self.current_indent -= 1
        self.emit("}")
        # cuerpo
        for stmt in node.body:
            self.visit(stmt)

        self.current_indent -= 1
        self.emit("}")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Suponemos retorno double/object si no hay Return claro
        # Tipamos args como 'double' por simplicidad (se puede mejorar)
        args = []
        for a in node.args.args:
            args.append(f"double {as_identifier(a.arg)}")
        args_s = ", ".join(args) if args else ""

        self.emit(f"public static Object {node.name}({args_s}) " + "{")
        self.current_indent += 1
        self.in_method = True
        self.push_locals()

        for stmt in node.body:
            self.visit(stmt)

        # si nunca se retornó nada:
        self.emit("return null;")

        self.pop_locals()
        self.in_method = False
        self.current_indent -= 1
        self.emit("}")

    def visit_Return(self, node: ast.Return):
        expr = self.expr(node.value)
        self.emit(f"return {expr};")

    def visit_Expr(self, node: ast.Expr):
        # expresiones sueltas (como print)
        if isinstance(node.value, ast.Call):
            self.visit(node.value)
        else:
            # ignoramos otras expresiones solitarias
            pass

    def visit_Call(self, node: ast.Call):
        # print(...)
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            args_java = " + \" \" + ".join([self.expr(a) for a in node.args]) if len(node.args) > 1 else (self.expr(node.args[0]) if node.args else "\"\"")
            self.emit(f"System.out.println({args_java});")
            return

        # len(x)
        if isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1:
            target = self.expr(node.args[0])
            self.emit(f"System.out.println(({target}).size());")
            return

        # llamadas genéricas -> comentario
        self.emit(f"// Llamada no mapeada: {ast.unparse(node)}")

    def visit_Assign(self, node: ast.Assign):
        # soporta: x = expr / a, b = ...
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = as_identifier(node.targets[0].id)
            value_java = self.expr(node.value)
            self.declare_local_if_needed(name, value_java)
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
                self.emit(f"for (int {v} = (int)({start}); {v} < {stop}; {v}+={step}) " + "{")
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
            # len(x)
            if isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1:
                t = self.expr(node.args[0])
                return f"({t}).size()"
            # print en expr (raro), devolvemos string
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                args_java = " + \" \" + ".join([self.expr(a) for a in node.args]) if node.args else "\"\""
                return f"(String)({args_java})"
            # por defecto:
            return f"/*call*/ {java_escape(ast.unparse(node))}"

        if isinstance(node, ast.List):
            # ArrayList raw
            items = [self.expr(elt) for elt in node.elts]
            return f"new java.util.ArrayList(java.util.Arrays.asList({', '.join(items)}))"

        if isinstance(node, ast.Subscript):
            # a[i]
            tgt = self.expr(node.value)
            idx = self.expr(node.slice)
            return f"({tgt}).get((int)({idx}))"

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
    r = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    # limpiamos fences si vienen
    if out.strip().startswith("```"):
        out = out.split("```", 2)[1]
        # puede venir como "java\n<code>"
        out = out.split("\n", 1)[1] if "\n" in out else out
    return out.strip()


@app.route("/api/transpile", methods=["POST"])
def transpile():
    """
    1) Intento por AST (rápido, local).
    2) Si falla o queda muy incompleto, y hay API key: fallback IA.
    """
    data = request.get_json(force=True)
    code = data.get("code", "")

    # 1) AST
    try:
        tree = ast.parse(code)
        tr = PyToJava()
        tr.visit(tree)
        java_code = "\n".join(tr.lines)

        # Heurística: si el resultado tiene demasiados comentarios de "no mapeado",
        # intentamos el fallback si hay API key.
        if "no mapeado" in java_code or "/*call*/" in java_code or "/*attr*/" in java_code:
            if OPENAI_API_KEY:
                ai_code = ai_translate_python_to_java(code)
                return jsonify({"javaCode": ai_code, "engine": "ai-fallback"})
        return jsonify({"javaCode": java_code, "engine": "ast"})
    except Exception as e:
        # fallback IA si está disponible
        if OPENAI_API_KEY:
            try:
                ai_code = ai_translate_python_to_java(code)
                return jsonify({"javaCode": ai_code, "engine": "ai-fallback"})
            except Exception as e2:
                return jsonify({"error": f"AST y IA fallaron: {e} / {e2}"}), 400
        return jsonify({"error": f"AST falló: {e}"}), 400


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "ai_enabled": bool(OPENAI_API_KEY)})


if __name__ == "__main__":
    app.run(port=8080, debug=True)
