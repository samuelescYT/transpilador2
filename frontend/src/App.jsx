import React, { useState } from "react";

export default function CapiTraduce() {
  const [pythonCode, setPythonCode] = useState(`# Ejemplo: función factorial\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\nprint(factorial(5))`);
  const [javaCode, setJavaCode] = useState("// Aquí aparecerá el código Java transpilado");
  const [processing, setProcessing] = useState(false);

  async function transpilar() {
    setProcessing(true);
    try {
      const res = await fetch("http://localhost:8080/api/transpile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: pythonCode }),
      });
      const data = await res.json();
      if (data.javaCode) setJavaCode(data.javaCode);
      else setJavaCode("// Error: " + data.error);
    } catch (e) {
      setJavaCode("// Error de conexión con el backend");
    }
    setProcessing(false);
  }

  function handleDownload() {
    const blob = new Blob([javaCode], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "TranspiledExample.java";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-100 to-slate-300 flex flex-col items-center justify-center p-6">
      {/* Contenedor principal */}
      <div className="bg-white shadow-2xl rounded-3xl w-full max-w-6xl p-8 border border-gray-200">
        {/* Encabezado */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-extrabold text-slate-800 flex justify-center items-center gap-3">
            <span role="img" aria-label="brain" className="text-5xl">🧠</span>
            Capi Traduce
          </h1>
          <p className="text-gray-500 text-sm">Transpilador Python → Java</p>
        </header>

        {/* Barra de controles */}
        <div className="flex flex-wrap justify-center gap-4 mb-8">
          <label className="flex items-center gap-2 px-4 py-2 bg-slate-50 text-gray-700 border border-gray-300 rounded-lg shadow-sm hover:shadow-md cursor-pointer transition">
            <span className="text-sm font-medium">📂 Archivo</span>
            <input type="file" accept=".py" className="hidden" />
          </label>

          <button
            onClick={handleDownload}
            className="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg shadow transition"
          >
            Descargar
          </button>

          <button
            onClick={transpilar}
            disabled={processing}
            className={`px-5 py-2 rounded-lg font-medium shadow transition ${
              processing
                ? "bg-gray-400 text-gray-800"
                : "bg-emerald-600 hover:bg-emerald-700 text-white"
            }`}
          >
            {processing ? "Traduciendo..." : "Traducir"}
          </button>
        </div>

        {/* Editores lado a lado */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Python */}
          <div>
            <h2 className="text-lg font-semibold text-gray-700 mb-2 text-center">Código Python</h2>
            <textarea
              value={pythonCode}
              onChange={(e) => setPythonCode(e.target.value)}
              className="w-full h-[34rem] p-4 border border-gray-300 rounded-lg bg-slate-50 font-mono text-base focus:outline-none focus:ring-2 focus:ring-emerald-400 resize-none shadow-inner"
            />
          </div>

          {/* Java */}
          <div>
            <h2 className="text-lg font-semibold text-gray-700 mb-2 text-center">Código Java</h2>
            <textarea
              value={javaCode}
              onChange={(e) => setJavaCode(e.target.value)}
              className="w-full h-[34rem] p-4 border border-gray-300 rounded-lg bg-slate-50 font-mono text-base focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none shadow-inner"
            />
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-sm text-gray-500">
          © 2025 — Proyecto académico • Universidad El Bosque
        </footer>
      </div>
    </div>
  );
}

