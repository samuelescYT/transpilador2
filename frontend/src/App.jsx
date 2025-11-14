import React, { useMemo, useRef, useState } from "react";
import CodeEditor from "./components/CodeEditor.jsx";

export default function CapiTraduce() {
  const [pythonCode, setPythonCode] = useState(`# Ejemplo: función factorial\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\nprint(factorial(5))`);
  const [javaCode, setJavaCode] = useState("// Aquí aparecerá el código Java transpilado");
  const [processing, setProcessing] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const fileInputRef = useRef(null);
  const apiBase = useMemo(() => {
    const configured = (import.meta.env?.VITE_API_BASE_URL || "").trim();
    if (!configured) {
      return "";
    }
    return configured.endsWith("/") ? configured.slice(0, -1) : configured;
  }, []);

  async function transpilar(sourceOverride) {
    const codeToSend =
      typeof sourceOverride === "string" ? sourceOverride : pythonCode;

    if (!codeToSend.trim()) {
      setFeedback({
        type: "warning",
        message: "Agrega código Python para iniciar la traducción.",
        details: [],
      });
      return;
    }

    setProcessing(true);
    setFeedback(null);
    try {
      const endpoint = apiBase ? `${apiBase}/api/transpile` : "/api/transpile";
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: codeToSend }),
      });
      const data = await res.json();
      if (!res.ok) {
        setJavaCode((prev) => prev || "// No se pudo generar código Java");
        const issues = Array.isArray(data.issues) ? data.issues : [];
        const warnings = Array.isArray(data.warnings) ? data.warnings : [];
        setFeedback({
          type: "error",
          message: data.error || "La transpilación no pudo completarse.",
          details: [...issues, ...warnings],
        });
      } else {
        setJavaCode(data.javaCode || "// El backend no devolvió código Java");
        setFeedback({
          type: data.warnings && data.warnings.length ? "warning" : "success",
          message:
            data.engine === "ai-fallback"
              ? "El código se generó usando el asistente inteligente."
              : "Transpilación realizada localmente.",
          details: data.warnings || [],
        });
      }
    } catch (e) {
      setJavaCode("// Error de conexión con el backend");
      setFeedback({
        type: "error",
        message: "No se pudo contactar el backend.",
        details: [e.message],
      });
    }
    setProcessing(false);
  }

  function readFileAsText(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : "");
      reader.onerror = () =>
        reject(
          reader.error || new Error("No se pudo leer el archivo seleccionado.")
        );
      reader.readAsText(file);
    });
  }

  async function handleFileUpload(event) {
    const file = event.target.files && event.target.files.length ? event.target.files[0] : null;
    if (!file) {
      return;
    }

    try {
      const text = await readFileAsText(file);
      setUploadedFileName(file.name);
      setPythonCode(text);
      await transpilar(text);
    } catch (err) {
      setFeedback({
        type: "error",
        message: "No se pudo leer el archivo seleccionado.",
        details: [err.message],
      });
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
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

  const feedbackStyles = useMemo(() => {
    if (!feedback) return "";
    switch (feedback.type) {
      case "success":
        return "bg-emerald-100 text-emerald-900 border border-emerald-200";
      case "warning":
        return "bg-amber-100 text-amber-900 border border-amber-200";
      default:
        return "bg-rose-100 text-rose-900 border border-rose-200";
    }
  }, [feedback]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-100 via-slate-200 to-slate-300 flex flex-col items-center justify-center p-6">
      {/* Contenedor principal */}
      <div className="bg-white/95 backdrop-blur shadow-[0_35px_120px_-45px_rgba(15,23,42,0.45)] rounded-[2.5rem] w-full max-w-6xl p-8 border border-white/60">
        {/* Encabezado */}
        <header className="text-center mb-8 space-y-2">
          <div className="flex items-center justify-center gap-3">
            <span role="img" aria-label="brain" className="text-5xl drop-shadow">🧠</span>
            <h1 className="text-4xl font-black text-slate-800 tracking-tight">Capi Traduce</h1>
          </div>
          <p className="text-gray-500 text-sm uppercase tracking-[0.35em]">Transpilador Python → Java</p>
        </header>

        {/* Barra de controles */}
        <div className="flex flex-wrap justify-center gap-4 mb-6">
          <input
            ref={fileInputRef}
            type="file"
            accept=".py,.txt"
            className="hidden"
            onChange={handleFileUpload}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
            className="flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50/70 px-4 py-2 text-gray-700 shadow-sm transition hover:shadow-md"
          >
            <span className="text-sm font-medium">📂 Archivo</span>
            <span className="text-xs text-slate-500 max-w-[12rem] truncate">
              {uploadedFileName || "Selecciona un archivo .py"}
            </span>
          </button>

          <button
            onClick={handleDownload}
            className="rounded-xl bg-blue-600 px-5 py-2 text-white shadow transition hover:bg-blue-700"
          >
            Descargar
          </button>

          <button
            onClick={transpilar}
            disabled={processing}
            className={`rounded-xl px-5 py-2 font-semibold shadow transition ${
              processing ? "bg-gray-400 text-gray-700" : "bg-emerald-600 text-white hover:bg-emerald-700"
            }`}
          >
            {processing ? "Traduciendo..." : "Traducir"}
          </button>
        </div>

        {feedback ? (
          <div className={`mb-6 rounded-2xl px-6 py-4 text-sm shadow-sm ${feedbackStyles}`}>
            <div className="flex items-center justify-between gap-3">
              <p className="font-semibold">{feedback.message}</p>
              <span className="rounded-full bg-white/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                {processing ? "..." : "Listo"}
              </span>
            </div>
            {feedback.details && feedback.details.length ? (
              <ul className="mt-3 list-disc space-y-1 pl-5 text-xs">
                {feedback.details.map((detail, idx) => (
                  <li key={`${detail}-${idx}`}>{detail}</li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}

        {/* Editores lado a lado */}
        <div className="flex flex-col gap-6 sm:flex-row">
          {/* Python */}
          <div className="flex-1 min-w-0">
            <h2 className="mb-3 text-center text-lg font-semibold text-slate-600">Código Python</h2>
            <CodeEditor
              value={pythonCode}
              onChange={(e) => setPythonCode(e.target.value)}
              language="Python"
              height="34rem"
              placeholder="Escribe o pega tu código Python aquí"
            />
          </div>

          {/* Java */}
          <div className="flex-1 min-w-0">
            <h2 className="mb-3 text-center text-lg font-semibold text-slate-600">Código Java</h2>
            <CodeEditor
              value={javaCode}
              onChange={(e) => setJavaCode(e.target.value)}
              language="Java"
              readOnly
              height="34rem"
              placeholder="Aquí verás el resultado en Java"
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

