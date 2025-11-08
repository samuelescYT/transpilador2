import React, { useMemo, useRef } from "react";

function buildLineNumbers(count) {
  return Array.from({ length: count }, (_, idx) => idx + 1);
}

export default function CodeEditor({
  value,
  onChange = () => {},
  language,
  readOnly = false,
  placeholder = "",
  height = "32rem",
}) {
  const gutterRef = useRef(null);

  const totalLines = useMemo(() => {
    if (!value) return 1;
    return value.split(/\r?\n/).length || 1;
  }, [value]);

  const lineNumbers = useMemo(() => buildLineNumbers(totalLines), [totalLines]);

  const handleScroll = (event) => {
    if (gutterRef.current) {
      gutterRef.current.scrollTop = event.target.scrollTop;
    }
  };

  return (
    <div
      className="relative flex w-full rounded-2xl border border-slate-800 bg-slate-950/95 text-slate-100 shadow-[0_25px_60px_-30px_rgba(15,23,42,0.9)]"
      style={{ height }}
    >
      <div
        ref={gutterRef}
        className="select-none overflow-hidden border-r border-slate-800/60 bg-slate-900/80 px-4 py-4 text-right text-xs font-semibold text-slate-500"
        style={{ width: "3.5rem" }}
      >
        <pre className="leading-6">
          {lineNumbers.map((line) => (
            <span key={line} className="block">
              {line}
            </span>
          ))}
        </pre>
      </div>
      <textarea
        value={value}
        onChange={onChange}
        onScroll={handleScroll}
        readOnly={readOnly}
        placeholder={placeholder}
        spellCheck={false}
        className="h-full w-full flex-1 resize-none bg-transparent px-5 py-4 font-mono text-sm leading-6 text-slate-100 outline-none focus:outline-none"
      />
      <div className="pointer-events-none absolute inset-0 rounded-2xl ring-1 ring-inset ring-white/5" aria-hidden />
      {language ? (
        <span className="pointer-events-none absolute right-4 top-3 select-none text-[0.65rem] font-semibold uppercase tracking-[0.3em] text-slate-500">
          {language}
        </span>
      ) : null}
    </div>
  );
}
