"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { Highlight, HighlightColor } from "./types";
import { createId, loadHighlights, saveHighlights } from "./storage";
import {
  extractText,
  forEachSlice,
  rangeToOffsets,
} from "./anchor";
import {
  LANGUAGES,
  loadTargetLang,
  saveTargetLang,
  languageLabel,
  translateText,
} from "./translate";
import "./highlights.css";

interface Props {
  slug: string;
  containerRef: React.RefObject<HTMLElement | null>;
}

const COLORS: HighlightColor[] = ["yellow", "green", "blue", "pink", "purple"];

interface ToolbarState {
  x: number;
  y: number;
  range: Range;
}

interface PopoverState {
  id: string;
  x: number;
  y: number;
}

interface TranslateState {
  x: number;
  y: number;
  text: string;
}

export default function BlogHighlighter({ slug, containerRef }: Props) {
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const [toolbar, setToolbar] = useState<ToolbarState | null>(null);
  const [popover, setPopover] = useState<PopoverState | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [noteDraft, setNoteDraft] = useState("");

  // Translate tool state
  const [translate, setTranslate] = useState<TranslateState | null>(null);
  const [targetLang, setTargetLang] = useState<string>("vi");
  const [translation, setTranslation] = useState<string>("");
  const [detectedLang, setDetectedLang] = useState<string>("");
  const [translateStatus, setTranslateStatus] = useState<
    "idle" | "loading" | "error"
  >("idle");
  const [translateError, setTranslateError] = useState<string>("");
  const [copied, setCopied] = useState(false);

  const popoverRef = useRef<HTMLDivElement | null>(null);
  const toolbarRef = useRef<HTMLDivElement | null>(null);
  const translateRef = useRef<HTMLDivElement | null>(null);
  // Guards against out-of-order responses when the language is switched fast.
  const translateReqId = useRef(0);
  const translateAbort = useRef<AbortController | null>(null);

  // Restore the session-saved target language on mount.
  useEffect(() => {
    setTargetLang(loadTargetLang());
  }, []);

  // Load from storage when slug changes
  useEffect(() => {
    setHighlights(loadHighlights(slug));
  }, [slug]);

  // Persist on change
  useEffect(() => {
    saveHighlights(slug, highlights);
  }, [slug, highlights]);

  // Apply/unapply highlight wrappers to DOM
  const applyHighlights = useCallback(() => {
    const root = containerRef.current;
    if (!root) return;

    // Clean previous wrappers
    const existing = root.querySelectorAll<HTMLElement>(".bh-mark");
    existing.forEach((el) => {
      const parent = el.parentNode;
      if (!parent) return;
      while (el.firstChild) parent.insertBefore(el.firstChild, el);
      parent.removeChild(el);
    });
    root.normalize();

    // Apply current highlights
    for (const h of highlights) {
      forEachSlice(root, h.start, h.end, (textNode, from, to) => {
        try {
          const range = document.createRange();
          range.setStart(textNode, from);
          range.setEnd(textNode, to);
          const mark = document.createElement("mark");
          mark.className = "bh-mark";
          mark.dataset.id = h.id;
          mark.dataset.color = h.color;
          mark.dataset.hasNote = h.note.trim() ? "true" : "false";
          range.surroundContents(mark);
        } catch {
          // range crosses boundaries; skip this slice
        }
      });
    }
  }, [highlights, containerRef]);

  // Re-apply when highlights change or content loads
  useEffect(() => {
    applyHighlights();
  }, [applyHighlights]);

  // Handle selection on the article
  useEffect(() => {
    const root = containerRef.current;
    if (!root) return;

    let timer: number | null = null;
    const onSelChange = () => {
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        const sel = window.getSelection();
        if (!sel || sel.rangeCount === 0 || sel.isCollapsed) {
          setToolbar(null);
          return;
        }
        const range = sel.getRangeAt(0);
        if (!root.contains(range.commonAncestorContainer)) {
          setToolbar(null);
          return;
        }
        // Skip if inside a saved highlight (user probably wants to edit)
        const anc = range.commonAncestorContainer;
        const el = (anc.nodeType === 1 ? (anc as Element) : anc.parentElement);
        if (el?.closest(".bh-mark")) {
          setToolbar(null);
          return;
        }
        const rect = range.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) {
          setToolbar(null);
          return;
        }
        const TOOLBAR_W = 236;
        const x = Math.max(
          8,
          Math.min(
            window.innerWidth - TOOLBAR_W - 8,
            rect.left + rect.width / 2 - TOOLBAR_W / 2,
          ),
        );
        const y = Math.max(8, rect.top - 44);
        setToolbar({ x, y, range: range.cloneRange() });
      }, 80);
    };

    document.addEventListener("selectionchange", onSelChange);
    return () => {
      document.removeEventListener("selectionchange", onSelChange);
      if (timer) window.clearTimeout(timer);
    };
  }, [containerRef]);

  // Click on a saved highlight opens popover
  useEffect(() => {
    const root = containerRef.current;
    if (!root) return;
    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (!target) return;
      const mark = target.closest<HTMLElement>(".bh-mark");
      if (!mark) return;
      const id = mark.dataset.id;
      if (!id) return;
      const rect = mark.getBoundingClientRect();
      const POP_W = 360;
      const x = Math.max(
        8,
        Math.min(
          window.innerWidth - POP_W - 8,
          rect.left + rect.width / 2 - POP_W / 2,
        ),
      );
      const spaceBelow = window.innerHeight - rect.bottom;
      const y = spaceBelow > 240 ? rect.bottom + 8 : Math.max(8, rect.top - 240);
      const h = highlights.find((x) => x.id === id);
      setNoteDraft(h?.note ?? "");
      setPopover({ id, x, y });
      setToolbar(null);
    };
    root.addEventListener("click", onClick);
    return () => root.removeEventListener("click", onClick);
  }, [containerRef, highlights]);

  // Dismiss popover when clicking outside
  useEffect(() => {
    if (!popover) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (popoverRef.current?.contains(t)) return;
      if ((t as HTMLElement).closest?.(".bh-mark")) return;
      setPopover(null);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPopover(null);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [popover]);

  // Dismiss toolbar on scroll/resize
  useEffect(() => {
    if (!toolbar) return;
    const hide = () => setToolbar(null);
    window.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
    return () => {
      window.removeEventListener("scroll", hide, true);
      window.removeEventListener("resize", hide);
    };
  }, [toolbar]);

  const addHighlight = useCallback(
    (color: HighlightColor) => {
      const root = containerRef.current;
      if (!root || !toolbar) return;
      const offsets = rangeToOffsets(root, toolbar.range);
      if (!offsets) {
        setToolbar(null);
        return;
      }
      const text = extractText(root, offsets.start, offsets.end);
      if (!text.trim()) {
        setToolbar(null);
        return;
      }
      const now = Date.now();
      const h: Highlight = {
        id: createId(),
        slug,
        text,
        note: "",
        color,
        start: offsets.start,
        end: offsets.end,
        createdAt: now,
        updatedAt: now,
      };
      // Prevent overlapping highlights (stacked colors). Replace any existing
      // highlights whose range intersects the new one — one color per span.
      setHighlights((prev) => [
        ...prev.filter((p) => p.end <= h.start || p.start >= h.end),
        h,
      ]);
      window.getSelection()?.removeAllRanges();
      setToolbar(null);
    },
    [containerRef, toolbar, slug],
  );

  const openNoteFromToolbar = useCallback(() => {
    if (!toolbar) return;
    const root = containerRef.current;
    if (!root) return;
    const offsets = rangeToOffsets(root, toolbar.range);
    if (!offsets) return;
    const text = extractText(root, offsets.start, offsets.end);
    if (!text.trim()) return;
    const now = Date.now();
    const id = createId();
    const h: Highlight = {
      id,
      slug,
      text,
      note: "",
      color: "yellow",
      start: offsets.start,
      end: offsets.end,
      createdAt: now,
      updatedAt: now,
    };
    setHighlights((prev) => [
      ...prev.filter((p) => p.end <= h.start || p.start >= h.end),
      h,
    ]);
    window.getSelection()?.removeAllRanges();
    const rect = toolbar.range.getBoundingClientRect();
    const POP_W = 360;
    const x = Math.max(
      8,
      Math.min(
        window.innerWidth - POP_W - 8,
        rect.left + rect.width / 2 - POP_W / 2,
      ),
    );
    const spaceBelow = window.innerHeight - rect.bottom;
    const y = spaceBelow > 240 ? rect.bottom + 8 : Math.max(8, rect.top - 240);
    setNoteDraft("");
    setToolbar(null);
    // schedule popover on next tick so wrap applies first
    setTimeout(() => setPopover({ id, x, y }), 0);
  }, [toolbar, containerRef, slug]);

  // Fetch a translation for the current selection into `lang`.
  const runTranslate = useCallback(async (text: string, lang: string) => {
    const reqId = ++translateReqId.current;
    translateAbort.current?.abort();
    const controller = new AbortController();
    translateAbort.current = controller;
    setTranslateStatus("loading");
    setTranslation("");
    setDetectedLang("");
    setTranslateError("");
    setCopied(false);
    try {
      const result = await translateText(text, lang, controller.signal);
      if (reqId !== translateReqId.current) return; // a newer request superseded
      setTranslation(result.translation);
      setDetectedLang(result.detected);
      setTranslateStatus("idle");
    } catch (err) {
      if (controller.signal.aborted || reqId !== translateReqId.current) return;
      setTranslateError(
        err instanceof Error ? err.message : "Translation failed.",
      );
      setTranslateStatus("error");
    }
  }, []);

  // Open the translate popover from the selection toolbar.
  const openTranslate = useCallback(() => {
    if (!toolbar) return;
    // Collapse runs of spaces/tabs but PRESERVE line/paragraph breaks — Google
    // translates with proper sentence boundaries and returns structured text,
    // instead of one run-on blob. (.bh-translate-text renders with pre-wrap.)
    const text = toolbar.range
      .toString()
      .replace(/[ \t]+/g, " ")
      .replace(/[ \t]*\n[ \t]*/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    if (!text) {
      setToolbar(null);
      return;
    }
    const rect = toolbar.range.getBoundingClientRect();
    const POP_W = 380;
    const x = Math.max(
      8,
      Math.min(
        window.innerWidth - POP_W - 8,
        rect.left + rect.width / 2 - POP_W / 2,
      ),
    );
    const spaceBelow = window.innerHeight - rect.bottom;
    const y = spaceBelow > 260 ? rect.bottom + 8 : Math.max(8, rect.top - 260);
    setTranslate({ x, y, text });
    setToolbar(null);
    window.getSelection()?.removeAllRanges();
    runTranslate(text, targetLang);
  }, [toolbar, targetLang, runTranslate]);

  // Switching language re-translates and persists the choice for the session.
  const changeTargetLang = useCallback(
    (lang: string) => {
      setTargetLang(lang);
      saveTargetLang(lang);
      if (translate) runTranslate(translate.text, lang);
    },
    [translate, runTranslate],
  );

  const closeTranslate = useCallback(() => {
    translateReqId.current++;
    translateAbort.current?.abort();
    setTranslate(null);
  }, []);

  const copyTranslation = useCallback(() => {
    if (!translation) return;
    navigator.clipboard?.writeText(translation).then(
      () => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      },
      () => {},
    );
  }, [translation]);

  // Dismiss translate popover on outside click / Escape.
  useEffect(() => {
    if (!translate) return;
    const onDown = (e: MouseEvent) => {
      if (translateRef.current?.contains(e.target as Node)) return;
      closeTranslate();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeTranslate();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [translate, closeTranslate]);

  const updateHighlight = useCallback(
    (id: string, patch: Partial<Highlight>) => {
      setHighlights((prev) =>
        prev.map((h) =>
          h.id === id ? { ...h, ...patch, updatedAt: Date.now() } : h,
        ),
      );
    },
    [],
  );

  const deleteHighlight = useCallback((id: string) => {
    setHighlights((prev) => prev.filter((h) => h.id !== id));
    setPopover(null);
  }, []);

  const saveNote = useCallback(() => {
    if (!popover) return;
    updateHighlight(popover.id, { note: noteDraft });
    setPopover(null);
  }, [popover, noteDraft, updateHighlight]);

  const currentHighlight = useMemo(
    () => (popover ? highlights.find((h) => h.id === popover.id) : null),
    [popover, highlights],
  );

  const scrollToHighlight = useCallback(
    (id: string) => {
      const root = containerRef.current;
      if (!root) return;
      const mark = root.querySelector<HTMLElement>(`.bh-mark[data-id="${id}"]`);
      if (!mark) return;
      mark.scrollIntoView({ behavior: "smooth", block: "center" });
      mark.animate(
        [
          { filter: "brightness(1)" },
          { filter: "brightness(1.3)" },
          { filter: "brightness(1)" },
        ],
        { duration: 900 },
      );
    },
    [containerRef],
  );

  return (
    <>
      {/* Selection toolbar */}
      {toolbar && (
        <div
          ref={toolbarRef}
          className="bh-toolbar"
          style={{ left: toolbar.x, top: toolbar.y }}
          role="toolbar"
          aria-label="Highlight colors"
          onMouseDown={(e) => e.preventDefault()}
        >
          {COLORS.map((c) => (
            <button
              key={c}
              className="bh-swatch"
              data-color={c}
              aria-label={`Highlight ${c}`}
              onClick={() => addHighlight(c)}
            />
          ))}
          <div className="bh-toolbar-sep" />
          <button
            className="bh-iconbtn"
            aria-label="Add note"
            title="Add note"
            onClick={openNoteFromToolbar}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 20h9" />
              <path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4 12.5-12.5z" />
            </svg>
          </button>
          <button
            className="bh-iconbtn"
            aria-label={`Translate to ${languageLabel(targetLang)}`}
            title={`Translate to ${languageLabel(targetLang)}`}
            onClick={openTranslate}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 5h7M9 3v2c0 4.4-2.7 8-6 9" />
              <path d="M5 9c0 2.5 3.3 4.8 6 5.5" />
              <path d="M14 19l3-7 3 7M14.7 17h4.6" />
            </svg>
          </button>
        </div>
      )}

      {/* Note popover */}
      {popover && currentHighlight && (
        <div
          ref={popoverRef}
          className="bh-popover"
          style={{ left: popover.x, top: popover.y }}
          role="dialog"
          aria-label="Edit note"
        >
          <div
            className="bh-popover-quote"
            style={{ color: swatchColor(currentHighlight.color) }}
          >
            {currentHighlight.text}
          </div>
          <textarea
            value={noteDraft}
            autoFocus
            onChange={(e) => setNoteDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                saveNote();
              }
            }}
            placeholder="Write a note... (⌘/Ctrl + Enter to save)"
          />
          <div className="bh-popover-row">
            <div className="bh-popover-colors">
              {COLORS.map((c) => (
                <button
                  key={c}
                  className="bh-swatch"
                  data-color={c}
                  aria-label={`Set color ${c}`}
                  style={{
                    outline:
                      currentHighlight.color === c
                        ? "2px solid var(--accent, #3b82f6)"
                        : undefined,
                    outlineOffset: 1,
                  }}
                  onClick={() =>
                    updateHighlight(currentHighlight.id, { color: c })
                  }
                />
              ))}
            </div>
            <div className="bh-popover-actions">
              <button
                className="bh-btn bh-btn-danger"
                onClick={() => deleteHighlight(currentHighlight.id)}
              >
                Delete
              </button>
              <button className="bh-btn bh-btn-primary" onClick={saveNote}>
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Translate popover */}
      {translate && (
        <div
          ref={translateRef}
          className="bh-popover bh-translate"
          style={{ left: translate.x, top: translate.y }}
          role="dialog"
          aria-label="Translate selection"
        >
          <div className="bh-translate-head">
            <div className="bh-translate-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 5h7M9 3v2c0 4.4-2.7 8-6 9" />
                <path d="M5 9c0 2.5 3.3 4.8 6 5.5" />
                <path d="M14 19l3-7 3 7M14.7 17h4.6" />
              </svg>
              <span>Translate</span>
            </div>
            <select
              className="bh-translate-lang"
              aria-label="Target language"
              value={targetLang}
              onChange={(e) => changeTargetLang(e.target.value)}
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label} · {l.native}
                </option>
              ))}
            </select>
            <button
              className="bh-iconbtn"
              aria-label="Close"
              title="Close"
              onClick={closeTranslate}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M6 6l12 12M18 6L6 18" />
              </svg>
            </button>
          </div>

          <div className="bh-translate-source">{translate.text}</div>

          <div className="bh-translate-result" aria-live="polite">
            {translateStatus === "loading" && (
              <div className="bh-translate-loading">
                <span className="bh-spinner" aria-hidden="true" />
                <span>Translating…</span>
              </div>
            )}
            {translateStatus === "error" && (
              <div className="bh-translate-errmsg">
                {translateError || "Translation failed."}
                <button
                  className="bh-btn bh-btn-primary bh-translate-retry"
                  onClick={() => runTranslate(translate.text, targetLang)}
                >
                  Retry
                </button>
              </div>
            )}
            {translateStatus === "idle" && translation && (
              <p className="bh-translate-text">{translation}</p>
            )}
          </div>

          <div className="bh-translate-foot">
            <span className="bh-translate-detected">
              {detectedLang
                ? `Detected: ${languageLabel(detectedLang)}`
                : "Powered by Google Translate"}
            </span>
            {translateStatus === "idle" && translation && (
              <button
                className="bh-btn"
                onClick={copyTranslation}
                aria-label="Copy translation"
              >
                {copied ? "Copied!" : "Copy"}
              </button>
            )}
          </div>
        </div>
      )}

      {/* Floating toggle (default hidden panel) */}
      <button
        className="bh-fab"
        aria-label="Toggle notes panel"
        onClick={() => setPanelOpen((v) => !v)}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
          <path d="M14 2v6h6" />
          <path d="M8 13h8M8 17h5" />
        </svg>
        <span>Notes</span>
        {highlights.length > 0 && (
          <span className="bh-fab-count">{highlights.length}</span>
        )}
      </button>

      {/* Side panel + backdrop */}
      {panelOpen && (
        <div className="bh-backdrop" onClick={() => setPanelOpen(false)} />
      )}
      <aside
        className={`bh-panel${panelOpen ? " open" : ""}`}
        aria-hidden={!panelOpen}
        aria-label="Saved notes"
      >
        <div className="bh-panel-head">
          <span>Notes ({highlights.length})</span>
          <button
            className="bh-iconbtn"
            aria-label="Close"
            onClick={() => setPanelOpen(false)}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>
        <div className="bh-panel-list">
          {highlights.length === 0 && (
            <div className="bh-panel-empty">
              No highlights yet.
              <br />
              Select any text to get started.
            </div>
          )}
          {highlights
            .slice()
            .sort((a, b) => a.start - b.start)
            .map((h) => (
              <div
                key={h.id}
                className="bh-card"
                onClick={() => {
                  scrollToHighlight(h.id);
                  setPanelOpen(false);
                }}
              >
                <div
                  className={`bh-card-quote bh-card-color-${h.color}`}
                >
                  {h.text}
                </div>
                <div
                  className={`bh-card-note${h.note.trim() ? "" : " empty"}`}
                >
                  {h.note.trim() || "No note"}
                </div>
              </div>
            ))}
        </div>
      </aside>
    </>
  );
}

function swatchColor(c: HighlightColor): string {
  switch (c) {
    case "yellow": return "#ca8a04";
    case "green": return "#16a34a";
    case "blue": return "#2563eb";
    case "pink": return "#db2777";
    case "purple": return "#9333ea";
  }
}
