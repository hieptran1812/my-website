// Target-language picker for the in-article translate tool.
// The chosen language is remembered for the lifetime of the browser session
// (sessionStorage) so it follows the reader across articles but resets when
// they close the tab — matching "saved while the user stays on the site".

export interface Language {
  code: string;
  /** English name shown in the dropdown. */
  label: string;
  /** Native/short name shown next to it. */
  native: string;
}

export const LANGUAGES: Language[] = [
  { code: "vi", label: "Vietnamese", native: "Tiếng Việt" },
  { code: "en", label: "English", native: "English" },
  { code: "zh-CN", label: "Chinese (Simpl.)", native: "简体中文" },
  { code: "zh-TW", label: "Chinese (Trad.)", native: "繁體中文" },
  { code: "ja", label: "Japanese", native: "日本語" },
  { code: "ko", label: "Korean", native: "한국어" },
  { code: "fr", label: "French", native: "Français" },
  { code: "de", label: "German", native: "Deutsch" },
  { code: "es", label: "Spanish", native: "Español" },
  { code: "pt", label: "Portuguese", native: "Português" },
  { code: "it", label: "Italian", native: "Italiano" },
  { code: "ru", label: "Russian", native: "Русский" },
  { code: "hi", label: "Hindi", native: "हिन्दी" },
  { code: "ar", label: "Arabic", native: "العربية" },
  { code: "th", label: "Thai", native: "ไทย" },
  { code: "id", label: "Indonesian", native: "Indonesia" },
];

const DEFAULT_LANG = "vi";
const STORAGE_KEY = "blog-translate-lang";

export function loadTargetLang(): string {
  if (typeof window === "undefined") return DEFAULT_LANG;
  try {
    const saved = window.sessionStorage.getItem(STORAGE_KEY);
    if (saved && LANGUAGES.some((l) => l.code === saved)) return saved;
  } catch {
    // sessionStorage disabled / unavailable
  }
  return DEFAULT_LANG;
}

export function saveTargetLang(code: string): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(STORAGE_KEY, code);
  } catch {
    // ignore quota / disabled
  }
}

export function languageLabel(code: string): string {
  return LANGUAGES.find((l) => l.code === code)?.label ?? code;
}

export interface TranslateResult {
  translation: string;
  detected: string;
  target: string;
}

/** Calls the same-origin proxy (CSP allows only 'self' for connect-src). */
export async function translateText(
  text: string,
  target: string,
  signal?: AbortSignal,
): Promise<TranslateResult> {
  // Trailing slash matches next.config `trailingSlash: true` (avoids a 308).
  const res = await fetch("/api/translate/", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, target }),
    signal,
  });
  if (!res.ok) {
    let message = "Translation failed.";
    try {
      const data = await res.json();
      if (data?.error) message = data.error;
    } catch {
      // non-JSON error body
    }
    throw new Error(message);
  }
  return (await res.json()) as TranslateResult;
}
