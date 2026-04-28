/* eslint-disable @next/next/no-img-element */
import { ImageResponse } from "next/og";
import { getPostMeta } from "@/lib/getPostMeta";
import { COVER_HEIGHT, COVER_WIDTH } from "@/lib/getPostCover";

export const runtime = "nodejs";
// Treat as fully static — re-render only when the file changes (rare).
export const revalidate = 31536000;

// ─────────────── Font loading (module-cached) ───────────────

interface FontPair {
  regular: ArrayBuffer;
  bold: ArrayBuffer;
}

let fontPromise: Promise<FontPair> | null = null;

async function fetchGoogleFontBinary(family: string, weight: number): Promise<ArrayBuffer> {
  const cssUrl = `https://fonts.googleapis.com/css2?family=${encodeURIComponent(family)}:wght@${weight}&display=swap`;
  const css = await fetch(cssUrl, {
    headers: {
      // Required: without a real UA, Google returns TTF instead of woff2,
      // which Satori does support, but woff2 is smaller.
      "User-Agent":
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    },
    cache: "force-cache",
  }).then((r) => r.text());
  const match = /url\((https:\/\/fonts\.gstatic\.com[^)]+)\)/.exec(css);
  if (!match) throw new Error(`Could not find ${family} ${weight} in Google CSS`);
  return await fetch(match[1], { cache: "force-cache" }).then((r) => r.arrayBuffer());
}

function loadFonts(): Promise<FontPair> {
  if (fontPromise) return fontPromise;
  fontPromise = (async () => {
    const [regular, bold] = await Promise.all([
      fetchGoogleFontBinary("Inter", 500),
      fetchGoogleFontBinary("Inter", 700),
    ]);
    return { regular, bold };
  })().catch((err) => {
    fontPromise = null; // allow retry on next request
    throw err;
  });
  return fontPromise;
}

// ─────────────── Visual primitives ───────────────

function hashString(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

interface Palette {
  bgFrom: string;
  bgTo: string;
  blob: string;
  ring: string;
}

const PALETTES: Palette[] = [
  // amber → rose
  { bgFrom: "#1f1411", bgTo: "#3a1d1d", blob: "rgba(245, 158, 11, 0.45)", ring: "rgba(245, 158, 11, 0.18)" },
  // emerald → teal
  { bgFrom: "#0f1d18", bgTo: "#0e2a2a", blob: "rgba(16, 185, 129, 0.4)", ring: "rgba(16, 185, 129, 0.18)" },
  // indigo → violet
  { bgFrom: "#15172e", bgTo: "#241836", blob: "rgba(139, 92, 246, 0.42)", ring: "rgba(139, 92, 246, 0.2)" },
  // cyan → blue
  { bgFrom: "#0d1c2c", bgTo: "#0f2436", blob: "rgba(56, 189, 248, 0.4)", ring: "rgba(56, 189, 248, 0.18)" },
  // rose → fuchsia
  { bgFrom: "#1f0f1c", bgTo: "#2c1330", blob: "rgba(236, 72, 153, 0.4)", ring: "rgba(236, 72, 153, 0.2)" },
  // lime → emerald
  { bgFrom: "#131c0e", bgTo: "#16291a", blob: "rgba(132, 204, 22, 0.4)", ring: "rgba(132, 204, 22, 0.2)" },
];

function paletteFor(seed: string): Palette {
  return PALETTES[hashString(seed) % PALETTES.length];
}

function formatDate(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

/** Pick a font size for the title that keeps the box readable. */
function pickTitleSize(title: string): number {
  const len = title.length;
  if (len <= 38) return 88;
  if (len <= 60) return 76;
  if (len <= 90) return 64;
  if (len <= 130) return 54;
  return 46;
}

// ─────────────── Route handler ───────────────

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ slug: string[] }> },
) {
  const { slug } = await params;
  const slugStr = (slug || []).join("/");
  const meta = getPostMeta(slugStr);

  // If slug is unknown, render a generic cover so the route never 404s
  // (related cards/og share would otherwise show broken images).
  const title = meta?.title ?? "Hiep Tran — AI Engineer";
  const category = (meta?.subcategory || meta?.category || "Hiep Tran")
    .replace(/-/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
  const author = meta?.author || "Hiep Tran";
  const date = formatDate(meta?.publishDate || "");
  const palette = paletteFor(slugStr || title);

  let fonts: FontPair;
  try {
    fonts = await loadFonts();
  } catch {
    // If Google Fonts is unreachable in dev, fall back to no custom fonts —
    // Satori will use its default Latin fallback.
    fonts = { regular: new ArrayBuffer(0), bold: new ArrayBuffer(0) };
  }

  const titleSize = pickTitleSize(title);

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          padding: "64px 72px",
          color: "#f8fafc",
          fontFamily: "Inter",
          background: `linear-gradient(135deg, ${palette.bgFrom} 0%, ${palette.bgTo} 100%)`,
          position: "relative",
        }}
      >
        {/* Decorative blob bottom-right */}
        <div
          style={{
            position: "absolute",
            right: -180,
            bottom: -200,
            width: 640,
            height: 640,
            borderRadius: 9999,
            background: palette.blob,
            filter: "blur(20px)",
            display: "flex",
          }}
        />
        {/* Decorative ring top-right */}
        <div
          style={{
            position: "absolute",
            right: -120,
            top: -120,
            width: 360,
            height: 360,
            borderRadius: 9999,
            border: `2px solid ${palette.ring}`,
            display: "flex",
          }}
        />
        {/* Subtle grid overlay */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
            display: "flex",
          }}
        />

        {/* Top: category pill */}
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div
            style={{
              padding: "10px 22px",
              borderRadius: 9999,
              background: "rgba(255,255,255,0.10)",
              border: "1px solid rgba(255,255,255,0.18)",
              fontSize: 22,
              fontWeight: 500,
              letterSpacing: 1.5,
              textTransform: "uppercase",
              color: "rgba(255,255,255,0.92)",
              display: "flex",
            }}
          >
            {category}
          </div>
        </div>

        {/* Spacer */}
        <div style={{ flex: 1, display: "flex" }} />

        {/* Title */}
        <div
          style={{
            display: "flex",
            fontSize: titleSize,
            lineHeight: 1.08,
            fontWeight: 700,
            letterSpacing: -1.5,
            color: "#ffffff",
            // 3-line clamp via line-clamp inside Satori
            overflow: "hidden",
            maxHeight: titleSize * 3 * 1.08,
          }}
        >
          {title}
        </div>

        {/* Footer */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginTop: 36,
            paddingTop: 24,
            borderTop: "1px solid rgba(255,255,255,0.16)",
            fontSize: 24,
            color: "rgba(255,255,255,0.78)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div
              style={{
                width: 40,
                height: 40,
                borderRadius: 9999,
                background:
                  "linear-gradient(135deg, rgba(255,255,255,0.85), rgba(255,255,255,0.55))",
                color: palette.bgFrom,
                fontWeight: 700,
                fontSize: 20,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {(author.match(/\b\w/g) || []).slice(0, 2).join("").toUpperCase() || "HT"}
            </div>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span style={{ fontWeight: 600, color: "#ffffff" }}>{author}</span>
              {date ? <span style={{ fontSize: 18 }}>{date}</span> : null}
            </div>
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              fontSize: 20,
              fontWeight: 500,
              color: "rgba(255,255,255,0.85)",
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: 9999,
                background: palette.blob,
                display: "flex",
              }}
            />
            halleyverse.dev
          </div>
        </div>
      </div>
    ),
    {
      width: COVER_WIDTH,
      height: COVER_HEIGHT,
      fonts:
        fonts.regular.byteLength > 0 && fonts.bold.byteLength > 0
          ? [
              { name: "Inter", data: fonts.regular, weight: 500, style: "normal" },
              { name: "Inter", data: fonts.bold, weight: 700, style: "normal" },
            ]
          : undefined,
      headers: {
        "Cache-Control":
          "public, max-age=31536000, s-maxage=31536000, immutable, stale-while-revalidate=86400",
        "Content-Type": "image/png",
      },
    },
  );
}
