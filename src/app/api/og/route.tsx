import { ImageResponse } from "next/og";
import { NextRequest } from "next/server";
import type React from "react";

export const runtime = "edge";

// Category → short icon label. Accent color now comes from the per-post palette.
const CATEGORY_ICONS: Record<string, string> = {
  "machine-learning": "ML",
  "paper-reading": "PR",
  "software-development": "SD",
  notes: "NT",
  trading: "TR",
  mlops: "MO",
  default: "HT",
};

// Diverse palette pool. Each post picks one deterministically via hash(title+category).
// bg is kept dark-leaning so light text stays legible over generated shapes.
const PALETTES: {
  bg: string;
  accents: [string, string, string];
}[] = [
  { bg: "#0f172a", accents: ["#0ea5e9", "#38bdf8", "#22d3ee"] }, // deep navy / sky
  { bg: "#1a0a2e", accents: ["#8b5cf6", "#a78bfa", "#c084fc"] }, // plum / violet
  { bg: "#0a1f16", accents: ["#10b981", "#34d399", "#6ee7b7"] }, // forest / emerald
  { bg: "#1c0a0a", accents: ["#f97316", "#fb923c", "#f59e0b"] }, // ember / amber
  { bg: "#0c1220", accents: ["#06b6d4", "#22d3ee", "#67e8f9"] }, // teal / cyan
  { bg: "#1f0a1a", accents: ["#ec4899", "#f472b6", "#fb7185"] }, // rose / pink
  { bg: "#0f0f1a", accents: ["#6366f1", "#818cf8", "#a78bfa"] }, // indigo
  { bg: "#0b1a1f", accents: ["#14b8a6", "#2dd4bf", "#5eead4"] }, // deep teal
  { bg: "#1a1410", accents: ["#eab308", "#facc15", "#fde047"] }, // amber night
  { bg: "#0a1a0f", accents: ["#22c55e", "#4ade80", "#86efac"] }, // jungle green
  { bg: "#120a1f", accents: ["#d946ef", "#e879f9", "#c084fc"] }, // magenta / fuchsia
  { bg: "#0f1a2a", accents: ["#3b82f6", "#60a5fa", "#93c5fd"] }, // cobalt blue
  { bg: "#1a0f0a", accents: ["#ef4444", "#f87171", "#fb7185"] }, // crimson ember
  { bg: "#0a0f1a", accents: ["#a78bfa", "#c4b5fd", "#ddd6fe"] }, // midnight lavender
];

function hashString(s: string): number {
  let h = 0x811c9dc5; // FNV-1a 32-bit offset basis
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Shape = {
  left: number;
  top: number;
  width: number;
  height: number;
  rotate: number;
  radius: number;
  color: string;
  opacity: number;
  outline: boolean;
};

function generateShapes(
  rng: () => number,
  accents: [string, string, string],
  width: number,
  height: number,
): Shape[] {
  const count = 6 + Math.floor(rng() * 4); // 6–9 shapes
  const shapes: Shape[] = [];
  for (let i = 0; i < count; i++) {
    const kind = rng();
    let w: number;
    let h: number;
    let radius: number;
    if (kind < 0.35) {
      const d = Math.round(80 + rng() * 240);
      w = d;
      h = d;
      radius = Math.round(d / 2);
    } else if (kind < 0.6) {
      const d = Math.round(70 + rng() * 180);
      w = d;
      h = d;
      radius = Math.round(14 + rng() * 28);
    } else if (kind < 0.85) {
      w = Math.round(120 + rng() * 280);
      h = Math.round(8 + rng() * 22);
      radius = Math.round(h / 2);
    } else {
      w = Math.round(40 + rng() * 80);
      h = Math.round(120 + rng() * 220);
      radius = Math.round(10 + rng() * 20);
    }
    const left = Math.round(-w * 0.3 + rng() * (width + w * 0.6 - w));
    const top = Math.round(-h * 0.3 + rng() * (height + h * 0.6 - h));
    const color = accents[Math.floor(rng() * accents.length)];
    shapes.push({
      left,
      top,
      width: w,
      height: h,
      rotate: Math.round(rng() * 360),
      radius,
      color,
      opacity: Math.round((0.28 + rng() * 0.32) * 100) / 100,
      outline: rng() < 0.35,
    });
  }
  return shapes;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const title = searchParams.get("title") || "Hiep Tran";
    const category = searchParams.get("category") || "";
    const subcategory = searchParams.get("subcategory") || "";
    const type = searchParams.get("type") || "default";

    // Article thumbnail mode
    if (type === "article") {
      const WIDTH = 672;
      const HEIGHT = 366;

      // Deterministic seed from title+category → same post always same image,
      // different posts look distinct (palette + shape layout).
      const seed = hashString(`${title}::${category}::${subcategory}`);
      const rng = mulberry32(seed);

      const palette = PALETTES[seed % PALETTES.length];
      const icon = CATEGORY_ICONS[category] || CATEGORY_ICONS["default"];
      const primaryAccent = palette.accents[0];
      const shapes = generateShapes(rng, palette.accents, WIDTH, HEIGHT);

      // Truncate title for display
      const displayTitle =
        title.length > 80 ? title.substring(0, 77) + "..." : title;

      // Adaptive font size
      const fontSize =
        displayTitle.length > 60
          ? "28px"
          : displayTitle.length > 40
            ? "34px"
            : "40px";

      const textColor = "#f1f5f9";
      const textMuted = "#94a3b8";
      const gridColor = "rgba(255,255,255,0.04)";

      return new ImageResponse(
        (
          <div
            style={{
              height: "100%",
              width: "100%",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              backgroundColor: palette.bg,
              padding: "0",
              fontFamily: "system-ui, -apple-system, sans-serif",
              overflow: "hidden",
              position: "relative",
            }}
          >
            {/* Grid pattern */}
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundImage: `linear-gradient(${gridColor} 1px, transparent 1px), linear-gradient(90deg, ${gridColor} 1px, transparent 1px)`,
                backgroundSize: "48px 48px",
                display: "flex",
              }}
            />

            {/* Randomized shape layer */}
            {shapes.map((s, i) => {
              const base: React.CSSProperties = {
                position: "absolute",
                left: s.left,
                top: s.top,
                width: s.width,
                height: s.height,
                borderRadius: s.radius,
                opacity: s.opacity,
                transform: `rotate(${s.rotate}deg)`,
                display: "flex",
              };
              const style: React.CSSProperties = s.outline
                ? { ...base, border: `2px solid ${s.color}` }
                : { ...base, backgroundColor: s.color };
              return <div key={i} style={style} />;
            })}

            {/* Contrast scrim so text stays legible over shapes */}
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: `linear-gradient(180deg, ${palette.bg}aa 0%, ${palette.bg}00 30%, ${palette.bg}00 70%, ${palette.bg}aa 100%)`,
                display: "flex",
              }}
            />

            {/* Top: category badge */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                padding: "40px 48px 0",
                gap: "14px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: "44px",
                  height: "44px",
                  borderRadius: "12px",
                  backgroundColor: `${primaryAccent}20`,
                  border: `1.5px solid ${primaryAccent}40`,
                  color: primaryAccent,
                  fontSize: "16px",
                  fontWeight: 700,
                  letterSpacing: "1px",
                }}
              >
                {icon}
              </div>
              {subcategory && (
                <div
                  style={{
                    display: "flex",
                    fontSize: "14px",
                    color: primaryAccent,
                    letterSpacing: "2px",
                    textTransform: "uppercase",
                    fontWeight: 600,
                    opacity: 0.85,
                  }}
                >
                  {subcategory}
                </div>
              )}
            </div>

            {/* Center: Title */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                padding: "0 48px",
                flex: 1,
                justifyContent: "center",
              }}
            >
              <div
                style={{
                  fontSize,
                  fontWeight: 700,
                  color: textColor,
                  lineHeight: 1.35,
                  letterSpacing: "-0.3px",
                  maxWidth: "520px",
                  display: "flex",
                }}
              >
                {displayTitle}
              </div>
              <div
                style={{
                  display: "flex",
                  marginTop: "20px",
                  width: "60px",
                  height: "3px",
                  borderRadius: "2px",
                  backgroundColor: primaryAccent,
                }}
              />
            </div>

            {/* Bottom: branding */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "0 48px 32px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    width: "30px",
                    height: "30px",
                    borderRadius: "8px",
                    background: `linear-gradient(135deg, ${primaryAccent}, ${palette.accents[1]})`,
                    color: "#fff",
                    fontSize: "13px",
                    fontWeight: 700,
                  }}
                >
                  H
                </div>
                <div
                  style={{
                    display: "flex",
                    fontSize: "14px",
                    color: textMuted,
                    fontWeight: 500,
                  }}
                >
                  halleyverse.dev
                </div>
              </div>
              <div style={{ display: "flex", gap: "5px" }}>
                {[0.6, 0.4, 0.2].map((opacity, i) => (
                  <div
                    key={i}
                    style={{
                      width: "6px",
                      height: "6px",
                      borderRadius: "50%",
                      backgroundColor: primaryAccent,
                      opacity,
                      display: "flex",
                    }}
                  />
                ))}
              </div>
            </div>
          </div>
        ),
        {
          width: WIDTH,
          height: HEIGHT,
          headers: {
            "Cache-Control":
              "public, max-age=31536000, s-maxage=31536000, immutable",
          },
        },
      );
    }

    // Default OG image (existing behavior)
    const description =
      searchParams.get("description") ||
      "AI Research Engineer & Full-Stack Developer";

    return new ImageResponse(
      (
        <div
          style={{
            height: "100%",
            width: "100%",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "#0f0f0f",
            backgroundImage:
              "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            fontSize: 32,
            fontWeight: 600,
          }}
        >
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              backgroundColor: "rgba(255, 255, 255, 0.95)",
              borderRadius: "24px",
              padding: "80px 60px",
              margin: "40px",
              boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)",
              maxWidth: "800px",
              textAlign: "center",
            }}
          >
            <div
              style={{
                fontSize: "48px",
                fontWeight: "bold",
                background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                backgroundClip: "text",
                color: "transparent",
                marginBottom: "20px",
                lineHeight: 1.2,
              }}
            >
              {title}
            </div>
            <div
              style={{
                fontSize: "24px",
                color: "#64748b",
                lineHeight: 1.4,
                maxWidth: "600px",
              }}
            >
              {description}
            </div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                marginTop: "40px",
                fontSize: "18px",
                color: "#94a3b8",
              }}
            >
              <div
                style={{
                  width: "32px",
                  height: "32px",
                  borderRadius: "8px",
                  background:
                    "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                  marginRight: "12px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "white",
                  fontSize: "16px",
                  fontWeight: "bold",
                }}
              >
                H
              </div>
              halleyverse.dev
            </div>
          </div>
        </div>
      ),
      {
        width: 1200,
        height: 630,
        headers: {
          "Cache-Control":
            "public, max-age=31536000, s-maxage=31536000, immutable",
        },
      },
    );
  } catch (e: unknown) {
    const errorMessage = e instanceof Error ? e.message : "Unknown error";
    console.log(`${errorMessage}`);
    return new Response(`Failed to generate the image`, {
      status: 500,
    });
  }
}
