import { ImageResponse } from "next/og";
import { NextRequest } from "next/server";

export const runtime = "edge";

// Category-based themes with light and dark variants
const CATEGORY_THEMES: Record<
  string,
  {
    dark: { bg: string; surface: string };
    light: { bg: string; surface: string };
    icon: string;
    accent: string;
  }
> = {
  "machine-learning": {
    dark: { bg: "#0f172a", surface: "#1e293b" },
    light: { bg: "#f0f9ff", surface: "#e0f2fe" },
    icon: "ML",
    accent: "#0ea5e9",
  },
  "paper-reading": {
    dark: { bg: "#1a0a2e", surface: "#2d1b4e" },
    light: { bg: "#faf5ff", surface: "#f3e8ff" },
    icon: "PR",
    accent: "#8b5cf6",
  },
  "software-development": {
    dark: { bg: "#0a1628", surface: "#1a3a2e" },
    light: { bg: "#f0fdf4", surface: "#dcfce7" },
    icon: "SD",
    accent: "#10b981",
  },
  notes: {
    dark: { bg: "#1c1917", surface: "#292524" },
    light: { bg: "#fefce8", surface: "#fef9c3" },
    icon: "NT",
    accent: "#f59e0b",
  },
  trading: {
    dark: { bg: "#0c1220", surface: "#1e293b" },
    light: { bg: "#fff7ed", surface: "#ffedd5" },
    icon: "TR",
    accent: "#f97316",
  },
  mlops: {
    dark: { bg: "#0f1729", surface: "#1e3a5f" },
    light: { bg: "#ecfeff", surface: "#cffafe" },
    icon: "MO",
    accent: "#06b6d4",
  },
  default: {
    dark: { bg: "#0f0f1a", surface: "#1e1e2e" },
    light: { bg: "#f8fafc", surface: "#e2e8f0" },
    icon: "HT",
    accent: "#6366f1",
  },
};

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const title = searchParams.get("title") || "Hiep Tran";
    const category = searchParams.get("category") || "";
    const subcategory = searchParams.get("subcategory") || "";
    const type = searchParams.get("type") || "default";
    const themeMode = searchParams.get("theme") === "light" ? "light" : "dark";

    // Article thumbnail mode
    if (type === "article") {
      const catTheme =
        CATEGORY_THEMES[category] || CATEGORY_THEMES["default"];
      const colors = catTheme[themeMode];
      const isDark = themeMode === "dark";

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

      const textColor = isDark ? "#f1f5f9" : "#0f172a";
      const textMuted = isDark ? "#94a3b8" : "#64748b";
      const gridColor = isDark
        ? "rgba(255,255,255,0.04)"
        : "rgba(0,0,0,0.04)";

      return new ImageResponse(
        (
          <div
            style={{
              height: "100%",
              width: "100%",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              backgroundColor: colors.bg,
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

            {/* Accent glow - top right */}
            <div
              style={{
                position: "absolute",
                top: "-80px",
                right: "-80px",
                width: "350px",
                height: "350px",
                borderRadius: "50%",
                background: `radial-gradient(circle, ${catTheme.accent}${isDark ? "18" : "20"} 0%, transparent 70%)`,
                display: "flex",
              }}
            />

            {/* Accent glow - bottom left */}
            <div
              style={{
                position: "absolute",
                bottom: "-60px",
                left: "-60px",
                width: "250px",
                height: "250px",
                borderRadius: "50%",
                background: `radial-gradient(circle, ${catTheme.accent}${isDark ? "10" : "15"} 0%, transparent 70%)`,
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
                  backgroundColor: `${catTheme.accent}${isDark ? "20" : "18"}`,
                  border: `1.5px solid ${catTheme.accent}${isDark ? "40" : "30"}`,
                  color: catTheme.accent,
                  fontSize: "16px",
                  fontWeight: 700,
                  letterSpacing: "1px",
                }}
              >
                {catTheme.icon}
              </div>
              {subcategory && (
                <div
                  style={{
                    display: "flex",
                    fontSize: "14px",
                    color: catTheme.accent,
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
                  backgroundColor: catTheme.accent,
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
                    background: `linear-gradient(135deg, ${catTheme.accent}, ${catTheme.accent}88)`,
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
                      backgroundColor: catTheme.accent,
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
          width: 672,
          height: 366,
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
