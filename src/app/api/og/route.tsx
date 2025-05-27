import { ImageResponse } from "next/og";
import { NextRequest } from "next/server";

export const runtime = "edge";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const title = searchParams.get("title") || "Hiep Tran";
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
              hieptran.dev
            </div>
          </div>
        </div>
      ),
      {
        width: 1200,
        height: 630,
      }
    );
  } catch (e: any) {
    console.log(`${e.message}`);
    return new Response(`Failed to generate the image`, {
      status: 500,
    });
  }
}
