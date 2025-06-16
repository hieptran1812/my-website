"use client";

import React, { useState } from "react";
import { useTheme } from "../ThemeProvider";

interface BlogShareSectionProps {
  postSlug: string;
  title: string;
  url?: string;
}

interface ShareButton {
  platform: string;
  label: string;
  icon: React.ReactNode;
  getUrl: (url: string, title: string) => string;
  color: string;
}

const shareButtons: ShareButton[] = [
  {
    platform: "twitter",
    label: "Twitter",
    icon: (
      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
      </svg>
    ),
    getUrl: (url: string, title: string) =>
      `https://twitter.com/intent/tweet?url=${encodeURIComponent(
        url
      )}&text=${encodeURIComponent(title)}&via=hieptran1812`,
    color: "#000000",
  },
  {
    platform: "facebook",
    label: "Facebook",
    icon: (
      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z" />
      </svg>
    ),
    getUrl: (url: string) =>
      `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`,
    color: "#1877F2",
  },
  {
    platform: "linkedin",
    label: "LinkedIn",
    icon: (
      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
      </svg>
    ),
    getUrl: (url: string, title: string) =>
      `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(
        url
      )}&title=${encodeURIComponent(title)}`,
    color: "#0A66C2",
  },
  {
    platform: "reddit",
    label: "Reddit",
    icon: (
      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.561 12 8 12.562 8 13.25c0 .687.561 1.248 1.25 1.248.687 0 1.248-.561 1.248-1.249 0-.688-.561-1.249-1.249-1.249zm5.5 0c-.687 0-1.248.561-1.248 1.25 0 .687.561 1.248 1.249 1.248.688 0 1.249-.561 1.249-1.249 0-.687-.562-1.249-1.25-1.249zm-5.466 3.99a.327.327 0 0 0-.231.094.33.33 0 0 0 0 .463c.842.842 2.484.913 2.961.913.477 0 2.105-.056 2.961-.913a.361.361 0 0 0 .029-.463.33.33 0 0 0-.464 0c-.547.533-1.684.73-2.512.73-.828 0-1.979-.196-2.512-.73a.326.326 0 0 0-.232-.095z" />
      </svg>
    ),
    getUrl: (url: string, title: string) =>
      `https://reddit.com/submit?url=${encodeURIComponent(
        url
      )}&title=${encodeURIComponent(title)}`,
    color: "#FF4500",
  },
  {
    platform: "telegram",
    label: "Telegram",
    icon: (
      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M11.944 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0a12 12 0 0 0-.056 0zm4.962 7.224c.1-.002.321.023.465.14a.506.506 0 0 1 .171.325c.016.093.036.306.02.472-.18 1.898-.962 6.502-1.36 8.627-.168.9-.499 1.201-.82 1.23-.696.065-1.225-.46-1.9-.902-1.056-.693-1.653-1.124-2.678-1.8-1.185-.78-.417-1.21.258-1.91.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.14-5.061 3.345-.48.33-.913.49-1.302.48-.428-.008-1.252-.241-1.865-.44-.752-.245-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.83-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635z" />
      </svg>
    ),
    getUrl: (url: string, title: string) =>
      `https://t.me/share/url?url=${encodeURIComponent(
        url
      )}&text=${encodeURIComponent(title)}`,
    color: "#0088CC",
  },
  {
    platform: "whatsapp",
    label: "WhatsApp",
    icon: (
      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893A11.821 11.821 0 0020.886 3.488" />
      </svg>
    ),
    getUrl: (url: string, title: string) =>
      `https://wa.me/?text=${encodeURIComponent(`${title} ${url}`)}`,
    color: "#25D366",
  },
  {
    platform: "email",
    label: "Email",
    icon: (
      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
        <path d="M24 5.457v13.909c0 .904-.732 1.636-1.636 1.636h-3.819V11.73L12 16.64l-6.545-4.91v9.273H1.636A1.636 1.636 0 0 1 0 19.366V5.457c0-.904.732-1.636 1.636-1.636h.749L12 10.845 21.615 3.821h.749c.904 0 1.636.732 1.636 1.636z" />
      </svg>
    ),
    getUrl: (url: string, title: string) =>
      `mailto:?subject=${encodeURIComponent(title)}&body=${encodeURIComponent(
        `Check out this article: ${title}\n\n${url}`
      )}`,
    color: "#EA4335",
  },
  {
    platform: "copy-link",
    label: "Copy Link",
    icon: (
      <svg
        className="w-5 h-5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M13.19 8.688a4.5 4.5 0 0 1 1.242 7.244l-4.5 4.5a4.5 4.5 0 0 1-6.364-6.364l1.757-1.757m13.35-.622 1.757-1.757a4.5 4.5 0 0 0-6.364-6.364l-4.5 4.5a4.5 4.5 0 0 0 1.242 7.244"
        />
      </svg>
    ),
    getUrl: () => "",
    color: "#6B7280",
  },
];

export default function BlogShareSection({
  postSlug,
  title,
  url,
}: BlogShareSectionProps) {
  const { theme, isReadingMode } = useTheme();
  const [copySuccess, setCopySuccess] = useState(false);
  const [shareCount, setShareCount] = useState(0);
  const [nativeShareSupported, setNativeShareSupported] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSharing, setIsSharing] = useState(false);

  const currentUrl =
    url || (typeof window !== "undefined" ? window.location.href : "");

  // Check for native share support and load initial share count
  React.useEffect(() => {
    if (typeof window !== "undefined" && "share" in navigator) {
      setNativeShareSupported(true);
    }

    // Load initial share count
    const loadShareCount = async () => {
      try {
        const response = await fetch(
          `/api/shares?postSlug=${encodeURIComponent(postSlug)}`
        );
        if (response.ok) {
          const data = await response.json();
          setShareCount(data.shareCount || 0);
        }
      } catch (error) {
        console.error("Error loading share count:", error);
      } finally {
        setIsLoading(false);
      }
    };

    loadShareCount();
  }, [postSlug]);

  const handleNativeShare = async () => {
    if (isSharing) return; // Prevent multiple simultaneous shares

    setIsSharing(true);
    try {
      if (navigator.share) {
        await navigator.share({
          title: title,
          text: `Check out this article: ${title}`,
          url: currentUrl,
        });

        // Record the share action
        await fetch("/api/shares", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ postSlug, platform: "native" }),
        });

        setShareCount((prev) => prev + 1);
      }
    } catch (error) {
      // Only log if it's not a user cancellation
      if (error instanceof Error && error.name !== "AbortError") {
        console.error("Error with native share:", error);
      }
    } finally {
      setIsSharing(false);
    }
  };

  const handleShare = async (platform: string) => {
    try {
      // Record the share action
      await fetch("/api/shares", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ postSlug, platform }),
      });

      // Handle different share platforms
      if (platform === "copy-link") {
        await navigator.clipboard.writeText(currentUrl);
        setCopySuccess(true);
        setTimeout(() => setCopySuccess(false), 2000);
      } else {
        const shareButton = shareButtons.find(
          (btn) => btn.platform === platform
        );
        if (shareButton) {
          const shareUrl = shareButton.getUrl(currentUrl, title);

          // For email, use window.location to avoid popup blockers
          if (platform === "email") {
            window.location.href = shareUrl;
          } else {
            window.open(shareUrl, "_blank", "width=600,height=400");
          }
        }
      }

      // Update share count (optional)
      setShareCount((prev) => prev + 1);
    } catch (error) {
      console.error("Error sharing:", error);

      // Fallback for copy link
      if (platform === "copy-link") {
        try {
          // Fallback method for older browsers
          const textArea = document.createElement("textarea");
          textArea.value = currentUrl;
          document.body.appendChild(textArea);
          textArea.select();
          document.execCommand("copy");
          document.body.removeChild(textArea);
          setCopySuccess(true);
          setTimeout(() => setCopySuccess(false), 2000);
        } catch (fallbackError) {
          console.error("Fallback copy failed:", fallbackError);
        }
      }
    }
  };

  return (
    <div
      className="border-t mt-12 pt-8"
      style={{
        borderColor: isReadingMode
          ? theme === "dark"
            ? "#52403d"
            : "#fef3c7"
          : "var(--border)",
      }}
    >
      <div className="text-center">
        <h3
          className="text-xl font-semibold mb-6"
          style={{
            color: isReadingMode
              ? theme === "dark"
                ? "#f5e6d3"
                : "#92400e"
              : "var(--text-primary)",
          }}
        >
          Share this article
        </h3>

        {/* Native Share Button for Mobile */}
        {nativeShareSupported && (
          <div className="mb-4">
            <button
              onClick={handleNativeShare}
              disabled={isSharing}
              className="group flex items-center gap-3 px-8 py-3 rounded-xl transition-all duration-300 border mx-auto disabled:opacity-50 disabled:cursor-not-allowed"
              style={{
                backgroundColor: isReadingMode
                  ? theme === "dark"
                    ? "rgba(82, 64, 61, 0.3)"
                    : "rgba(254, 243, 199, 0.5)"
                  : "var(--surface-accent)",
                borderColor: "var(--accent)",
                color: "var(--accent)",
              }}
              onMouseEnter={(e) => {
                if (!isSharing) {
                  e.currentTarget.style.backgroundColor = "var(--accent)";
                  e.currentTarget.style.color = "white";
                }
              }}
              onMouseLeave={(e) => {
                if (!isSharing) {
                  e.currentTarget.style.backgroundColor = isReadingMode
                    ? theme === "dark"
                      ? "rgba(82, 64, 61, 0.3)"
                      : "rgba(254, 243, 199, 0.5)"
                    : "var(--surface-accent)";
                  e.currentTarget.style.color = "var(--accent)";
                }
              }}
              title="Share using your device's native share menu"
            >
              {isSharing ? (
                <svg
                  className="w-5 h-5 animate-spin"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
              ) : (
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z"
                  />
                </svg>
              )}
              <span className="font-medium">
                {isSharing ? "Sharing..." : "Share"}
              </span>
            </button>
          </div>
        )}

        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3 mb-6">
          {shareButtons.map((button) => (
            <button
              key={button.platform}
              onClick={() => handleShare(button.platform)}
              className="group flex flex-col items-center gap-2 px-3 py-4 rounded-xl transition-all duration-300 border hover:scale-105"
              style={{
                backgroundColor: isReadingMode
                  ? theme === "dark"
                    ? "rgba(82, 64, 61, 0.3)"
                    : "rgba(254, 243, 199, 0.5)"
                  : "var(--surface)",
                borderColor: isReadingMode
                  ? theme === "dark"
                    ? "#52403d"
                    : "#f3e8ff"
                  : "var(--border)",
                color: isReadingMode
                  ? theme === "dark"
                    ? "#e8d5b7"
                    : "#78350f"
                  : "var(--text-secondary)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = isReadingMode
                  ? theme === "dark"
                    ? "rgba(82, 64, 61, 0.5)"
                    : "rgba(254, 243, 199, 0.8)"
                  : "var(--surface-accent)";
                e.currentTarget.style.borderColor = button.color;
                e.currentTarget.style.color = button.color;
                e.currentTarget.style.transform = "scale(1.05)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = isReadingMode
                  ? theme === "dark"
                    ? "rgba(82, 64, 61, 0.3)"
                    : "rgba(254, 243, 199, 0.5)"
                  : "var(--surface)";
                e.currentTarget.style.borderColor = isReadingMode
                  ? theme === "dark"
                    ? "#52403d"
                    : "#f3e8ff"
                  : "var(--border)";
                e.currentTarget.style.color = isReadingMode
                  ? theme === "dark"
                    ? "#e8d5b7"
                    : "#78350f"
                  : "var(--text-secondary)";
                e.currentTarget.style.transform = "scale(1)";
              }}
              title={`Share on ${button.label}`}
            >
              <div
                className="flex items-center justify-center w-6 h-6"
                style={{ color: "inherit" }}
              >
                {button.icon}
              </div>
              <span className="text-xs font-medium text-center leading-tight">
                {button.platform === "copy-link" && copySuccess
                  ? "Copied!"
                  : button.label}
              </span>
            </button>
          ))}
        </div>

        {shareCount > 0 && !isLoading && (
          <p
            className="text-sm mb-4"
            style={{
              color: isReadingMode
                ? theme === "dark"
                  ? "#d97706"
                  : "#78350f"
                : "var(--text-secondary)",
            }}
          >
            📊 This article has been shared {shareCount}{" "}
            {shareCount === 1 ? "time" : "times"}
          </p>
        )}

        <div
          className="mt-6 pt-6 border-t"
          style={{
            borderColor: isReadingMode
              ? theme === "dark"
                ? "#52403d"
                : "#fef3c7"
              : "var(--border)",
          }}
        >
          <p
            className="text-sm"
            style={{
              color: isReadingMode
                ? theme === "dark"
                  ? "#d97706"
                  : "#78350f"
                : "var(--text-secondary)",
            }}
          >
            💡 Found this article helpful? Share it with your network and help
            others discover valuable content!
          </p>
        </div>
      </div>
    </div>
  );
}
