/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "selector",
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "Inter", "system-ui", "sans-serif"],
        mono: [
          "var(--font-jetbrains-mono)",
          "JetBrains Mono",
          "Menlo",
          "Monaco",
          "monospace",
        ],
      },
      colors: {
        primary: {
          50: "#f0f9ff",
          100: "#e0f2fe",
          200: "#bae6fd",
          300: "#7dd3fc",
          400: "#38bdf8",
          500: "#0ea5e9",
          600: "#0284c7",
          700: "#0369a1",
          800: "#075985",
          900: "#0c4a6e",
        },
        gray: {
          950: "#0a0a0a",
        },
      },
      animation: {
        "pulse-slow": "pulse-slow 4s ease-in-out infinite",
        "pulse-slower": "pulse-slower 6s ease-in-out infinite",
        "pulse-slowest": "pulse-slowest 8s ease-in-out infinite",
        "spin-slow": "spin-slow 3s linear infinite",
        twinkle: "twinkle 4s ease-in-out infinite",
        "float-slow": "float-slow 8s ease-in-out infinite",
        "glow-pulse": "glow-pulse 3s ease-in-out infinite",
        "shooting-star": "shooting-star 5s linear infinite",
      },
      keyframes: {
        "pulse-slow": {
          "0%, 100%": {
            opacity: "0.4",
            transform: "scale(1)",
          },
          "50%": {
            opacity: "0.6",
            transform: "scale(1.05)",
          },
        },
        "pulse-slower": {
          "0%, 100%": {
            opacity: "0.3",
            transform: "scale(1)",
          },
          "50%": {
            opacity: "0.5",
            transform: "scale(1.08)",
          },
        },
        "pulse-slowest": {
          "0%, 100%": {
            opacity: "0.2",
            transform: "scale(1)",
          },
          "50%": {
            opacity: "0.4",
            transform: "scale(1.03)",
          },
        },
        "spin-slow": {
          from: {
            transform: "rotate(0deg)",
          },
          to: {
            transform: "rotate(360deg)",
          },
        },
        twinkle: {
          "0%, 100%": {
            opacity: 0.2,
            transform: "scale(0.8)",
          },
          "50%": {
            opacity: 0.8,
            transform: "scale(1)",
          },
        },
        "float-slow": {
          "0%, 100%": {
            transform: "translateY(0) translateX(0)",
          },
          "25%": {
            transform: "translateY(-5px) translateX(5px)",
          },
          "50%": {
            transform: "translateY(5px) translateX(10px)",
          },
          "75%": {
            transform: "translateY(10px) translateX(-5px)",
          },
        },
        "glow-pulse": {
          "0%, 100%": {
            boxShadow: "0 0 20px 5px rgba(14, 165, 233, 0.3)",
          },
          "50%": {
            boxShadow: "0 0 30px 8px rgba(14, 165, 233, 0.5)",
          },
        },
        "shooting-star": {
          "0%": {
            transform: "translateX(0) translateY(0) rotate(45deg)",
            opacity: 0,
          },
          "10%": {
            opacity: 1,
          },
          "100%": {
            transform: "translateX(500px) translateY(500px) rotate(45deg)",
            opacity: 0,
          },
        },
      },
      colors: {
        gray: {
          950: "#0a0a0a",
        },
      },
      backdropBlur: {
        xs: "2px",
      },
    },
  },
  plugins: [],
};
