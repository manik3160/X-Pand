/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "rgb(var(--bg) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        border: "rgb(var(--line) / 0.08)",
        "text-primary": "rgb(var(--text) / <alpha-value>)",
        "text-secondary": "rgb(var(--text-muted) / <alpha-value>)",
        "text-muted": "rgb(var(--text-subtle) / <alpha-value>)",
        accent: "rgb(var(--accent) / <alpha-value>)",
        "accent-fg": "rgb(var(--accent-fg) / <alpha-value>)",
        success: "rgb(var(--profit-high) / <alpha-value>)",
        warning: "rgb(var(--profit-mid) / <alpha-value>)",
        danger: "rgb(var(--profit-low) / <alpha-value>)",
        "profit-high": "rgb(var(--profit-high) / <alpha-value>)",
        "profit-mid": "rgb(var(--profit-mid) / <alpha-value>)",
        "profit-low": "rgb(var(--profit-low) / <alpha-value>)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        heading: ["'Space Grotesk'", "Inter", "system-ui", "sans-serif"],
        mono: ["'Space Grotesk'", "'JetBrains Mono'", "monospace"],
      },
      borderRadius: {
        panel: "14px",
        control: "10px",
      },
      boxShadow: {
        panel: "0 8px 24px rgb(0 0 0 / .35)",
      },
      zIndex: {
        panel: "10",
        sheet: "20",
        banner: "30",
      },
      transitionTimingFunction: {
        out: "cubic-bezier(.2, .8, .2, 1)",
      },
      animation: {
        "fade-up": "fadeUp 0.6s ease-out both",
        "fade-up-1": "fadeUp 0.6s ease-out 0.1s both",
        "fade-up-2": "fadeUp 0.6s ease-out 0.2s both",
        "fade-up-3": "fadeUp 0.6s ease-out 0.3s both",
        "fade-up-4": "fadeUp 0.6s ease-out 0.4s both",
        "fade-up-5": "fadeUp 0.6s ease-out 0.5s both",
        "fade-up-6": "fadeUp 0.6s ease-out 0.6s both",
        "fade-up-7": "fadeUp 0.6s ease-out 0.7s both",
        "fade-right": "fadeRight 0.6s ease-out 0.4s both",
        "pulse-live": "pulseLive 2s ease-in-out infinite",
        "shimmer": "shimmer 2s ease-in-out infinite",
        "spin-slow": "spin 1s linear infinite",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeRight: {
          "0%": { opacity: "0", transform: "translateX(20px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        pulseLive: {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%": { opacity: "0.5", transform: "scale(0.85)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
    },
  },
  plugins: [],
};
