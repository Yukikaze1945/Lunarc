/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        davinci: {
          dark: "#121214",
          surface: "#18181b",
          panel: "#1e1e22",
          panelHover: "#25252b",
          card: "#26262c",
          border: "#32323a",
          borderLight: "#42424e",
          orange: "#e58e26",
          orangeHover: "#f39c12",
          cyan: "#00cec9",
          red: "#eb4d4b",
          green: "#2ecc71",
          text: "#f1f2f6",
          muted: "#8a8f9d",
          dim: "#575b66"
        }
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'Consolas', 'monospace'],
        sans: ['"Segoe UI"', '-apple-system', 'BlinkMacSystemFont', 'Roboto', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
