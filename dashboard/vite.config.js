import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite config tuned for both standalone `vite dev` and the
// Tauri shell. Tauri's `beforeDevCommand` runs `npm run dev`
// which boots vite on the port configured here.

export default defineConfig({
  plugins: [react()],
  // Tauri expects the dev server on 5173 by default.
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
  // Tauri uses Chromium 96+ on Windows / WebKit on macOS;
  // safari14 covers both.
  build: {
    target: ['es2021', 'chrome100', 'safari14'],
    minify: !process.env.TAURI_DEBUG ? 'esbuild' : false,
    sourcemap: !!process.env.TAURI_DEBUG,
  },
})
