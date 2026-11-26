// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The page the desktop shell loads: `pnpm dev` serves it for `tauri dev`,
// `pnpm build` writes dist/ for the installer.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  clearScreen: false,
  server: { port: 5174, strictPort: true },
  build: { target: "es2022", outDir: "dist", emptyOutDir: true },
  test: { environment: "jsdom" },
});
