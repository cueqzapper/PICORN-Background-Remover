import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  base: "./",
  optimizeDeps: { noDiscovery: true },
  server: {
    host: "127.0.0.1",
    port: 4185,
    strictPort: true,
  },
  preview: {
    host: "127.0.0.1",
    port: 4185,
    strictPort: true,
  },
  build: {
    rollupOptions: {
      input: { index: resolve(process.cwd(), "index.html") },
    },
  },
});
