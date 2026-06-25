import { copyFileSync, existsSync, mkdirSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const src = resolve(__dirname, "../node_modules/onnxruntime-web/dist");
const dest = resolve(__dirname, "../public");

mkdirSync(dest, { recursive: true });

const files = [
  "ort-wasm.wasm",
  "ort-wasm-simd.wasm",
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-threaded.wasm",
  "ort-wasm-threaded.js",
  "ort-wasm-threaded.worker.js",
];

for (const f of files) {
  const srcPath = resolve(src, f);
  if (existsSync(srcPath)) {
    copyFileSync(srcPath, resolve(dest, f));
    console.log(`Copied ${f} -> public/`);
  }
}
