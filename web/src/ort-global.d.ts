import type * as OrtType from "onnxruntime-web";

declare global {
  const ort: typeof OrtType;
}
