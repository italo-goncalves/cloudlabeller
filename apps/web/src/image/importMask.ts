// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// Reading an external mask file (the Import Mask… button): the user picks a
// .jpg or .png, the browser decodes it, and its pixels go to `maskFromPixels`.

import { pickFile, readFile } from "../desktop/bridge";
import { extension, mimeFor } from "./usePhoto";

export interface MaskPixels {
  kind: "jpg" | "png";
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
}

/** Ask for a mask file and decode it; null when the user cancels. */
export async function pickMaskPixels(): Promise<MaskPixels | null> {
  const path = await pickFile("Import mask", [
    { name: "Masks", extensions: ["jpg", "jpeg", "png"] },
  ]);
  if (!path) return null;
  const ext = extension(path);
  if (ext !== "jpg" && ext !== "jpeg" && ext !== "png") {
    throw new Error(`Unsupported mask format ".${ext}": use .jpg (intensity) or .png (alpha).`);
  }
  const bytes = await readFile(path);
  // No colour management: the intensity must be the file's own.
  const bitmap = await createImageBitmap(new Blob([bytes], { type: mimeFor(path) }), {
    colorSpaceConversion: "none",
    imageOrientation: "none",
    premultiplyAlpha: "none",
  });
  try {
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("The window could not read the mask's pixels.");
    ctx.drawImage(bitmap, 0, 0);
    const data = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    return {
      kind: ext === "png" ? "png" : "jpg",
      rgba: data.data,
      width: bitmap.width,
      height: bitmap.height,
    };
  } finally {
    bitmap.close();
  }
}
