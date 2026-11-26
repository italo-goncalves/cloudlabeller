// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The photo as a blob URL the page can show, or the reason it can't.

import { useEffect, useState } from "react";
import { readPhoto } from "../desktop/bridge";

const MIME: Record<string, string> = {
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  png: "image/png",
  webp: "image/webp",
  gif: "image/gif",
  bmp: "image/bmp",
  avif: "image/avif",
  tif: "image/tiff",
  tiff: "image/tiff",
};

export function extension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot + 1).toLowerCase();
}

/** The MIME type a photo's file name implies. */
export function mimeFor(name: string): string {
  return MIME[extension(name)] ?? "application/octet-stream";
}

export interface Photo {
  /** Blob URL once the browser has decoded the photo. */
  url: string | null;
  loading: boolean;
  /** Why it can't be shown (a format the browser can't decode, a missing file…). */
  problem: string | null;
}

const NONE: Photo = { url: null, loading: false, problem: null };

export function usePhoto(name: string | null): Photo {
  const [photo, setPhoto] = useState<Photo>(NONE);

  useEffect(() => {
    if (!name) {
      setPhoto(NONE);
      return;
    }
    let live = true;
    let url: string | null = null;
    setPhoto({ url: null, loading: true, problem: null });
    (async () => {
      let decoding = false;
      try {
        const bytes = await readPhoto(name);
        if (!live) return;
        decoding = true;
        url = URL.createObjectURL(new Blob([bytes], { type: mimeFor(name) }));
        const img = new Image();
        img.src = url;
        await img.decode();
        if (live) setPhoto({ url, loading: false, problem: null });
      } catch (e) {
        if (!live) return;
        const ext = extension(name).toUpperCase();
        setPhoto({
          url: null,
          loading: false,
          problem: decoding
            ? `The window can't display ${ext || "this"} photos. You can still label on the mask, but not see the photo.`
            : `The photo could not be read: ${e instanceof Error ? e.message : String(e)}`,
        });
      }
    })();
    return () => {
      live = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [name]);

  return photo;
}
