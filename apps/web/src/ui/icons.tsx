// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The few icons the window's toolbars use, drawn here as line art on a
// 24-unit grid so no icon library is needed. They take the text colour.

import type { ReactNode } from "react";

function Icon({ children }: { children: ReactNode }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  );
}

export const HomeIcon = () => (
  <Icon>
    <path d="M3 11.5 12 4l9 7.5" />
    <path d="M5.5 10v10h13V10" />
    <path d="M10 20v-5h4v5" />
  </Icon>
);

export const LassoIcon = () => (
  <Icon>
    <ellipse cx="12" cy="9.5" rx="8.5" ry="5.5" strokeDasharray="3 2.5" />
    <path d="M7.5 14.2c-1.2 1.6-.6 3.8 1.5 3.8 1.6 0 2 1.3 1.2 3" />
  </Icon>
);

/** Give the selection the active class. */
export const LabelIcon = () => (
  <Icon>
    <path d="M3.5 12.5V4h8.5l8.5 8.5-8 8z" />
    <circle cx="8" cy="8.5" r="1.3" />
  </Icon>
);

export const TrashIcon = () => (
  <Icon>
    <path d="M4 7h16" />
    <path d="M9.5 7V4.5h5V7" />
    <path d="M6 7l1 13h10l1-13" />
    <path d="M10 11v5.5M14 11v5.5" />
  </Icon>
);

export const ClearIcon = () => (
  <Icon>
    <path d="M6.5 6.5l11 11M17.5 6.5l-11 11" />
  </Icon>
);

/** Place a GCP's Mark: a survey target. */
export const GcpIcon = () => (
  <Icon>
    <path d="M12 3.5 20.5 12 12 20.5 3.5 12z" />
    <path d="M12 8.5v7M8.5 12h7" />
  </Icon>
);
