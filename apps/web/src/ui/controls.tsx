// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Ítalo Gomes Gonçalves

// The window's small controls, on the kit's tokens: buttons, a modal dialog,
// labelled fields, number and range inputs, selects, checks, a progress bar.
// Each screen builds from these, so the look is decided once.

import { cn } from "@geokit/ui";
import { type ReactNode, useEffect, useId, useRef } from "react";

type ButtonKind = "primary" | "plain" | "danger" | "quiet";

const BUTTON: Record<ButtonKind, string> = {
  primary: "bg-accent text-accent-fg hover:bg-accent-hover",
  plain: "bg-surface-2 hover:bg-surface-3",
  danger: "bg-danger text-white hover:opacity-90",
  quiet: "hover:bg-surface-2",
};

export function Button({
  kind = "plain",
  className,
  children,
  ...rest
}: { kind?: ButtonKind } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={cn(
        "h-control rounded-md px-3 whitespace-nowrap disabled:cursor-default disabled:opacity-50",
        BUTTON[kind],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

/** A small square button showing an icon; `title` names it, for the tooltip and screen readers.
 *  `pressed` makes it a toggle. */
export function IconButton({
  title,
  kind = "quiet",
  pressed,
  className,
  children,
  ...rest
}: { title: string; kind?: ButtonKind; pressed?: boolean } & Omit<
  React.ButtonHTMLAttributes<HTMLButtonElement>,
  "title"
>) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      aria-pressed={pressed}
      className={cn(
        "inline-flex size-8 items-center justify-center rounded-md disabled:cursor-default disabled:opacity-40",
        pressed ? BUTTON.primary : BUTTON[kind],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

/** A modal dialog: Escape and the backdrop close it, unless it is `busy`. */
export function Dialog({
  title,
  onClose,
  children,
  footer,
  wide,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  const box = useRef<HTMLDivElement>(null);
  // the latest onClose, so a parent handing a new function on each render
  // doesn't re-run the effect below (which would pull focus back to the
  // first field on every keystroke)
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        closeRef.current();
      }
    };
    window.addEventListener("keydown", key, true);
    box.current?.querySelector<HTMLElement>("input, select, button")?.focus();
    return () => window.removeEventListener("keydown", key, true);
  }, []);
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: a click on the backdrop closes, as Escape does
    <div
      role="presentation"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        ref={box}
        role="dialog"
        aria-label={title}
        className={cn(
          "flex max-h-[90vh] flex-col rounded-lg border border-border bg-surface-1 shadow-xl",
          wide ? "w-[min(56rem,95vw)]" : "w-[min(32rem,95vw)]",
        )}
      >
        <h2 className="border-b border-border px-4 py-3 font-semibold">{title}</h2>
        <div className="min-h-0 flex-1 overflow-auto px-4 py-3">{children}</div>
        {footer && (
          <div className="flex justify-end gap-2 border-t border-border px-4 py-3">{footer}</div>
        )}
      </div>
    </div>
  );
}

/** A label above (or beside) its control, with an optional hint under it. */
export function Field({
  label,
  hint,
  children,
  inline,
}: {
  label: string;
  hint?: ReactNode;
  children: (id: string) => ReactNode;
  inline?: boolean;
}) {
  const id = useId();
  return (
    <div className={cn("mb-3", inline && "flex items-center gap-3")}>
      <label htmlFor={id} className={cn("block text-fg-muted", !inline && "mb-1")}>
        {label}
      </label>
      {children(id)}
      {hint && <p className="mt-1 text-fg-faint">{hint}</p>}
    </div>
  );
}

const INPUT =
  "h-control rounded-md border border-border bg-surface-0 px-2 focus:border-accent focus:outline-none";

export function NumberInput({
  id,
  value,
  onChange,
  min,
  max,
  step = 1,
  className,
}: {
  id?: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
}) {
  return (
    <input
      id={id}
      type="number"
      className={cn(INPUT, "w-28", className)}
      value={Number.isFinite(value) ? value : ""}
      min={min}
      max={max}
      step={step}
      onChange={(e) => {
        const v = Number(e.target.value);
        if (e.target.value !== "" && Number.isFinite(v)) onChange(v);
      }}
    />
  );
}

export function TextInput({
  id,
  value,
  onChange,
  placeholder,
  className,
  list,
}: {
  id?: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
  list?: string;
}) {
  return (
    <input
      id={id}
      type="text"
      list={list}
      className={cn(INPUT, "w-full", className)}
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

export function Slider({
  id,
  value,
  onChange,
  min,
  max,
  step = 1,
  className,
  label,
}: {
  id?: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  className?: string;
  label?: string;
}) {
  return (
    <input
      id={id}
      type="range"
      aria-label={label}
      className={cn("w-full accent-[var(--gk-accent)]", className)}
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}

export function Select<T extends string>({
  id,
  value,
  onChange,
  options,
  className,
}: {
  id?: string;
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string; disabled?: boolean }[];
  className?: string;
}) {
  return (
    <select
      id={id}
      className={cn(INPUT, "w-full", className)}
      value={value}
      onChange={(e) => onChange(e.target.value as T)}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value} disabled={o.disabled}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function Check({
  checked,
  onChange,
  children,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  children: ReactNode;
  disabled?: boolean;
}) {
  return (
    <label className={cn("mb-2 flex items-center gap-2", disabled && "opacity-50")}>
      <input
        type="checkbox"
        className="accent-[var(--gk-accent)]"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>{children}</span>
    </label>
  );
}

export function Radios<T extends string>({
  value,
  onChange,
  options,
  name,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string; disabled?: boolean }[];
  name: string;
}) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1" role="radiogroup">
      {options.map((o) => (
        <label
          key={o.value}
          className={cn("flex items-center gap-1.5", o.disabled && "opacity-50")}
        >
          <input
            type="radio"
            name={name}
            className="accent-[var(--gk-accent)]"
            checked={value === o.value}
            disabled={o.disabled}
            onChange={() => onChange(o.value)}
          />
          {o.label}
        </label>
      ))}
    </div>
  );
}

export function ProgressBar({ fraction }: { fraction: number | null }) {
  return (
    <div className="h-1.5 w-32 overflow-hidden rounded bg-surface-3">
      {fraction === null ? (
        <div className="h-full w-1/3 animate-pulse bg-accent" />
      ) : (
        <div className="h-full bg-accent" style={{ width: `${Math.round(fraction * 100)}%` }} />
      )}
    </div>
  );
}

/** A pane's heading. */
export function PaneTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex h-bar items-center gap-2 border-b border-border px-3 font-semibold">
      <span className="mr-auto">{children}</span>
      {right}
    </div>
  );
}
