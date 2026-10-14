/**
 * Shared Tailwind class strings: the design system's controls and text roles. Colors are the
 * semantic tokens from globals.css (ink, muted, faint, line, accent...), which switch with the
 * color scheme on their own, so nothing here needs a `dark:` variant.
 */

/** Page width for reading: prose never runs wider than this. */
export const proseWidth = "max-w-[70ch]";

/** A section heading (h2) inside a page. */
export const sectionHeadingClass = "text-heading font-semibold text-ink";

/** A small label above a value or a group (sentence case, never uppercase). */
export const labelTextClass = "text-label font-medium text-faint";

/** The one card style, for data panels (charts, tables, the selected meeting...). Use sparingly. */
export const panelClass = "rounded-md border border-line bg-canvas";

/** An inline link in running text. */
export const linkClass =
  "rounded-sm text-accent underline decoration-accent/35 underline-offset-[3px] transition-colors hover:decoration-accent";

/** A link or text button that stands on its own (no underline until hovered). */
export const textButtonClass =
  "rounded-sm text-small font-medium text-accent underline-offset-4 hover:underline";

const buttonBase =
  "inline-flex shrink-0 items-center justify-center gap-1.5 rounded-md font-medium whitespace-nowrap transition-colors select-none aria-disabled:cursor-not-allowed aria-disabled:opacity-50";

/** The main action: solid ink (near-black on light, near-white on dark). */
export const primaryButtonClass = `${buttonBase} h-9 bg-ink px-4 text-body text-canvas hover:bg-ink/85 aria-disabled:hover:bg-ink`;

/** A secondary action at the primary button's height. */
export const outlineButtonClass = `${buttonBase} h-9 border border-line-strong bg-canvas px-4 text-body text-ink hover:bg-subtle`;

/** A small secondary action (Stop, Retry, Ask again...). */
export const secondaryButtonClass = `${buttonBase} h-8 border border-line-strong bg-canvas px-3 text-small text-ink hover:bg-subtle`;

/** A text field's look without its size: add padding and a text size (fieldClass has the default). */
export const fieldBaseClass =
  "block w-full rounded-md border border-line-strong bg-canvas text-ink placeholder:text-faint aria-invalid:border-bad";

export const fieldClass = `${fieldBaseClass} px-3 py-2 text-body`;

export const labelClass = "block text-small font-medium text-ink";

export const helpClass = "text-small text-faint";

export const errorTextClass = "text-small text-bad";

/** A compact select or input (filters above a chart or a list): add a width. */
export const smallFieldClass = `${fieldBaseClass} h-8 px-2.5 text-small`;

/** A placeholder bar while something loads. */
export const skeletonClass = "rounded-sm bg-line/70 motion-safe:animate-pulse";

/** A floating readout (chart tooltips): the one surface that casts a shadow. */
export const overlayClass = "rounded-md border border-line bg-canvas px-3 py-2 text-label text-muted shadow-lg";

/** One quiet line of context under a table or a chart (bold meaning, units, source). */
export const noteClass = "mt-3 max-w-[70ch] text-label text-faint";
