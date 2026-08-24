// Minimal typings for the vanilla-JS lets-scroll engine.
export interface ScrollSection {
  id: string;
  label: string;
  still?: string;
  stillMobile?: string;
  clip: string;
  clipMobile?: string;
  accent?: string;
  scroll?: number;
  linger?: number;
  eyebrow?: string;
  title?: string;
  body?: string;
  tags?: string[];
  cta?: {
    primary?: { label: string; href: string };
    secondary?: { label: string; href: string };
  };
}

export interface ScrollConfig {
  brand?: { name: string; href?: string };
  diveScroll?: number;
  connScroll?: number;
  crossfade?: number;
  hint?: string;
  nav?: boolean;
  atmosphere?: boolean;
  cta?: { label: string; href: string };
  sections: ScrollSection[];
  connectors?: (string | null)[];
  connectorsMobile?: (string | null)[];
}

export function mountLetsScroll(container: HTMLElement, config: ScrollConfig): void;
declare const _default: typeof mountLetsScroll;
export default _default;
