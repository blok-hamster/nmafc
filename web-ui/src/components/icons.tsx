"use client";

/* Minimal inline SVG icon set. 16x16 viewBox, currentColor for fill. */

function Icon({ d, className = "w-4 h-4" }: { d: string; className?: string }) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d={d} />
    </svg>
  );
}

/* ── Navigation ── */

export function IconDashboard(props: IconProps) {
  return (
    <Icon
      className={props.className}
      d="M2 2h5v5H2V2zm7 0h5v5H9V2zM2 9h5v5H2V9zm7 0h5v5H9V9z"
    />
  );
}

export function IconMemory(props: IconProps) {
  return (
    <Icon
      className={props.className}
      d="M3 3h4v4H3V3zm6 0h4v4H9V3zM3 9h4v4H3V9zm6 0h4v4H9V9z"
    />
  );
}

export function IconGraph(props: IconProps) {
  return (
    <svg className={props.className ?? "w-4 h-4"} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <circle cx="4" cy="4" r="2" />
      <circle cx="12" cy="4" r="2" />
      <circle cx="8" cy="12" r="2" />
      <line x1="5.5" y1="5" x2="10.5" y2="5" stroke="currentColor" strokeWidth="1.5" />
      <line x1="5" y1="5.5" x2="7" y2="10.5" stroke="currentColor" strokeWidth="1.5" />
      <line x1="11" y1="5.5" x2="9" y2="10.5" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

export function IconDecay(props: IconProps) {
  return (
    <svg className={props.className ?? "w-4 h-4"} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <polyline points="1,4 5,4 5,2" />
      <polyline points="11,2 11,4 15,4" />
      <path d="M5,4 Q8,5 9,14" />
    </svg>
  );
}

export function IconEvents(props: IconProps) {
  return (
    <Icon
      className={props.className}
      d="M2 2h12v1H2V2zm0 3h12v1H2V5zm0 3h12v1H2V8zm0 3h8v1H2v-1zm0 3h10v1H2v-1z"
    />
  );
}

export function IconDocs(props: IconProps) {
  return (
    <Icon
      className={props.className}
      d="M3 1h7l4 4v10a1 1 0 01-1 1H3a1 1 0 01-1-1V2a1 1 0 011-1zm6.5 0v4h4"
    />
  );
}

/* ── Docs section icons ── */

export function IconOverview(props: IconProps) {
  return (
    <Icon
      className={props.className}
      d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 2a5 5 0 110 10A5 5 0 018 3zm0 2a3 3 0 100 6 3 3 0 000-6z"
    />
  );
}

export function IconPlay(props: IconProps) {
  return (
    <Icon className={props.className} d="M4 2l10 6-10 6V2z" />
  );
}

export function IconTerminal(props: IconProps) {
  return (
    <Icon className={props.className} d="M1 2h14v12H1V2zm2 2v2l3 2-3 2v2l4-3-4-3V4z" />
  );
}

export function IconApi(props: IconProps) {
  return (
    <Icon
      className={props.className}
      d="M1 3h4l2-2h4l2 2h4v10H1V3zm3 1v8h8V4H6l-1-1H5L4 4z"
    />
  );
}

export function IconConfig(props: IconProps) {
  return (
    <svg className={props.className ?? "w-4 h-4"} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M8 1l1.5 1.7a5.5 5.5 0 011.6.7L13 2.5l.5 2.2-1.8 1.2a5.5 5.5 0 010 1.4l1.8 1.2-.5 2.2-1.9-.9a5.5 5.5 0 01-1.6.7L8 15l-1.5-1.7a5.5 5.5 0 01-1.6-.7L3 13.5l-.5-2.2 1.8-1.2a5.5 5.5 0 010-1.4L2.5 7.5l.5-2.2 1.9.9a5.5 5.5 0 011.6-.7L8 1zm0 3.5a2.5 2.5 0 100 5 2.5 2.5 0 000-5z" />
    </svg>
  );
}

export function IconLibrary(props: IconProps) {
  return (
    <Icon
      className={props.className}
      d="M2 1h5l1 1 1-1h5v13H9l-1-1-1 1H2V1zm1 1v11h5v-1h4v1h5V2H9V1H8L7 2H3z"
    />
  );
}

export function IconProviders(props: IconProps) {
  return (
    <Icon
      className={props.className}
      d="M2 2h5v5H2V2zm7 0h5v5H9V2zM2 9h5v5H2V9zm7 0h5v5H9V9zm-2 2h2v2H7v-2z"
    />
  );
}

/* ── UI utility icons ── */

export function IconX(props: IconProps) {
  return (
    <svg className={props.className ?? "w-4 h-4"} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <path d="M4 4l8 8M12 4l-8 8" />
    </svg>
  );
}

export function IconArrowUp(props: IconProps) {
  return (
    <svg className={props.className ?? "w-3 h-3"} viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <path d="M6 10V2M3 5l3-3 3 3" />
    </svg>
  );
}

export function IconArrowDown(props: IconProps) {
  return (
    <svg className={props.className ?? "w-3 h-3"} viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <path d="M6 2v8M3 7l3 3 3-3" />
    </svg>
  );
}

export function IconChevronDown(props: IconProps) {
  return (
    <svg className={props.className ?? "w-4 h-4"} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <path d="M4 6l4 4 4-4" />
    </svg>
  );
}

type IconProps = { className?: string };
