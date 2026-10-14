import { ImageResponse } from "next/og";

import { SHARE_IMAGE_SIZE } from "@/lib/site";

// The site's tokens (globals.css, light): ink, muted, faint, line, accent.
const INK = "#111418";
const MUTED = "#4b525c";
const SUBTLE = "#69707b";
const RULE = "#e4e7eb";
const ACCENT = "#1d4ed8";

const TABS = ["Ask the Fed", "Policy Stance", "Statement Comparison", "RAG Performance"];

function Marker({ n }: { n: number }) {
  return (
    <span
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: 30,
        height: 30,
        marginLeft: 6,
        borderRadius: 4,
        background: "rgba(29, 78, 216, 0.08)",
        color: ACCENT,
        fontSize: 18,
      }}
    >
      {n}
    </span>
  );
}

/**
 * The link-preview card for every page (Open Graph and Twitter), rendered once at build time.
 * Geist Regular only (the font next/og bundles), so hierarchy comes from size and color.
 */
export function shareImage(): ImageResponse {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "64px 72px",
          background: "#ffffff",
          color: INK,
        }}
      >
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ display: "flex", fontSize: 40, letterSpacing: -1 }}>FedRAG</div>
            <span style={{ fontSize: 22, color: SUBTLE }}>github.com/hoedongkim/fedrag</span>
          </div>
          <div style={{ marginTop: 56, fontSize: 68, lineHeight: 1.1, letterSpacing: -2, maxWidth: 1000 }}>
            Ask what the Fed actually said.
          </div>
          <div style={{ marginTop: 16, fontSize: 28, lineHeight: 1.4, color: MUTED, maxWidth: 980 }}>
            Answers built only from FOMC statements and minutes, with a citation for every claim.
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              marginTop: 36,
              paddingLeft: 20,
              borderLeft: `2px solid ${RULE}`,
              fontSize: 24,
              color: MUTED,
            }}
          >
            <span>&ldquo;&hellip;the U.S. banking system was sound and resilient&rdquo;</span>
            <Marker n={1} />
            <Marker n={2} />
          </div>
        </div>

        <div style={{ display: "flex", paddingTop: 24, borderTop: `1px solid ${RULE}`, fontSize: 24, color: MUTED }}>
          {TABS.map((tab) => (
            <span key={tab} style={{ marginRight: 40 }}>
              {tab}
            </span>
          ))}
        </div>
      </div>
    ),
    SHARE_IMAGE_SIZE,
  );
}
