import type { NextConfig } from "next";

const rawBackendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";
// Avoid macOS IPv6 localhost collision by mapping localhost to 127.0.0.1
const backendUrl = rawBackendUrl.replace("//localhost", "//127.0.0.1");

let backendHost = "127.0.0.1";
let backendPort = "8000";
let backendProtocol: "http" | "https" = "http";

try {
  const parsed = new URL(backendUrl);
  backendHost = parsed.hostname;
  backendPort = parsed.port || (parsed.protocol === "https:" ? "443" : "80");
  backendProtocol = parsed.protocol.replace(":", "") as "http" | "https";
} catch {
  // fallback defaults
}

const nextConfig: NextConfig = {
  devIndicators: false,
  turbopack: {
    root: process.cwd(),
  },
  experimental: {
    proxyClientMaxBodySize: "500mb",
    serverActions: {
      bodySizeLimit: "500mb",
    },
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "Strict-Transport-Security", value: "max-age=63072000" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${backendUrl}/api/v1/:path*`,
      },
      {
        source: "/api/v1/ws/:path*",
        destination: `${backendUrl}/api/v1/ws/:path*`,
      },
    ];
  },
  images: {
    remotePatterns: [
      {
        protocol: backendProtocol,
        hostname: backendHost,
        port: backendPort,
      },
    ],
  },
};

export default nextConfig;


