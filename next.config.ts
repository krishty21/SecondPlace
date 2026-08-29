import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* Vercel-compatible config — no standalone output */
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
};

export default nextConfig;
