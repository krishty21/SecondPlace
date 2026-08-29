import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CipherMind Sentinel — AI SOC Copilot",
  description:
    "AI Security Operations Copilot for network intrusion detection, trained on UNSW-NB15. Live inference, incident correlation, explainability and real-time detection replay.",
  keywords: ["SOC", "UNSW-NB15", "intrusion detection", "explainable AI", "security operations"],
  authors: [{ name: "CipherMind AI" }],
  icons: {
    icon: "/logo.svg",
  },
  openGraph: {
    title: "CipherMind Sentinel — AI SOC Copilot",
    description: "AI Security Operations Copilot · UNSW-NB15 · research prototype",
    siteName: "CipherMind Sentinel",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
