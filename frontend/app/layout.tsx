import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { ShellLayout } from "@/components/shell/ShellLayout";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Evidra — Autonomous AI Data Scientist & ML Research Engine",
    template: "%s | Evidra",
  },
  description:
    "Evidra is an evidence-driven autonomous AI data scientist that profiles datasets, plans hypotheses, executes multi-model ML experiments, and delivers production-grade recommendations in real time.",
  keywords: [
    "Evidra",
    "Autonomous AI Data Scientist",
    "Automated Machine Learning",
    "AutoML",
    "AI Research Agent",
    "LangGraph Orchestration",
    "Predictive Modeling",
    "Data Science Pipeline",
    "Multi-Model Evaluation",
    "Feature Engineering",
  ],
  authors: [{ name: "Evidra AI" }],
  creator: "Evidra AI",
  publisher: "Evidra AI",
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/evidra-icon-v2.png", type: "image/png" },
    ],
    shortcut: "/favicon.ico",
    apple: "/evidra-icon-v2.png",
  },
  openGraph: {
    title: "Evidra — Autonomous AI Data Scientist & ML Research Engine",
    description:
      "Transform raw datasets into high-performance, evidence-backed machine learning pipelines autonomously.",
    type: "website",
    siteName: "Evidra",
  },
  twitter: {
    card: "summary_large_image",
    title: "Evidra — Autonomous AI Data Scientist & ML Research Engine",
    description:
      "Transform raw datasets into high-performance, evidence-backed machine learning pipelines autonomously.",
  },
};

export const viewport: Viewport = {
  themeColor: "#080C0E",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-scroll-behavior="smooth" className={`dark bg-bg ${inter.variable} ${jetbrainsMono.variable}`}>
      <body className={`${inter.className} antialiased bg-bg text-text min-h-screen`}>
        <Providers>
          <ShellLayout>{children}</ShellLayout>
        </Providers>
      </body>
    </html>
  );
}
