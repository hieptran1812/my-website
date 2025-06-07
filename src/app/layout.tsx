import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "./ThemeProvider";
import Navigation from "./components/Navigation";
import Footer from "./Footer";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { Analytics } from "@vercel/analytics/next";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://halleyverse.dev"),
  title: "Hiep Tran - AI Engineer & Full-Stack Developer Portfolio",
  description:
    "Experienced AI Engineer and Full-Stack Developer specializing in Machine Learning, Deep Learning, and scalable web applications. Explore my projects, blog posts, and technical insights.",
  openGraph: {
    title: "Hiep Tran - AI Engineer & Full-Stack Developer",
    description:
      "Experienced AI Engineer and Full-Stack Developer specializing in Machine Learning, Deep Learning, and scalable web applications.",
    url: "https://halleyverse.dev",
    type: "website",
    images: [
      {
        url: "https://halleyverse.dev/about-profile.png",
        width: 1200,
        height: 1200,
        alt: "Hiep Tran - AI Engineer Profile",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Hiep Tran - AI Engineer & Full-Stack Developer",
    description:
      "Experienced AI Engineer and Full-Stack Developer specializing in Machine Learning and scalable web applications.",
    images: ["https://halleyverse.dev/about-profile.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ThemeProvider>
          <Navigation />
          {children}
          <Footer />
        </ThemeProvider>
        <SpeedInsights />
        <Analytics />
      </body>
    </html>
  );
}
