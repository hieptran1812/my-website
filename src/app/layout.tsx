import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "./ThemeProvider";
import Navigation from "./components/Navigation";
import Footer from "./Footer";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL('https://hieptran.dev'),
  title: "Hiep Tran - AI Engineer & Full-Stack Developer Portfolio",
  description: "Experienced AI Engineer and Full-Stack Developer specializing in Machine Learning, Deep Learning, and scalable web applications. Explore my projects, blog posts, and technical insights.",
  openGraph: {
    title: "Hiep Tran - AI Engineer & Full-Stack Developer",
    description: "Experienced AI Engineer and Full-Stack Developer specializing in Machine Learning, Deep Learning, and scalable web applications.",
    url: "https://hieptran.dev",
    type: "website",
    images: [
      {
        url: "/og-image.jpg",
        width: 1200,
        height: 630,
        alt: "Hiep Tran Portfolio",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Hiep Tran - AI Engineer & Full-Stack Developer",
    description: "Experienced AI Engineer and Full-Stack Developer specializing in Machine Learning and scalable web applications.",
    images: ["/twitter-image.jpg"],
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
      </body>
    </html>
  );
}
