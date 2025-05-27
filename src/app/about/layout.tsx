import { Metadata } from "next";

export const metadata: Metadata = {
  title: "About Hiep Tran - AI Engineer & Software Developer",
  description:
    "Learn about Hiep Tran's background in AI engineering, machine learning, and full-stack development. Discover his skills, experience, education, and professional journey in technology.",
  keywords: [
    "Hiep Tran About",
    "AI Engineer Background",
    "Machine Learning Experience",
    "Software Developer Skills",
    "Computer Science Education",
    "Programming Skills",
    "Professional Experience",
    "Technology Expertise",
  ],
  openGraph: {
    title: "About Hiep Tran - AI Engineer & Software Developer",
    description:
      "Learn about Hiep Tran's background in AI engineering, machine learning, and full-stack development. Discover his skills and professional journey.",
    url: "https://hieptran.dev/about",
    type: "profile",
    images: [
      {
        url: "/og-about.jpg",
        width: 1200,
        height: 630,
        alt: "About Hiep Tran - AI Engineer & Software Developer",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "About Hiep Tran - AI Engineer & Software Developer",
    description:
      "Learn about Hiep Tran's background in AI engineering, machine learning, and full-stack development.",
    images: ["/twitter-about.jpg"],
  },
  alternates: {
    canonical: "https://hieptran.dev/about",
  },
};

export default function AboutLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
