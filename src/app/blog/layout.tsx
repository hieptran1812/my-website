import { Metadata } from "next";

export const metadata: Metadata = {
  title:
    "Blog - Hiep Tran | AI, Machine Learning & Software Development Insights",
  description:
    "Read Hiep Tran's latest blog posts about AI, machine learning, deep learning, software development, and emerging technologies. Technical insights, tutorials, and industry analysis from an experienced AI engineer.",
  keywords: [
    "Hiep Tran Blog",
    "AI Blog",
    "Machine Learning Articles",
    "Software Development Blog",
    "Deep Learning Tutorials",
    "Technology Insights",
    "Programming Blog",
    "AI Engineering Articles",
    "Tech Industry Analysis",
    "Computer Science Blog",
    "Cryptocurrency Analysis",
    "Paper Reading",
    "Software Engineering",
  ],
  authors: [{ name: "Hiep Tran", url: "https://halleyverse.dev" }],
  openGraph: {
    title:
      "Blog - Hiep Tran | AI, Machine Learning & Software Development Insights",
    description:
      "Read Hiep Tran's latest blog posts about AI, machine learning, and software development. Technical insights and tutorials from an experienced AI engineer.",
    url: "https://halleyverse.dev/blog",
    type: "website",
    siteName: "Hiep Tran Portfolio",
    images: [
      {
        url: "/og-blog.jpg",
        width: 1200,
        height: 630,
        alt: "Hiep Tran Blog - AI & Software Development",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Blog - Hiep Tran | AI & Software Development Insights",
    description:
      "Read Hiep Tran's latest blog posts about AI, machine learning, and software development.",
    creator: "@hieptran1812",
    images: ["/twitter-blog.jpg"],
  },
  alternates: {
    canonical: "https://halleyverse.dev/blog",
    types: {
      "application/rss+xml": [
        { url: "/blog/rss.xml", title: "Hiep Tran Blog RSS Feed" },
      ],
    },
  },
};

export default function BlogLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
