import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Projects - Hiep Tran | AI & Web Development Portfolio",
  description:
    "Explore Hiep Tran's portfolio of AI, machine learning, and web development projects. View detailed case studies of innovative solutions in artificial intelligence, data science, and full-stack development.",
  keywords: [
    "Hiep Tran Projects",
    "AI Projects Portfolio",
    "Machine Learning Projects",
    "Web Development Portfolio",
    "Software Projects",
    "Open Source Projects",
    "Data Science Projects",
    "Programming Portfolio",
    "Technology Showcase",
    "Development Work",
  ],
  openGraph: {
    title: "Projects - Hiep Tran | AI & Web Development Portfolio",
    description:
      "Explore Hiep Tran's portfolio of AI, machine learning, and web development projects. View innovative solutions and detailed case studies.",
    url: "https://halleyverse.dev/projects",
    type: "website",
    images: [
      {
        url: "/og-projects.jpg",
        width: 1200,
        height: 630,
        alt: "Hiep Tran Projects Portfolio",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Projects - Hiep Tran | AI & Web Development Portfolio",
    description:
      "Explore Hiep Tran's portfolio of AI, machine learning, and web development projects.",
    images: ["/twitter-projects.jpg"],
  },
  alternates: {
    canonical: "https://halleyverse.dev/projects",
  },
};

export default function ProjectsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
