import HeroSection from "./HeroSection";
import LatestProjectsSection from "./LatestProjectsSection";
import BlogSection from "./BlogSection";
import ContactSection from "./ContactSection";
import { Metadata } from "next";

export const metadata: Metadata = {
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
    description:
      "Experienced AI Engineer and Full-Stack Developer specializing in Machine Learning and scalable web applications.",
    images: ["/twitter-image.jpg"],
  },
};

export default function Home() {
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "Person",
    name: "Hiep Tran",
    jobTitle: "AI Engineer & Full-Stack Developer",
    description:
      "Experienced AI Engineer and Full-Stack Developer specializing in Machine Learning, Deep Learning, and scalable web applications.",
    url: "https://halleyverse.dev",
    image: "https://halleyverse.dev/profile-image.jpg",
    sameAs: [
      "https://twitter.com/hieptran1812",
      "https://github.com/hieptran1812",
      "https://linkedin.com/in/hieptran1812",
    ],
    knowsAbout: [
      "Machine Learning",
      "Deep Learning",
      "Artificial Intelligence",
      "Full-Stack Development",
      "Python",
      "JavaScript",
      "React",
      "Next.js",
      "Software Engineering",
    ],
    worksFor: {
      "@type": "Organization",
      name: "Freelance",
    },
    address: {
      "@type": "PostalAddress",
      addressLocality: "Ho Chi Minh City",
      addressCountry: "Vietnam",
    },
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify(structuredData),
        }}
      />
      <HeroSection />
      <LatestProjectsSection />
      <BlogSection />
      <ContactSection />
    </>
  );
}
