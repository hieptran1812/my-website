import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Contact Hiep Tran - AI Engineer & Full-Stack Developer",
  description:
    "Get in touch with Hiep Tran for AI engineering projects, machine learning consulting, full-stack development, or collaboration opportunities. Available for freelance work and technical partnerships.",
  keywords: [
    "Contact Hiep Tran",
    "AI Engineer Contact",
    "Machine Learning Consultant",
    "Freelance Developer",
    "Technical Collaboration",
    "AI Project Consultation",
    "Software Development Services",
    "Hire AI Engineer",
    "Contact Information",
  ],
  openGraph: {
    title: "Contact Hiep Tran - AI Engineer & Full-Stack Developer",
    description:
      "Get in touch with Hiep Tran for AI engineering projects, machine learning consulting, or collaboration opportunities.",
    url: "https://hieptran.dev/contact",
    type: "website",
    images: [
      {
        url: "/og-contact.jpg",
        width: 1200,
        height: 630,
        alt: "Contact Hiep Tran",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Contact Hiep Tran - AI Engineer & Full-Stack Developer",
    description:
      "Get in touch for AI engineering projects, machine learning consulting, or collaboration opportunities.",
    images: ["/twitter-contact.jpg"],
  },
  alternates: {
    canonical: "https://hieptran.dev/contact",
  },
};

export default function ContactLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
