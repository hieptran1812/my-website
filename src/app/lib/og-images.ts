// OpenGraph Image Generator
// This creates a simple text-based OG image for each page

export function generateOGImage(title: string, description?: string): string {
  // For now, return a placeholder URL - in production, you could use:
  // - Vercel's OG Image Generation API
  // - Canvas API to generate images
  // - Pre-generated static images

  const params = new URLSearchParams({
    title: title,
    description: description || "",
  });

  // Return a placeholder for now - replace with actual OG image service
  return `/api/og?${params.toString()}`;
}

export const defaultOGImage = "/about-profile.webp"; // Fallback image

export const pageOGImages = {
  home: "/about-profile.webp",
  about: "/about-profile.webp",
  projects: "/about-profile.webp",
  contact: "/about-profile.webp",
  blog: "/about-profile.webp",
  "machine-learning": "/about-profile.webp",
  "paper-reading": "/about-profile.webp",
  "software-development": "/about-profile.webp",
  trading: "/about-profile.webp",
  notes: "/about-profile.webp",
  privacy: "/about-profile.webp",
  terms: "/about-profile.webp",
} as const;
