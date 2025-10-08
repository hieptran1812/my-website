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

export const defaultOGImage = "/about-profile.png"; // Fallback image

export const pageOGImages = {
  home: "/about-profile.png",
  about: "/about-profile.png",
  projects: "/about-profile.png",
  contact: "/about-profile.png",
  blog: "/about-profile.png",
  "machine-learning": "/about-profile.png",
  "paper-reading": "/about-profile.png",
  "software-development": "/about-profile.png",
  trading: "/about-profile.png",
  notes: "/about-profile.png",
  privacy: "/about-profile.png",
  terms: "/about-profile.png",
} as const;
