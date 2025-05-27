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

export const defaultOGImage = "/image.png"; // Fallback image

export const pageOGImages = {
  home: "/image.png",
  about: "/image.png",
  projects: "/image.png",
  contact: "/image.png",
  blog: "/image.png",
  "machine-learning": "/image.png",
  "paper-reading": "/image.png",
  "software-development": "/image.png",
  crypto: "/image.png",
  notes: "/image.png",
  privacy: "/image.png",
  terms: "/image.png",
} as const;
