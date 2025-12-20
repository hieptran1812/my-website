// Project data and utilities for the portfolio
export interface Project {
  id: number;
  title: string;
  description: string;
  category: string;
  technologies: string[];
  githubUrl?: string;
  liveUrl?: string | null;
  featured: boolean;
  status: string;
  highlights: string[];
  image?: string;
}

// Centralized project data
export const projects: Project[] = [
  {
    id: 1,
    title: "AI-Powered Natural Language Processing Platform",
    description:
      "A comprehensive NLP platform built with transformer models for text analysis, sentiment analysis, and language understanding. Supports multiple languages and real-time processing.",
    image: "/project-placeholder.jpg",
    category: "Machine Learning",
    technologies: [
      "Python",
      "PyTorch",
      "Transformers",
      "FastAPI",
      "Docker",
      "Kubernetes",
    ],
    githubUrl: "https://github.com/hieptran1812/nlp-platform",
    liveUrl: "https://nlp-platform-demo.com",
    featured: true,
    status: "Production",
    highlights: [
      "98% accuracy on sentiment analysis",
      "Supports 15+ languages",
      "10K+ daily active users",
      "Sub-100ms response time",
    ],
  },
];

// Utility functions for project data
export const getProjectById = (id: number): Project | undefined => {
  return projects.find((project) => project.id === id);
};

// Utility functions for projects
export const getAllProjects = (): Project[] => {
  return projects;
};

export const getFeaturedProjects = (): Project[] => {
  return projects.filter((project) => project.featured);
};

export const getProjectsByCategory = (category: string): Project[] => {
  return projects.filter(
    (project) => project.category.toLowerCase() === category.toLowerCase()
  );
};

export const getProjectsByStatus = (status: string): Project[] => {
  return projects.filter(
    (project) => project.status.toLowerCase() === status.toLowerCase()
  );
};

export const getProjectCategories = (): string[] => {
  const categories = new Set(projects.map((project) => project.category));
  return Array.from(categories);
};

export const searchProjects = (query: string): Project[] => {
  const lowercaseQuery = query.toLowerCase();
  return projects.filter(
    (project) =>
      project.title.toLowerCase().includes(lowercaseQuery) ||
      project.description.toLowerCase().includes(lowercaseQuery) ||
      project.category.toLowerCase().includes(lowercaseQuery) ||
      project.technologies.some((tech) =>
        tech.toLowerCase().includes(lowercaseQuery)
      ) ||
      project.highlights.some((highlight) =>
        highlight.toLowerCase().includes(lowercaseQuery)
      )
  );
};
