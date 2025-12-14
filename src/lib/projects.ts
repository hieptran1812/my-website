// Utility functions for processing markdown project files
import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { remark } from "remark";
import html from "remark-html";
import remarkGfm from "remark-gfm";

// Project interface for type safety
export interface ProjectData {
  id: string;
  title: string;
  excerpt: string;
  description: string;
  content: string;
  category: string;
  subcategory?: string;
  technologies: string[];
  status: string;
  featured: boolean;
  publishDate: string;
  lastUpdated: string;
  githubUrl?: string | null;
  liveUrl?: string | null;
  stars?: number;
  image?: string | null;
  highlights: string[];
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  slug: string;
}

export interface ProjectMetadata {
  id: string;
  title: string;
  excerpt: string;
  description: string;
  category: string;
  subcategory?: string;
  technologies: string[];
  status: string;
  featured: boolean;
  publishDate: string;
  lastUpdated: string;
  githubUrl?: string | null;
  liveUrl?: string | null;
  stars?: number;
  image?: string | null;
  highlights: string[];
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  slug: string;
}

const projectsDirectory = path.join(process.cwd(), "content/projects");

// Get all project files
export function getProjectFiles(): string[] {
  try {
    if (!fs.existsSync(projectsDirectory)) {
      console.warn("Projects directory does not exist:", projectsDirectory);
      return [];
    }
    return fs
      .readdirSync(projectsDirectory)
      .filter((file) => file.endsWith(".md"));
  } catch (error) {
    console.error("Error reading projects directory:", error);
    return [];
  }
}

// Get project data by slug
export async function getProjectBySlug(
  slug: string
): Promise<ProjectData | null> {
  try {
    const fullPath = path.join(projectsDirectory, `${slug}.md`);

    if (!fs.existsSync(fullPath)) {
      return null;
    }

    const fileContents = fs.readFileSync(fullPath, "utf8");
    const { data, content } = matter(fileContents);

    // Process markdown content to HTML
    const processedContent = await remark()
      .use(remarkGfm)
      .use(html)
      .process(content);

    return {
      id: slug,
      slug,
      title: data.title || "",
      excerpt: data.excerpt || "",
      description: data.description || "",
      content: processedContent.toString(),
      category: data.category || "",
      subcategory: data.subcategory,
      technologies: data.technologies || [],
      status: data.status || "",
      featured: data.featured || false,
      publishDate: data.publishDate || "",
      lastUpdated: data.lastUpdated || "",
      githubUrl: data.githubUrl || null,
      liveUrl: data.liveUrl || null,
      stars: data.stars,
      image: data.image || null,
      highlights: data.highlights || [],
      difficulty: data.difficulty || "Intermediate",
    };
  } catch (error) {
    console.error(`Error reading project ${slug}:`, error);
    return null;
  }
}

// Get all projects metadata
export async function getAllProjects(): Promise<ProjectMetadata[]> {
  const projectFiles = getProjectFiles();
  const projects: ProjectMetadata[] = [];

  for (const fileName of projectFiles) {
    try {
      const slug = fileName.replace(/\.md$/, "");
      const fullPath = path.join(projectsDirectory, fileName);
      const fileContents = fs.readFileSync(fullPath, "utf8");
      const { data } = matter(fileContents);

      projects.push({
        id: slug,
        slug,
        title: data.title || "",
        excerpt: data.excerpt || "",
        description: data.description || "",
        category: data.category || "",
        subcategory: data.subcategory,
        technologies: data.technologies || [],
        status: data.status || "",
        featured: data.featured || false,
        publishDate: data.publishDate || "",
        lastUpdated: data.lastUpdated || "",
        githubUrl: data.githubUrl || null,
        liveUrl: data.liveUrl || null,
        stars: data.stars,
        image: data.image || null,
        highlights: data.highlights || [],
        difficulty: data.difficulty || "Intermediate",
      });
    } catch (error) {
      console.error(`Error processing project file ${fileName}:`, error);
    }
  }

  // Sort by publishDate (most recent first)
  return projects.sort(
    (a, b) =>
      new Date(b.publishDate).getTime() - new Date(a.publishDate).getTime()
  );
}

// Get featured projects
export async function getFeaturedProjects(): Promise<ProjectMetadata[]> {
  const allProjects = await getAllProjects();
  return allProjects.filter((project) => project.featured);
}

// Get latest projects (most recent 6)
export async function getLatestProjects(
  limit: number = 6
): Promise<ProjectMetadata[]> {
  const allProjects = await getAllProjects();
  return allProjects.slice(0, limit);
}

// Get projects by category
export async function getProjectsByCategory(
  category: string
): Promise<ProjectMetadata[]> {
  const allProjects = await getAllProjects();
  return allProjects.filter(
    (project) => project.category.toLowerCase() === category.toLowerCase()
  );
}

// Get projects by status
export async function getProjectsByStatus(
  status: string
): Promise<ProjectMetadata[]> {
  const allProjects = await getAllProjects();
  return allProjects.filter(
    (project) => project.status.toLowerCase() === status.toLowerCase()
  );
}

// Get project categories
export async function getProjectCategories(): Promise<string[]> {
  const allProjects = await getAllProjects();
  const categories = new Set(allProjects.map((project) => project.category));
  return Array.from(categories);
}

// Search projects
export async function searchProjects(
  query: string
): Promise<ProjectMetadata[]> {
  const allProjects = await getAllProjects();
  const lowercaseQuery = query.toLowerCase();

  return allProjects.filter(
    (project) =>
      project.title.toLowerCase().includes(lowercaseQuery) ||
      project.description.toLowerCase().includes(lowercaseQuery) ||
      project.excerpt.toLowerCase().includes(lowercaseQuery) ||
      project.technologies.some((tech) =>
        tech.toLowerCase().includes(lowercaseQuery)
      ) ||
      project.category.toLowerCase().includes(lowercaseQuery)
  );
}

// Get project stats
export async function getProjectStats() {
  const allProjects = await getAllProjects();
  const categories = await getProjectCategories();

  return {
    total: allProjects.length,
    featured: allProjects.filter((p) => p.featured).length,
    categories: categories.length,
    totalStars: allProjects.reduce(
      (sum, project) => sum + (project.stars || 0),
      0
    ),
    statusCounts: {
      production: allProjects.filter(
        (p) => p.status.toLowerCase() === "production"
      ).length,
      development: allProjects.filter((p) =>
        p.status.toLowerCase().includes("development")
      ).length,
      beta: allProjects.filter((p) => p.status.toLowerCase() === "beta").length,
    },
  };
}
