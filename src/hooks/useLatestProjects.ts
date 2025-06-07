import { useState, useEffect, useCallback } from "react";
import type { ProjectMetadata } from "@/lib/projects";

interface UseLatestProjectsOptions {
  limit?: number;
  featured?: boolean;
  category?: string;
}

interface UseLatestProjectsReturn {
  projects: ProjectMetadata[];
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

/**
 * Custom hook for fetching latest projects with comprehensive error handling and caching
 *
 * @param options - Configuration options for fetching projects
 * @returns Object containing projects data, loading state, error state, and refetch function
 */
export function useLatestProjects({
  limit = 6,
  featured,
  category,
}: UseLatestProjectsOptions = {}): UseLatestProjectsReturn {
  const [projects, setProjects] = useState<ProjectMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const validateProject = useCallback((project: ProjectMetadata): boolean => {
    // Essential fields validation
    const hasEssentialFields =
      project &&
      typeof project.id === "string" &&
      typeof project.title === "string" &&
      typeof project.slug === "string" &&
      typeof project.excerpt === "string" &&
      typeof project.category === "string";

    // Additional safety checks
    const hasValidArrays =
      Array.isArray(project.technologies) && Array.isArray(project.highlights);

    // Date validation
    const hasValidDate =
      project.publishDate && !isNaN(new Date(project.publishDate).getTime());

    return Boolean(hasEssentialFields && hasValidArrays && hasValidDate);
  }, []);

  const fetchProjects = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Build query parameters
      const searchParams = new URLSearchParams();
      if (limit) searchParams.append("limit", limit.toString());
      if (featured !== undefined)
        searchParams.append("featured", featured.toString());
      if (category) searchParams.append("category", category);

      // Enhanced fetch with timeout and better error handling
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

      const response = await fetch(`/api/projects?${searchParams.toString()}`, {
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
        },
        next: { revalidate: 300 }, // Cache for 5 minutes
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        if (response.status === 404) {
          throw new Error("Projects API endpoint not found");
        } else if (response.status >= 500) {
          throw new Error("Server error while fetching projects");
        } else {
          throw new Error(
            `Failed to fetch projects: ${response.status} ${response.statusText}`
          );
        }
      }

      const data = await response.json();

      // Enhanced validation
      if (!data || typeof data !== "object") {
        throw new Error("Invalid response format from API");
      }

      if (!Array.isArray(data.projects)) {
        throw new Error("Projects data is not in expected array format");
      }

      // Validate project data structure
      const validatedProjects = data.projects.filter(validateProject);

      if (validatedProjects.length === 0 && data.projects.length > 0) {
        throw new Error("No valid projects found in response");
      }

      setProjects(validatedProjects);

      // Log success for debugging in development
      if (process.env.NODE_ENV === "development") {
        console.log(
          `Successfully loaded ${validatedProjects.length} projects`,
          { limit, featured, category }
        );
      }
    } catch (err) {
      let errorMessage = "Failed to fetch projects";

      if (err instanceof Error) {
        if (err.name === "AbortError") {
          errorMessage = "Request timeout - please check your connection";
        } else {
          errorMessage = err.message;
        }
      }

      setError(errorMessage);
      console.error("Error fetching projects:", err);
      setProjects([]);
    } finally {
      setLoading(false);
    }
  }, [limit, featured, category, validateProject]);

  const refetch = useCallback(() => {
    fetchProjects();
  }, [fetchProjects]);

  useEffect(() => {
    fetchProjects();
  }, [fetchProjects]);

  return {
    projects,
    loading,
    error,
    refetch,
  };
}
