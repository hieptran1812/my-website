import { NextResponse } from "next/server";
import {
  getAllProjects,
  getProjectsByCategory as getProjectsByCategoryFromLib,
  getProjectsByStatus,
  getFeaturedProjects,
  getLatestProjects,
} from "@/lib/projects";
import type { ProjectMetadata } from "@/lib/projects";

// Implement caching for projects
const CACHE_DURATION = 60 * 60 * 1000; // 1 hour in milliseconds
let cachedProjects: ProjectMetadata[] | null = null;
let lastCacheTime = 0;

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const category = searchParams.get("category");
    const featured = searchParams.get("featured");
    const status = searchParams.get("status");
    const limit = searchParams.get("limit");

    // Check cache validity
    const now = Date.now();
    if (!cachedProjects || now - lastCacheTime > CACHE_DURATION) {
      console.log("Loading projects from markdown files...");
      cachedProjects = await getAllProjects();
      lastCacheTime = now;
    }

    let filteredProjects: ProjectMetadata[] = cachedProjects;

    // Apply filters using the appropriate utility functions
    if (featured === "true") {
      filteredProjects = await getFeaturedProjects();
    } else if (category && category !== "all") {
      filteredProjects = await getProjectsByCategoryFromLib(category);
    } else if (status) {
      filteredProjects = await getProjectsByStatus(status);
    }

    // Apply limit - handle latest projects optimally
    if (limit) {
      const limitNum = parseInt(limit, 10);
      if (!isNaN(limitNum) && limitNum > 0) {
        if (!category && !featured && !status) {
          // Use optimized getLatestProjects for better performance
          filteredProjects = await getLatestProjects(limitNum);
        } else {
          filteredProjects = filteredProjects.slice(0, limitNum);
        }
      }
    }

    // Get all projects for metadata
    const allProjects = cachedProjects;

    return NextResponse.json({
      projects: filteredProjects,
      total: filteredProjects.length,
      categories: [...new Set(allProjects.map((p) => p.category))],
      statuses: [...new Set(allProjects.map((p) => p.status))],
      cached: true, // Indicate that data is cached
      lastUpdated: new Date(lastCacheTime).toISOString(),
    });
  } catch (error) {
    console.error("Error fetching projects:", error);
    return NextResponse.json(
      { error: "Failed to fetch projects" },
      { status: 500 }
    );
  }
}
