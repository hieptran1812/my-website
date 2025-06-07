import { NextResponse } from "next/server";
import { projects, getProjectsByCategory } from "@/data/projects";
import type { Project } from "@/data/projects";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const category = searchParams.get("category");
    const featured = searchParams.get("featured");
    const status = searchParams.get("status");
    const limit = searchParams.get("limit");

    let filteredProjects: Project[] = projects;

    // Apply filters
    if (category && category !== "all") {
      filteredProjects = getProjectsByCategory(category);
    }

    if (featured === "true") {
      filteredProjects = filteredProjects.filter((project) => project.featured);
    }

    if (status) {
      filteredProjects = filteredProjects.filter(
        (project) => project.status === status
      );
    }

    // Apply limit
    if (limit) {
      const limitNum = parseInt(limit, 10);
      if (!isNaN(limitNum) && limitNum > 0) {
        filteredProjects = filteredProjects.slice(0, limitNum);
      }
    }

    return NextResponse.json({
      projects: filteredProjects,
      total: filteredProjects.length,
      categories: [...new Set(projects.map((p) => p.category))],
      statuses: [...new Set(projects.map((p) => p.status))],
    });
  } catch (error) {
    console.error("Error fetching projects:", error);
    return NextResponse.json(
      { error: "Failed to fetch projects" },
      { status: 500 }
    );
  }
}
