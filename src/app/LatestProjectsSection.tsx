"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import type { ProjectMetadata } from "@/lib/projects";

export default function LatestProjectsSection() {
  const [projects, setProjects] = useState<ProjectMetadata[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch latest projects from API
  useEffect(() => {
    async function fetchLatestProjects() {
      try {
        setIsLoading(true);
        setError(null);

        // Fetch latest 6 projects from our API
        const response = await fetch("/api/projects?limit=6");

        if (!response.ok) {
          throw new Error(
            `Failed to fetch projects: ${response.status} ${response.statusText}`
          );
        }

        const data = await response.json();

        if (!data || !Array.isArray(data.projects)) {
          throw new Error("Invalid response format from API");
        }

        const fetchedProjects = data.projects;

        if (fetchedProjects.length === 0) {
          console.warn("No projects found in the database");
        }

        // Set projects directly from API response
        setProjects(fetchedProjects);
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to fetch projects";
        setError(errorMessage);
        console.error("Error fetching latest projects:", err);

        // Fallback to empty array
        setProjects([]);
      } finally {
        setIsLoading(false);
      }
    }

    fetchLatestProjects();
  }, []);

  if (isLoading) {
    return (
      <section className="py-16 md:py-24">
        <div className="container mx-auto px-6 max-w-6xl">
          <div className="text-center">Loading latest projects...</div>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="py-16 md:py-24">
        <div className="container mx-auto px-6 max-w-6xl">
          <div className="text-center text-red-500">Error: {error}</div>
        </div>
      </section>
    );
  }

  return (
    <section
      id="projects"
      className="py-16 md:py-24 transition-colors duration-300 section-pattern section-elevated"
      style={{ backgroundColor: "var(--surface-secondary)" }}
    >
      <div className="container mx-auto px-6 max-w-6xl">
        <div className="text-center mb-12">
          <h3 className="section-heading text-3xl md:text-4xl lg:text-5xl font-bold mb-4 transition-colors duration-300 relative">
            <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent">
              Latest Projects
            </span>
            <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-24 h-1 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 rounded-full"></div>
          </h3>
          <p
            className="text-lg max-w-2xl mx-auto transition-colors duration-300 mt-6"
            style={{ color: "var(--text-secondary)" }}
          >
            Explore my recent work in AI, web development, and open-source
            contributions.
          </p>
          <Link
            href="/projects"
            className="inline-flex items-center mt-4 font-medium transition-colors"
            style={{ color: "var(--accent)" }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = "var(--accent-hover)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = "var(--accent)";
            }}
          >
            View my projects
            <svg
              className="ml-2 w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M9 5l7 7-7 7"
              />
            </svg>
          </Link>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map((project, idx) => {
            // Ensure href is never undefined
            const href =
              project.liveUrl ||
              project.githubUrl ||
              `/projects/${project.slug}`;

            return (
              <Link
                href={href}
                key={idx}
                className="group block rounded-xl hover:shadow-lg transition-all duration-300 overflow-hidden border card-enhanced"
              >
                <div className="relative w-full h-48 overflow-hidden">
                  <Image
                    src={project.image || "/project-placeholder.jpg"}
                    alt={project.title}
                    fill
                    style={{ objectFit: "cover" }}
                    className="transition-transform duration-300 group-hover:scale-105"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
                  <div className="absolute top-3 right-3">
                    <div className="flex items-center gap-1 text-white text-xs bg-black/50 px-2 py-1 rounded-full">
                      <svg
                        className="w-3.5 h-3.5"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"
                          clipRule="evenodd"
                        />
                      </svg>
                      {project.stars}
                    </div>
                  </div>
                </div>
                <div className="p-5">
                  <h4
                    className="text-lg font-semibold mb-2 transition-colors"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {project.title}
                  </h4>
                  <p
                    className="text-sm mb-3 line-clamp-2"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {project.description}
                  </p>
                  <div className="flex flex-wrap gap-1.5 mb-3">
                    {project.technologies?.map((tech: string) => (
                      <span
                        key={tech}
                        className="text-xs px-2 py-0.5 rounded-full font-medium"
                        style={{
                          backgroundColor: "var(--accent-subtle)",
                          color: "var(--accent)",
                        }}
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                  <span
                    className="inline-flex items-center text-sm font-medium transition-colors"
                    style={{ color: "var(--accent)" }}
                  >
                    View Project
                    <svg
                      className="ml-1.5 w-3.5 h-3.5 transition-transform group-hover:translate-x-0.5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M9 5l7 7-7 7"
                      />
                    </svg>
                  </span>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </section>
  );
}
