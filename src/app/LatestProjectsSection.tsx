"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";
import Image from "next/image";
import type { ProjectMetadata } from "@/lib/projects";

export default function LatestProjectsSection() {
  const [projects, setProjects] = useState<ProjectMetadata[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);

  // Intersection Observer for scroll animation
  useEffect(() => {
    // Only setup observer after loading is complete
    if (isLoading || error) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      {
        threshold: 0.15,
        rootMargin: "0px 0px -40% 0px",
      }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, [isLoading, error]);

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
      <section
        className="py-16 md:py-24 transition-colors duration-300"
        style={{ backgroundColor: "var(--surface)" }}
      >
        <div className="container mx-auto px-6 max-w-6xl">
          <div className="text-center mb-12">
            <h3 className="section-heading text-3xl md:text-4xl lg:text-5xl font-bold mb-4 transition-colors duration-300 relative">
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent">
                Latest Projects
              </span>
            </h3>
            <div className="flex justify-center items-center mt-6">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
          </div>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="py-16 md:py-24 section-secondary">
        <div className="container mx-auto px-6 max-w-6xl">
          <div className="text-center mb-12">
            <h3 className="section-heading text-3xl md:text-4xl lg:text-5xl font-bold mb-4 transition-colors duration-300 relative">
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent">
                Latest Projects
              </span>
            </h3>
            <p
              className="text-center mt-6"
              style={{ color: "var(--text-secondary)" }}
            >
              Unable to load projects at the moment. Please try again later.
            </p>
            <Link
              href="/projects"
              className="inline-flex items-center mt-4 font-medium transition-colors"
              style={{ color: "var(--accent)" }}
            >
              Browse All Projects
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
        </div>
      </section>
    );
  }

  return (
    <section
      ref={sectionRef}
      id="projects"
      className="py-20 md:py-28 section-secondary relative overflow-hidden"
    >
      {/* Background decorative elements */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-20 left-10 w-32 h-32 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-full blur-xl"></div>
        <div className="absolute bottom-20 right-10 w-40 h-40 bg-gradient-to-br from-purple-500/10 to-blue-500/10 rounded-full blur-xl"></div>
      </div>

      <div className="container mx-auto px-6 max-w-7xl">
        {/* Enhanced Header Section */}
        <div
          className="text-center mb-16"
          style={{
            opacity: isVisible ? 1 : 0,
            transform: isVisible ? "translateY(0)" : "translateY(32px)",
            filter: isVisible ? "blur(0)" : "blur(8px)",
            transition:
              "opacity 1000ms cubic-bezier(0.25, 0.46, 0.45, 0.94), transform 1000ms cubic-bezier(0.25, 0.46, 0.45, 0.94), filter 1000ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
          }}
        >
          <div
            className="inline-flex items-center gap-2 px-4 py-2 mb-6 rounded-full border"
            style={{
              backgroundColor: "var(--surface)",
              borderColor: "var(--card-border)",
              color: "var(--text-muted)",
            }}
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
              />
            </svg>
            <span className="text-sm font-medium">Portfolio</span>
          </div>

          <h3 className="section-heading text-4xl md:text-5xl lg:text-6xl font-bold mb-6 transition-colors duration-300 relative">
            <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent">
              Latest Projects
            </span>
            <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2 w-32 h-1.5 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 rounded-full"></div>
          </h3>

          <p
            className="text-lg md:text-xl max-w-3xl mx-auto transition-colors duration-300 leading-relaxed"
            style={{ color: "var(--text-secondary)" }}
          >
            A selection of projects I have worked on across{" "}
            <span className="font-semibold" style={{ color: "var(--accent)" }}>
              Artificial Intelligence
            </span>
            . These projects reflect hands on experience from idea development
            through to implementation, with a focus on practical and real world
            applications.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 items-center justify-center mt-8">
            <Link
              href="/projects"
              className="inline-flex items-center px-6 py-3 font-semibold rounded-lg transition-all duration-300 hover:shadow-lg hover:scale-105"
              style={{
                backgroundColor: "var(--accent)",
                color: "white",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "var(--accent-hover)";
                e.currentTarget.style.transform =
                  "translateY(-2px) scale(1.05)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "var(--accent)";
                e.currentTarget.style.transform = "translateY(0) scale(1)";
              }}
            >
              <svg
                className="w-5 h-5 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                />
              </svg>
              Explore All Projects
            </Link>
            <Link
              href="https://github.com/hieptran1812"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-6 py-3 font-medium rounded-lg border transition-all duration-300 hover:shadow-md"
              style={{
                borderColor: "var(--card-border)",
                color: "var(--text-primary)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "var(--surface)";
                e.currentTarget.style.transform = "translateY(-1px)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = "transparent";
                e.currentTarget.style.transform = "translateY(0)";
              }}
            >
              <svg
                className="w-5 h-5 mr-2"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
              View on GitHub
            </Link>
          </div>
        </div>

        {/* Projects Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project, idx) => {
            return (
              <Link
                href={`/projects?project=${project.slug}`}
                key={idx}
                className="group block rounded-xl hover:shadow-xl hover:shadow-blue-500/10 overflow-hidden border card-enhanced"
                style={{
                  opacity: isVisible ? 1 : 0,
                  transform: isVisible
                    ? "translateY(0) scale(1)"
                    : "translateY(48px) scale(0.95)",
                  filter: isVisible ? "blur(0)" : "blur(8px)",
                  transition:
                    "opacity 700ms cubic-bezier(0.25, 0.46, 0.45, 0.94), transform 700ms cubic-bezier(0.25, 0.46, 0.45, 0.94), filter 700ms cubic-bezier(0.25, 0.46, 0.45, 0.94)",
                  transitionDelay: isVisible ? `${200 + idx * 120}ms` : "0ms",
                }}
              >
                {project.image && (
                  <div className="relative w-full h-48 overflow-hidden">
                    <Image
                      src={project.image}
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
                )}
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
                    Explore Project
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
                        d="M13 7l5 5m0 0l-5 5m5-5H6"
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
