"use client";

import React, { useState, useEffect, Suspense } from "react";
import Image from "next/image";
import { useSearchParams } from "next/navigation";
import type { ProjectMetadata } from "@/lib/projects";
import ScrollReveal, { CounterReveal } from "@/components/ScrollReveal";

// Define fallback for statusColors
const statusColors: Record<string, React.CSSProperties> = {
  Completed: { backgroundColor: "rgba(16, 185, 129, 0.8)", color: "#fff" },
  "In Progress": { backgroundColor: "rgba(245, 158, 11, 0.8)", color: "#fff" },
  Planned: { backgroundColor: "rgba(99, 102, 241, 0.8)", color: "#fff" },
  "": { backgroundColor: "rgba(209, 213, 219, 0.8)", color: "#111" },
};

const projectCategories = [
  "All",
  "Machine Learning",
  "Web Development",
  "Data Science",
  "Open Source",
  "Research",
];

// Component that uses useSearchParams - must be wrapped in Suspense
function ProjectsContent() {
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [selectedProject, setSelectedProject] =
    useState<ProjectMetadata | null>(null);
  const [projects, setProjects] = useState<ProjectMetadata[]>([]);
  const searchParams = useSearchParams();

  useEffect(() => {
    async function fetchProjects() {
      try {
        const res = await fetch("/api/projects");
        if (!res.ok) throw new Error("Failed to fetch projects");
        const data = await res.json();
        setProjects(data.projects || []);
      } catch {
        setProjects([]);
      }
    }
    fetchProjects();
  }, []);

  // Handle query parameter to open project popup
  useEffect(() => {
    const projectSlug = searchParams.get("project");
    if (projectSlug && projects.length > 0) {
      const project = projects.find((p) => p.slug === projectSlug);
      if (project) {
        setSelectedProject(project);
      }
    }
  }, [searchParams, projects]);

  const filteredProjects =
    selectedCategory === "All"
      ? projects
      : projects.filter((project) => project.category === selectedCategory);

  const featuredProjects = projects.filter((project) => project.featured);

  return (
    <div
      className="flex flex-col min-h-screen transition-colors duration-300"
      style={{
        backgroundColor: "var(--background)",
        color: "var(--text-primary)",
      }}
    >
      <main className="flex-1">
        <div className="max-w-7xl mx-auto px-6 py-16">
          {/* Hero Section */}
          <div className="text-center mb-20 relative">
            {/* Background decoration */}
            <div className="absolute inset-0 -z-10">
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-indigo-500/10 rounded-full blur-3xl animate-pulse"></div>
              <div className="absolute top-1/3 left-1/3 w-64 h-64 bg-gradient-to-br from-cyan-400/5 to-blue-600/5 rounded-full blur-2xl animate-pulse delay-700"></div>
              <div className="absolute bottom-1/3 right-1/3 w-48 h-48 bg-gradient-to-tl from-purple-400/5 to-pink-600/5 rounded-full blur-2xl animate-pulse delay-1000"></div>
            </div>

            {/* Animated title with enhanced effects */}
            <ScrollReveal animation="fade-up" duration={800}>
              <div className="relative inline-block group">
                <h1 className="text-3xl md:text-5xl lg:text-6xl font-black mb-6 relative overflow-hidden">
                  {/* Main gradient text */}
                  <span className="relative z-10 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent animate-gradient bg-300% font-extrabold tracking-tight">
                    My Projects
                  </span>

                  {/* Glowing shadow effect */}
                  <span className="absolute inset-0 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent blur-sm opacity-30 scale-105">
                    My Projects
                  </span>

                  {/* Sparkle decorations */}
                  <div className="absolute -top-4 left-8 w-2 h-2 bg-blue-400 rounded-full animate-pulse opacity-60"></div>
                  <div className="absolute -top-2 right-12 w-1.5 h-1.5 bg-purple-400 rounded-full animate-pulse delay-300 opacity-80"></div>
                  <div className="absolute top-4 left-4 w-1 h-1 bg-indigo-400 rounded-full animate-pulse delay-500 opacity-70"></div>
                  <div className="absolute -bottom-2 right-8 w-2 h-2 bg-cyan-400 rounded-full animate-pulse delay-700 opacity-60"></div>
                  <div className="absolute bottom-4 left-16 w-1.5 h-1.5 bg-pink-400 rounded-full animate-pulse delay-1000 opacity-75"></div>
                </h1>

                {/* Floating particles */}
                <div className="absolute inset-0 pointer-events-none">
                  <div className="absolute top-1/4 left-1/4 w-1 h-1 bg-blue-300 rounded-full animate-bounce opacity-40"></div>
                  <div className="absolute top-3/4 right-1/4 w-1 h-1 bg-purple-300 rounded-full animate-bounce delay-300 opacity-50"></div>
                  <div className="absolute bottom-1/4 left-3/4 w-1 h-1 bg-indigo-300 rounded-full animate-bounce delay-500 opacity-30"></div>
                </div>
              </div>
            </ScrollReveal>

            {/* Enhanced subtitle with typing effect appearance */}
            <ScrollReveal animation="fade-up" delay={200} duration={800}>
              <div className="relative">
                <p
                  className="text-xl md:text-2xl mb-8 max-w-4xl mx-auto leading-relaxed"
                  style={{ color: "var(--text-secondary)" }}
                >
                  pin{" "}
                  <span className="font-semibold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
                    artificial intelligence
                  </span>
                  ,{" "}
                  <span className="font-semibold bg-gradient-to-r from-purple-500 to-indigo-600 bg-clip-text text-transparent">
                    data science
                  </span>
                  , and{" "}
                  <span className="font-semibold bg-gradient-to-r from-cyan-500 to-purple-600 bg-clip-text text-transparent">
                    research
                  </span>
                  , where each project reflects an effort to address real
                  problems through thoughtful technology and careful engineering
                </p>
              </div>
            </ScrollReveal>

            {/* Enhanced animated stats badges */}
            <ScrollReveal animation="fade-up" delay={400} duration={800}>
              <div className="flex flex-wrap justify-center gap-4">
                <span
                  className="group px-6 py-3 rounded-full text-sm font-semibold transition-all duration-300 hover:scale-110 hover:shadow-lg cursor-default border"
                  style={{
                    backgroundColor: "var(--surface-accent)",
                    color: "var(--accent)",
                    borderColor: "var(--accent)/20",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "scale(1.1) rotate(1deg)";
                    e.currentTarget.style.boxShadow =
                      "0 10px 25px rgba(14, 165, 233, 0.3)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "scale(1)";
                    e.currentTarget.style.boxShadow = "none";
                  }}
                >
                  🎯{" "}
                  <CounterReveal
                    end={projects.length}
                    duration={1500}
                    delay={600}
                    className="font-bold"
                  />{" "}
                  Total Projects
                </span>
                <span
                  className="group px-6 py-3 rounded-full text-sm font-semibold transition-all duration-300 hover:scale-110 hover:shadow-lg cursor-default border"
                  style={{
                    backgroundColor: "var(--surface-accent)",
                    color: "var(--accent)",
                    borderColor: "var(--accent)/20",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform =
                      "scale(1.1) rotate(-1deg)";
                    e.currentTarget.style.boxShadow =
                      "0 10px 25px rgba(147, 51, 234, 0.3)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "scale(1)";
                    e.currentTarget.style.boxShadow = "none";
                  }}
                >
                  ⭐{" "}
                  <CounterReveal
                    end={featuredProjects.length}
                    duration={1500}
                    delay={800}
                    className="font-bold"
                  />{" "}
                  Featured
                </span>
                <span
                  className="group px-6 py-3 rounded-full text-sm font-semibold transition-all duration-300 hover:scale-110 hover:shadow-lg cursor-default border"
                  style={{
                    backgroundColor: "var(--surface-accent)",
                    color: "var(--accent)",
                    borderColor: "var(--accent)/20",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = "scale(1.1) rotate(1deg)";
                    e.currentTarget.style.boxShadow =
                      "0 10px 25px rgba(59, 130, 246, 0.3)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = "scale(1)";
                    e.currentTarget.style.boxShadow = "none";
                  }}
                >
                  🚀{" "}
                  <CounterReveal
                    end={
                      projects.filter((p) => p.status === "Production").length
                    }
                    duration={1500}
                    delay={1000}
                    className="font-bold"
                  />{" "}
                  In Production
                </span>
              </div>
            </ScrollReveal>
          </div>

          {/* Featured Projects */}
          <div className="mb-20">
            <ScrollReveal animation="fade-up" duration={700}>
              <h2
                className="text-4xl font-bold text-center mb-4"
                style={{ color: "var(--text-primary)" }}
              >
                Featured Projects
              </h2>
            </ScrollReveal>
            <ScrollReveal animation="fade-up" delay={100} duration={700}>
              <p
                className="text-lg text-center mb-12 max-w-3xl mx-auto"
                style={{ color: "var(--text-secondary)" }}
              >
                Highlighting my most impactful and innovative work
              </p>
            </ScrollReveal>
            <div className="grid lg:grid-cols-3 gap-8">
              {featuredProjects.map((project, index) => (
                <ScrollReveal
                  key={project.id}
                  animation="fade-up"
                  delay={200 + index * 150}
                  duration={700}
                >
                  <div
                    className="group relative overflow-hidden rounded-2xl border transition-all duration-300 hover:scale-105 hover:shadow-2xl cursor-pointer h-full"
                    style={{
                      backgroundColor: "var(--card-bg)",
                      borderColor: "var(--card-border)",
                    }}
                    onClick={() => setSelectedProject(project)}
                  >
                    {project.image && (
                      <div className="relative h-48 overflow-hidden">
                        <Image
                          src={project.image}
                          alt={project.title}
                          fill
                          className="object-cover group-hover:scale-110 transition-transform duration-300"
                        />
                        <div className="absolute top-4 right-4 flex gap-2">
                          <span
                            className="px-3 py-1 rounded-full text-xs font-semibold"
                            style={
                              statusColors[
                                project.status as keyof typeof statusColors
                              ]
                            }
                          >
                            {project.status}
                          </span>
                          <span className="px-3 py-1 rounded-full text-xs font-semibold bg-orange-100 text-orange-800">
                            Featured
                          </span>
                        </div>
                      </div>
                    )}
                    {!project.image && (
                      <div className="absolute top-4 right-4 flex gap-2">
                        <span
                          className="px-3 py-1 rounded-full text-xs font-semibold"
                          style={
                            statusColors[
                              project.status as keyof typeof statusColors
                            ]
                          }
                        >
                          {project.status}
                        </span>
                        <span className="px-3 py-1 rounded-full text-xs font-semibold bg-orange-100 text-orange-800">
                          Featured
                        </span>
                      </div>
                    )}
                    <div className="p-6">
                      <span
                        className="text-sm font-medium"
                        style={{ color: "var(--accent)" }}
                      >
                        {project.category}
                      </span>
                      <h3
                        className="text-xl font-bold mb-3 group-hover:text-[var(--accent)] transition-colors duration-200"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {project.title}
                      </h3>
                      <p
                        className="text-sm leading-relaxed mb-4"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {project.description}
                      </p>
                      <div className="flex flex-wrap gap-2 mb-4">
                        {project.technologies
                          .slice(0, 3)
                          .map((tech, techIndex) => (
                            <span
                              key={techIndex}
                              className="px-2 py-1 rounded text-xs"
                              style={{
                                backgroundColor: "var(--surface)",
                                color: "var(--text-secondary)",
                              }}
                            >
                              {tech}
                            </span>
                          ))}
                        {project.technologies.length > 3 && (
                          <span
                            className="px-2 py-1 rounded text-xs"
                            style={{
                              backgroundColor: "var(--surface)",
                              color: "var(--text-secondary)",
                            }}
                          >
                            +{project.technologies.length - 3}
                          </span>
                        )}
                      </div>
                      {(project.githubUrl || project.liveUrl) && (
                        <div className="flex gap-3">
                          {project.githubUrl && (
                            <a
                              href={project.githubUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200"
                              style={{
                                backgroundColor: "var(--surface)",
                                color: "var(--text-secondary)",
                              }}
                              onClick={(e) => e.stopPropagation()}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.backgroundColor =
                                  "var(--surface-accent)";
                                e.currentTarget.style.color = "var(--accent)";
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.backgroundColor =
                                  "var(--surface)";
                                e.currentTarget.style.color =
                                  "var(--text-secondary)";
                              }}
                            >
                              <span>👨‍💻</span>
                              Code
                            </a>
                          )}
                          {project.liveUrl && (
                            <a
                              href={project.liveUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 text-white"
                              style={{ backgroundColor: "var(--accent)" }}
                              onClick={(e) => e.stopPropagation()}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.backgroundColor =
                                  "var(--accent-hover)";
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.backgroundColor =
                                  "var(--accent)";
                              }}
                            >
                              <span>🚀</span>
                              Live Demo
                            </a>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </ScrollReveal>
              ))}
            </div>
          </div>

          {/* Category Filter */}
          <div className="mb-12">
            <ScrollReveal animation="fade-up" duration={700}>
              <h2
                className="text-4xl font-bold text-center mb-12"
                style={{ color: "var(--text-primary)" }}
              >
                All Projects
              </h2>
            </ScrollReveal>
            <ScrollReveal animation="fade-up" delay={100} duration={700}>
              <div className="flex flex-wrap justify-center gap-3">
                {projectCategories.map((category, index) => (
                  <button
                    key={category}
                    onClick={() => setSelectedCategory(category)}
                    className="px-6 py-3 rounded-xl font-medium transition-all duration-200 transform hover:scale-105"
                    style={{
                      backgroundColor:
                        selectedCategory === category
                          ? "var(--accent)"
                          : "var(--surface)",
                      color:
                        selectedCategory === category
                          ? "white"
                          : "var(--text-secondary)",
                      animationDelay: `${index * 50}ms`,
                    }}
                    onMouseEnter={(e) => {
                      if (selectedCategory !== category) {
                        e.currentTarget.style.backgroundColor =
                          "var(--surface-accent)";
                        e.currentTarget.style.color = "var(--accent)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (selectedCategory !== category) {
                        e.currentTarget.style.backgroundColor =
                          "var(--surface)";
                        e.currentTarget.style.color = "var(--text-secondary)";
                      }
                    }}
                  >
                    {category}
                  </button>
                ))}
              </div>
            </ScrollReveal>
          </div>

          {/* Projects Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-20">
            {filteredProjects.map((project, index) => (
              <ScrollReveal
                key={project.id}
                animation="fade-up"
                delay={100 + (index % 6) * 100}
                duration={600}
              >
                <div
                  id={`project-${project.id}`}
                  className="group overflow-hidden rounded-2xl border transition-all duration-300 hover:scale-105 hover:shadow-xl cursor-pointer h-full"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--card-border)",
                  }}
                  onClick={() => setSelectedProject(project)}
                >
                  {project.image && (
                    <div className="relative h-48 overflow-hidden">
                      <Image
                        src={project.image}
                        alt={project.title}
                        fill
                        className="object-cover group-hover:scale-110 transition-transform duration-300"
                      />
                      <div className="absolute top-4 right-4">
                        <span
                          className="px-3 py-1 rounded-full text-xs font-semibold"
                          style={
                            statusColors[
                              project.status as keyof typeof statusColors
                            ]
                          }
                        >
                          {project.status}
                        </span>
                      </div>
                    </div>
                  )}
                  {!project.image && (
                    <div className="relative pt-4 pr-4 flex justify-end">
                      <span
                        className="px-3 py-1 rounded-full text-xs font-semibold"
                        style={
                          statusColors[
                            project.status as keyof typeof statusColors
                          ]
                        }
                      >
                        {project.status}
                      </span>
                    </div>
                  )}
                  <div className="p-6">
                    <span
                      className="text-sm font-medium"
                      style={{ color: "var(--accent)" }}
                    >
                      {project.category}
                    </span>
                    <h3
                      className="text-xl font-bold mb-3 group-hover:text-[var(--accent)] transition-colors duration-200"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {project.title}
                    </h3>
                    <p
                      className="text-sm leading-relaxed mb-4"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {project.description}
                    </p>
                    <div className="flex flex-wrap gap-2 mb-4">
                      {project.technologies
                        .slice(0, 3)
                        .map((tech, techIndex) => (
                          <span
                            key={techIndex}
                            className="px-2 py-1 rounded text-xs"
                            style={{
                              backgroundColor: "var(--surface)",
                              color: "var(--text-secondary)",
                            }}
                          >
                            {tech}
                          </span>
                        ))}
                      {project.technologies.length > 3 && (
                        <span
                          className="px-2 py-1 rounded text-xs"
                          style={{
                            backgroundColor: "var(--surface)",
                            color: "var(--text-secondary)",
                          }}
                        >
                          +{project.technologies.length - 3}
                        </span>
                      )}
                    </div>
                    {(project.githubUrl || project.liveUrl) && (
                      <div className="flex gap-3">
                        {project.githubUrl && (
                          <a
                            href={project.githubUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200"
                            style={{
                              backgroundColor: "var(--surface)",
                              color: "var(--text-secondary)",
                            }}
                            onClick={(e) => e.stopPropagation()}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.backgroundColor =
                                "var(--surface-accent)";
                              e.currentTarget.style.color = "var(--accent)";
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.backgroundColor =
                                "var(--surface)";
                              e.currentTarget.style.color = "var(--text-secondary)";
                            }}
                          >
                            👨‍💻 Code
                          </a>
                        )}
                        {project.liveUrl && (
                          <a
                            href={project.liveUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 text-white"
                            style={{ backgroundColor: "var(--accent)" }}
                            onClick={(e) => e.stopPropagation()}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.backgroundColor =
                                "var(--accent-hover)";
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.backgroundColor =
                                "var(--accent)";
                            }}
                          >
                            🚀 Live
                          </a>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </ScrollReveal>
            ))}
          </div>

          {/* Call to Action */}
          <ScrollReveal animation="zoom-in" duration={800}>
            <div
              className="p-10 rounded-2xl border text-center"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
              }}
            >
              <ScrollReveal animation="fade-up" delay={200} duration={600}>
                <h2
                  className="text-3xl font-bold mb-6"
                  style={{ color: "var(--text-primary)" }}
                >
                  Interested in Collaboration?
                </h2>
              </ScrollReveal>
              <ScrollReveal animation="fade-up" delay={300} duration={600}>
                <p
                  className="text-lg mb-8 max-w-3xl mx-auto"
                  style={{ color: "var(--text-secondary)" }}
                >
                  I&apos;m always open to discussing new projects, innovative
                  ideas, or opportunities to contribute to exciting initiatives.
                  Let&apos;s build something amazing together!
                </p>
              </ScrollReveal>
              <ScrollReveal animation="fade-up" delay={400} duration={600}>
                <div className="flex flex-wrap justify-center gap-4">
                  <a
                    href="/contact"
                    className="inline-flex items-center px-8 py-4 font-semibold rounded-xl transition-all duration-200 text-white shadow-lg hover:shadow-xl transform hover:scale-105"
                    style={{ backgroundColor: "var(--accent)" }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor =
                        "var(--accent-hover)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = "var(--accent)";
                    }}
                  >
                    <span className="mr-2">💬</span>
                    Let&apos;s Talk
                  </a>
                  <a
                    href="https://github.com/hieptran1812"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center px-8 py-4 font-semibold rounded-xl transition-all duration-200 border shadow-lg hover:shadow-xl transform hover:scale-105"
                    style={{
                      backgroundColor: "var(--card-bg)",
                      borderColor: "var(--border)",
                      color: "var(--text-primary)",
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = "var(--surface)";
                      e.currentTarget.style.borderColor = "var(--accent)";
                      e.currentTarget.style.color = "var(--accent)";
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = "var(--card-bg)";
                      e.currentTarget.style.borderColor = "var(--border)";
                      e.currentTarget.style.color = "var(--text-primary)";
                    }}
                  >
                    <span className="mr-2">👨‍💻</span>
                    View More on GitHub
                  </a>
                </div>
              </ScrollReveal>
            </div>
          </ScrollReveal>
        </div>

        {/* Project Detail Modal */}
        {selectedProject && (
          <div
            className="fixed inset-0 bg-black flex items-center justify-center p-6 z-50 animate-fade-in"
            style={{
              backgroundColor: "rgba(0, 0, 0, 0)",
              animation: "fadeInBg 0.3s ease-out forwards",
            }}
            onClick={() => setSelectedProject(null)}
          >
            <div
              className="max-w-4xl w-full max-h-[90vh] overflow-y-auto rounded-2xl border"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
                animation:
                  "modalSlideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards",
                opacity: 0,
                transform: "scale(0.95) translateY(20px)",
              }}
              onClick={(e) => e.stopPropagation()}
            >
              {selectedProject.image ? (
                <div className="relative h-64 md:h-80">
                  <Image
                    src={selectedProject.image}
                    alt={selectedProject.title}
                    fill
                    className="object-cover"
                  />
                  <button
                    onClick={() => setSelectedProject(null)}
                    className="absolute top-4 right-4 w-10 h-10 rounded-full bg-black bg-opacity-50 text-white flex items-center justify-center hover:bg-opacity-70 transition-colors"
                  >
                    ×
                  </button>
                  <div className="absolute bottom-4 left-4 flex gap-2">
                    <span
                      className="px-3 py-1 rounded-full text-sm font-semibold"
                      style={
                        statusColors[
                          selectedProject.status as keyof typeof statusColors
                        ]
                      }
                    >
                      {selectedProject.status}
                    </span>
                    <span
                      className="px-3 py-1 rounded-full text-sm font-semibold"
                      style={{
                        backgroundColor: "var(--surface-accent)",
                        color: "var(--accent)",
                      }}
                    >
                      {selectedProject.category}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="relative p-4 flex justify-between items-start">
                  <div className="flex gap-2">
                    <span
                      className="px-3 py-1 rounded-full text-sm font-semibold"
                      style={
                        statusColors[
                          selectedProject.status as keyof typeof statusColors
                        ]
                      }
                    >
                      {selectedProject.status}
                    </span>
                    <span
                      className="px-3 py-1 rounded-full text-sm font-semibold"
                      style={{
                        backgroundColor: "var(--surface-accent)",
                        color: "var(--accent)",
                      }}
                    >
                      {selectedProject.category}
                    </span>
                  </div>
                  <button
                    onClick={() => setSelectedProject(null)}
                    className="w-10 h-10 rounded-full flex items-center justify-center hover:bg-opacity-70 transition-colors"
                    style={{
                      backgroundColor: "var(--surface)",
                      color: "var(--text-primary)",
                    }}
                  >
                    ×
                  </button>
                </div>
              )}
              <div className="p-8">
                <h3
                  className="text-3xl font-bold mb-4"
                  style={{ color: "var(--text-primary)" }}
                >
                  {selectedProject.title}
                </h3>
                <p
                  className="text-lg leading-relaxed mb-6"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {selectedProject.description}
                </p>

                <div className="mb-6">
                  <h4
                    className="text-lg font-semibold mb-3"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Key Highlights
                  </h4>
                  <div className="grid md:grid-cols-2 gap-2">
                    {selectedProject.highlights.map((highlight, index) => (
                      <div
                        key={index}
                        className="flex items-center gap-2"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        <span style={{ color: "var(--accent)" }}>✓</span>
                        {highlight}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mb-6">
                  <h4
                    className="text-lg font-semibold mb-3"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Technologies Used
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedProject.technologies.map((tech, index) => (
                      <span
                        key={index}
                        className="px-3 py-1.5 rounded-lg text-sm font-medium"
                        style={{
                          backgroundColor: "var(--surface)",
                          color: "var(--text-secondary)",
                        }}
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>

                {(selectedProject.githubUrl || selectedProject.liveUrl) && (
                  <div className="flex gap-4">
                    {selectedProject.githubUrl && (
                      <a
                        href={selectedProject.githubUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors duration-200"
                        style={{
                          backgroundColor: "var(--surface)",
                          color: "var(--text-secondary)",
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor =
                            "var(--surface-accent)";
                          e.currentTarget.style.color = "var(--accent)";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = "var(--surface)";
                          e.currentTarget.style.color = "var(--text-secondary)";
                        }}
                      >
                        <span>👨‍💻</span>
                        View Source Code
                      </a>
                    )}
                    {selectedProject.liveUrl && (
                      <a
                        href={selectedProject.liveUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors duration-200 text-white"
                        style={{ backgroundColor: "var(--accent)" }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor =
                            "var(--accent-hover)";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = "var(--accent)";
                        }}
                      >
                        <span>🚀</span>
                        View Live Demo
                      </a>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// Main export wrapped in Suspense for useSearchParams
export default function ProjectsPage() {
  return (
    <Suspense fallback={
      <div className="flex flex-col min-h-screen items-center justify-center" style={{ backgroundColor: "var(--background)" }}>
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2" style={{ borderColor: "var(--accent)" }}></div>
      </div>
    }>
      <ProjectsContent />
    </Suspense>
  );
}
