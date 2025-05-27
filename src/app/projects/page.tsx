"use client";

import React, { useState } from "react";
import Image from "next/image";

const projectCategories = [
  "All",
  "Machine Learning",
  "Web Development",
  "Data Science",
  "Open Source",
  "Research",
];

const projects = [
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
  {
    id: 2,
    title: "Distributed Machine Learning Training System",
    description:
      "Scalable ML training infrastructure using distributed computing to train large-scale deep learning models efficiently across multiple GPUs and nodes.",
    image: "/project-placeholder.jpg",
    category: "Machine Learning",
    technologies: [
      "Python",
      "TensorFlow",
      "Apache Spark",
      "Kubernetes",
      "Redis",
      "PostgreSQL",
    ],
    githubUrl: "https://github.com/hieptran1812/distributed-ml-training",
    liveUrl: null,
    featured: true,
    status: "Active Development",
    highlights: [
      "80% faster training time",
      "Auto-scaling capabilities",
      "Fault-tolerant architecture",
      "MLflow integration",
    ],
  },
  {
    id: 3,
    title: "Modern E-commerce Platform",
    description:
      "Full-stack e-commerce solution with advanced features including real-time inventory management, AI-powered recommendations, and comprehensive analytics dashboard.",
    image: "/project-placeholder.jpg",
    category: "Web Development",
    technologies: [
      "Next.js",
      "TypeScript",
      "PostgreSQL",
      "Prisma",
      "Stripe",
      "Tailwind CSS",
    ],
    githubUrl: "https://github.com/hieptran1812/ecommerce-platform",
    liveUrl: "https://ecommerce-demo.hieptran.dev",
    featured: true,
    status: "Production",
    highlights: [
      "Mobile-first responsive design",
      "AI-powered product recommendations",
      "Real-time inventory tracking",
      "Integrated payment processing",
    ],
  },
  {
    id: 4,
    title: "Computer Vision Object Detection API",
    description:
      "RESTful API service for real-time object detection and classification using state-of-the-art YOLO and R-CNN models with custom training capabilities.",
    image: "/project-placeholder.jpg",
    category: "Machine Learning",
    technologies: ["Python", "OpenCV", "YOLO", "FastAPI", "Docker", "AWS"],
    githubUrl: "https://github.com/hieptran1812/object-detection-api",
    liveUrl: "https://api.object-detection.com",
    featured: false,
    status: "Production",
    highlights: [
      "Real-time processing",
      "Custom model training",
      "RESTful API design",
      "Cloud deployment",
    ],
  },
  {
    id: 5,
    title: "Big Data Analytics Pipeline",
    description:
      "End-to-end data processing pipeline for analyzing large-scale datasets with real-time streaming capabilities and interactive visualization dashboards.",
    image: "/project-placeholder.jpg",
    category: "Data Science",
    technologies: [
      "Apache Kafka",
      "Spark",
      "Hadoop",
      "Elasticsearch",
      "Kibana",
      "Python",
    ],
    githubUrl: "https://github.com/hieptran1812/big-data-pipeline",
    liveUrl: null,
    featured: false,
    status: "Production",
    highlights: [
      "Processes 1M+ records/day",
      "Real-time streaming",
      "Interactive dashboards",
      "Scalable architecture",
    ],
  },
  {
    id: 6,
    title: "Open Source ML Model Registry",
    description:
      "Community-driven platform for sharing, versioning, and managing machine learning models with automated testing and deployment capabilities.",
    image: "/project-placeholder.jpg",
    category: "Open Source",
    technologies: [
      "Python",
      "Django",
      "PostgreSQL",
      "Redis",
      "Docker",
      "GitHub Actions",
    ],
    githubUrl: "https://github.com/hieptran1812/ml-model-registry",
    liveUrl: "https://ml-registry.org",
    featured: false,
    status: "Production",
    highlights: [
      "500+ community models",
      "Automated testing",
      "Version control",
      "API integration",
    ],
  },
  {
    id: 7,
    title: "Research: Few-Shot Learning for NLP",
    description:
      "Academic research project investigating novel approaches to few-shot learning in natural language processing, with published results in top-tier conferences.",
    image: "/project-placeholder.jpg",
    category: "Research",
    technologies: [
      "Python",
      "PyTorch",
      "Transformers",
      "Weights & Biases",
      "LaTeX",
    ],
    githubUrl: "https://github.com/hieptran1812/few-shot-nlp-research",
    liveUrl: null,
    featured: false,
    status: "Published",
    highlights: [
      "Published in ACL 2024",
      "15% improvement over baseline",
      "Open-sourced implementation",
      "Reproducible experiments",
    ],
  },
];

const statusColors = {
  Production: {
    backgroundColor: "var(--surface-accent)",
    color: "var(--accent)",
  },
  "Active Development": { backgroundColor: "#fef3c7", color: "#d97706" },
  Beta: { backgroundColor: "#dbeafe", color: "#2563eb" },
  Published: { backgroundColor: "#f3e8ff", color: "#7c3aed" },
};

export default function ProjectsPage() {
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [selectedProject, setSelectedProject] = useState<
    (typeof projects)[0] | null
  >(null);

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

                {/* Animated underline */}
                <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-0 h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-indigo-500 rounded-full group-hover:w-full transition-all duration-1000 ease-out"></div>
              </h1>

              {/* Floating particles */}
              <div className="absolute inset-0 pointer-events-none">
                <div className="absolute top-1/4 left-1/4 w-1 h-1 bg-blue-300 rounded-full animate-bounce opacity-40"></div>
                <div className="absolute top-3/4 right-1/4 w-1 h-1 bg-purple-300 rounded-full animate-bounce delay-300 opacity-50"></div>
                <div className="absolute bottom-1/4 left-3/4 w-1 h-1 bg-indigo-300 rounded-full animate-bounce delay-500 opacity-30"></div>
              </div>
            </div>

            {/* Enhanced subtitle with typing effect appearance */}
            <div className="relative">
              <p
                className="text-xl md:text-2xl mb-8 max-w-4xl mx-auto leading-relaxed opacity-0 animate-fade-in-up"
                style={{
                  color: "var(--text-secondary)",
                  animationDelay: "0.5s",
                  animationFillMode: "forwards",
                }}
              >
                A showcase of my work in{" "}
                <span className="font-semibold bg-gradient-to-r from-blue-500 to-purple-600 bg-clip-text text-transparent">
                  artificial intelligence
                </span>
                ,{" "}
                <span className="font-semibold bg-gradient-to-r from-purple-500 to-indigo-600 bg-clip-text text-transparent">
                  machine learning
                </span>
                ,{" "}
                <span className="font-semibold bg-gradient-to-r from-indigo-500 to-blue-600 bg-clip-text text-transparent">
                  web development
                </span>
                , and{" "}
                <span className="font-semibold bg-gradient-to-r from-cyan-500 to-purple-600 bg-clip-text text-transparent">
                  research
                </span>
                . Each project represents a unique challenge solved with
                innovative technology and engineering excellence.
              </p>
            </div>

            {/* Enhanced animated stats badges */}
            <div
              className="flex flex-wrap justify-center gap-4 opacity-0 animate-fade-in-up"
              style={{ animationDelay: "1s", animationFillMode: "forwards" }}
            >
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
                🎯 <span className="font-bold">{projects.length}</span> Total
                Projects
              </span>
              <span
                className="group px-6 py-3 rounded-full text-sm font-semibold transition-all duration-300 hover:scale-110 hover:shadow-lg cursor-default border"
                style={{
                  backgroundColor: "var(--surface-accent)",
                  color: "var(--accent)",
                  borderColor: "var(--accent)/20",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = "scale(1.1) rotate(-1deg)";
                  e.currentTarget.style.boxShadow =
                    "0 10px 25px rgba(147, 51, 234, 0.3)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = "scale(1)";
                  e.currentTarget.style.boxShadow = "none";
                }}
              >
                ⭐ <span className="font-bold">{featuredProjects.length}</span>{" "}
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
                <span className="font-bold">
                  {projects.filter((p) => p.status === "Production").length}
                </span>{" "}
                In Production
              </span>
            </div>
          </div>

          {/* Featured Projects */}
          <div className="mb-20">
            <h2
              className="text-4xl font-bold text-center mb-4"
              style={{ color: "var(--text-primary)" }}
            >
              Featured Projects
            </h2>
            <p
              className="text-lg text-center mb-12 max-w-3xl mx-auto"
              style={{ color: "var(--text-secondary)" }}
            >
              Highlighting my most impactful and innovative work
            </p>
            <div className="grid lg:grid-cols-3 gap-8">
              {featuredProjects.map((project) => (
                <div
                  key={project.id}
                  className="group relative overflow-hidden rounded-2xl border transition-all duration-300 hover:scale-105 hover:shadow-2xl cursor-pointer"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--card-border)",
                  }}
                  onClick={() => setSelectedProject(project)}
                >
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
                      {project.technologies.slice(0, 3).map((tech, index) => (
                        <span
                          key={index}
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
                    <div className="flex gap-3">
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
                        <span>👨‍💻</span>
                        Code
                      </a>
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
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Category Filter */}
          <div className="mb-12">
            <h2
              className="text-4xl font-bold text-center mb-12"
              style={{ color: "var(--text-primary)" }}
            >
              All Projects
            </h2>
            <div className="flex flex-wrap justify-center gap-3">
              {projectCategories.map((category) => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className="px-6 py-3 rounded-xl font-medium transition-all duration-200"
                  style={{
                    backgroundColor:
                      selectedCategory === category
                        ? "var(--accent)"
                        : "var(--surface)",
                    color:
                      selectedCategory === category
                        ? "white"
                        : "var(--text-secondary)",
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
                      e.currentTarget.style.backgroundColor = "var(--surface)";
                      e.currentTarget.style.color = "var(--text-secondary)";
                    }
                  }}
                >
                  {category}
                </button>
              ))}
            </div>
          </div>

          {/* Projects Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-20">
            {filteredProjects.map((project) => (
              <div
                key={project.id}
                className="group overflow-hidden rounded-2xl border transition-all duration-300 hover:scale-105 hover:shadow-xl cursor-pointer"
                style={{
                  backgroundColor: "var(--card-bg)",
                  borderColor: "var(--card-border)",
                }}
                onClick={() => setSelectedProject(project)}
              >
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
                    {project.technologies.slice(0, 3).map((tech, index) => (
                      <span
                        key={index}
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
                  <div className="flex gap-3">
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
                </div>
              </div>
            ))}
          </div>

          {/* Call to Action */}
          <div
            className="p-10 rounded-2xl border text-center"
            style={{
              backgroundColor: "var(--card-bg)",
              borderColor: "var(--card-border)",
            }}
          >
            <h2
              className="text-3xl font-bold mb-6"
              style={{ color: "var(--text-primary)" }}
            >
              Interested in Collaboration?
            </h2>
            <p
              className="text-lg mb-8 max-w-3xl mx-auto"
              style={{ color: "var(--text-secondary)" }}
            >
              I&apos;m always open to discussing new projects, innovative ideas,
              or opportunities to contribute to exciting initiatives. Let&apos;s
              build something amazing together!
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <a
                href="/contact"
                className="inline-flex items-center px-8 py-4 font-semibold rounded-xl transition-all duration-200 text-white shadow-lg hover:shadow-xl transform hover:scale-105"
                style={{ backgroundColor: "var(--accent)" }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = "var(--accent-hover)";
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
          </div>
        </div>

        {/* Project Detail Modal */}
        {selectedProject && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-6 z-50"
            onClick={() => setSelectedProject(null)}
          >
            <div
              className="max-w-4xl w-full max-h-[90vh] overflow-y-auto rounded-2xl border"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
              }}
              onClick={(e) => e.stopPropagation()}
            >
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

                <div className="flex gap-4">
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
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
