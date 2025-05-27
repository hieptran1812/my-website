"use client";

import React from "react";
import Image from "next/image";
import Link from "next/link";

const projects = [
  {
    title: "AnyLabeling",
    description:
      "Smart image labeling with Segment Anything and YOLO. A modern tool for efficient data annotation.",
    link: "/projects/anylabeling",
    image: "/project-placeholder.jpg",
    tags: ["AI", "Computer Vision", "Python"],
    stars: "2.6K",
  },
  {
    title: "Llama Assistant",
    description:
      "Privacy-focused AI assistant running locally. Open-source alternative to ChatGPT.",
    link: "/projects/llama-assistant",
    image: "/project-placeholder.jpg",
    tags: ["LLM", "Privacy", "TypeScript"],
    stars: "505",
  },
  {
    title: "Open ADAS",
    description:
      "Open-source Advanced Driver Assistance System for autonomous driving research.",
    link: "/projects/open-adas",
    image: "/project-placeholder.jpg",
    tags: ["C++", "Computer Vision", "Automotive"],
    stars: "452",
  },
];

export default function LatestProjectsSection() {
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
          {projects.map((project, idx) => (
            <Link
              href={project.link}
              key={idx}
              className="group block rounded-xl hover:shadow-lg transition-all duration-300 overflow-hidden border card-enhanced"
            >
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
                  {project.tags?.map((tag) => (
                    <span
                      key={tag}
                      className="text-xs px-2 py-0.5 rounded-full font-medium"
                      style={{
                        backgroundColor: "var(--accent-subtle)",
                        color: "var(--accent)",
                      }}
                    >
                      {tag}
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
          ))}
        </div>
      </div>
    </section>
  );
}
