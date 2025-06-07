import React from "react";
import Image from "next/image";
import Link from "next/link";
import { ProjectMetadata } from "@/lib/projects";

interface ProjectCardProps {
  project: ProjectMetadata;
  className?: string;
  compact?: boolean;
}

export default function ProjectCard({
  project,
  className = "",
  compact = false,
}: ProjectCardProps) {
  return (
    <Link
      href={`/projects/${project.slug}`}
      className={`group flex flex-col overflow-hidden rounded-xl transition-all duration-300 
                 shadow-lg hover:shadow-xl bg-white dark:bg-gray-800 
                 ${compact ? "h-[320px]" : "h-full"} ${className}`}
    >
      {/* Project Image */}
      <div className="relative h-40 overflow-hidden">
        <Image
          src={project.image || "/project-placeholder.jpg"}
          alt={project.title}
          fill
          className="object-cover transition-transform duration-500 group-hover:scale-110"
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
        />
        {project.status && (
          <div
            className="absolute top-3 right-3 px-2 py-1 text-xs font-medium rounded-full bg-opacity-80 backdrop-blur-sm"
            style={{
              backgroundColor:
                project.status === "Completed"
                  ? "rgba(16, 185, 129, 0.8)"
                  : project.status === "In Progress"
                  ? "rgba(245, 158, 11, 0.8)"
                  : project.status === "Planned"
                  ? "rgba(99, 102, 241, 0.8)"
                  : "rgba(209, 213, 219, 0.8)",
            }}
          >
            {project.status}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex flex-col flex-grow p-4">
        <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2 line-clamp-2">
          {project.title}
        </h3>

        {!compact && (
          <p className="text-sm text-gray-600 dark:text-gray-300 mb-4 line-clamp-3">
            {project.excerpt}
          </p>
        )}

        {/* Technologies */}
        <div className="mt-auto">
          <div className="flex flex-wrap gap-2">
            {project.technologies
              .slice(0, compact ? 2 : 3)
              .map((tech, index) => (
                <span
                  key={index}
                  className="px-2 py-1 text-xs font-medium rounded-full 
                         bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                >
                  {tech}
                </span>
              ))}
            {project.technologies.length > (compact ? 2 : 3) && (
              <span
                className="px-2 py-1 text-xs font-medium rounded-full 
                              bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200"
              >
                +{project.technologies.length - (compact ? 2 : 3)}
              </span>
            )}
          </div>
        </div>
      </div>
    </Link>
  );
}
