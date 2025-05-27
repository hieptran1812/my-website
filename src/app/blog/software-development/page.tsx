import React from "react";
import Link from "next/link";

const softwareDevPosts = [
  {
    id: 1,
    title: "Building Scalable Microservices with Node.js and Docker",
    excerpt:
      "A comprehensive guide to designing, implementing, and deploying microservices architecture using modern tools and best practices.",
    date: "2024-03-22",
    readTime: "15 min read",
    tags: ["Microservices", "Node.js", "Docker", "Architecture"],
    difficulty: "Advanced",
    featured: true,
  },
  {
    id: 2,
    title: "Modern React Patterns: Hooks, Context, and State Management",
    excerpt:
      "Exploring advanced React patterns and state management strategies for building maintainable applications.",
    date: "2024-03-18",
    readTime: "12 min read",
    tags: ["React", "Hooks", "State Management", "Frontend"],
    difficulty: "Intermediate",
  },
  {
    id: 3,
    title: "API Design Best Practices: RESTful Services and GraphQL",
    excerpt:
      "Comprehensive guide to designing robust APIs, comparing REST and GraphQL approaches with real-world examples.",
    date: "2024-03-14",
    readTime: "10 min read",
    tags: ["API Design", "REST", "GraphQL", "Backend"],
    difficulty: "Intermediate",
  },
  {
    id: 4,
    title: "DevOps Pipeline: CI/CD with GitHub Actions and AWS",
    excerpt:
      "Setting up automated deployment pipelines using GitHub Actions, Docker, and AWS services for continuous delivery.",
    date: "2024-03-09",
    readTime: "14 min read",
    tags: ["DevOps", "CI/CD", "GitHub Actions", "AWS"],
    difficulty: "Advanced",
  },
  {
    id: 5,
    title: "Database Design Patterns for High-Performance Applications",
    excerpt:
      "Exploring database design patterns, indexing strategies, and optimization techniques for scalable applications.",
    date: "2024-03-05",
    readTime: "11 min read",
    tags: ["Database", "Performance", "SQL", "NoSQL"],
    difficulty: "Intermediate",
  },
  {
    id: 6,
    title: "Full-Stack TypeScript: End-to-End Type Safety",
    excerpt:
      "Building type-safe applications from frontend to backend using TypeScript, including API contracts and validation.",
    date: "2024-02-28",
    readTime: "13 min read",
    tags: ["TypeScript", "Full-Stack", "Type Safety", "Development"],
    difficulty: "Advanced",
  },
  {
    id: 7,
    title: "Testing Strategies: Unit, Integration, and E2E Testing",
    excerpt:
      "Comprehensive testing strategies for modern web applications, including tools, patterns, and best practices.",
    date: "2024-02-22",
    readTime: "9 min read",
    tags: ["Testing", "Quality Assurance", "Jest", "Cypress"],
    difficulty: "Intermediate",
  },
  {
    id: 8,
    title: "Performance Optimization: Frontend and Backend Techniques",
    excerpt:
      "Practical techniques for optimizing application performance across the entire stack, from code splitting to caching.",
    date: "2024-02-18",
    readTime: "16 min read",
    tags: ["Performance", "Optimization", "Caching", "Monitoring"],
    difficulty: "Advanced",
  },
];

export default function SoftwareDevelopmentBlogPage() {
  const featuredPost = softwareDevPosts.find((post) => post.featured);
  const regularPosts = softwareDevPosts.filter((post) => !post.featured);

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case "Beginner":
        return "#22c55e";
      case "Intermediate":
        return "#f59e0b";
      case "Advanced":
        return "#ef4444";
      default:
        return "var(--text-secondary)";
    }
  };

  return (
    <div
      className="flex flex-col min-h-screen transition-colors duration-300"
      style={{
        backgroundColor: "var(--background)",
        color: "var(--text-primary)",
      }}
    >
      <main className="flex-1 flex flex-col transition-colors duration-300">
        <div className="container mx-auto py-16 px-4 sm:px-6 lg:px-8">
          <div className="max-w-6xl mx-auto">
            {/* Header */}
            <div className="text-center mb-16">
              <div className="flex items-center justify-center gap-3 mb-6">
                <svg
                  className="w-8 h-8"
                  style={{ color: "var(--accent)" }}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                  />
                </svg>
                <h1
                  className="text-4xl md:text-5xl font-bold"
                  style={{
                    background:
                      "linear-gradient(135deg, var(--accent), var(--accent-hover))",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    backgroundClip: "text",
                  }}
                >
                  Software Development
                </h1>
              </div>
              <p
                className="text-xl max-w-3xl mx-auto leading-relaxed mb-8"
                style={{ color: "var(--text-secondary)" }}
              >
                In-depth tutorials, architectural insights, and best practices
                for modern software development. From frontend frameworks to
                backend systems and DevOps practices.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "Full-Stack",
                  "Backend",
                  "Frontend",
                  "DevOps",
                  "Architecture",
                  "Testing",
                ].map((tag) => (
                  <span
                    key={tag}
                    className="px-3 py-1 text-sm rounded-full border transition-colors duration-200 hover:bg-[var(--surface)] cursor-pointer"
                    style={{
                      borderColor: "var(--border)",
                      backgroundColor: "var(--surface)",
                      color: "var(--text-secondary)",
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            {/* Stats Overview */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
              <div
                className="rounded-lg p-6 text-center border"
                style={{
                  backgroundColor: "var(--surface)",
                  borderColor: "var(--border)",
                }}
              >
                <div
                  className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
                  style={{ backgroundColor: "var(--accent)" }}
                >
                  <svg
                    className="w-6 h-6 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                    />
                  </svg>
                </div>
                <div
                  className="text-2xl font-bold mb-2"
                  style={{ color: "var(--text-primary)" }}
                >
                  8+
                </div>
                <div
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Comprehensive Guides
                </div>
              </div>
              <div
                className="rounded-lg p-6 text-center border"
                style={{
                  backgroundColor: "var(--surface)",
                  borderColor: "var(--border)",
                }}
              >
                <div
                  className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
                  style={{ backgroundColor: "#f59e0b" }}
                >
                  <svg
                    className="w-6 h-6 text-white"
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
                </div>
                <div
                  className="text-2xl font-bold mb-2"
                  style={{ color: "var(--text-primary)" }}
                >
                  6+
                </div>
                <div
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Tech Categories
                </div>
              </div>
              <div
                className="rounded-lg p-6 text-center border"
                style={{
                  backgroundColor: "var(--surface)",
                  borderColor: "var(--border)",
                }}
              >
                <div
                  className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
                  style={{ backgroundColor: "#22c55e" }}
                >
                  <svg
                    className="w-6 h-6 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M13 10V3L4 14h7v7l9-11h-7z"
                    />
                  </svg>
                </div>
                <div
                  className="text-2xl font-bold mb-2"
                  style={{ color: "var(--text-primary)" }}
                >
                  Modern
                </div>
                <div
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Best Practices
                </div>
              </div>
            </div>

            {/* Featured Article */}
            {featuredPost && (
              <div className="mb-16">
                <h2
                  className="text-2xl font-bold mb-8 text-center"
                  style={{ color: "var(--text-primary)" }}
                >
                  Featured Article
                </h2>
                <div
                  className="rounded-lg p-8 border-2 transition-all duration-300 hover:shadow-lg relative overflow-hidden"
                  style={{
                    borderColor: "var(--accent)",
                    backgroundColor: "var(--surface)",
                  }}
                >
                  <div className="relative z-10">
                    <div className="flex flex-wrap items-center gap-4 mb-4">
                      <span
                        className="px-3 py-1 text-xs font-medium rounded-full"
                        style={{
                          backgroundColor: "var(--accent)",
                          color: "var(--background)",
                        }}
                      >
                        Featured
                      </span>
                      <span
                        className="px-2 py-1 text-xs rounded-md font-medium"
                        style={{
                          backgroundColor: getDifficultyColor(
                            featuredPost.difficulty
                          ),
                          color: "white",
                        }}
                      >
                        {featuredPost.difficulty}
                      </span>
                    </div>
                    <h3
                      className="text-2xl md:text-3xl font-bold mb-4"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {featuredPost.title}
                    </h3>
                    <p
                      className="text-lg mb-6 leading-relaxed"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {featuredPost.excerpt}
                    </p>
                    <div className="flex flex-wrap items-center justify-between gap-4">
                      <div className="flex flex-wrap gap-2">
                        {featuredPost.tags.map((tag) => (
                          <span
                            key={tag}
                            className="px-2 py-1 text-xs rounded-md"
                            style={{
                              backgroundColor: "var(--background)",
                              color: "var(--accent)",
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                      <div className="flex items-center gap-4">
                        <span
                          className="text-sm"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {featuredPost.date} • {featuredPost.readTime}
                        </span>
                        <Link
                          href={`/blog/software-development/${featuredPost.id}`}
                          className="px-4 py-2 rounded-md font-medium transition-all duration-200 hover:scale-105"
                          style={{
                            backgroundColor: "var(--accent)",
                            color: "var(--background)",
                          }}
                        >
                          Read Article
                        </Link>
                      </div>
                    </div>
                  </div>
                  <div
                    className="absolute top-0 right-0 w-32 h-32 opacity-5"
                    style={{
                      background:
                        "radial-gradient(circle, var(--accent) 0%, transparent 70%)",
                    }}
                  />
                </div>
              </div>
            )}

            {/* All Articles Grid */}
            <div className="mb-16">
              <h2
                className="text-2xl font-bold mb-8 text-center"
                style={{ color: "var(--text-primary)" }}
              >
                Latest Articles
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {regularPosts.map((post) => (
                  <div
                    key={post.id}
                    className="rounded-lg p-6 border transition-all duration-300 hover:shadow-lg hover:scale-105 group"
                    style={{
                      borderColor: "var(--border)",
                      backgroundColor: "var(--surface)",
                    }}
                  >
                    <div className="mb-4">
                      <span
                        className="text-xs px-2 py-1 rounded-full font-medium"
                        style={{
                          backgroundColor: getDifficultyColor(post.difficulty),
                          color: "white",
                        }}
                      >
                        {post.difficulty}
                      </span>
                    </div>
                    <h3
                      className="text-lg font-bold mb-3 group-hover:text-[var(--accent)] transition-colors duration-200"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {post.title}
                    </h3>
                    <p
                      className="text-sm mb-4 leading-relaxed"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {post.excerpt}
                    </p>
                    <div className="flex flex-wrap gap-1 mb-4">
                      {post.tags.map((tag) => (
                        <span
                          key={tag}
                          className="px-2 py-1 text-xs rounded-md"
                          style={{
                            backgroundColor: "var(--background)",
                            color: "var(--text-secondary)",
                          }}
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    <div className="flex items-center justify-between">
                      <span
                        className="text-xs"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {post.date} • {post.readTime}
                      </span>
                      <Link
                        href={`/blog/software-development/${post.id}`}
                        className="text-sm font-medium hover:underline transition-colors duration-200"
                        style={{ color: "var(--accent)" }}
                      >
                        Read Article →
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Learning Path Section */}
            <div
              className="rounded-lg p-8 border"
              style={{
                borderColor: "var(--border)",
                backgroundColor: "var(--surface)",
              }}
            >
              <h2
                className="text-2xl font-bold mb-6 text-center"
                style={{ color: "var(--text-primary)" }}
              >
                Recommended Learning Path
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div
                    className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
                    style={{ backgroundColor: "#22c55e" }}
                  >
                    <span className="text-white font-bold">1</span>
                  </div>
                  <h3
                    className="font-bold mb-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Fundamentals
                  </h3>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Start with basic concepts, testing strategies, and API
                    design patterns.
                  </p>
                </div>
                <div className="text-center">
                  <div
                    className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
                    style={{ backgroundColor: "#f59e0b" }}
                  >
                    <span className="text-white font-bold">2</span>
                  </div>
                  <h3
                    className="font-bold mb-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Intermediate
                  </h3>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Learn modern frameworks, state management, and database
                    design.
                  </p>
                </div>
                <div className="text-center">
                  <div
                    className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
                    style={{ backgroundColor: "#ef4444" }}
                  >
                    <span className="text-white font-bold">3</span>
                  </div>
                  <h3
                    className="font-bold mb-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Advanced
                  </h3>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Master microservices, DevOps pipelines, and system
                    architecture.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
