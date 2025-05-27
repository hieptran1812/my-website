import React from "react";
import Link from "next/link";

const notesPosts = [
  {
    id: 1,
    title: "Docker Best Practices: From Development to Production",
    excerpt:
      "Essential Docker practices, optimization techniques, and deployment strategies for scalable applications.",
    date: "2024-03-20",
    readTime: "7 min read",
    tags: ["Docker", "DevOps", "Containerization"],
    category: "Development",
    featured: true,
  },
  {
    id: 2,
    title: "Git Workflow Strategies for Team Collaboration",
    excerpt:
      "Comparing different Git workflows and best practices for maintaining clean, collaborative codebases.",
    date: "2024-03-15",
    readTime: "5 min read",
    tags: ["Git", "Version Control", "Collaboration"],
    category: "Development",
  },
  {
    id: 3,
    title: "System Design Interview Prep: Scalability Patterns",
    excerpt:
      "Key architectural patterns and design principles for building scalable distributed systems.",
    date: "2024-03-10",
    readTime: "12 min read",
    tags: ["System Design", "Architecture", "Scalability"],
    category: "Architecture",
  },
  {
    id: 4,
    title: "TypeScript Advanced Types: Utility Types Deep Dive",
    excerpt:
      "Exploring TypeScript's powerful utility types and how to leverage them for better type safety.",
    date: "2024-03-05",
    readTime: "8 min read",
    tags: ["TypeScript", "Types", "JavaScript"],
    category: "Development",
  },
  {
    id: 5,
    title: "Database Indexing Strategies and Performance Optimization",
    excerpt:
      "Understanding different types of database indexes and when to use them for optimal query performance.",
    date: "2024-02-28",
    readTime: "10 min read",
    tags: ["Database", "Performance", "SQL"],
    category: "Backend",
  },
  {
    id: 6,
    title: "React Performance Optimization Techniques",
    excerpt:
      "Practical techniques for optimizing React applications: memoization, code splitting, and more.",
    date: "2024-02-22",
    readTime: "9 min read",
    tags: ["React", "Performance", "Frontend"],
    category: "Frontend",
  },
  {
    id: 7,
    title: "Kubernetes Fundamentals: Pods, Services, and Deployments",
    excerpt:
      "Understanding core Kubernetes concepts and how to deploy applications effectively.",
    date: "2024-02-15",
    readTime: "11 min read",
    tags: ["Kubernetes", "Container Orchestration", "DevOps"],
  },
  {
    id: 8,
    title: "AWS Lambda Best Practices and Cost Optimization",
    excerpt:
      "Maximizing performance and minimizing costs when building serverless applications with AWS Lambda.",
    date: "2024-02-08",
    readTime: "6 min read",
    tags: ["AWS", "Serverless", "Cloud"],
    category: "Cloud",
  },
];

const categories = [
  "All",
  "Development",
  "Architecture",
  "Frontend",
  "Backend",
  "DevOps",
  "Cloud",
];

export default function NotesBlogPage() {
  const featuredPost = notesPosts.find((post) => post.featured);
  const regularPosts = notesPosts.filter((post) => !post.featured);

  const getCategoryColor = (category: string) => {
    const colors = {
      Development: "from-blue-500 to-blue-600",
      Architecture: "from-purple-500 to-purple-600",
      Frontend: "from-green-500 to-green-600",
      Backend: "from-orange-500 to-orange-600",
      DevOps: "from-red-500 to-red-600",
      Cloud: "from-sky-500 to-sky-600",
    };
    return (
      colors[category as keyof typeof colors] || "from-gray-500 to-gray-600"
    );
  };

  return (
    <div
      className="flex flex-col min-h-screen transition-colors duration-300"
      style={{
        backgroundColor: "var(--background)",
        color: "var(--text-primary)",
      }}
    >
      <main className="flex-1">
        <div className="max-w-6xl mx-auto px-6 py-16">
          {/* Header */}
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-3 mb-6">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center text-white text-xl font-bold">
                📝
              </div>
              <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-indigo-600 to-blue-600 bg-clip-text text-transparent">
                Notes & Quick Thoughts
              </h1>
            </div>
            <p
              className="text-xl max-w-3xl mx-auto leading-relaxed"
              style={{ color: "var(--text-secondary)" }}
            >
              Quick insights, learnings, and thoughts from my development
              journey. Bite-sized knowledge covering various tech topics and
              practical tips.
            </p>
          </div>

          {/* Stats Overview */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
            <div
              className="p-4 rounded-xl border text-center"
              style={{
                backgroundColor: "var(--surface)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="text-2xl font-bold mb-1"
                style={{ color: "var(--accent)" }}
              >
                {notesPosts.length}
              </div>
              <div
                className="text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                Total Notes
              </div>
            </div>
            <div
              className="p-4 rounded-xl border text-center"
              style={{
                backgroundColor: "var(--surface)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="text-2xl font-bold mb-1"
                style={{ color: "var(--accent)" }}
              >
                6
              </div>
              <div
                className="text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                Categories
              </div>
            </div>
            <div
              className="p-4 rounded-xl border text-center"
              style={{
                backgroundColor: "var(--surface)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="text-2xl font-bold mb-1"
                style={{ color: "var(--accent)" }}
              >
                {Math.round(
                  notesPosts.reduce(
                    (acc, post) => acc + parseInt(post.readTime),
                    0
                  ) / notesPosts.length
                )}
              </div>
              <div
                className="text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                Avg. Read Time
              </div>
            </div>
            <div
              className="p-4 rounded-xl border text-center"
              style={{
                backgroundColor: "var(--surface)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="text-2xl font-bold mb-1"
                style={{ color: "var(--accent)" }}
              >
                2024
              </div>
              <div
                className="text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                Latest Year
              </div>
            </div>
          </div>

          {/* Featured Note */}
          {featuredPost && (
            <div className="mb-16">
              <div className="flex items-center gap-3 mb-8">
                <svg
                  className="w-6 h-6"
                  style={{ color: "var(--accent)" }}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
                  />
                </svg>
                <h2
                  className="text-2xl font-bold"
                  style={{ color: "var(--text-primary)" }}
                >
                  Featured Note
                </h2>
              </div>

              <div
                className="rounded-2xl p-8 border transition-all duration-300 hover:shadow-xl"
                style={{
                  backgroundColor: "var(--surface)",
                  borderColor: "var(--border)",
                  background:
                    "linear-gradient(145deg, var(--surface), var(--surface-hover))",
                }}
              >
                <div className="flex flex-wrap gap-2 mb-4">
                  <span
                    className={`px-3 py-1 text-xs font-medium rounded-full text-white bg-gradient-to-r ${getCategoryColor(
                      featuredPost.category
                    )}`}
                  >
                    {featuredPost.category}
                  </span>
                  {featuredPost.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-3 py-1 text-xs font-medium rounded-full"
                      style={{
                        backgroundColor: "var(--accent-subtle)",
                        color: "var(--accent)",
                      }}
                    >
                      {tag}
                    </span>
                  ))}
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

                <div className="flex items-center justify-between">
                  <div
                    className="flex items-center gap-4 text-sm"
                    style={{ color: "var(--text-muted)" }}
                  >
                    <span>{featuredPost.date}</span>
                    <span>•</span>
                    <span>{featuredPost.readTime}</span>
                  </div>
                  <Link
                    href={`/blog/notes/${featuredPost.id}`}
                    className="inline-flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-200 hover:scale-105"
                    style={{
                      backgroundColor: "var(--accent)",
                      color: "white",
                    }}
                  >
                    Read Note
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
                        d="M9 5l7 7-7 7"
                      />
                    </svg>
                  </Link>
                </div>
              </div>
            </div>
          )}

          {/* Category Filter */}
          <div className="mb-12">
            <h2
              className="text-2xl font-bold mb-6"
              style={{ color: "var(--text-primary)" }}
            >
              Browse All Notes
            </h2>
            <div className="flex flex-wrap gap-3 mb-8">
              <span
                className="text-sm font-medium"
                style={{ color: "var(--text-secondary)" }}
              >
                Filter by category:
              </span>
              {categories.map((category) => (
                <button
                  key={category}
                  className="px-4 py-2 text-sm font-medium rounded-lg border transition-all duration-200 hover:shadow-md hover:scale-105"
                  style={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                >
                  {category}
                </button>
              ))}
            </div>
          </div>

          {/* Notes Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
            {regularPosts.map((post) => (
              <article
                key={post.id}
                className="group rounded-xl border transition-all duration-300 hover:shadow-xl hover:scale-105 overflow-hidden"
                style={{
                  backgroundColor: "var(--card-bg)",
                  borderColor: "var(--card-border)",
                }}
              >
                <div className="p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <span
                      className={`px-2 py-1 text-xs font-medium rounded-full text-white bg-gradient-to-r ${getCategoryColor(
                        post.category
                      )}`}
                    >
                      {post.category}
                    </span>
                    <span
                      className="text-xs"
                      style={{ color: "var(--text-muted)" }}
                    >
                      {post.readTime}
                    </span>
                  </div>

                  <h3
                    className="text-lg font-semibold mb-3 group-hover:text-sky-600 dark:group-hover:text-sky-400 transition-colors line-clamp-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {post.title}
                  </h3>

                  <p
                    className="text-sm mb-4 leading-relaxed line-clamp-3"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {post.excerpt}
                  </p>

                  <div className="flex flex-wrap gap-1 mb-4">
                    {post.tags.slice(0, 3).map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-1 text-xs rounded-full"
                        style={{
                          backgroundColor: "var(--accent-subtle)",
                          color: "var(--accent)",
                        }}
                      >
                        {tag}
                      </span>
                    ))}
                    {post.tags.length > 3 && (
                      <span
                        className="px-2 py-1 text-xs rounded-full"
                        style={{
                          backgroundColor: "var(--surface)",
                          color: "var(--text-muted)",
                        }}
                      >
                        +{post.tags.length - 3}
                      </span>
                    )}
                  </div>

                  <div
                    className="flex items-center justify-between pt-4 border-t"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <span
                      className="text-xs"
                      style={{ color: "var(--text-muted)" }}
                    >
                      {post.date}
                    </span>
                    <Link
                      href={`/blog/notes/${post.id}`}
                      className="text-sm font-medium transition-colors hover:underline"
                      style={{ color: "var(--accent)" }}
                    >
                      Read More →
                    </Link>
                  </div>
                </div>
              </article>
            ))}
          </div>

          {/* Newsletter CTA */}
          <div
            className="p-8 rounded-2xl border text-center"
            style={{
              backgroundColor: "var(--surface)",
              borderColor: "var(--border)",
            }}
          >
            <div className="max-w-2xl mx-auto">
              <h3
                className="text-2xl font-bold mb-4"
                style={{ color: "var(--text-primary)" }}
              >
                💡 Stay Updated with Quick Insights
              </h3>
              <p className="mb-6" style={{ color: "var(--text-secondary)" }}>
                Get notified when I publish new notes and quick thoughts.
                Bite-sized learning delivered to your inbox.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto">
                <input
                  type="email"
                  placeholder="Enter your email"
                  className="flex-1 px-4 py-3 rounded-lg border focus:outline-none focus:ring-2 focus:ring-sky-500 focus:border-transparent"
                  style={{
                    backgroundColor: "var(--background)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                />
                <button
                  className="px-6 py-3 font-medium rounded-lg transition-colors duration-200"
                  style={{
                    backgroundColor: "var(--accent)",
                    color: "white",
                  }}
                >
                  Subscribe
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
