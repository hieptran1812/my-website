import React from "react";
import Link from "next/link";

const mlPosts = [
  {
    id: 1,
    title: "Transformer Architecture: Building GPT from Scratch",
    excerpt:
      "A comprehensive implementation guide to building a GPT-style transformer model from scratch using PyTorch, with detailed explanations of attention mechanisms.",
    date: "2024-03-25",
    readTime: "20 min read",
    tags: ["Transformers", "GPT", "PyTorch", "NLP"],
    level: "Advanced",
    type: "Tutorial",
    featured: true,
  },
  {
    id: 2,
    title: "Fine-tuning Large Language Models with LoRA",
    excerpt:
      "Learn how to efficiently fine-tune large language models using Low-Rank Adaptation (LoRA) for specific tasks while maintaining performance.",
    date: "2024-03-20",
    readTime: "15 min read",
    tags: ["LLM", "Fine-tuning", "LoRA", "Efficiency"],
    level: "Intermediate",
    type: "Tutorial",
  },
  {
    id: 3,
    title: "Computer Vision: Object Detection with YOLO v8",
    excerpt:
      "Step-by-step guide to implementing real-time object detection using the latest YOLO v8 architecture with custom dataset training.",
    date: "2024-03-16",
    readTime: "12 min read",
    tags: ["Computer Vision", "Object Detection", "YOLO", "Real-time"],
    level: "Intermediate",
    type: "Tutorial",
  },
  {
    id: 4,
    title: "MLOps: Production-Ready ML Pipelines with MLflow",
    excerpt:
      "Building scalable machine learning pipelines for production environments using MLflow for experiment tracking and model deployment.",
    date: "2024-03-12",
    readTime: "18 min read",
    tags: ["MLOps", "MLflow", "Production", "Pipeline"],
    level: "Advanced",
    type: "Guide",
  },
  {
    id: 5,
    title: "Reinforcement Learning: Deep Q-Networks (DQN) Implementation",
    excerpt:
      "Understanding and implementing Deep Q-Networks for solving complex sequential decision-making problems in reinforcement learning.",
    date: "2024-03-08",
    readTime: "16 min read",
    tags: ["Reinforcement Learning", "DQN", "Q-Learning", "Gaming"],
    level: "Advanced",
    type: "Tutorial",
  },
  {
    id: 6,
    title: "Generative AI: Building a Text-to-Image Model",
    excerpt:
      "Creating a text-to-image generation model using diffusion techniques, exploring the mathematics behind stable diffusion.",
    date: "2024-03-04",
    readTime: "14 min read",
    tags: ["Generative AI", "Diffusion Models", "Text-to-Image", "GANs"],
    level: "Advanced",
    type: "Tutorial",
  },
  {
    id: 7,
    title: "Feature Engineering for Machine Learning",
    excerpt:
      "Advanced feature engineering techniques including feature selection, dimensionality reduction, and handling categorical variables.",
    date: "2024-02-28",
    readTime: "10 min read",
    tags: ["Feature Engineering", "Data Science", "Preprocessing", "Analytics"],
    level: "Beginner",
    type: "Guide",
  },
  {
    id: 8,
    title: "Model Interpretability: SHAP and LIME Explained",
    excerpt:
      "Understanding black-box machine learning models using SHAP values and LIME for model interpretability and explainable AI.",
    date: "2024-02-24",
    readTime: "11 min read",
    tags: ["Explainable AI", "SHAP", "LIME", "Interpretability"],
    level: "Intermediate",
    type: "Analysis",
  },
];

export default function MachineLearningBlogPage() {
  const featuredPost = mlPosts.find((post) => post.featured);
  const regularPosts = mlPosts.filter((post) => !post.featured);

  const getLevelColor = (level: string) => {
    switch (level) {
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

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "Tutorial":
        return (
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
              d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
            />
          </svg>
        );
      case "Guide":
        return (
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
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        );
      case "Analysis":
        return (
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
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
        );
      default:
        return (
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
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        );
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
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
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
                  Machine Learning
                </h1>
              </div>
              <p
                className="text-xl max-w-3xl mx-auto leading-relaxed mb-8"
                style={{ color: "var(--text-secondary)" }}
              >
                Comprehensive guides, tutorials, and insights into machine
                learning, deep learning, and artificial intelligence. From
                foundational concepts to cutting-edge research implementations.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "Deep Learning",
                  "Computer Vision",
                  "NLP",
                  "MLOps",
                  "Generative AI",
                  "Reinforcement Learning",
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
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
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
                  In-depth Tutorials
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
                      d="M13 10V3L4 14h7v7l9-11h-7z"
                    />
                  </svg>
                </div>
                <div
                  className="text-2xl font-bold mb-2"
                  style={{ color: "var(--text-primary)" }}
                >
                  3
                </div>
                <div
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Difficulty Levels
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
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </svg>
                </div>
                <div
                  className="text-2xl font-bold mb-2"
                  style={{ color: "var(--text-primary)" }}
                >
                  Latest
                </div>
                <div
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  AI Research
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
                  Featured Tutorial
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
                          backgroundColor: getLevelColor(featuredPost.level),
                          color: "white",
                        }}
                      >
                        {featuredPost.level}
                      </span>
                      <div
                        className="flex items-center gap-1 px-2 py-1 text-xs rounded-md"
                        style={{
                          backgroundColor: "var(--background)",
                          color: "var(--accent)",
                        }}
                      >
                        {getTypeIcon(featuredPost.type)}
                        {featuredPost.type}
                      </div>
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
                          href={`/blog/machine-learning/${featuredPost.id}`}
                          className="px-4 py-2 rounded-md font-medium transition-all duration-200 hover:scale-105"
                          style={{
                            backgroundColor: "var(--accent)",
                            color: "var(--background)",
                          }}
                        >
                          Start Tutorial
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
                    <div className="mb-4 flex items-center gap-2">
                      <span
                        className="text-xs px-2 py-1 rounded-full font-medium"
                        style={{
                          backgroundColor: getLevelColor(post.level),
                          color: "white",
                        }}
                      >
                        {post.level}
                      </span>
                      <div
                        className="flex items-center gap-1 px-2 py-1 text-xs rounded-md"
                        style={{
                          backgroundColor: "var(--background)",
                          color: "var(--accent)",
                        }}
                      >
                        {getTypeIcon(post.type)}
                        {post.type}
                      </div>
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
                        href={`/blog/machine-learning/${post.id}`}
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

            {/* Learning Resources Section */}
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
                ML Learning Resources
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="text-center">
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
                        d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
                      />
                    </svg>
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
                    Mathematics, statistics, and core ML algorithms
                  </p>
                </div>
                <div className="text-center">
                  <div
                    className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
                    style={{ backgroundColor: "#3b82f6" }}
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
                        d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                      />
                    </svg>
                  </div>
                  <h3
                    className="font-bold mb-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Deep Learning
                  </h3>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Neural networks, transformers, and modern architectures
                  </p>
                </div>
                <div className="text-center">
                  <div
                    className="w-12 h-12 rounded-full mx-auto mb-4 flex items-center justify-center"
                    style={{ backgroundColor: "#8b5cf6" }}
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
                        d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 7.172V5L8 4z"
                      />
                    </svg>
                  </div>
                  <h3
                    className="font-bold mb-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Specializations
                  </h3>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Computer vision, NLP, and generative AI
                  </p>
                </div>
                <div className="text-center">
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
                  <h3
                    className="font-bold mb-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Production
                  </h3>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    MLOps, deployment, and scalable ML systems
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
