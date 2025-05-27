import React from "react";
import Link from "next/link";

const cryptoPosts = [
  {
    id: 1,
    title: "Understanding Blockchain Technology: A Comprehensive Guide",
    excerpt:
      "Dive deep into the fundamentals of blockchain technology, its applications, and how it's revolutionizing various industries from finance to supply chain management.",
    date: "2024-03-15",
    readTime: "8 min read",
    tags: ["Blockchain", "Technology", "Cryptocurrency", "Web3"],
    category: "Fundamentals",
    difficulty: "Beginner",
    featured: true,
  },
  {
    id: 2,
    title: "DeFi Protocols: The Future of Finance",
    excerpt:
      "Explore the world of Decentralized Finance and how smart contracts are reshaping traditional financial services with automated market makers and yield farming.",
    date: "2024-03-10",
    readTime: "12 min read",
    tags: ["DeFi", "Smart Contracts", "Finance", "AMM"],
    category: "DeFi",
    difficulty: "Intermediate",
  },
  {
    id: 3,
    title: "Bitcoin vs Ethereum: Technical Deep Dive",
    excerpt:
      "A comprehensive technical analysis comparing Bitcoin and Ethereum architectures, consensus mechanisms, scalability solutions, and their respective ecosystems.",
    date: "2024-03-05",
    readTime: "15 min read",
    tags: ["Bitcoin", "Ethereum", "Analysis", "Consensus"],
    category: "Technology",
    difficulty: "Advanced",
  },
  {
    id: 4,
    title: "NFTs and Digital Ownership Revolution",
    excerpt:
      "Understanding Non-Fungible Tokens, their technical implementation using ERC-721 and ERC-1155 standards, and implications for digital ownership and creative economies.",
    date: "2024-02-28",
    readTime: "10 min read",
    tags: ["NFT", "Digital Rights", "Blockchain", "ERC-721"],
    category: "NFTs",
    difficulty: "Intermediate",
  },
  {
    id: 5,
    title: "Crypto Security: Advanced Protection Strategies",
    excerpt:
      "Essential security measures for cryptocurrency holders, from hardware wallet management to multi-signature setups and avoiding sophisticated attack vectors.",
    date: "2024-02-20",
    readTime: "8 min read",
    tags: ["Security", "Wallets", "Best Practices", "Hardware"],
    category: "Security",
    difficulty: "Intermediate",
  },
  {
    id: 6,
    title: "Layer 2 Solutions: Scaling Blockchain Networks",
    excerpt:
      "Comprehensive guide to Layer 2 scaling solutions including optimistic rollups, zk-rollups, and sidechains for improving blockchain throughput and reducing costs.",
    date: "2024-02-15",
    readTime: "14 min read",
    tags: ["Layer 2", "Scaling", "Rollups", "Performance"],
    category: "Technology",
    difficulty: "Advanced",
  },
  {
    id: 7,
    title: "Smart Contract Development with Solidity",
    excerpt:
      "Learn to build and deploy smart contracts on Ethereum using Solidity, covering best practices, security considerations, and gas optimization techniques.",
    date: "2024-02-10",
    readTime: "18 min read",
    tags: ["Solidity", "Smart Contracts", "Development", "Ethereum"],
    category: "Development",
    difficulty: "Advanced",
  },
  {
    id: 8,
    title: "Tokenomics: Designing Sustainable Crypto Economies",
    excerpt:
      "Understanding token economics, distribution models, governance mechanisms, and how to design sustainable cryptocurrency ecosystems.",
    date: "2024-02-05",
    readTime: "11 min read",
    tags: ["Tokenomics", "Economics", "Governance", "Design"],
    category: "Economics",
    difficulty: "Intermediate",
  },
];

export default function CryptoBlogPage() {
  const featuredPost = cryptoPosts.find((post) => post.featured);
  const regularPosts = cryptoPosts.filter((post) => !post.featured);

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

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case "Fundamentals":
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
      case "DeFi":
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
              d="M13 10V3L4 14h7v7l9-11h-7z"
            />
          </svg>
        );
      case "Technology":
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
              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
            />
          </svg>
        );
      case "Security":
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
              d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
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
              d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
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
                    d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
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
                  Crypto & Blockchain
                </h1>
              </div>
              <p
                className="text-xl max-w-3xl mx-auto leading-relaxed mb-8"
                style={{ color: "var(--text-secondary)" }}
              >
                Exploring the intersection of technology and finance through
                cryptocurrency, blockchain innovations, DeFi protocols, and the
                decentralized web. From fundamentals to advanced development.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {[
                  "Web3",
                  "DeFi",
                  "Smart Contracts",
                  "NFTs",
                  "Layer 2",
                  "Security",
                  "Development",
                  "Economics",
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
                  In-depth Articles
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
                  5+
                </div>
                <div
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Technical Categories
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
                  Latest
                </div>
                <div
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Web3 Insights
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
                      <div
                        className="flex items-center gap-1 px-2 py-1 text-xs rounded-md"
                        style={{
                          backgroundColor: "var(--background)",
                          color: "var(--accent)",
                        }}
                      >
                        {getCategoryIcon(featuredPost.category)}
                        {featuredPost.category}
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
                          href={`/blog/crypto/${featuredPost.id}`}
                          className="px-4 py-2 rounded-md font-medium transition-all duration-200 hover:scale-105"
                          style={{
                            backgroundColor: "var(--accent)",
                            color: "var(--background)",
                          }}
                        >
                          Start Reading
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

            {/* Recent Articles */}
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
                          backgroundColor: getDifficultyColor(post.difficulty),
                          color: "white",
                        }}
                      >
                        {post.difficulty}
                      </span>
                      <div
                        className="flex items-center gap-1 px-2 py-1 text-xs rounded-md"
                        style={{
                          backgroundColor: "var(--background)",
                          color: "var(--accent)",
                        }}
                      >
                        {getCategoryIcon(post.category)}
                        {post.category}
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
                      {post.tags.slice(0, 3).map((tag) => (
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
                        href={`/blog/crypto/${post.id}`}
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
                Crypto Learning Resources
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
                    Blockchain basics, cryptography, and economic principles
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
                        d="M13 10V3L4 14h7v7l9-11h-7z"
                      />
                    </svg>
                  </div>
                  <h3
                    className="font-bold mb-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    DeFi & Web3
                  </h3>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Decentralized finance protocols and Web3 development
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
                        d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                      />
                    </svg>
                  </div>
                  <h3
                    className="font-bold mb-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Development
                  </h3>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Smart contract development and dApp creation
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
                        d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                      />
                    </svg>
                  </div>
                  <h3
                    className="font-bold mb-2"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Security
                  </h3>
                  <p
                    className="text-sm"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Best practices and advanced security measures
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
