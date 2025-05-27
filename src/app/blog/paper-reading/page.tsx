import React from "react";
import Link from "next/link";

const paperPosts = [
  {
    id: 1,
    title:
      "Attention Is All You Need: Breaking Down the Transformer Architecture",
    excerpt:
      "A comprehensive analysis of the groundbreaking paper that introduced the Transformer model and revolutionized NLP.",
    date: "2024-03-18",
    readTime: "12 min read",
    tags: ["Deep Learning", "Transformers", "NLP"],
    authors: "Vaswani et al.",
    venue: "NIPS 2017",
    featured: true,
  },
  {
    id: 2,
    title: "BERT: Bidirectional Encoder Representations from Transformers",
    excerpt:
      "Exploring how BERT's bidirectional training approach changed the landscape of language understanding tasks.",
    date: "2024-03-12",
    readTime: "10 min read",
    tags: ["BERT", "Language Models", "Pre-training"],
    authors: "Devlin et al.",
    venue: "NAACL 2019",
  },
  {
    id: 3,
    title: "GPT-3: Language Models are Few-Shot Learners",
    excerpt:
      "Understanding the scaling laws and emergent abilities demonstrated in OpenAI's GPT-3 research.",
    date: "2024-03-08",
    readTime: "15 min read",
    tags: ["GPT", "Few-Shot Learning", "Scaling"],
    authors: "Brown et al.",
    venue: "ArXiv 2020",
  },
  {
    id: 4,
    title: "ResNet: Deep Residual Learning for Image Recognition",
    excerpt:
      "How residual connections solved the vanishing gradient problem and enabled training of very deep networks.",
    date: "2024-03-01",
    readTime: "8 min read",
    tags: ["Computer Vision", "CNN", "Residual Networks"],
    authors: "He et al.",
    venue: "CVPR 2016",
  },
  {
    id: 5,
    title: "Diffusion Models: Denoising Diffusion Probabilistic Models",
    excerpt:
      "Understanding the mathematical foundations behind modern text-to-image generation models.",
    date: "2024-02-25",
    readTime: "14 min read",
    tags: ["Generative Models", "Diffusion", "DDPM"],
    authors: "Ho et al.",
    venue: "NIPS 2020",
  },
  {
    id: 6,
    title: "Vision Transformer: An Image is Worth 16x16 Words",
    excerpt:
      "How the Transformer architecture was successfully adapted for computer vision tasks.",
    date: "2024-02-18",
    readTime: "9 min read",
    tags: ["Vision Transformer", "Computer Vision", "Attention"],
    authors: "Dosovitskiy et al.",
    venue: "ICLR 2021",
  },
];

export default function PaperReadingBlogPage() {
  const featuredPost = paperPosts.find((post) => post.featured);
  const regularPosts = paperPosts.filter((post) => !post.featured);

  const getVenueColor = (venue: string) => {
    const colors = {
      NIPS: "from-purple-500 to-purple-600",
      NAACL: "from-blue-500 to-blue-600",
      ArXiv: "from-green-500 to-green-600",
      CVPR: "from-red-500 to-red-600",
      ICML: "from-yellow-500 to-orange-600",
      ICLR: "from-pink-500 to-pink-600",
    };
    const venueKey = venue.split(" ")[0] as keyof typeof colors;
    return colors[venueKey] || "from-gray-500 to-gray-600";
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
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center text-white text-xl font-bold">
                📄
              </div>
              <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                Paper Reading
              </h1>
            </div>
            <p
              className="text-xl max-w-3xl mx-auto leading-relaxed"
              style={{ color: "var(--text-secondary)" }}
            >
              Deep dives into influential research papers in AI, ML, and
              computer science. Breaking down complex ideas into digestible
              insights and practical takeaways.
            </p>
          </div>

          {/* Research Stats */}
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
                {paperPosts.length}
              </div>
              <div
                className="text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                Papers Reviewed
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
                5
              </div>
              <div
                className="text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                Research Areas
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
                2016-2024
              </div>
              <div
                className="text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                Publication Range
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
                  paperPosts.reduce(
                    (acc, post) => acc + parseInt(post.readTime),
                    0
                  ) / paperPosts.length
                )}
              </div>
              <div
                className="text-sm"
                style={{ color: "var(--text-secondary)" }}
              >
                Avg. Read Time
              </div>
            </div>
          </div>

          {/* Featured Paper */}
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
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <h2
                  className="text-2xl font-bold"
                  style={{ color: "var(--text-primary)" }}
                >
                  Featured Paper Analysis
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
                <div className="grid md:grid-cols-3 gap-6 mb-6">
                  <div>
                    <div
                      className="text-sm font-medium mb-1"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      Authors
                    </div>
                    <div
                      className="font-semibold"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {featuredPost.authors}
                    </div>
                  </div>
                  <div>
                    <div
                      className="text-sm font-medium mb-1"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      Published
                    </div>
                    <span
                      className={`px-3 py-1 text-sm font-medium rounded-full text-white bg-gradient-to-r ${getVenueColor(
                        featuredPost.venue
                      )}`}
                    >
                      {featuredPost.venue}
                    </span>
                  </div>
                  <div>
                    <div
                      className="text-sm font-medium mb-1"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      Reading Time
                    </div>
                    <div
                      className="font-semibold"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {featuredPost.readTime}
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2 mb-4">
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
                    <span>📝 {featuredPost.date}</span>
                    <span>•</span>
                    <span>🔬 Research Analysis</span>
                  </div>
                  <Link
                    href={`/blog/paper-reading/${featuredPost.id}`}
                    className="inline-flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all duration-200 hover:scale-105"
                    style={{
                      backgroundColor: "var(--accent)",
                      color: "white",
                    }}
                  >
                    Read Analysis
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

          {/* Research Areas */}
          <div className="mb-12">
            <h2
              className="text-2xl font-bold mb-6"
              style={{ color: "var(--text-primary)" }}
            >
              Research Areas
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
              {[
                "Deep Learning",
                "NLP",
                "Computer Vision",
                "Generative AI",
                "Reinforcement Learning",
              ].map((area) => (
                <button
                  key={area}
                  className="p-4 rounded-lg border transition-all duration-200 hover:shadow-md hover:scale-105 text-center"
                  style={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                >
                  <div className="font-medium text-sm">{area}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Papers Grid */}
          <div className="mb-16">
            <h2
              className="text-2xl font-bold mb-8"
              style={{ color: "var(--text-primary)" }}
            >
              Recent Paper Analyses
            </h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
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
                    <div className="flex items-center justify-between mb-4">
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded-full text-white bg-gradient-to-r ${getVenueColor(
                          post.venue
                        )}`}
                      >
                        {post.venue}
                      </span>
                      <span
                        className="text-xs"
                        style={{ color: "var(--text-muted)" }}
                      >
                        {post.readTime}
                      </span>
                    </div>

                    <h3
                      className="text-lg font-semibold mb-3 group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors line-clamp-2"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {post.title}
                    </h3>

                    <div
                      className="text-sm mb-3"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      <span className="font-medium">Authors:</span>{" "}
                      {post.authors}
                    </div>

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
                        href={`/blog/paper-reading/${post.id}`}
                        className="text-sm font-medium transition-colors hover:underline"
                        style={{ color: "var(--accent)" }}
                      >
                        Read Analysis →
                      </Link>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </div>

          {/* Research Resources */}
          <div
            className="p-8 rounded-2xl border"
            style={{
              backgroundColor: "var(--surface)",
              borderColor: "var(--border)",
            }}
          >
            <h3
              className="text-xl font-bold mb-4"
              style={{ color: "var(--text-primary)" }}
            >
              🔬 Research Resources
            </h3>
            <p className="mb-6" style={{ color: "var(--text-secondary)" }}>
              Useful resources for staying updated with the latest research:
            </p>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { name: "ArXiv", desc: "Latest preprints", icon: "📚" },
                {
                  name: "Papers With Code",
                  desc: "Code implementations",
                  icon: "💻",
                },
                {
                  name: "Google Scholar",
                  desc: "Citation tracking",
                  icon: "🎓",
                },
                {
                  name: "Semantic Scholar",
                  desc: "AI-powered search",
                  icon: "🔍",
                },
              ].map((resource) => (
                <a
                  key={resource.name}
                  href="#"
                  className="flex items-center gap-3 p-4 rounded-lg border transition-colors hover:shadow-md"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--card-border)",
                  }}
                >
                  <span className="text-2xl">{resource.icon}</span>
                  <div>
                    <div
                      className="font-medium"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {resource.name}
                    </div>
                    <div
                      className="text-sm"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {resource.desc}
                    </div>
                  </div>
                </a>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
