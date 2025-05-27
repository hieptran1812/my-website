"use client";

// This would ideally be in a separate metadata file, but for client components we'll handle it differently
// We'll add the metadata to the parent layout or create a wrapper component

const skills = [
  {
    category: "Programming Languages",
    items: [
      "Python",
      "JavaScript/TypeScript",
      "Java",
      "C++",
      "Go",
      "SQL",
      "R",
      "Scala",
    ],
    icon: "🔧",
  },
  {
    category: "AI/ML Technologies",
    items: [
      "TensorFlow",
      "PyTorch",
      "Scikit-learn",
      "OpenCV",
      "NLTK",
      "Transformers",
      "Hugging Face",
      "Keras",
    ],
    icon: "🤖",
  },
  {
    category: "Web Technologies",
    items: [
      "React/Next.js",
      "Node.js",
      "Express",
      "FastAPI",
      "PostgreSQL",
      "MongoDB",
      "Redis",
      "GraphQL",
    ],
    icon: "🌐",
  },
  {
    category: "Tools & Platforms",
    items: [
      "Docker",
      "Kubernetes",
      "AWS",
      "Git",
      "Linux",
      "Jupyter",
      "Apache Spark",
      "Elasticsearch",
    ],
    icon: "⚙️",
  },
  {
    category: "Data Science",
    items: [
      "Pandas",
      "NumPy",
      "Matplotlib",
      "Seaborn",
      "Apache Airflow",
      "MLflow",
      "Tableau",
      "Power BI",
    ],
    icon: "📊",
  },
  {
    category: "DevOps & Cloud",
    items: [
      "CI/CD",
      "Terraform",
      "Azure",
      "GCP",
      "Jenkins",
      "Prometheus",
      "Grafana",
      "ELK Stack",
    ],
    icon: "☁️",
  },
];

const experiences = [
  {
    title: "AI/ML Research & Development",
    description:
      "Designing and implementing cutting-edge machine learning algorithms for NLP, computer vision, and data analytics. Published research in top-tier conferences and journals.",
    icon: "🔬",
    highlights: ["Deep Learning", "Research Publications", "Algorithm Design"],
  },
  {
    title: "Full-Stack Software Engineering",
    description:
      "Building scalable, production-ready applications using modern frameworks and cloud technologies. Focus on microservices architecture and DevOps best practices.",
    icon: "💻",
    highlights: ["System Design", "Scalable Architecture", "Cloud Computing"],
  },
  {
    title: "Open Source & Community",
    description:
      "Active contributor to open-source projects in the AI/ML ecosystem. Maintaining libraries, mentoring developers, and sharing knowledge through technical blogs.",
    icon: "🌐",
    highlights: ["Open Source", "Technical Writing", "Community Building"],
  },
];

const achievements = [
  {
    title: "Academic Excellence",
    description:
      "Graduated with honors in Computer Science, specializing in Machine Learning and Software Engineering. Strong foundation in theoretical CS and practical applications.",
    metric: "Summa Cum Laude",
    icon: "🎓",
  },
  {
    title: "Research Impact",
    description:
      "Published research papers in machine learning, AI, and software engineering. Work cited by international research community.",
    metric: "10+ Publications",
    icon: "📚",
  },
  {
    title: "Industry Experience",
    description:
      "Successfully delivered production ML systems and web applications serving thousands of users. Led technical teams and mentored junior developers.",
    metric: "3+ Years",
    icon: "🏆",
  },
  {
    title: "Project Portfolio",
    description:
      "Developed and deployed multiple full-stack applications, ML models, and research prototypes across various domains and technologies.",
    metric: "25+ Projects",
    icon: "🚀",
  },
];

const education = [
  {
    degree: "Master of Science in Computer Science",
    institution: "University Name",
    year: "2023-2025",
    specialization: "Machine Learning & AI",
    coursework: [
      "Advanced Machine Learning",
      "Deep Learning",
      "Computer Vision",
      "Natural Language Processing",
      "Distributed Systems",
      "Advanced Algorithms",
    ],
    gpa: "3.9/4.0",
  },
  {
    degree: "Bachelor of Science in Computer Science",
    institution: "University Name",
    year: "2019-2023",
    specialization: "Software Engineering",
    coursework: [
      "Data Structures & Algorithms",
      "Software Engineering",
      "Database Systems",
      "Operating Systems",
      "Computer Networks",
      "Web Development",
    ],
    gpa: "3.8/4.0",
  },
];

const certifications = [
  {
    name: "AWS Certified Solutions Architect",
    issuer: "Amazon Web Services",
    year: "2024",
  },
  {
    name: "Google Cloud Professional ML Engineer",
    issuer: "Google Cloud",
    year: "2024",
  },
  {
    name: "TensorFlow Developer Certificate",
    issuer: "TensorFlow",
    year: "2023",
  },
  { name: "Kubernetes Application Developer", issuer: "CNCF", year: "2023" },
];

export default function About() {
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
          <div className="text-center mb-20">
            <div className="relative w-40 h-40 mx-auto mb-8">
              <div
                className="w-full h-full rounded-full border-4 flex items-center justify-center text-5xl font-bold shadow-xl"
                style={{
                  backgroundColor: "var(--card-bg)",
                  borderColor: "var(--accent)",
                  color: "var(--accent)",
                  boxShadow: "0 20px 40px rgba(0,0,0,0.1)",
                }}
              >
                HT
              </div>
              <div
                className="absolute -bottom-2 -right-2 w-8 h-8 rounded-full flex items-center justify-center text-lg border-2"
                style={{
                  backgroundColor: "var(--accent)",
                  borderColor: "var(--background)",
                  color: "white",
                }}
              >
                🚀
              </div>
            </div>
            <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent">
              Hiep Tran
            </h1>
            <p
              className="text-2xl md:text-3xl mb-6 font-semibold"
              style={{ color: "var(--accent)" }}
            >
              Computer Science Graduate & AI Research Engineer
            </p>
            <p
              className="text-lg max-w-4xl mx-auto leading-relaxed mb-8"
              style={{ color: "var(--text-secondary)" }}
            >
              Passionate AI researcher and software engineer with expertise in
              machine learning, distributed systems, and full-stack development.
              Dedicated to building intelligent systems that solve real-world
              problems and advance the field of artificial intelligence.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <span
                className="px-4 py-2 rounded-full text-sm font-medium"
                style={{
                  backgroundColor: "var(--surface-accent)",
                  color: "var(--accent)",
                }}
              >
                🎓 MS Computer Science
              </span>
              <span
                className="px-4 py-2 rounded-full text-sm font-medium"
                style={{
                  backgroundColor: "var(--surface-accent)",
                  color: "var(--accent)",
                }}
              >
                🔬 AI Researcher
              </span>
              <span
                className="px-4 py-2 rounded-full text-sm font-medium"
                style={{
                  backgroundColor: "var(--surface-accent)",
                  color: "var(--accent)",
                }}
              >
                💻 Full-Stack Engineer
              </span>
            </div>
          </div>

          {/* Experience Cards */}
          <div className="mb-20">
            <h2
              className="text-4xl font-bold text-center mb-4"
              style={{ color: "var(--text-primary)" }}
            >
              What I Do
            </h2>
            <p
              className="text-lg text-center mb-12 max-w-3xl mx-auto"
              style={{ color: "var(--text-secondary)" }}
            >
              Combining theoretical knowledge with practical experience to
              create innovative solutions
            </p>
            <div className="grid lg:grid-cols-3 gap-8">
              {experiences.map((exp, index) => (
                <div
                  key={index}
                  className="group p-8 rounded-2xl border transition-all duration-300 hover:scale-105 hover:shadow-xl"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--card-border)",
                  }}
                >
                  <div className="text-4xl mb-6 group-hover:scale-110 transition-transform duration-300">
                    {exp.icon}
                  </div>
                  <h3
                    className="text-2xl font-bold mb-4"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {exp.title}
                  </h3>
                  <p
                    className="leading-relaxed mb-4"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {exp.description}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {exp.highlights.map((highlight, idx) => (
                      <span
                        key={idx}
                        className="px-3 py-1 rounded-full text-xs font-medium"
                        style={{
                          backgroundColor: "var(--surface)",
                          color: "var(--accent)",
                        }}
                      >
                        {highlight}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Skills Section */}
          <div className="mb-20">
            <h2
              className="text-4xl font-bold text-center mb-4"
              style={{ color: "var(--text-primary)" }}
            >
              Technical Expertise
            </h2>
            <p
              className="text-lg text-center mb-12 max-w-3xl mx-auto"
              style={{ color: "var(--text-secondary)" }}
            >
              Comprehensive skill set spanning multiple domains and technologies
            </p>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {skills.map((skillSet, index) => (
                <div
                  key={index}
                  className="group p-6 rounded-2xl border hover:shadow-lg transition-all duration-300"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--card-border)",
                  }}
                >
                  <div className="flex items-center mb-4">
                    <span className="text-2xl mr-3 group-hover:scale-110 transition-transform duration-300">
                      {skillSet.icon}
                    </span>
                    <h3
                      className="text-lg font-bold"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {skillSet.category}
                    </h3>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {skillSet.items.map((skill, skillIndex) => (
                      <span
                        key={skillIndex}
                        className="inline-block px-3 py-1.5 rounded-lg text-sm font-medium transition-colors duration-200 hover:scale-105"
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
                          e.currentTarget.style.backgroundColor =
                            "var(--surface)";
                          e.currentTarget.style.color = "var(--text-secondary)";
                        }}
                      >
                        {skill}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Achievements Section */}
          <div className="mb-20">
            <h2
              className="text-4xl font-bold text-center mb-4"
              style={{ color: "var(--text-primary)" }}
            >
              Key Achievements
            </h2>
            <p
              className="text-lg text-center mb-12 max-w-3xl mx-auto"
              style={{ color: "var(--text-secondary)" }}
            >
              Milestones that reflect my commitment to excellence and continuous
              growth
            </p>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {achievements.map((achievement, index) => (
                <div
                  key={index}
                  className="group text-center p-6 rounded-2xl border hover:shadow-lg transition-all duration-300 hover:scale-105"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--card-border)",
                  }}
                >
                  <div className="text-3xl mb-3 group-hover:scale-110 transition-transform duration-300">
                    {achievement.icon}
                  </div>
                  <div
                    className="text-2xl font-bold mb-2"
                    style={{ color: "var(--accent)" }}
                  >
                    {achievement.metric}
                  </div>
                  <h3
                    className="text-lg font-semibold mb-3"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {achievement.title}
                  </h3>
                  <p
                    className="text-sm leading-relaxed"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {achievement.description}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Education Section */}
          <div className="mb-20">
            <h2
              className="text-4xl font-bold text-center mb-4"
              style={{ color: "var(--text-primary)" }}
            >
              Education & Academic Background
            </h2>
            <p
              className="text-lg text-center mb-12 max-w-3xl mx-auto"
              style={{ color: "var(--text-secondary)" }}
            >
              Strong academic foundation with focus on advanced computer science
              concepts
            </p>
            <div className="space-y-6">
              {education.map((edu, index) => (
                <div
                  key={index}
                  className="p-8 rounded-2xl border"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--card-border)",
                  }}
                >
                  <div className="grid md:grid-cols-3 gap-6">
                    <div className="md:col-span-2">
                      <h3
                        className="text-2xl font-bold mb-2"
                        style={{ color: "var(--text-primary)" }}
                      >
                        {edu.degree}
                      </h3>
                      <p
                        className="text-lg mb-2"
                        style={{ color: "var(--accent)" }}
                      >
                        {edu.institution}
                      </p>
                      <p
                        className="text-sm mb-4"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        {edu.year} • Specialization: {edu.specialization} • GPA:{" "}
                        {edu.gpa}
                      </p>
                      <div>
                        <h4
                          className="font-semibold mb-2"
                          style={{ color: "var(--text-primary)" }}
                        >
                          Relevant Coursework:
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {edu.coursework.map((course, idx) => (
                            <span
                              key={idx}
                              className="px-3 py-1 rounded-lg text-sm"
                              style={{
                                backgroundColor: "var(--surface)",
                                color: "var(--text-secondary)",
                              }}
                            >
                              {course}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center justify-center">
                      <div
                        className="w-24 h-24 rounded-full flex items-center justify-center text-3xl"
                        style={{
                          backgroundColor: "var(--surface-accent)",
                          color: "var(--accent)",
                        }}
                      >
                        🎓
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Certifications */}
          <div className="mb-20">
            <h2
              className="text-4xl font-bold text-center mb-12"
              style={{ color: "var(--text-primary)" }}
            >
              Professional Certifications
            </h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
              {certifications.map((cert, index) => (
                <div
                  key={index}
                  className="p-4 rounded-xl border text-center hover:shadow-lg transition-all duration-300"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--card-border)",
                  }}
                >
                  <div className="text-2xl mb-2">🏅</div>
                  <h4
                    className="font-semibold text-sm mb-1"
                    style={{ color: "var(--text-primary)" }}
                  >
                    {cert.name}
                  </h4>
                  <p
                    className="text-xs mb-1"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    {cert.issuer}
                  </p>
                  <p className="text-xs" style={{ color: "var(--accent)" }}>
                    {cert.year}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Research Interests */}
          <div className="mb-20">
            <h2
              className="text-4xl font-bold text-center mb-12"
              style={{ color: "var(--text-primary)" }}
            >
              Research Interests & Focus Areas
            </h2>
            <div
              className="max-w-5xl mx-auto p-8 rounded-2xl border"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
              }}
            >
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3
                    className="text-2xl font-semibold mb-6 flex items-center"
                    style={{ color: "var(--text-primary)" }}
                  >
                    <span className="mr-3">🔬</span>
                    Current Research Focus
                  </h3>
                  <ul
                    className="space-y-4"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    <li className="flex items-start">
                      <span
                        className="mr-3 mt-1"
                        style={{ color: "var(--accent)" }}
                      >
                        •
                      </span>
                      <div>
                        <strong>Large Language Models & NLP:</strong> Advancing
                        transformer architectures, few-shot learning, and
                        multilingual understanding
                      </div>
                    </li>
                    <li className="flex items-start">
                      <span
                        className="mr-3 mt-1"
                        style={{ color: "var(--accent)" }}
                      >
                        •
                      </span>
                      <div>
                        <strong>Computer Vision & Multimodal AI:</strong>{" "}
                        Developing efficient vision models and cross-modal
                        learning systems
                      </div>
                    </li>
                    <li className="flex items-start">
                      <span
                        className="mr-3 mt-1"
                        style={{ color: "var(--accent)" }}
                      >
                        •
                      </span>
                      <div>
                        <strong>MLOps & AI Infrastructure:</strong> Building
                        scalable ML systems and automated deployment pipelines
                      </div>
                    </li>
                    <li className="flex items-start">
                      <span
                        className="mr-3 mt-1"
                        style={{ color: "var(--accent)" }}
                      >
                        •
                      </span>
                      <div>
                        <strong>Distributed Systems:</strong> Designing
                        fault-tolerant, high-performance computing systems for
                        AI workloads
                      </div>
                    </li>
                  </ul>
                </div>
                <div>
                  <h3
                    className="text-2xl font-semibold mb-6 flex items-center"
                    style={{ color: "var(--text-primary)" }}
                  >
                    <span className="mr-3">🎯</span>
                    Research Impact & Goals
                  </h3>
                  <p
                    className="leading-relaxed mb-6"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    My research focuses on developing practical AI solutions
                    that bridge the gap between academic innovation and
                    real-world applications. I&apos;m particularly interested in
                    making advanced AI techniques more accessible and efficient
                    for deployment in production environments.
                  </p>
                  <p
                    className="leading-relaxed"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    Current work involves optimizing transformer models for
                    resource-constrained environments, developing novel
                    approaches to multimodal learning, and creating robust MLOps
                    frameworks that enable seamless integration of AI into
                    existing software systems.
                  </p>
                  <div
                    className="mt-6 p-4 rounded-xl"
                    style={{ backgroundColor: "var(--surface-accent)" }}
                  >
                    <p
                      className="text-sm font-medium"
                      style={{ color: "var(--accent)" }}
                    >
                      🌟 &quot;Building AI systems that are not just
                      intelligent, but also practical, ethical, and accessible
                      to everyone.&quot;
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Contact CTA */}
          <div
            className="p-10 rounded-2xl border text-center"
            style={{
              backgroundColor: "var(--card-bg)",
              borderColor: "var(--card-border)",
            }}
          >
            <div className="max-w-3xl mx-auto">
              <h2
                className="text-3xl font-bold mb-6"
                style={{ color: "var(--text-primary)" }}
              >
                Let&apos;s Build Something Amazing Together
              </h2>
              <p
                className="text-lg mb-8 leading-relaxed"
                style={{ color: "var(--text-secondary)" }}
              >
                I&apos;m always excited to collaborate on innovative projects,
                discuss research opportunities, or explore how AI can solve
                complex challenges. Whether you&apos;re looking for a technical
                partner, research collaborator, or want to discuss the latest in
                AI and software engineering.
              </p>
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
                  <span className="mr-2">✉️</span>
                  Get in Touch
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
                  GitHub Profile
                </a>
                <a
                  href="/projects"
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
                  <span className="mr-2">🚀</span>
                  View Projects
                </a>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
