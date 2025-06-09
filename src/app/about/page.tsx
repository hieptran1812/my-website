"use client";

import FadeInWrapper from "../../components/FadeInWrapper";
import { useState } from "react";
import Image from "next/image";

// This would ideally be in a separate metadata file, but for client components we'll handle it differently
// We'll add the metadata to the parent layout or create a wrapper component

// Work Experience Timeline
const workExperience = [
  {
    title: "AI Engineer",
    company: "Torilab",
    location: "Hanoi, Vietnam",
    period: "Jan 2024 - Present",
    type: "Full-time",
    description: "Developing and maintaining AI Agent for chit-chat",
    responsibilities: [
      "Developing and maintaining AI Agent for chit-chat including long-term memory, knowledge base, tool use features",
      "Implementing and optimizing LLM-based systems for natural language understanding and generation",
      "Collaborating with cross-functional teams to integrate AI solutions into products",
    ],
    technologies: ["Python", "PyTorch", "LLM", "AI Agent"],
    icon: "🔬",
  },
  {
    title: "AI Engineer",
    company: "VCCorp Corporation",
    location: "Hanoi, Vietnam",
    period: "March 2022 - Dec 2023",
    type: "Full-time",
    description:
      "Developing and maintaining AI systems for Vietnamese document understanding and computer vision applications for social media platforms.",
    responsibilities: [
      "Developing and maintaining Automated Information Extraction module for Vietnamese document understanding, often achieve 99% of accuracy",
      "Developing and optimizing automatic face attendance system with face detection and recognition, mask wearing check, person tracking modules",
      "Development of image and video classification system for Lotus social network",
      "Deploy and manage Machine learning models using Docker, Kubeflow",
    ],
    technologies: [
      "Python",
      "PyTorch",
      "Computer Vision",
      "NLP",
      "Airflow",
      "Docker",
      "Kubeflow",
    ],
    icon: "🔬",
  },
  {
    title: "AI Engineer",
    company: "A.L.I.S VIETNAM",
    location: "Hanoi, Vietnam",
    period: "April 2022 - Dec 2022",
    type: "Personal Project",
    description:
      "Developing a system to convert speech and text into sign language to assist Deaf people in accessing information.",
    responsibilities: [
      "Study of speech-to-text, text-to-speech models, GANs, recognition and conversion models to human actions",
      "Implement pipeline for machine learning model using FastAPI, Docker, Kubeflow",
      "Research and development of multimodal AI systems for accessibility",
      "Design and implement real-time processing systems for sign language conversion",
    ],
    technologies: [
      "Python",
      "GANs",
      "Speech Recognition",
      "Computer Vision",
      "FastAPI",
      "Docker",
      "Kubeflow",
    ],
    icon: "🤝",
  },
  {
    title: "Viblo Algorithm Content Creator",
    company: "Sun* Vietnam",
    location: "Hanoi, Vietnam",
    period: "June 2021 - Dec 2023",
    type: "Content Creator",
    description:
      "Contributing articles and algorithmic challenges on the Viblo ecosystem, focusing on AI/ML education and research.",
    responsibilities: [
      "Contributing articles, algorithmic challenges on the Viblo ecosystem",
      "Research and publish articles on Computer Vision, Natural Language Processing, Recommendation system and MLOps",
      "Creating educational content for the developer community",
      "Sharing insights on latest AI/ML trends and technologies",
    ],
    technologies: [
      "Computer Vision",
      "Natural Language Processing",
      "Recommendation Systems",
      "MLOps",
      "Technical Writing",
    ],
    icon: "✍️",
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

// Honors & Awards categorized by type
const awards = {
  hackathons: [
    {
      title: "GDSC HACKATHON VIETNAM 2023",
      organizer: "Google Developer Student Clubs and Coderschool",
      achievement: "Second Prize",
      icon: "🥈",
      year: "2023",
    },
    {
      title: "KO Hackathon",
      organizer: "Kambria",
      achievement: "First Prize",
      icon: "🥇",
      year: "2022",
    },
    {
      title: "Developer Circles Vietnam Innovation Challenge",
      organizer: "Facebook and Coderschool",
      achievement: "FPT Track Winner",
      icon: "🏅",
      year: "2022",
    },
  ],
  innovation: [
    {
      title: "Build On Vietnam 2022",
      organizer: "Amazon Web Services",
      achievement: "Best Innovation Award",
      icon: "🏆",
      year: "2022",
    },
    {
      title: "Creative Idea Challenge",
      organizer:
        "School of Information and Communication Technology - Hanoi University of Science and Technology",
      achievement: "First Prize",
      icon: "💡",
      year: "2022",
    },
  ],
  entrepreneurship: [
    {
      title: "Vietnam Social Innovation Challenge (VSIC)",
      organizer: "Foreign Trade University",
      achievement: "National Runner-up",
      icon: "🏆",
      year: "2022",
    },
    {
      title: "P-Startup Student Startup Idea Contest",
      organizer: "Posts and Telecommunications Institute of Technology",
      achievement: "First Prize",
      icon: "🥇",
      year: "2022",
    },
    {
      title: "Social Business Creation",
      organizer: "HEC Montréal",
      achievement: "Finalist",
      icon: "🌟",
      year: "2022",
    },
  ],
  technical: [
    {
      title: "VAIPE Medicine Pill Image Recognition Challenge",
      organizer: "VinUni-Illinois Smart Health Center (VISHC) at VinUniversity",
      achievement: "Encouragement Prize",
      icon: "🔬",
      year: "2022",
    },
    {
      title: "Mathematics Olympiad for Vietnamese Universities",
      organizer: "National Mathematics Committee",
      achievement: "Bronze Medal",
      icon: "🥉",
      year: "2022",
    },
  ],
};

const education = [
  {
    degree: "Master of Science in Computer Science",
    institution: "Posts And Telecommunications Institute of Technology",
    year: "2024-2026",
    specialization: "Machine Learning & AI",
    coursework: [
      "Advanced Machine Learning",
      "Deep Learning",
      "Computer Vision",
      "Natural Language Processing",
      "Distributed Systems",
      "Advanced Algorithms",
    ],
  },
  {
    degree: "Bachelor of Science in Computer Science",
    institution: "Posts And Telecommunications Institute of Technology",
    year: "2018-2023",
    specialization: "Software Engineering",
    coursework: [
      "Data Structures & Algorithms",
      "Software Engineering",
      "Database Systems",
      "Operating Systems",
      "Computer Networks",
      "Web Development",
      "Artificial Intelligence",
    ],
    gpa: "3.25/4.0",
  },
];

const AwardsCarousel = ({
  awards,
}: {
  awards: Record<
    string,
    Array<{
      title: string;
      organizer: string;
      achievement: string;
      icon: string;
      year: string;
    }>
  >;
}) => {
  const categories = [
    { id: "hackathons", name: "Hackathons", icon: "🚀" },
    { id: "innovation", name: "Innovation", icon: "💡" },
    { id: "entrepreneurship", name: "Entrepreneurship", icon: "💼" },
    { id: "technical", name: "Technical Competitions", icon: "🔬" },
  ];

  const [activeCategory, setActiveCategory] = useState("hackathons");
  const [animationDirection, setAnimationDirection] = useState("right");

  const handleCategoryChange = (newCategory: string) => {
    const currentIndex = categories.findIndex((c) => c.id === activeCategory);
    const newIndex = categories.findIndex((c) => c.id === newCategory);

    // Determine if we're going forward or backward for animation direction
    setAnimationDirection(newIndex > currentIndex ? "right" : "left");
    setActiveCategory(newCategory);
  };

  const handleNext = () => {
    const currentIndex = categories.findIndex((c) => c.id === activeCategory);
    const nextIndex = (currentIndex + 1) % categories.length;
    setAnimationDirection("right");
    setActiveCategory(categories[nextIndex].id);
  };

  const handlePrev = () => {
    const currentIndex = categories.findIndex((c) => c.id === activeCategory);
    const prevIndex =
      (currentIndex - 1 + categories.length) % categories.length;
    setAnimationDirection("left");
    setActiveCategory(categories[prevIndex].id);
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Category Tabs */}
      <div className="flex flex-wrap justify-center gap-4 mb-8">
        {categories.map((category, index) => (
          <FadeInWrapper
            key={category.id}
            direction="up"
            delay={350 + index * 50}
          >
            <button
              onClick={() => handleCategoryChange(category.id)}
              className={`px-4 py-2 rounded-full text-sm font-medium cursor-pointer transition-all duration-300 ${
                activeCategory === category.id ? "ring-2 ring-offset-2" : ""
              }`}
              style={
                {
                  backgroundColor:
                    activeCategory === category.id
                      ? "var(--accent)"
                      : "var(--surface-accent)",
                  color:
                    activeCategory === category.id ? "white" : "var(--accent)",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
                  "--ring-color": "var(--accent)",
                  "--ring-offset-color": "var(--background)",
                  transform: "translate3d(0, 0, 0)",
                  willChange: "transform",
                  backfaceVisibility: "hidden",
                } as React.CSSProperties
              }
              onMouseEnter={(e) => {
                e.currentTarget.style.transform =
                  "translate3d(0, -1px, 0) scale(1.05)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform =
                  "translate3d(0, 0, 0) scale(1)";
              }}
            >
              <span className="mr-2">{category.icon}</span>
              {category.name}
            </button>
          </FadeInWrapper>
        ))}
      </div>

      {/* Awards Content */}
      <div
        className="p-8 rounded-2xl border transition-all duration-300 relative overflow-hidden"
        style={{
          backgroundColor: "var(--card-bg)",
          borderColor: "var(--card-border)",
          minHeight: "450px", // Set a minimum height to prevent layout shifts
          boxShadow: "0 10px 30px rgba(0,0,0,0.05)",
          transform: "translate3d(0, 0, 0)",
          willChange: "auto",
          backfaceVisibility: "hidden",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = "translate3d(0, -2px, 0)";
          e.currentTarget.style.boxShadow = "0 25px 50px rgba(0,0,0,0.15)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = "translate3d(0, 0, 0)";
          e.currentTarget.style.boxShadow = "0 10px 30px rgba(0,0,0,0.05)";
        }}
      >
        {categories.map((category) => (
          <div
            key={category.id}
            className={`transition-all duration-700 absolute top-0 left-0 w-full h-full p-8 ${
              activeCategory === category.id
                ? "opacity-100 pointer-events-auto"
                : "opacity-0 pointer-events-none"
            }`}
            style={{
              transform:
                activeCategory === category.id
                  ? "translate3d(0, 0, 0)"
                  : animationDirection === "right"
                  ? "translate3d(50px, 0, 0)"
                  : "translate3d(-50px, 0, 0)",
              willChange: activeCategory === category.id ? "auto" : "transform",
              backfaceVisibility: "hidden",
            }}
          >
            <FadeInWrapper direction="up" delay={150}>
              <h3
                className="text-2xl font-semibold mb-6 flex items-center"
                style={{ color: "var(--text-primary)" }}
              >
                <span className="mr-3 text-2xl">{category.icon}</span>
                {category.name}
              </h3>

              <div className="space-y-4 overflow-y-auto max-h-[300px] pr-2 pb-12 styled-scrollbar">
                {awards[category.id] &&
                  awards[category.id].map((award, index) => (
                    <div
                      key={index}
                      className="flex items-start p-4 rounded-lg hover:bg-opacity-50 transition-all duration-300 hover:shadow-md transform hover:scale-[1.01]"
                      style={{
                        backgroundColor: "var(--surface)",
                        borderLeft: "3px solid var(--accent)",
                      }}
                    >
                      <div className="text-3xl mr-4 flex-shrink-0 transform transition-all duration-300 hover:scale-110 hover:rotate-12">
                        {award.icon}
                      </div>
                      <div>
                        <h4
                          className="text-lg font-semibold"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {award.achievement} - {award.title}
                        </h4>
                        <p
                          className="text-sm mt-1"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {award.organizer}
                        </p>
                        <div className="mt-2">
                          <span
                            className="text-xs px-2 py-1 rounded-full transition-all duration-300 hover:scale-105"
                            style={{
                              backgroundColor: "var(--surface-accent)",
                              color: "var(--accent)",
                              boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
                            }}
                          >
                            {award.year}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </FadeInWrapper>
          </div>
        ))}

        {/* Navigation Controls */}
        <div className="flex justify-between items-center mt-8 absolute bottom-4 left-0 right-0 px-8">
          <div className="flex space-x-3">
            {categories.map((category) => (
              <button
                key={category.id}
                onClick={() => handleCategoryChange(category.id)}
                className="flex flex-col items-center transition-all duration-300 relative"
                aria-label={`View ${category.name}`}
              >
                <span
                  className={`w-2.5 h-2.5 rounded-full transition-all duration-500 ${
                    activeCategory === category.id ? "bg-accent w-8" : ""
                  }`}
                  style={{
                    backgroundColor:
                      activeCategory === category.id
                        ? "var(--accent)"
                        : "var(--card-border)",
                    transform:
                      activeCategory === category.id ? "scale(1)" : "scale(1)",
                    height: "4px",
                    opacity: activeCategory === category.id ? 1 : 0.5,
                    borderRadius: "2px",
                  }}
                />
              </button>
            ))}
          </div>

          <div className="flex space-x-3">
            <button
              onClick={handlePrev}
              className="w-10 h-10 rounded-full flex items-center justify-center transition-all duration-200 hover:scale-110 active:scale-95"
              style={{
                backgroundColor: "var(--surface)",
                color: "var(--accent)",
                boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
              }}
              aria-label="Previous category"
            >
              ←
            </button>
            <button
              onClick={handleNext}
              className="w-10 h-10 rounded-full flex items-center justify-center transition-all duration-200 hover:scale-110 active:scale-95"
              style={{
                backgroundColor: "var(--accent)",
                color: "white",
                boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
              }}
              aria-label="Next category"
            >
              →
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

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
          {" "}
          {/* Hero Section */}
          <FadeInWrapper direction="up">
            <div className="text-center mb-20">
              <div className="relative w-40 h-40 mx-auto mb-8">
                <div
                  className="w-full h-full rounded-full border-4 flex items-center justify-center overflow-hidden"
                  style={{
                    backgroundColor: "var(--card-bg)",
                    borderColor: "var(--accent)",
                    boxShadow: "0 20px 40px rgba(0,0,0,0.1)",
                  }}
                >
                  <Image
                    src="/about-profile.webp"
                    alt="Profile"
                    width={160}
                    height={160}
                    className="w-full h-full object-cover"
                    priority
                  />
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
                machine learning, distributed systems, and full-stack
                development. Dedicated to building intelligent systems that
                solve real-world problems and advance the field of artificial
                intelligence.
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
          </FadeInWrapper>{" "}
          {/* Experience Cards */}
          <FadeInWrapper direction="up" delay={150}>
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
                  <FadeInWrapper
                    key={index}
                    direction="up"
                    delay={150 + index * 100}
                  >
                    <div
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
                  </FadeInWrapper>
                ))}
              </div>
            </div>
          </FadeInWrapper>{" "}
          {/* Work Experience Timeline */}
          <FadeInWrapper direction="up" delay={200}>
            <div className="mb-20">
              <h2
                className="text-4xl font-bold text-center mb-4"
                style={{ color: "var(--text-primary)" }}
              >
                Professional Experience
              </h2>
              <p
                className="text-lg text-center mb-12 max-w-3xl mx-auto"
                style={{ color: "var(--text-secondary)" }}
              >
                Timeline of my professional journey in AI research and software
                development
              </p>
              <div className="space-y-8">
                {workExperience.map((job, index) => (
                  <FadeInWrapper
                    key={index}
                    direction="up"
                    delay={200 + index * 100}
                  >
                    <div className="relative">
                      {/* Timeline connector */}
                      {index < workExperience.length - 1 && (
                        <div
                          className="absolute left-8 top-20 w-0.5 h-16 z-0"
                          style={{ backgroundColor: "var(--card-border)" }}
                        />
                      )}

                      <div
                        className="relative p-8 rounded-2xl border transform hover:shadow-lg hover:scale-[1.01] transition-all duration-300"
                        style={{
                          backgroundColor: "var(--card-bg)",
                          borderColor: "var(--card-border)",
                        }}
                      >
                        <div className="grid md:grid-cols-4 gap-6">
                          {/* Timeline icon and period */}
                          <div className="md:col-span-1">
                            <FadeInWrapper
                              direction="up"
                              delay={250 + index * 100}
                            >
                              <div className="flex flex-col items-start">
                                <div
                                  className="w-16 h-16 rounded-full flex items-center justify-center text-2xl mb-4 border-4 transition-all duration-300 hover:scale-110"
                                  style={{
                                    backgroundColor: "var(--card-bg)",
                                    borderColor: "var(--accent)",
                                    color: "var(--accent)",
                                  }}
                                >
                                  {job.icon}
                                </div>
                                <div
                                  className="text-sm font-bold mb-1"
                                  style={{ color: "var(--accent)" }}
                                >
                                  {job.period}
                                </div>
                                <div
                                  className="text-xs"
                                  style={{ color: "var(--text-secondary)" }}
                                >
                                  {job.type}
                                </div>
                              </div>
                            </FadeInWrapper>
                          </div>

                          {/* Job details */}
                          <div className="md:col-span-3">
                            <FadeInWrapper
                              direction="up"
                              delay={300 + index * 100}
                            >
                              <div className="mb-4">
                                <h3
                                  className="text-2xl font-bold mb-2"
                                  style={{ color: "var(--text-primary)" }}
                                >
                                  {job.title}
                                </h3>
                                <p
                                  className="text-lg font-semibold mb-1"
                                  style={{ color: "var(--accent)" }}
                                >
                                  {job.company}
                                </p>
                                <p
                                  className="text-sm mb-4"
                                  style={{ color: "var(--text-secondary)" }}
                                >
                                  📍 {job.location}
                                </p>
                                <p
                                  className="leading-relaxed mb-4"
                                  style={{ color: "var(--text-secondary)" }}
                                >
                                  {job.description}
                                </p>
                              </div>
                            </FadeInWrapper>

                            {/* Responsibilities */}
                            <FadeInWrapper
                              direction="up"
                              delay={350 + index * 100}
                            >
                              <div className="mb-4">
                                <h4
                                  className="font-semibold mb-3"
                                  style={{ color: "var(--text-primary)" }}
                                >
                                  Key Responsibilities:
                                </h4>
                                <ul className="space-y-2">
                                  {job.responsibilities.map((resp, idx) => (
                                    <li
                                      key={idx}
                                      className="flex items-start text-sm transition-all duration-300 hover:translate-x-1"
                                      style={{ color: "var(--text-secondary)" }}
                                    >
                                      <span
                                        className="mr-3 mt-1 text-xs"
                                        style={{ color: "var(--accent)" }}
                                      >
                                        •
                                      </span>
                                      {resp}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </FadeInWrapper>

                            {/* Technologies */}
                            <FadeInWrapper
                              direction="up"
                              delay={400 + index * 100}
                            >
                              <div>
                                <h4
                                  className="font-semibold mb-3"
                                  style={{ color: "var(--text-primary)" }}
                                >
                                  Technologies & Tools:
                                </h4>
                                <div className="flex flex-wrap gap-2">
                                  {job.technologies.map((tech, idx) => (
                                    <span
                                      key={idx}
                                      className="px-3 py-1 rounded-lg text-xs font-medium transition-all duration-200 hover:scale-105"
                                      style={{
                                        backgroundColor: "var(--surface)",
                                        color: "var(--text-secondary)",
                                      }}
                                      onMouseEnter={(e) => {
                                        e.currentTarget.style.backgroundColor =
                                          "var(--surface-accent)";
                                        e.currentTarget.style.color =
                                          "var(--accent)";
                                      }}
                                      onMouseLeave={(e) => {
                                        e.currentTarget.style.backgroundColor =
                                          "var(--surface)";
                                        e.currentTarget.style.color =
                                          "var(--text-secondary)";
                                      }}
                                    >
                                      {tech}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            </FadeInWrapper>
                          </div>
                        </div>
                      </div>
                    </div>
                  </FadeInWrapper>
                ))}
              </div>
            </div>
          </FadeInWrapper>
          <FadeInWrapper direction="up" delay={300}>
            <div className="mb-20">
              <h2
                className="text-4xl font-bold text-center mb-4"
                style={{ color: "var(--text-primary)" }}
              >
                Honors & Awards
              </h2>
              <p
                className="text-lg text-center mb-12 max-w-3xl mx-auto"
                style={{ color: "var(--text-secondary)" }}
              >
                Recognition for innovation, entrepreneurship, and technical
                excellence
              </p>

              <AwardsCarousel awards={awards} />
            </div>
          </FadeInWrapper>
          {/* Education Section */}
          <FadeInWrapper direction="up" delay={300}>
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
                Strong academic foundation with focus on advanced computer
                science concepts
              </p>
              <div className="space-y-6">
                {education.map((edu, index) => (
                  <FadeInWrapper
                    key={index}
                    direction="up"
                    delay={300 + index * 100}
                  >
                    <div
                      className="p-8 rounded-2xl border hover:shadow-lg transition-all duration-300"
                      style={{
                        backgroundColor: "var(--card-bg)",
                        borderColor: "var(--card-border)",
                      }}
                    >
                      <div className="grid md:grid-cols-3 gap-6">
                        <div className="md:col-span-2">
                          <FadeInWrapper
                            direction="up"
                            delay={350 + index * 100}
                          >
                            <h3
                              className="text-2xl font-bold mb-2"
                              style={{ color: "var(--text-primary)" }}
                            >
                              {edu.degree}
                            </h3>
                          </FadeInWrapper>
                          <FadeInWrapper
                            direction="up"
                            delay={400 + index * 100}
                          >
                            <p
                              className="text-lg mb-2"
                              style={{ color: "var(--accent)" }}
                            >
                              {edu.institution}
                            </p>
                          </FadeInWrapper>
                          <FadeInWrapper
                            direction="up"
                            delay={450 + index * 100}
                          >
                            <p
                              className="text-sm mb-4"
                              style={{ color: "var(--text-secondary)" }}
                            >
                              {edu.year} • Specialization: {edu.specialization}{" "}
                            </p>
                          </FadeInWrapper>
                          <FadeInWrapper
                            direction="up"
                            delay={500 + index * 100}
                          >
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
                                    className="px-3 py-1 rounded-lg text-sm transition-all duration-200 hover:scale-105"
                                    style={{
                                      backgroundColor: "var(--surface)",
                                      color: "var(--text-secondary)",
                                    }}
                                    onMouseEnter={(e) => {
                                      e.currentTarget.style.backgroundColor =
                                        "var(--surface-accent)";
                                      e.currentTarget.style.color =
                                        "var(--accent)";
                                    }}
                                    onMouseLeave={(e) => {
                                      e.currentTarget.style.backgroundColor =
                                        "var(--surface)";
                                      e.currentTarget.style.color =
                                        "var(--text-secondary)";
                                    }}
                                  >
                                    {course}
                                  </span>
                                ))}
                              </div>
                            </div>
                          </FadeInWrapper>
                        </div>
                        <div className="flex items-center justify-center">
                          <FadeInWrapper
                            direction="up"
                            delay={550 + index * 100}
                          >
                            <div
                              className="w-24 h-24 rounded-full flex items-center justify-center text-3xl transform transition-all duration-300 hover:scale-110"
                              style={{
                                backgroundColor: "var(--surface-accent)",
                                color: "var(--accent)",
                              }}
                            >
                              🎓
                            </div>
                          </FadeInWrapper>
                        </div>
                      </div>
                    </div>
                  </FadeInWrapper>
                ))}
              </div>
            </div>
          </FadeInWrapper>
          {/* Research Interests */}
          <FadeInWrapper direction="up" delay={350}>
            <div className="mb-20">
              <h2
                className="text-4xl font-bold text-center mb-12"
                style={{ color: "var(--text-primary)" }}
              >
                Research Interests & Focus Areas
              </h2>
              <div
                className="max-w-5xl mx-auto p-8 rounded-2xl border hover:shadow-lg transition-all duration-300"
                style={{
                  backgroundColor: "var(--card-bg)",
                  borderColor: "var(--card-border)",
                }}
              >
                <div className="grid md:grid-cols-2 gap-8">
                  <div>
                    <FadeInWrapper direction="up" delay={400}>
                      <h3
                        className="text-2xl font-semibold mb-6 flex items-center"
                        style={{ color: "var(--text-primary)" }}
                      >
                        <span className="mr-3">🔬</span>
                        Current Research Focus
                      </h3>
                    </FadeInWrapper>
                    <ul
                      className="space-y-4"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      <FadeInWrapper direction="up" delay={450}>
                        <li className="flex items-start">
                          <span
                            className="mr-3 mt-1"
                            style={{ color: "var(--accent)" }}
                          >
                            •
                          </span>
                          <div>
                            <strong>Large Language Models & NLP:</strong>{" "}
                            Advancing transformer architectures, few-shot
                            learning, and multilingual understanding
                          </div>
                        </li>
                      </FadeInWrapper>
                      <FadeInWrapper direction="up" delay={500}>
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
                      </FadeInWrapper>
                      <FadeInWrapper direction="up" delay={550}>
                        <li className="flex items-start">
                          <span
                            className="mr-3 mt-1"
                            style={{ color: "var(--accent)" }}
                          >
                            •
                          </span>
                          <div>
                            <strong>MLOps & AI Infrastructure:</strong> Building
                            scalable ML systems and automated deployment
                            pipelines
                          </div>
                        </li>
                      </FadeInWrapper>
                      <FadeInWrapper direction="up" delay={600}>
                        <li className="flex items-start">
                          <span
                            className="mr-3 mt-1"
                            style={{ color: "var(--accent)" }}
                          >
                            •
                          </span>
                          <div>
                            <strong>Distributed Systems:</strong> Designing
                            fault-tolerant, high-performance computing systems
                            for AI workloads
                          </div>
                        </li>
                      </FadeInWrapper>
                    </ul>
                  </div>
                  <div>
                    <FadeInWrapper direction="up" delay={400}>
                      <h3
                        className="text-2xl font-semibold mb-6 flex items-center"
                        style={{ color: "var(--text-primary)" }}
                      >
                        <span className="mr-3">🎯</span>
                        Research Impact & Goals
                      </h3>
                    </FadeInWrapper>
                    <FadeInWrapper direction="up" delay={450}>
                      <p
                        className="leading-relaxed mb-6"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        My research focuses on developing practical AI solutions
                        that bridge the gap between academic innovation and
                        real-world applications. I&apos;m particularly
                        interested in making advanced AI techniques more
                        accessible and efficient for deployment in production
                        environments.
                      </p>
                    </FadeInWrapper>
                    <FadeInWrapper direction="up" delay={500}>
                      <p
                        className="leading-relaxed"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        Current work involves optimizing transformer models for
                        resource-constrained environments, developing novel
                        approaches to multimodal learning, and creating robust
                        MLOps frameworks that enable seamless integration of AI
                        into existing software systems.
                      </p>
                    </FadeInWrapper>
                    <FadeInWrapper direction="up" delay={550}>
                      <div
                        className="mt-6 p-4 rounded-xl transition-all duration-300 hover:shadow-md"
                        style={{ backgroundColor: "var(--surface-accent)" }}
                      >
                        <p
                          className="text-sm font-medium"
                          style={{ color: "var(--accent)" }}
                        >
                          🌟 &quot;Building AI systems that are not just
                          intelligent, but also practical, ethical, and
                          accessible to everyone.&quot;
                        </p>
                      </div>
                    </FadeInWrapper>
                  </div>
                </div>
              </div>
            </div>
          </FadeInWrapper>
          {/* Contact CTA */}
          <FadeInWrapper direction="up" delay={400}>
            <div
              className="p-10 rounded-2xl border text-center hover:shadow-lg transition-all duration-300"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
              }}
            >
              <div className="max-w-3xl mx-auto">
                <FadeInWrapper direction="up" delay={450}>
                  <h2
                    className="text-3xl font-bold mb-6"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Let&apos;s Build Something Amazing Together
                  </h2>
                </FadeInWrapper>
                <FadeInWrapper direction="up" delay={500}>
                  <p
                    className="text-lg mb-8 leading-relaxed"
                    style={{ color: "var(--text-secondary)" }}
                  >
                    I&apos;m always excited to collaborate on innovative
                    projects, discuss research opportunities, or explore how AI
                    can solve complex challenges. Whether you&apos;re looking
                    for a technical partner, research collaborator, or want to
                    discuss the latest in AI and software engineering.
                  </p>
                </FadeInWrapper>
                <FadeInWrapper direction="up" delay={550}>
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
                        e.currentTarget.style.backgroundColor =
                          "var(--surface)";
                        e.currentTarget.style.borderColor = "var(--accent)";
                        e.currentTarget.style.color = "var(--accent)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor =
                          "var(--card-bg)";
                        e.currentTarget.style.borderColor = "var(--border)";
                        e.currentTarget.style.color = "var(--text-primary)";
                      }}
                    >
                      <span className="mr-2">👨‍💻</span>
                      GitHub Profile
                    </a>
                    <a
                      href="https://linkedin.com/in/hieptran1812"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center px-8 py-4 font-semibold rounded-xl transition-all duration-200 border shadow-lg hover:shadow-xl transform hover:scale-105"
                      style={{
                        backgroundColor: "var(--card-bg)",
                        borderColor: "var(--border)",
                        color: "var(--text-primary)",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor =
                          "var(--surface)";
                        e.currentTarget.style.borderColor = "var(--accent)";
                        e.currentTarget.style.color = "var(--accent)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor =
                          "var(--card-bg)";
                        e.currentTarget.style.borderColor = "var(--border)";
                        e.currentTarget.style.color = "var(--text-primary)";
                      }}
                    >
                      <span className="mr-2">📊</span>
                      LinkedIn
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
                        e.currentTarget.style.backgroundColor =
                          "var(--surface)";
                        e.currentTarget.style.borderColor = "var(--accent)";
                        e.currentTarget.style.color = "var(--accent)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor =
                          "var(--card-bg)";
                        e.currentTarget.style.borderColor = "var(--border)";
                        e.currentTarget.style.color = "var(--text-primary)";
                      }}
                    >
                      <span className="mr-2">🚀</span>
                      View Projects
                    </a>
                  </div>
                </FadeInWrapper>
              </div>
            </div>
          </FadeInWrapper>
        </div>
      </main>
    </div>
  );
}
