"use client";

import ScrollReveal from "../../components/ScrollReveal";
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
    description:
      "Building agent frameworks, multimodal AI companions, RAG pipelines and the training and serving stack behind them, from fine-tuning to multi-node GPU training.",
    responsibilities: [
      "Agent Harness: Built agent framework with long-term memory, tool calling, dynamic prompts and multi-turn interactions; used the Jev System One model for fast routing and guardrailing; built agent evaluation and benchmarks",
      "Multimodal AI Companion: Integrated vision-language models for real-time camera interaction and conversations about images",
      "RAG: Architected end-to-end pipeline (ingestion, chunking, embedding, vector search, generation) for financial reports and market data",
      "Fine-tuning & Serving: SFT, DPO and multimodal alignment for LLMs and vision models; serving with vLLM and SGLang",
      "Distributed Training: Multi-GPU, multi-node model training on NVIDIA H100 clusters",
      "Research: Activation steering to control LLM behavior without prompting or fine-tuning",
      "Other: Emotion detection from text and images; image generation with diffusion models and LoRA",
    ],
    technologies: [
      "Python",
      "PyTorch",
      "LLM",
      "AI Agent",
      "RAG",
      "vLLM",
      "SGLang",
      "H100",
    ],
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
      "Document Understanding: Built and maintained information extraction module for Vietnamese documents, often reaching 99% accuracy",
      "Face Recognition: Developed and optimized face attendance system with face detection and recognition, mask checking and person tracking",
      "Image & Video Classification: Built content classification system for the Lotus social network",
      "Generative AI: Fine-tuned Stable Diffusion to automatically generate banners",
      "MLOps: Deployed models with Kubernetes, Docker, Triton Inference Server and FastAPI; automated training and deployment pipelines with Airflow",
    ],
    technologies: [
      "Python",
      "PyTorch",
      "Computer Vision",
      "NLP",
      "Stable Diffusion",
      "Kubernetes",
      "Triton",
      "Airflow",
    ],
    icon: "🔬",
  },
  {
    title: "Viblo Algorithm Content Creator",
    company: "Sun* Vietnam",
    location: "Hanoi, Vietnam",
    period: "June 2021 - June 2024",
    type: "Content Creator",
    description:
      "Contributing articles and algorithmic challenges on the Viblo ecosystem, focusing on AI/ML education and research.",
    responsibilities: [
      "Contributing articles, algorithmic challenges on the Viblo ecosystem",
      "Research and publish articles on Computer Vision, Large Language Model, Natural Language Processing, Recommendation system and MLOps",
    ],
    technologies: [
      "Computer Vision",
      "LLM",
      "Natural Language Processing",
      "Recommendation Systems",
      "MLOps",
      "Technical Writing",
    ],
    icon: "✍️",
  },
  {
    title: "AI Researcher",
    company: "PTIT Lab X Naver AI Center",
    location: "Hanoi, Vietnam",
    period: "Sep 2020 - Dec 2021",
    type: "Research",
    description:
      "Conducting deep learning research on audio classification and time-series forecasting at the PTIT Lab in collaboration with Naver AI Center.",
    responsibilities: [
      "Developed an EfficientNet-based deep learning model for cough sound classification",
      "Developed a CNN-LSTM model to forecast air quality sensor values",
    ],
    technologies: [
      "Python",
      "PyTorch",
      "EfficientNet",
      "CNN-LSTM",
      "Audio Classification",
      "Time Series",
    ],
    icon: "🔬",
  },
];

// Personal Projects
const personalProjects = [
  {
    title: "Founder & Solo Developer",
    company: "BandPeak",
    url: "https://bandpeak.com/",
    location: "Hanoi, Vietnam",
    period: "Sep 2026 - Present",
    type: "Personal Project",
    description:
      "Building a comprehensive AI-powered English learning platform covering IELTS mock exams, vocabulary, grammar, pronunciation and listening.",
    responsibilities: [
      "Building a comprehensive AI-powered English learning platform covering IELTS mock exams, vocabulary, grammar, pronunciation and listening",
      "Developed AI grading for Writing and Speaking, AI writing coach, spoken roleplay and chat tutor",
    ],
    technologies: ["LLM", "Speech AI", "AI Grading", "EdTech"],
    icon: "🚀",
  },
  {
    title: "AI Engineer",
    company: "A.L.I.S VIETNAM",
    url: "",
    location: "Hanoi, Vietnam",
    period: "April 2022 - 2023",
    type: "Personal Project",
    description:
      "Developing a system to convert speech and text into sign language to assist Deaf people in accessing information.",
    responsibilities: [
      "Developing a system to convert speech and text into sign language to assist Deaf people in accessing information",
      "Studying of speech-to-text, text-to-speech models, GANs, recognition and conversion models to human actions",
      "Implement pipeline for machine learning model using FastAPI, Docker",
    ],
    technologies: [
      "Python",
      "GANs",
      "Speech Recognition",
      "Text-to-Speech",
      "FastAPI",
      "Docker",
    ],
    icon: "🤝",
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
      achievement: "FPT Track Challenge Winner",
      icon: "🏅",
      year: "2022",
    },
    {
      title: "Build On Vietnam 2022",
      organizer: "Amazon Web Services",
      achievement: "Best Innovation Award",
      icon: "🏆",
      year: "2022",
    },
    {
      title: "Creative Idea Challenge 2022",
      organizer:
        "School of Information and Communication Technology, Hanoi University of Science and Technology",
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
      title: "P-Startup Student Startup Idea Contest 2022",
      organizer: "Posts and Telecommunications Institute of Technology",
      achievement: "First Prize",
      icon: "🥇",
      year: "2022",
    },
    {
      title: "Social Business Creation 2022",
      organizer: "HEC Montréal",
      achievement: "Finalist",
      icon: "🌟",
      year: "2022",
    },
  ],
  technical: [
    {
      title: "VAIPE Medicine Pill Image Recognition Challenge",
      organizer: "VinUni-Illinois Smart Health Center (VISHC), VinUniversity",
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
    degree: "Master of Computer Science",
    institution: "Posts and Telecommunications Institute of Technology",
    year: "2024-2026",
    specialization: "Machine Learning & AI",
    highlights: [
      "Researching AI safety and alignment, including controlling LLM behavior through activation steering",
      "Studying interpretability and robustness of open-weight foundation models, including the Qwen, DeepSeek, Kimi and Gemma families",
    ],
    coursework: [
      "Machine Learning",
      "Deep Learning",
      "Large Language Models",
      "Natural Language Processing",
      "Computer Vision",
      "Reinforcement Learning",
      "Advanced Algorithms",
      "Distributed Systems",
    ],
  },
  {
    degree: "Bachelor of Computer Science",
    institution: "Posts and Telecommunications Institute of Technology",
    year: "2018-2023",
    specialization: "Software Engineering",
    highlights: [
      "CPA: 3.25/4",
      "Merit-based Scholarship for Excellent Students in the semesters of freshman, sophomore and junior",
    ],
    coursework: [
      "Data Structures & Algorithms",
      "Software Engineering",
      "Database Systems",
      "Operating Systems",
      "Computer Networks",
      "Web Development",
      "Artificial Intelligence",
    ],
  },
];

// Technical Skills
const technicalSkills = [
  { category: "Programming Languages", items: ["Python", "Java"] },
  {
    category: "AI Frameworks/Libraries",
    items: [
      "PyTorch",
      "Transformers",
      "LangChain",
      "Haystack",
      "LlamaIndex",
      "Scikit-learn",
      "OpenCV",
      "TRL",
      "Unsloth",
    ],
  },
  {
    category: "Backend and Database",
    items: ["FastAPI", "Redis", "SQL", "Qdrant"],
  },
  { category: "Cloud and Tools", items: ["Git", "GCP"] },
  {
    category: "MLOps",
    items: [
      "Docker",
      "Kubernetes",
      "Triton Inference Server",
      "vLLM",
      "SGLang",
    ],
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
          <ScrollReveal
            key={category.id}
            animation="zoom-in"
            delay={350 + index * 80}
            duration={600}
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
          </ScrollReveal>
        ))}
      </div>

      {/* Awards Content */}
      <div
        className="p-4 sm:p-8 rounded-2xl border transition-all duration-300 relative overflow-hidden min-h-[480px] sm:min-h-[400px]"
        style={{
          backgroundColor: "var(--card-bg)",
          borderColor: "var(--card-border)",
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
            className={`transition-all duration-700 absolute top-0 left-0 w-full h-full p-4 sm:p-8 ${
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
            <ScrollReveal animation="fade-up" delay={150} duration={600}>
              <h3
                className="text-xl sm:text-2xl font-semibold mb-6 flex items-center"
                style={{ color: "var(--text-primary)" }}
              >
                <span className="mr-3 text-xl sm:text-2xl">
                  {category.icon}
                </span>
                {category.name}
              </h3>

              <div className="space-y-3 sm:space-y-4 overflow-y-auto max-h-[320px] sm:max-h-[300px] pr-2 pb-12 styled-scrollbar">
                {awards[category.id] &&
                  awards[category.id].map((award, index) => (
                    <div
                      key={index}
                      className="flex items-start gap-4 p-4 rounded-xl border transition-all duration-300 hover:shadow-md"
                      style={{
                        backgroundColor: "var(--surface)",
                        borderColor: "var(--card-border)",
                      }}
                    >
                      <div
                        className="w-11 h-11 rounded-lg flex items-center justify-center text-xl flex-shrink-0"
                        style={{ backgroundColor: "var(--surface-accent)" }}
                      >
                        {award.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-3">
                          <h4
                            className="text-base font-semibold leading-snug"
                            style={{ color: "var(--text-primary)" }}
                          >
                            {award.achievement} - {award.title}
                          </h4>
                          <span
                            className="text-xs font-medium px-2 py-0.5 rounded-md flex-shrink-0 mt-0.5"
                            style={{
                              backgroundColor: "var(--surface-accent)",
                              color: "var(--accent)",
                            }}
                          >
                            {award.year}
                          </span>
                        </div>
                        <p
                          className="text-sm mt-1"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          {award.organizer}
                        </p>
                      </div>
                    </div>
                  ))}
              </div>
            </ScrollReveal>
          </div>
        ))}

        {/* Navigation Controls */}
        <div className="flex justify-between items-center mt-8 absolute bottom-4 left-0 right-0 px-4 sm:px-8">
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
      className="flex flex-col min-h-screen overflow-x-clip transition-colors duration-300"
      style={{
        backgroundColor: "transparent",
        color: "var(--text-primary)",
      }}
    >
      <main className="flex-1">
        <div className="max-w-7xl mx-auto px-6 py-16">
          {" "}
          {/* Hero Section */}
          <div className="text-center mb-20">
            <ScrollReveal animation="zoom-in" duration={800}>
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
                    borderColor: "var(--card-bg)",
                    color: "white",
                  }}
                >
                  🚀
                </div>
              </div>
            </ScrollReveal>

            <ScrollReveal animation="blur-in" delay={200} duration={900}>
              <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold mb-6 bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 bg-clip-text text-transparent">
                Hiep Tran
              </h1>
            </ScrollReveal>

            <ScrollReveal animation="fade-up" delay={400} duration={700}>
              <p
                className="text-2xl md:text-3xl mb-6 font-semibold"
                style={{ color: "var(--accent)" }}
              >
                AI Research Engineer & Software Developer
              </p>
            </ScrollReveal>

            <ScrollReveal animation="fade-up" delay={550} duration={700}>
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
            </ScrollReveal>

            <div className="flex flex-wrap justify-center gap-4">
              <ScrollReveal animation="slide-up" delay={700} duration={600}>
                <span
                  className="px-4 py-2 rounded-full text-sm font-medium inline-block"
                  style={{
                    backgroundColor: "var(--surface-accent)",
                    color: "var(--accent)",
                  }}
                >
                  🎓 MS Computer Science
                </span>
              </ScrollReveal>
              <ScrollReveal animation="slide-up" delay={800} duration={600}>
                <span
                  className="px-4 py-2 rounded-full text-sm font-medium inline-block"
                  style={{
                    backgroundColor: "var(--surface-accent)",
                    color: "var(--accent)",
                  }}
                >
                  🔬 AI Engineer
                </span>
              </ScrollReveal>
              <ScrollReveal animation="slide-up" delay={900} duration={600}>
                <span
                  className="px-4 py-2 rounded-full text-sm font-medium inline-block"
                  style={{
                    backgroundColor: "var(--surface-accent)",
                    color: "var(--accent)",
                  }}
                >
                  💻 Software Developer
                </span>
              </ScrollReveal>
            </div>
          </div>{" "}
          {/* Experience Cards */}
          <div className="mb-20">
            <ScrollReveal animation="fade-up" duration={700}>
              <h2
                className="text-3xl sm:text-4xl font-bold text-center mb-4"
                style={{ color: "var(--text-primary)" }}
              >
                What I Do
              </h2>
            </ScrollReveal>
            <ScrollReveal animation="fade-up" delay={100} duration={700}>
              <p
                className="text-lg text-center mb-12 max-w-3xl mx-auto"
                style={{ color: "var(--text-secondary)" }}
              >
                Combining theoretical knowledge with practical experience to
                create innovative solutions
              </p>
            </ScrollReveal>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {experiences.map((exp, index) => (
                <ScrollReveal
                  key={index}
                  animation="flip-up"
                  delay={200 + index * 150}
                  duration={800}
                >
                  <div
                    className="group p-5 sm:p-8 rounded-2xl border transition-all duration-300 hover:scale-105 hover:shadow-xl h-full"
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
                </ScrollReveal>
              ))}
            </div>
          </div>{" "}
          {/* Work Experience Timeline + Personal Projects */}
          {[
            {
              heading: "Professional Experience",
              subtitle:
                "Timeline of my professional journey in AI research and software development",
              items: workExperience,
            },
            {
              heading: "Personal Projects",
              subtitle: "Products and side projects I build on my own",
              items: personalProjects,
            },
          ].map((section) => (
            <div className="mb-20" key={section.heading}>
              <ScrollReveal animation="fade-up" duration={700}>
                <h2
                  className="text-3xl sm:text-4xl font-bold text-center mb-4"
                  style={{ color: "var(--text-primary)" }}
                >
                  {section.heading}
                </h2>
              </ScrollReveal>
              <ScrollReveal animation="fade-up" delay={100} duration={700}>
                <p
                  className="text-lg text-center mb-12 max-w-3xl mx-auto"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {section.subtitle}
                </p>
              </ScrollReveal>
              <div className="space-y-8">
                {section.items.map(
                  (
                    job: (typeof section.items)[number] & { url?: string },
                    index,
                  ) => (
                    <ScrollReveal
                      key={index}
                      animation="slide-left"
                      delay={150 + index * 100}
                      duration={800}
                    >
                      <div className="relative">
                        {/* Timeline connector */}
                        {index < section.items.length - 1 && (
                          <div
                            className="absolute left-8 top-20 w-0.5 h-16 z-0"
                            style={{ backgroundColor: "var(--card-border)" }}
                          />
                        )}

                        <div
                          className="relative p-5 sm:p-8 rounded-2xl border transform hover:shadow-lg hover:scale-[1.01] transition-all duration-300"
                          style={{
                            backgroundColor: "var(--card-bg)",
                            borderColor: "var(--card-border)",
                          }}
                        >
                          <div className="grid md:grid-cols-4 gap-6">
                            {/* Timeline icon and period */}
                            <div className="md:col-span-1">
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
                            </div>

                            {/* Job details */}
                            <div className="md:col-span-3">
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
                                  {job.url ? (
                                    <a
                                      href={job.url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="hover:underline"
                                    >
                                      {job.company}
                                    </a>
                                  ) : (
                                    job.company
                                  )}
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

                              {/* Responsibilities */}
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

                              {/* Technologies */}
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
                            </div>
                          </div>
                        </div>
                      </div>
                    </ScrollReveal>
                  ),
                )}
              </div>
            </div>
          ))}
          <div className="mb-20">
            <ScrollReveal animation="fade-up" duration={700}>
              <h2
                className="text-3xl sm:text-4xl font-bold text-center mb-4"
                style={{ color: "var(--text-primary)" }}
              >
                Honors & Awards
              </h2>
            </ScrollReveal>
            <ScrollReveal animation="fade-up" delay={100} duration={700}>
              <p
                className="text-lg text-center mb-12 max-w-3xl mx-auto"
                style={{ color: "var(--text-secondary)" }}
              >
                Recognition in hackathons, entrepreneurship, and technical
                competitions
              </p>
            </ScrollReveal>

            <ScrollReveal animation="scale-up" delay={200} duration={800}>
              <AwardsCarousel awards={awards} />
            </ScrollReveal>
          </div>
          {/* Education Section */}
          <div className="mb-20">
            <ScrollReveal animation="fade-up" duration={700}>
              <h2
                className="text-3xl sm:text-4xl font-bold text-center mb-4"
                style={{ color: "var(--text-primary)" }}
              >
                Education & Academic Background
              </h2>
            </ScrollReveal>
            <ScrollReveal animation="fade-up" delay={100} duration={700}>
              <p
                className="text-lg text-center mb-12 max-w-3xl mx-auto"
                style={{ color: "var(--text-secondary)" }}
              >
                Strong academic foundation with focus on advanced computer
                science concepts
              </p>
            </ScrollReveal>
            <div className="space-y-6">
              {education.map((edu, index) => (
                <ScrollReveal
                  key={index}
                  animation="slide-right"
                  delay={200 + index * 150}
                  duration={800}
                >
                  <div
                    className="p-5 sm:p-8 rounded-2xl border hover:shadow-lg transition-all duration-300"
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
                          {edu.year} • Specialization: {edu.specialization}{" "}
                        </p>
                        <ul className="space-y-1 mb-4">
                          {edu.highlights.map((item, idx) => (
                            <li
                              key={idx}
                              className="flex items-start text-sm"
                              style={{ color: "var(--text-secondary)" }}
                            >
                              <span
                                className="mr-3 mt-1 text-xs"
                                style={{ color: "var(--accent)" }}
                              >
                                •
                              </span>
                              {item}
                            </li>
                          ))}
                        </ul>
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
                                  e.currentTarget.style.color = "var(--accent)";
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
                      </div>
                      <div className="flex items-center justify-center">
                        <div
                          className="w-24 h-24 rounded-full flex items-center justify-center text-3xl transform transition-all duration-300 hover:scale-110"
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
                </ScrollReveal>
              ))}
            </div>
          </div>
          {/* Technical Skills */}
          <div className="mb-20">
            <ScrollReveal animation="fade-up" duration={700}>
              <h2
                className="text-3xl sm:text-4xl font-bold text-center mb-12"
                style={{ color: "var(--text-primary)" }}
              >
                Technical Skills
              </h2>
            </ScrollReveal>
            <ScrollReveal animation="fade-up" delay={150} duration={800}>
              <div
                className="max-w-5xl mx-auto p-5 sm:p-8 rounded-2xl border space-y-5"
                style={{
                  backgroundColor: "var(--card-bg)",
                  borderColor: "var(--card-border)",
                }}
              >
                {technicalSkills.map((group) => (
                  <div
                    key={group.category}
                    className="grid md:grid-cols-4 gap-3 items-start"
                  >
                    <h3
                      className="font-semibold"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {group.category}
                    </h3>
                    <div className="md:col-span-3 flex flex-wrap gap-2">
                      {group.items.map((item) => (
                        <span
                          key={item}
                          className="px-3 py-1 rounded-lg text-sm"
                          style={{
                            backgroundColor: "var(--surface)",
                            color: "var(--text-secondary)",
                          }}
                        >
                          {item}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </ScrollReveal>
          </div>
          {/* Research Interests */}
          <div className="mb-20">
            <ScrollReveal animation="fade-up" duration={700}>
              <h2
                className="text-3xl sm:text-4xl font-bold text-center mb-12"
                style={{ color: "var(--text-primary)" }}
              >
                Research Interests & Focus Areas
              </h2>
            </ScrollReveal>
            <ScrollReveal animation="blur-in" delay={150} duration={900}>
              <div
                className="max-w-5xl mx-auto p-5 sm:p-8 rounded-2xl border hover:shadow-lg transition-all duration-300"
                style={{
                  backgroundColor: "var(--card-bg)",
                  borderColor: "var(--card-border)",
                }}
              >
                <div className="grid md:grid-cols-2 gap-8">
                  <div>
                    <ScrollReveal
                      animation="fade-left"
                      delay={250}
                      duration={700}
                    >
                      <h3
                        className="text-2xl font-semibold mb-6 flex items-center"
                        style={{ color: "var(--text-primary)" }}
                      >
                        <span className="mr-3">🔬</span>
                        Current Research Focus
                      </h3>
                    </ScrollReveal>
                    <ul
                      className="space-y-4"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      <ScrollReveal
                        animation="fade-up"
                        delay={350}
                        duration={600}
                      >
                        <li className="flex items-start">
                          <span
                            className="mr-3 mt-1"
                            style={{ color: "var(--accent)" }}
                          >
                            •
                          </span>
                          <div>
                            <strong>AI Safety & Interpretability:</strong>{" "}
                            Controlling LLM behavior through activation
                            steering; interpretability and robustness of
                            open-weight models (Qwen, DeepSeek, Kimi, Gemma)
                          </div>
                        </li>
                      </ScrollReveal>
                      <ScrollReveal
                        animation="fade-up"
                        delay={450}
                        duration={600}
                      >
                        <li className="flex items-start">
                          <span
                            className="mr-3 mt-1"
                            style={{ color: "var(--accent)" }}
                          >
                            •
                          </span>
                          <div>
                            <strong>LLM Agents & Multimodal AI:</strong> Agent
                            frameworks with memory and tool calling,
                            vision-language models, and RAG systems
                          </div>
                        </li>
                      </ScrollReveal>
                      <ScrollReveal
                        animation="fade-up"
                        delay={550}
                        duration={600}
                      >
                        <li className="flex items-start">
                          <span
                            className="mr-3 mt-1"
                            style={{ color: "var(--accent)" }}
                          >
                            •
                          </span>
                          <div>
                            <strong>Fine-tuning & Serving:</strong> SFT, DPO and
                            multimodal alignment; serving with vLLM, SGLang and
                            Triton Inference Server
                          </div>
                        </li>
                      </ScrollReveal>
                      <ScrollReveal
                        animation="fade-up"
                        delay={650}
                        duration={600}
                      >
                        <li className="flex items-start">
                          <span
                            className="mr-3 mt-1"
                            style={{ color: "var(--accent)" }}
                          >
                            •
                          </span>
                          <div>
                            <strong>Distributed Training:</strong> Multi-GPU,
                            multi-node model training on NVIDIA H100 clusters
                          </div>
                        </li>
                      </ScrollReveal>
                    </ul>
                  </div>
                  <div>
                    <ScrollReveal
                      animation="fade-right"
                      delay={250}
                      duration={700}
                    >
                      <h3
                        className="text-2xl font-semibold mb-6 flex items-center"
                        style={{ color: "var(--text-primary)" }}
                      >
                        <span className="mr-3">🎯</span>
                        AI Development & Research Goals
                      </h3>
                    </ScrollReveal>
                    <ScrollReveal
                      animation="fade-up"
                      delay={350}
                      duration={600}
                    >
                      <p
                        className="leading-relaxed mb-6"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        My primary focus is on building practical AI products
                        that deliver real value. I&apos;m deeply passionate
                        about AI Interpretability — understanding how neural
                        networks make decisions, visualizing internal
                        representations, and developing techniques to make AI
                        systems more transparent and trustworthy.
                      </p>
                    </ScrollReveal>
                    <ScrollReveal
                      animation="fade-up"
                      delay={450}
                      duration={600}
                    >
                      <p
                        className="leading-relaxed"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        I&apos;m also heavily invested in Generative AI,
                        including diffusion models, large language models, and
                        multimodal generation systems. My current work involves
                        exploring novel architectures, improving generation
                        quality, and developing efficient inference techniques
                        for deploying these models in production environments.
                      </p>
                    </ScrollReveal>
                    <ScrollReveal
                      animation="zoom-in"
                      delay={550}
                      duration={700}
                    >
                      <div
                        className="mt-6 p-4 rounded-xl transition-all duration-300 hover:shadow-md"
                        style={{ backgroundColor: "var(--surface-accent)" }}
                      >
                        <p
                          className="text-sm font-medium"
                          style={{ color: "var(--accent)" }}
                        >
                          🌟 &quot;Building AI products that are not only
                          powerful but also interpretable, reliable, and ready
                          for real-world deployment.&quot;
                        </p>
                      </div>
                    </ScrollReveal>
                  </div>
                </div>
              </div>
            </ScrollReveal>
          </div>
          {/* Contact CTA */}
          <ScrollReveal animation="fade-up" duration={800}>
            <div
              className="p-6 sm:p-10 rounded-2xl border text-center hover:shadow-lg transition-all duration-300"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--card-border)",
              }}
            >
              <div className="max-w-3xl mx-auto">
                <ScrollReveal animation="blur-in" delay={150} duration={700}>
                  <h2
                    className="text-3xl font-bold mb-6"
                    style={{ color: "var(--text-primary)" }}
                  >
                    Let&apos;s Build Something Amazing Together
                  </h2>
                </ScrollReveal>
                <ScrollReveal animation="fade-up" delay={250} duration={600}>
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
                </ScrollReveal>
                <div className="flex flex-wrap justify-center gap-4">
                  <ScrollReveal
                    animation="bounce-in"
                    delay={400}
                    duration={700}
                  >
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
                  </ScrollReveal>
                  <ScrollReveal
                    animation="bounce-in"
                    delay={500}
                    duration={700}
                  >
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
                  </ScrollReveal>
                  <ScrollReveal
                    animation="bounce-in"
                    delay={600}
                    duration={700}
                  >
                    <a
                      href="https://www.linkedin.com/in/hieptran01/"
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
                  </ScrollReveal>
                </div>
              </div>
            </div>
          </ScrollReveal>
        </div>
      </main>
    </div>
  );
}
