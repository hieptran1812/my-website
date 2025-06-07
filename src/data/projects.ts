// Project data and utilities for the portfolio
export interface Project {
  id: number;
  title: string;
  description: string;
  category: string;
  technologies: string[];
  githubUrl?: string;
  liveUrl?: string | null;
  featured: boolean;
  status: string;
  highlights: string[];
  image?: string;
}

// Centralized project data
export const projects: Project[] = [
  {
    id: 1,
    title: "AI-Powered Natural Language Processing Platform",
    description:
      "A comprehensive NLP platform built with transformer models for text analysis, sentiment analysis, and language understanding. Supports multiple languages and real-time processing.",
    image: "/project-placeholder.jpg",
    category: "Machine Learning",
    technologies: [
      "Python",
      "PyTorch",
      "Transformers",
      "FastAPI",
      "Docker",
      "Kubernetes",
    ],
    githubUrl: "https://github.com/hieptran1812/nlp-platform",
    liveUrl: "https://nlp-platform-demo.com",
    featured: true,
    status: "Production",
    highlights: [
      "98% accuracy on sentiment analysis",
      "Supports 15+ languages",
      "10K+ daily active users",
      "Sub-100ms response time",
    ],
  },
  {
    id: 2,
    title: "Distributed Machine Learning Training System",
    description:
      "Scalable ML training infrastructure using distributed computing to train large-scale deep learning models efficiently across multiple GPUs and nodes.",
    image: "/project-placeholder.jpg",
    category: "Machine Learning",
    technologies: [
      "Python",
      "TensorFlow",
      "Apache Spark",
      "Kubernetes",
      "Redis",
      "PostgreSQL",
    ],
    githubUrl: "https://github.com/hieptran1812/distributed-ml-training",
    liveUrl: null,
    featured: true,
    status: "Active Development",
    highlights: [
      "80% faster training time",
      "Auto-scaling capabilities",
      "Fault-tolerant architecture",
      "MLflow integration",
    ],
  },
  {
    id: 3,
    title: "Modern E-commerce Platform",
    description:
      "Full-stack e-commerce solution with advanced features including real-time inventory management, AI-powered recommendations, and comprehensive analytics dashboard.",
    image: "/project-placeholder.jpg",
    category: "Web Development",
    technologies: [
      "Next.js",
      "TypeScript",
      "PostgreSQL",
      "Prisma",
      "Stripe",
      "Tailwind CSS",
    ],
    githubUrl: "https://github.com/hieptran1812/ecommerce-platform",
    liveUrl: "https://ecommerce-demo.halleyverse.dev",
    featured: true,
    status: "Production",
    highlights: [
      "Mobile-first responsive design",
      "AI-powered product recommendations",
      "Real-time inventory tracking",
      "Integrated payment processing",
    ],
  },
  {
    id: 4,
    title: "Computer Vision Object Detection API",
    description:
      "RESTful API service for real-time object detection and classification using state-of-the-art YOLO and R-CNN models with custom training capabilities.",
    image: "/project-placeholder.jpg",
    category: "Machine Learning",
    technologies: ["Python", "OpenCV", "YOLO", "FastAPI", "Docker", "AWS"],
    githubUrl: "https://github.com/hieptran1812/object-detection-api",
    liveUrl: "https://api.object-detection.com",
    featured: false,
    status: "Production",
    highlights: [
      "Real-time processing",
      "Custom model training",
      "RESTful API design",
      "Cloud deployment",
    ],
  },
  {
    id: 5,
    title: "Big Data Analytics Pipeline",
    description:
      "End-to-end data processing pipeline for analyzing large-scale datasets with real-time streaming capabilities and interactive visualization dashboards.",
    image: "/project-placeholder.jpg",
    category: "Data Science",
    technologies: [
      "Apache Kafka",
      "Spark",
      "Hadoop",
      "Elasticsearch",
      "Kibana",
      "Python",
    ],
    githubUrl: "https://github.com/hieptran1812/big-data-pipeline",
    liveUrl: null,
    featured: false,
    status: "Production",
    highlights: [
      "Processes 1M+ records/day",
      "Real-time streaming",
      "Interactive dashboards",
      "Scalable architecture",
    ],
  },
  {
    id: 6,
    title: "Open Source ML Model Registry",
    description:
      "Community-driven platform for sharing, versioning, and managing machine learning models with automated testing and deployment capabilities.",
    image: "/project-placeholder.jpg",
    category: "Open Source",
    technologies: [
      "Python",
      "Django",
      "PostgreSQL",
      "Redis",
      "Docker",
      "GitHub Actions",
    ],
    githubUrl: "https://github.com/hieptran1812/ml-model-registry",
    liveUrl: "https://ml-registry.org",
    featured: false,
    status: "Production",
    highlights: [
      "500+ community models",
      "Automated testing",
      "Version control",
      "API integration",
    ],
  },
  {
    id: 7,
    title: "Blockchain Data Analytics Dashboard",
    description:
      "Real-time blockchain analytics platform providing insights into DeFi protocols, token metrics, and market trends with advanced visualization.",
    image: "/project-placeholder.jpg",
    category: "Data Science",
    technologies: [
      "React",
      "D3.js",
      "Node.js",
      "Web3.js",
      "MongoDB",
      "GraphQL",
    ],
    githubUrl: "https://github.com/hieptran1812/blockchain-analytics",
    liveUrl: "https://crypto-analytics.dev",
    featured: false,
    status: "Beta",
    highlights: [
      "Real-time data processing",
      "Interactive visualizations",
      "Multi-chain support",
      "Custom alerts system",
    ],
  },
  {
    id: 8,
    title: "AI Code Review Assistant",
    description:
      "Intelligent code review tool powered by large language models that provides automated code analysis, suggestions, and best practice recommendations.",
    image: "/project-placeholder.jpg",
    category: "Open Source",
    technologies: [
      "Python",
      "TypeScript",
      "OpenAI API",
      "GitHub API",
      "Docker",
      "FastAPI",
    ],
    githubUrl: "https://github.com/hieptran1812/ai-code-reviewer",
    liveUrl: null,
    featured: false,
    status: "Active Development",
    highlights: [
      "AI-powered analysis",
      "GitHub integration",
      "Multi-language support",
      "Custom rule engine",
    ],
  },
];

// Utility functions for project data
export const getProjectById = (id: number): Project | undefined => {
  return projects.find((project) => project.id === id);
};

// Utility functions for projects
export const getAllProjects = (): Project[] => {
  return projects;
};

export const getFeaturedProjects = (): Project[] => {
  return projects.filter((project) => project.featured);
};

export const getProjectsByCategory = (category: string): Project[] => {
  return projects.filter(
    (project) => project.category.toLowerCase() === category.toLowerCase()
  );
};

export const getProjectsByStatus = (status: string): Project[] => {
  return projects.filter(
    (project) => project.status.toLowerCase() === status.toLowerCase()
  );
};

export const getProjectCategories = (): string[] => {
  const categories = new Set(projects.map((project) => project.category));
  return Array.from(categories);
};

export const searchProjects = (query: string): Project[] => {
  const lowercaseQuery = query.toLowerCase();
  return projects.filter(
    (project) =>
      project.title.toLowerCase().includes(lowercaseQuery) ||
      project.description.toLowerCase().includes(lowercaseQuery) ||
      project.category.toLowerCase().includes(lowercaseQuery) ||
      project.technologies.some((tech) =>
        tech.toLowerCase().includes(lowercaseQuery)
      ) ||
      project.highlights.some((highlight) =>
        highlight.toLowerCase().includes(lowercaseQuery)
      )
  );
};
