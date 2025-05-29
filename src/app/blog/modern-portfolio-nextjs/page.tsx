import React from "react";
import Image from "next/image";
import BlogReader from "../../components/BlogReader";

export default function ModernPortfolioNextjsArticle() {
  const articleContent = (
    <>
      <Image
        src="/blog-placeholder.jpg"
        alt="Modern Portfolio with Next.js - Tutorial Preview"
        width={640}
        height={320}
        className="rounded-lg mb-8 object-cover w-full"
      />

      <p>
        This article guides you through building a personal portfolio website
        using Next.js, TypeScript, and Tailwind CSS. You&apos;ll learn about
        project structure, modular React components, and deploying your site.
      </p>

      <h2>Getting Started</h2>
      <p>
        Start by creating a new Next.js project and installing Tailwind CSS.
        Organize your code into reusable components for each section of your
        portfolio.
      </p>

      <div className="callout info">
        <strong>Prerequisites:</strong> Basic knowledge of React, TypeScript,
        and modern web development concepts.
      </div>

      <h3>Project Setup</h3>
      <p>First, create a new Next.js project with TypeScript:</p>

      <pre>
        <code className="language-bash">{`npx create-next-app@latest my-portfolio --typescript --tailwind --app
cd my-portfolio
npm run dev`}</code>
      </pre>

      <h2>Key Features</h2>
      <p>Your modern portfolio should include these essential sections:</p>

      <ul>
        <li>
          <strong>Hero section</strong> with intro and profile image
        </li>
        <li>
          <strong>Latest projects</strong> and open source highlights
        </li>
        <li>
          <strong>Blog/articles</strong> with previews
        </li>
        <li>
          <strong>Contact and collaboration</strong> form
        </li>
        <li>
          <strong>Responsive design</strong> for all devices
        </li>
        <li>
          <strong>Dark/light mode</strong> toggle
        </li>
      </ul>

      <h3>Component Structure</h3>
      <p>Organize your components for maintainability:</p>

      <pre>
        <code className="language-typescript">{`// components/HeroSection.tsx
export default function HeroSection() {
  return (
    <section className="py-20 px-4">
      <div className="max-w-4xl mx-auto text-center">
        <h1 className="text-4xl md:text-6xl font-bold mb-6">
          Your Name
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-300 mb-8">
          Your professional tagline
        </p>
      </div>
    </section>
  );
}`}</code>
      </pre>

      <h2>Styling with Tailwind CSS</h2>
      <p>
        Tailwind CSS provides utility-first styling that makes your portfolio
        both beautiful and maintainable. Use responsive classes and dark mode
        variants to create a polished experience.
      </p>

      <div className="callout success">
        <strong>Pro Tip:</strong> Use CSS custom properties for theme variables
        to enable smooth transitions between light and dark modes.
      </div>

      <h2>Deployment</h2>
      <p>
        Deploy your site easily with Vercel, which provides seamless integration
        with Next.js projects. Simply connect your GitHub repository and enjoy
        automatic deployments on every push.
      </p>

      <h3>Deployment Steps</h3>
      <ol>
        <li>Push your code to GitHub</li>
        <li>Connect your repository to Vercel</li>
        <li>Configure your domain (optional)</li>
        <li>Enjoy your live portfolio!</li>
      </ol>

      <div className="callout warning">
        <strong>Remember:</strong> Always test your site in production mode
        before deploying to catch any build-time issues.
      </div>

      <h2>Conclusion</h2>
      <p>
        Building a modern portfolio with Next.js gives you the flexibility and
        performance needed to showcase your work effectively. The combination of
        React components, TypeScript safety, and Tailwind styling creates a
        maintainable and impressive online presence.
      </p>
    </>
  );

  return (
    <BlogReader
      title="How to Build a Modern Portfolio with Next.js"
      publishDate="2025-05-01"
      readTime="8 min read"
      category="Tutorial"
      tags={[
        "Next.js",
        "TypeScript",
        "Tailwind CSS",
        "Portfolio",
        "Web Development",
      ]}
      author="Hiep Tran"
    >
      {articleContent}
    </BlogReader>
  );
}
