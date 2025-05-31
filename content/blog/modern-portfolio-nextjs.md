---
title: Building a Modern Portfolio with Next.js
publishDate: '2024-03-15'
readTime: 4 min read
category: Tutorial
author: Hiep Tran
tags:
  - Next.js
  - TypeScript
  - Tailwind CSS
  - Portfolio
  - Web Development
image: /blog-placeholder.jpg
excerpt: >-
  Learn how to build a stunning personal portfolio website using Next.js,
  TypeScript, and Tailwind CSS with modern design patterns.
---

# Building a Modern Portfolio with Next.js

![Modern Portfolio with Next.js - Tutorial Preview](/blog-placeholder.jpg)

This article guides you through building a personal portfolio website using Next.js, TypeScript, and Tailwind CSS. You'll learn about project structure, modular React components, and deploying your site.

## Getting Started

Start by creating a new Next.js project and installing Tailwind CSS. Organize your code into reusable components for each section of your portfolio.

<div className="callout callout-info">
<strong>Prerequisites:</strong> Basic knowledge of React, TypeScript, and modern web development concepts.
</div>

### Project Setup

First, create a new Next.js project with TypeScript:

```bash
npx create-next-app@latest my-portfolio --typescript --tailwind --app
cd my-portfolio
npm run dev
```

## Key Features

Your modern portfolio should include these essential sections:

- **Hero section** with intro and profile image
- **Latest projects** and open source highlights
- **Blog/articles** section with previews
- **Contact/collaboration** section
- **Footer** with navigation and social links

### Hero Section Implementation

```tsx
export default function HeroSection() {
  return (
    <section className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-6xl font-bold mb-6">
          Hi, I'm <span className="text-blue-600">Your Name</span>
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          Full-Stack Developer & AI Enthusiast
        </p>
      </div>
    </section>
  );
}
```

## Responsive Design

Ensure your portfolio looks great on all devices by using Tailwind's responsive utilities:

```css
/* Mobile-first approach */
.hero-title {
  @apply text-3xl md:text-4xl lg:text-6xl;
}
```

## Deployment

Deploy your portfolio to Vercel for free hosting with automatic deployments from GitHub.

<div className="callout callout-success">
<strong>Pro tip:</strong> Set up custom domain and SSL certificates for a professional look.
</div>

## Conclusion

Building a modern portfolio with Next.js gives you the performance and SEO benefits needed to stand out in today's competitive market.
