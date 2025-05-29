"use client";

import React from "react";
import BlogReader from "../../components/BlogReader";
import Image from "next/image";

export default function SampleBlogPost() {
  return (
    <BlogReader
      title="Building Modern Web Applications: A Comprehensive Guide"
      publishDate="2025-05-29"
      readTime="15 min read"
      category="Tutorial"
      author="Hiep Tran"
      tags={[
        "React",
        "TypeScript",
        "Next.js",
        "Web Development",
        "Modern Architecture",
      ]}
    >
      {/* Main Article Content */}
      <div className="mb-8">
        <Image
          src="/blog-placeholder.jpg"
          alt="Modern web development illustration"
          width={800}
          height={400}
          className="w-full h-64 object-cover rounded-lg shadow-lg"
        />
        <figcaption className="text-center text-sm text-gray-600 mt-2 italic">
          Modern web development with React and TypeScript
        </figcaption>
      </div>

      <p className="text-lg leading-relaxed mb-6">
        In today's rapidly evolving web development landscape, building modern,
        scalable, and maintainable web applications requires a deep
        understanding of contemporary frameworks, tools, and best practices.
        This comprehensive guide will walk you through the essential concepts
        and techniques needed to create exceptional web experiences.
      </p>

      <div className="callout callout-info mb-6">
        <p className="mb-0">
          <strong>💡 Pro Tip:</strong> This article covers advanced concepts. If
          you're new to web development, consider starting with our{" "}
          <a href="/blog/web-dev-basics">Web Development Fundamentals</a> guide
          first.
        </p>
      </div>

      <h2>Table of Contents</h2>
      <ul>
        <li>
          <a href="#foundation">Setting Up a Solid Foundation</a>
        </li>
        <li>
          <a href="#architecture">Application Architecture</a>
        </li>
        <li>
          <a href="#state-management">State Management Strategies</a>
        </li>
        <li>
          <a href="#performance">Performance Optimization</a>
        </li>
        <li>
          <a href="#testing">Testing Approaches</a>
        </li>
        <li>
          <a href="#deployment">Deployment and DevOps</a>
        </li>
      </ul>

      <h2 id="foundation">Setting Up a Solid Foundation</h2>

      <p>
        The foundation of any modern web application starts with choosing the
        right technology stack. In 2025, the combination of{" "}
        <strong>React</strong>, <strong>TypeScript</strong>, and{" "}
        <strong>Next.js</strong> provides an excellent starting point for most
        applications.
      </p>

      <h3>Why This Stack?</h3>

      <ul>
        <li>
          <strong>React</strong>: Component-based architecture for reusable UI
          elements
        </li>
        <li>
          <strong>TypeScript</strong>: Type safety and better developer
          experience
        </li>
        <li>
          <strong>Next.js</strong>: Full-stack framework with built-in
          optimizations
        </li>
      </ul>

      <div className="callout callout-success mb-6">
        <p className="mb-0">
          <strong>✅ Quick Start:</strong> Use{" "}
          <code>npx create-next-app@latest --typescript</code> to bootstrap your
          project with all the essentials.
        </p>
      </div>

      <h3>Essential Dependencies</h3>

      <p>
        Here's a typical <code>package.json</code> setup for a modern React
        application:
      </p>

      <div className="highlight" data-lang="json">
        <pre>
          <code>{`{
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "typescript": "^5.0.0",
    "@types/react": "^18.0.0",
    "@types/react-dom": "^18.0.0",
    "tailwindcss": "^3.4.0",
    "@next/font": "^14.0.0",
    "framer-motion": "^10.0.0"
  },
  "devDependencies": {
    "eslint": "^8.0.0",
    "eslint-config-next": "^14.0.0",
    "prettier": "^3.0.0",
    "@types/node": "^20.0.0"
  }
}`}</code>
        </pre>
      </div>

      <h2 id="architecture">Application Architecture</h2>

      <p>
        A well-structured application architecture is crucial for
        maintainability and scalability. Let's explore the key architectural
        patterns that work well with modern React applications.
      </p>

      <h3>Folder Structure</h3>

      <p>Organize your code with a clear, logical folder structure:</p>

      <div className="highlight" data-lang="bash">
        <pre>
          <code>{`src/
├── app/                 # Next.js 13+ App Router
│   ├── components/      # Reusable UI components
│   ├── hooks/          # Custom React hooks
│   ├── utils/          # Utility functions
│   ├── types/          # TypeScript type definitions
│   ├── styles/         # Global styles and themes
│   └── api/            # API routes (if using Next.js)
├── public/             # Static assets
└── tests/              # Test files`}</code>
        </pre>
      </div>

      <h3>Component Design Principles</h3>

      <p>Follow these principles when building React components:</p>

      <ol>
        <li>
          <strong>Single Responsibility</strong>: Each component should have one
          clear purpose
        </li>
        <li>
          <strong>Composition over Inheritance</strong>: Use composition to
          build complex UIs
        </li>
        <li>
          <strong>Props Interface Design</strong>: Design clear, well-typed
          props interfaces
        </li>
        <li>
          <strong>Accessibility First</strong>: Build with ARIA labels and
          semantic HTML
        </li>
      </ol>

      <blockquote>
        "The best components are those that are easy to understand, test, and
        maintain. They should do one thing well and be composable with other
        components." - Dan Abramov, React Team
      </blockquote>

      <h2 id="state-management">State Management Strategies</h2>

      <p>
        Effective state management is critical for building complex
        applications. The choice of state management solution depends on your
        application's complexity and requirements.
      </p>

      <h3>Local State vs Global State</h3>

      <table>
        <thead>
          <tr>
            <th>State Type</th>
            <th>Use Cases</th>
            <th>Tools</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Local State</td>
            <td>Component-specific data, form inputs, UI toggles</td>
            <td>useState, useReducer</td>
          </tr>
          <tr>
            <td>Global State</td>
            <td>User authentication, app settings, shared data</td>
            <td>Zustand, Redux Toolkit, Context API</td>
          </tr>
          <tr>
            <td>Server State</td>
            <td>API data, caching, synchronization</td>
            <td>React Query, SWR, Apollo Client</td>
          </tr>
        </tbody>
      </table>

      <h3>Recommended State Management Pattern</h3>

      <p>For most applications, I recommend this hybrid approach:</p>

      <div className="highlight" data-lang="typescript">
        <pre>
          <code>{`// hooks/useAuthStore.ts
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AuthState {
  user: User | null
  isAuthenticated: boolean
  login: (credentials: LoginCredentials) => Promise<void>
  logout: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      login: async (credentials) => {
        const user = await authAPI.login(credentials)
        set({ user, isAuthenticated: true })
      },
      logout: () => {
        set({ user: null, isAuthenticated: false })
      },
    }),
    {
      name: 'auth-storage',
    }
  )
)`}</code>
        </pre>
      </div>

      <div className="callout callout-warning mb-6">
        <p className="mb-0">
          <strong>⚠️ Warning:</strong> Avoid putting everything in global state.
          Keep component-specific state local and only elevate to global state
          when multiple components need access to the same data.
        </p>
      </div>

      <h2 id="performance">Performance Optimization</h2>

      <p>
        Performance optimization should be a consideration from the beginning of
        your project, not an afterthought. Here are the key strategies for
        building fast React applications.
      </p>

      <h3>Code Splitting and Lazy Loading</h3>

      <p>Split your code to reduce initial bundle size:</p>

      <div className="highlight" data-lang="typescript">
        <pre>
          <code>{`// Dynamic imports with React.lazy
const Dashboard = lazy(() => import('./Dashboard'))
const Settings = lazy(() => import('./Settings'))

function App() {
  return (
    <Router>
      <Suspense fallback={<LoadingSpinner />}>
        <Routes>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Suspense>
    </Router>
  )
}`}</code>
        </pre>
      </div>

      <h3>Memoization Strategies</h3>

      <p>Use React's built-in memoization hooks wisely:</p>

      <ul>
        <li>
          <code>React.memo()</code> for component memoization
        </li>
        <li>
          <code>useMemo()</code> for expensive calculations
        </li>
        <li>
          <code>useCallback()</code> for function references
        </li>
      </ul>

      <h2 id="testing">Testing Approaches</h2>

      <p>
        A comprehensive testing strategy ensures your application remains
        reliable as it grows. Focus on testing user behavior rather than
        implementation details.
      </p>

      <h3>Testing Pyramid</h3>

      <p>Follow the testing pyramid approach:</p>

      <ol>
        <li>
          <strong>Unit Tests (70%)</strong>: Test individual functions and
          components
        </li>
        <li>
          <strong>Integration Tests (20%)</strong>: Test component interactions
        </li>
        <li>
          <strong>End-to-End Tests (10%)</strong>: Test complete user workflows
        </li>
      </ol>

      <div className="highlight" data-lang="typescript">
        <pre>
          <code>{`// Example unit test with React Testing Library
import { render, screen, fireEvent } from '@testing-library/react'
import { Button } from './Button'

describe('Button', () => {
  it('calls onClick when clicked', () => {
    const handleClick = jest.fn()
    render(<Button onClick={handleClick}>Click me</Button>)
    
    fireEvent.click(screen.getByRole('button'))
    expect(handleClick).toHaveBeenCalledTimes(1)
  })
})`}</code>
        </pre>
      </div>

      <h2 id="deployment">Deployment and DevOps</h2>

      <p>
        Modern deployment strategies focus on automation, reliability, and quick
        rollbacks. Here's how to set up a robust deployment pipeline.
      </p>

      <h3>Deployment Platforms</h3>

      <p>Choose a platform that fits your needs:</p>

      <ul>
        <li>
          <strong>Vercel</strong>: Perfect for Next.js applications with zero
          configuration
        </li>
        <li>
          <strong>Netlify</strong>: Great for static sites and JAMstack
          applications
        </li>
        <li>
          <strong>AWS/Azure/GCP</strong>: Full control and scalability for
          enterprise applications
        </li>
      </ul>

      <div className="callout callout-info mb-6">
        <p className="mb-0">
          <strong>📦 Deployment Tip:</strong> Use preview deployments for
          feature branches to test changes before merging to production.
        </p>
      </div>

      <hr />

      <h2>Conclusion</h2>

      <p>
        Building modern web applications requires a thoughtful approach to
        architecture, performance, and developer experience. By following the
        patterns and practices outlined in this guide, you'll be well-equipped
        to create applications that are both powerful and maintainable.
      </p>

      <p>
        Remember that technology evolves rapidly, so stay curious and keep
        learning. The fundamentals of good software design remain constant, but
        the tools and techniques continue to improve.
      </p>

      <div className="callout callout-success mb-6">
        <p className="mb-0">
          <strong>🎉 Next Steps:</strong> Try implementing these concepts in a
          small project. Start with the foundation and gradually add complexity
          as you become more comfortable with each pattern.
        </p>
      </div>

      <p>
        Have questions or want to share your own experiences? Feel free to{" "}
        <a href="/contact">reach out</a> or connect with me on{" "}
        <a
          href="https://twitter.com/hieptran1812"
          target="_blank"
          rel="noopener noreferrer"
        >
          Twitter
        </a>
        .
      </p>
    </BlogReader>
  );
}
