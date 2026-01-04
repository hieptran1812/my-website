# Personal Website & Blog

A modern, feature-rich personal website and blog built with Next.js 15, React 19, and TypeScript. Features an interactive blog with D3.js graph visualization, math rendering, audio reading, and a comprehensive engagement system.

## Features

### Blog System
- **Markdown-based content**: Write blog posts in Markdown with frontmatter metadata
- **Category organization**: Posts organized by categories (Machine Learning, Paper Reading, Software Development, Notes, etc.)
- **Tag-based clustering**: Posts grouped by tags for easy navigation
- **Interactive Graph View**: D3.js-powered force-directed graph showing relationships between articles
  - Posts with shared tags are clustered together
  - Direct references between posts create connections
  - Expandable fullscreen modal view
  - Drag, zoom, and pan interactions
- **Table of Contents**: Auto-generated TOC with smooth scrolling
- **Reading progress indicator**: Visual progress bar while reading
- **Estimated read time**: Automatic calculation based on content length

### Reading Experience
- **Eye Comfort Mode**: Warm, sepia-toned background for reduced eye strain
- **Adjustable Font Size**: Customize text size (12px - 24px)
- **Line Spacing Control**: Adjust line height (1.2 - 2.0)
- **Audio Reading**: Text-to-speech functionality with play/pause/stop controls and progress tracking
- **Dark/Light Theme**: Full theme support with smooth transitions

### Math & Code
- **KaTeX Integration**: Beautiful math rendering with LaTeX syntax
- **Code Highlighting**: Syntax highlighting for code blocks
- **GitHub Flavored Markdown**: Support for GFM features (tables, task lists, etc.)

### Engagement Features
- **Reactions**: Like, love, laugh, wow, sad, angry reactions on posts
- **Comments**: Nested comment system with replies
- **Social Sharing**: Share to Twitter, Facebook, LinkedIn, or copy link
- **View Tracking**: Track article views and engagement metrics

### SEO & Performance
- **Vercel Analytics**: Built-in analytics integration
- **Speed Insights**: Performance monitoring
- **Sitemap Generation**: Automatic sitemap for search engines
- **Open Graph Images**: Dynamic OG image generation
- **Robots.txt**: Proper search engine directives

## Tech Stack

| Category | Technology |
|----------|------------|
| Framework | [Next.js 15](https://nextjs.org/) with App Router |
| Language | [TypeScript 5](https://www.typescriptlang.org/) |
| UI Library | [React 19](https://react.dev/) |
| Styling | [Tailwind CSS 4](https://tailwindcss.com/) |
| Database | [Prisma](https://www.prisma.io/) with SQLite |
| Visualization | [D3.js 7](https://d3js.org/) |
| Math Rendering | [KaTeX](https://katex.org/) |
| Markdown | [remark](https://github.com/remarkjs/remark) + [gray-matter](https://github.com/jonschlinkert/gray-matter) |
| Theming | [next-themes](https://github.com/pacocoursey/next-themes) |
| Email | [Nodemailer](https://nodemailer.com/) |
| Deployment | [Vercel](https://vercel.com/) |

## Project Structure

```
my-website/
├── content/
│   ├── blog/                    # Blog posts (Markdown)
│   │   ├── machine-learning/    # ML articles
│   │   ├── paper-reading/       # Paper reviews
│   │   ├── software-development/# Dev articles
│   │   └── notes/               # Personal notes
│   └── projects/                # Project descriptions
├── prisma/
│   └── schema.prisma            # Database schema
├── public/                      # Static assets
├── scripts/
│   ├── blogAnalytics.ts         # Blog analytics reports
│   ├── updateReadTime.ts        # Update read time metadata
│   └── updatePaperReferences.ts # Update paper references
├── src/
│   ├── app/
│   │   ├── api/                 # API routes
│   │   │   ├── blog/            # Blog APIs (graph, posts, etc.)
│   │   │   ├── comments/        # Comments API
│   │   │   ├── reactions/       # Reactions API
│   │   │   └── shares/          # Share tracking API
│   │   ├── blog/                # Blog pages
│   │   ├── components/          # Shared components
│   │   │   ├── BlogGraphView.tsx    # D3.js graph visualization
│   │   │   ├── BlogGraphSidebar.tsx # Reading options + graph widget
│   │   │   ├── BlogReader.tsx       # Main blog reader component
│   │   │   └── Navigation.tsx       # Site navigation
│   │   ├── about/               # About page
│   │   ├── projects/            # Projects page
│   │   └── contact/             # Contact page
│   ├── components/
│   │   ├── styles/              # CSS modules
│   │   └── utils/               # Utility functions
│   └── lib/                     # Shared libraries
└── package.json
```

## Getting Started

### Prerequisites
- Node.js 18+
- npm, yarn, pnpm, or bun

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/my-website.git
cd my-website
```

2. Install dependencies:
```bash
npm install
# or
yarn install
# or
pnpm install
```

3. Set up the database:
```bash
npx prisma generate
npx prisma db push
```

4. Create a `.env` file:
```env
DATABASE_URL="file:./prisma/dev.db"
```

5. Run the development server:
```bash
npm run dev
```

6. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server with Turbopack |
| `npm run build` | Build for production |
| `npm run start` | Start production server |
| `npm run lint` | Run ESLint |
| `npm run update-readtime` | Update read time for all blog posts |
| `npm run update-paper-references` | Update paper references |
| `npm run blog-report` | Generate blog analytics report |
| `npm run blog-validate` | Validate blog post metadata |

## Writing Blog Posts

Create a new Markdown file in `content/blog/[category]/`:

```markdown
---
title: "Your Post Title"
description: "A brief description of your post"
date: "2024-01-15"
tags: ["tag1", "tag2"]
category: "Machine Learning"
author: "Your Name"
image: "/imgs/blog/your-image.png"
---

Your content here...

## Math Example
$$E = mc^2$$

## Code Example
```python
def hello():
    print("Hello, World!")
```
```

### Frontmatter Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | Yes | Post title |
| `description` | string | Yes | Short description for SEO |
| `date` | string | Yes | Publication date (YYYY-MM-DD) |
| `tags` | array | No | List of tags |
| `category` | string | No | Post category |
| `author` | string | No | Author name |
| `image` | string | No | Cover image path |
| `readTime` | number | No | Auto-calculated read time |

## API Routes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/blog/graph` | GET | Get graph data for article relationships |
| `/api/blog/[slug]` | GET | Get single blog post |
| `/api/reactions` | GET/POST | Manage post reactions |
| `/api/comments` | GET/POST | Manage comments |
| `/api/shares` | POST | Track social shares |
| `/api/contact` | POST | Handle contact form submissions |

## Database Schema

The application uses Prisma with SQLite for storing engagement data:

- **BlogPost**: Core blog post entity
- **Reaction**: User reactions (like, love, laugh, wow, sad, angry)
- **Comment**: Nested comments with replies
- **Share**: Social share tracking

## Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Import the project in [Vercel](https://vercel.com)
3. Set environment variables
4. Deploy

The `vercel-build` script handles Prisma generation automatically.

### Manual Deployment

```bash
npm run build
npm run start
```

## Customization

### Theme Colors
Edit `src/app/globals.css` to customize the color scheme.

### Graph Colors
Modify the `TAG_COLORS` array in `src/app/api/blog/graph/route.ts` to change cluster colors.

### Reading Options
Adjust default values in `BlogGraphSidebar.tsx`:
- `fontSize`: Default font size (18px)
- `lineHeight`: Default line height (1.6)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- [Next.js](https://nextjs.org/) - The React framework
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS
- [D3.js](https://d3js.org/) - Data visualization library
- [Prisma](https://www.prisma.io/) - Database ORM
- [KaTeX](https://katex.org/) - Math typesetting
- [Vercel](https://vercel.com/) - Deployment platform
