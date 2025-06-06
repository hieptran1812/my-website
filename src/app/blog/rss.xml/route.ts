import { NextResponse } from "next/server";

const articles = [
  {
    title: "How to Build a Modern Portfolio with Next.js",
    summary:
      "A comprehensive step-by-step guide to creating a personal portfolio website using Next.js, TypeScript, and Tailwind CSS with modern design patterns and best practices.",
    date: "2025-05-01",
    link: "/blog/modern-portfolio-nextjs",
    category: "Tutorial",
    author: "Hiep Tran",
    guid: "hieptran-dev-modern-portfolio-nextjs",
  },
  {
    title: "Open Source: Why and How to Start",
    summary:
      "My personal journey into open source development and comprehensive tips for beginners looking to make their first contributions to open source projects.",
    date: "2025-04-15",
    link: "/blog/open-source-guide",
    category: "Guide",
    author: "Hiep Tran",
    guid: "hieptran-dev-open-source-guide",
  },
  {
    title: "TypeScript Best Practices for Large Applications",
    summary:
      "Essential patterns and practices for building scalable TypeScript applications with proper type safety and maintainable code architecture.",
    date: "2025-04-01",
    link: "/blog/typescript-best-practices",
    category: "Development",
    author: "Hiep Tran",
    guid: "hieptran-dev-typescript-best-practices",
  },
  {
    title: "DeFi Fundamentals: Understanding Decentralized Finance",
    summary:
      "A comprehensive guide to DeFi protocols, yield farming, and the future of decentralized financial services with real-world applications.",
    date: "2025-03-25",
    link: "/blog/defi-fundamentals",
    category: "Crypto",
    author: "Hiep Tran",
    guid: "hieptran-dev-defi-fundamentals",
  },
  {
    title: "Software Development Best Practices",
    summary:
      "Best practices and architectural patterns for building scalable, maintainable software applications using modern development methodologies.",
    date: "2025-03-10",
    link: "/blog/software-development-best-practices",
    category: "Development",
    author: "Hiep Tran",
    guid: "hieptran-dev-software-development-best-practices",
  },
];

export async function GET() {
  const baseUrl = "https://halleyverse.dev";
  const buildDate = new Date().toUTCString();

  const rssXml = `<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0" 
     xmlns:atom="http://www.w3.org/2005/Atom"
     xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Hiep Tran Blog - AI, Machine Learning &amp; Software Development</title>
    <description>Latest insights and tutorials about AI, machine learning, deep learning, software development, and emerging technologies by Hiep Tran, AI Engineer and Full-Stack Developer.</description>
    <link>${baseUrl}/blog</link>
    <atom:link href="${baseUrl}/blog/rss.xml" rel="self" type="application/rss+xml" />
    <language>en-us</language>
    <lastBuildDate>${buildDate}</lastBuildDate>
    <pubDate>${buildDate}</pubDate>
    <ttl>60</ttl>
    <generator>Next.js RSS Generator by Hiep Tran</generator>
    <managingEditor>hieptran.jobs@gmail.com (Hiep Tran)</managingEditor>
    <webMaster>hieptran.jobs@gmail.com (Hiep Tran)</webMaster>
    <copyright>Copyright ${new Date().getFullYear()} Hiep Tran. All rights reserved.</copyright>
    <category>Technology</category>
    <category>Programming</category>
    <category>Artificial Intelligence</category>
    <category>Machine Learning</category>
    <image>
      <url>${baseUrl}/og-image.jpg</url>
      <title>Hiep Tran Blog</title>
      <link>${baseUrl}/blog</link>
      <description>AI Engineer and Full-Stack Developer Blog</description>
      <width>1200</width>
      <height>630</height>
    </image>
    ${articles
      .map(
        (article) => `
    <item>
      <title><![CDATA[${article.title}]]></title>
      <description><![CDATA[${article.summary}]]></description>
      <link>${baseUrl}${article.link}</link>
      <guid isPermaLink="false">${article.guid}</guid>
      <pubDate>${new Date(article.date).toUTCString()}</pubDate>
      <category><![CDATA[${article.category}]]></category>
      <dc:creator><![CDATA[${article.author}]]></dc:creator>
      <author>hieptran.jobs@gmail.com (${article.author})</author>
      <source url="${baseUrl}/blog/rss.xml">Hiep Tran Blog</source>
    </item>`
      )
      .join("")}
  </channel>
</rss>`;

  return new NextResponse(rssXml, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8",
      "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400",
    },
  });
}
