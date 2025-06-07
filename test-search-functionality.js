#!/usr/bin/env node

/**
 * Test script to verify search counter functionality
 * This simulates the search component behavior with various test queries
 */

const fetch = require("node-fetch");

// Simulate the searchContent function from lib/search.ts
function searchContent(data, query, minRelevance = 0.1) {
  if (!query.trim()) return data;

  const queryWords = query.toLowerCase().split(/\s+/);

  return data.filter((item) => {
    let relevance = 0;
    const searchableText = `${item.title || ""} ${item.content || ""} ${
      item.description || ""
    }`.toLowerCase();

    for (const word of queryWords) {
      if (searchableText.includes(word)) {
        relevance += 1;
      }
    }

    return relevance >= minRelevance;
  });
}

// Test queries
const testQueries = [
  "react",
  "javascript",
  "project",
  "blog",
  "nextjs",
  "typescript",
  "portfolio",
  "web development",
];

async function testSearchCounters() {
  try {
    console.log("🧪 Testing Search Counter Functionality\n");

    // Fetch data from APIs
    const [articlesResponse, projectsResponse] = await Promise.all([
      fetch("http://localhost:3001/api/blog/articles"),
      fetch("http://localhost:3001/api/projects"),
    ]);

    const articlesData = await articlesResponse.json();
    const projectsData = await projectsResponse.json();

    // Extract arrays safely
    const articles = articlesData?.articles || [];
    const projects = projectsData?.projects || [];

    // Static pages (hardcoded like in SearchComponent)
    const pages = [
      { title: "Home", content: "homepage main landing", type: "page" },
      {
        title: "About",
        content: "about me personal information",
        type: "page",
      },
      { title: "Contact", content: "contact information email", type: "page" },
    ];

    console.log(`📊 Data loaded:
- Articles: ${articles.length}
- Projects: ${projects.length}  
- Pages: ${pages.length}
- Total: ${articles.length + projects.length + pages.length}\n`);

    // Test each query
    for (const query of testQueries) {
      console.log(`🔍 Testing query: "${query}"`);

      // Combine all data with type markers
      const allData = [
        ...articles.map((item) => ({ ...item, type: "article" })),
        ...projects.map((item) => ({ ...item, type: "project" })),
        ...pages,
      ];

      // Search with our improved threshold
      const results = searchContent(allData, query, 0.1);

      // Count by type
      const counts = {
        all: results.length,
        article: results.filter((item) => item.type === "article").length,
        project: results.filter((item) => item.type === "project").length,
        page: results.filter((item) => item.type === "page").length,
      };

      console.log(
        `  Results: All(${counts.all}) Articles(${counts.article}) Projects(${counts.project}) Pages(${counts.page})`
      );

      // Show some example matches if any
      if (results.length > 0 && results.length <= 3) {
        results.forEach((result) => {
          console.log(`    - ${result.type}: "${result.title}"`);
        });
      }
      console.log("");
    }

    console.log("✅ Search counter test completed successfully!");
  } catch (error) {
    console.error("❌ Test failed:", error.message);
  }
}

// Run the test
testSearchCounters();
