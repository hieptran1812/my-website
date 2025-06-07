// Comprehensive search debugging script
// import { searchContent, getAllSearchableContent } from './src/lib/search.js';

const debugSearchCounters = async () => {
  const baseUrl = "http://localhost:3002";

  console.log("🐛 Debugging Search Counter Issues...\n");

  try {
    // Fetch the same data that the component fetches
    const [articlesResponse, projectsResponse] = await Promise.all([
      fetch(`${baseUrl}/api/blog/articles`),
      fetch(`${baseUrl}/api/projects`),
    ]);

    const articlesData = await articlesResponse.json();
    const projectsData = await projectsResponse.json();

    console.log("📊 Raw API Data:");
    console.log(`   Articles: ${articlesData.articles?.length || 0}`);
    console.log(`   Projects: ${projectsData.projects?.length || 0}`);

    // Simulate the search component's data transformation
    const searchResults = [];

    // Add articles (simulate the component logic)
    (articlesData.articles || []).forEach((article) => {
      const blogUrl = `/blog/${article.slug}`;
      searchResults.push({
        title: article.title,
        description: article.excerpt,
        url: blogUrl,
        type: "blog",
        category: article.category,
        tags: article.tags,
        featured: article.featured,
        date: article.date,
        difficulty: article.difficulty,
        readTime: article.readTime,
      });
    });

    // Add projects (simulate the component logic)
    (projectsData.projects || []).forEach((project) => {
      searchResults.push({
        title: project.title,
        description: project.description,
        url: `/projects/${project.id}`,
        type: "project",
        category: project.category,
        technologies: project.technologies,
        featured: project.featured,
        status: project.status,
        highlights: project.highlights,
      });
    });

    // Add static pages
    const staticPages = [
      {
        title: "About",
        description:
          "Learn more about Hiep Tran - AI Research Engineer and Full-Stack Developer specializing in machine learning and web development.",
        url: "/about",
        type: "page",
        tags: ["About", "Profile", "Background"],
      },
      {
        title: "Projects",
        description:
          "Explore my portfolio of AI research projects, web applications, and open source contributions.",
        url: "/projects",
        type: "page",
        tags: ["Portfolio", "Projects", "AI", "Web Development"],
      },
      {
        title: "Contact",
        description:
          "Get in touch for collaboration opportunities, research discussions, or project inquiries.",
        url: "/contact",
        type: "page",
        tags: ["Contact", "Collaboration", "Hire"],
      },
    ];

    searchResults.push(...staticPages);

    console.log(`\n🔍 Total Searchable Content: ${searchResults.length}`);
    console.log(
      `   Blogs: ${searchResults.filter((r) => r.type === "blog").length}`
    );
    console.log(
      `   Projects: ${searchResults.filter((r) => r.type === "project").length}`
    );
    console.log(
      `   Pages: ${searchResults.filter((r) => r.type === "page").length}`
    );

    // Test different search queries
    const testQueries = ["machine learning", "react", "web", "ai"];

    console.log("\n🧪 Testing Search Queries:");

    for (const query of testQueries) {
      console.log(`\n📝 Query: "${query}"`);

      // This simulates what the component does - import searchContent from search.ts
      // Since we can't import in this Node.js script, let's simulate the logic

      const queryResults = searchResults.filter((item) => {
        const searchText = `${item.title} ${item.description} ${
          item.category || ""
        } ${(item.tags || []).join(" ")}`.toLowerCase();
        return searchText.includes(query.toLowerCase());
      });

      const counts = {
        all: queryResults.length,
        blog: queryResults.filter((r) => r.type === "blog").length,
        project: queryResults.filter((r) => r.type === "project").length,
        page: queryResults.filter((r) => r.type === "page").length,
      };

      console.log(`   Results: ${counts.all} total`);
      console.log(`   • Blog: ${counts.blog}`);
      console.log(`   • Project: ${counts.project}`);
      console.log(`   • Page: ${counts.page}`);

      if (counts.all === 0) {
        console.log(
          `   ⚠️  No results for "${query}" - might be relevance threshold issue`
        );
      }
    }

    console.log("\n✅ Counter Logic Validation:");
    console.log("   The counters should show:");
    console.log("   • All: Total number of search results matching the query");
    console.log(
      "   • Blog/Project/Page: Number of each type matching the query"
    );
    console.log("   • When you click a filter, it shows only that type");
    console.log(
      "   • The counter numbers don't change when you switch filters"
    );
  } catch (error) {
    console.error("❌ Debug failed:", error.message);
  }
};

// Run the debug
debugSearchCounters();
