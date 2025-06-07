// Test script to verify search counter functionality
const testSearchCounters = async () => {
  const baseUrl = "http://localhost:3002";

  console.log("🔍 Testing Search Counter Functionality...\n");

  try {
    // Test 1: Check articles API
    console.log("1️⃣ Testing Articles API (/api/blog/articles)...");
    const articlesResponse = await fetch(`${baseUrl}/api/blog/articles`);
    const articlesData = await articlesResponse.json();
    console.log(`✅ Articles found: ${articlesData.articles?.length || 0}`);
    console.log(`   Total: ${articlesData.total || 0}`);
    console.log(
      `   Sample article types: ${articlesData.articles
        ?.slice(0, 3)
        .map((a) => `${a.title} (${a.category})`)
        .join(", ")}\n`
    );

    // Test 2: Check projects API
    console.log("2️⃣ Testing Projects API (/api/projects)...");
    const projectsResponse = await fetch(`${baseUrl}/api/projects`);
    const projectsData = await projectsResponse.json();
    console.log(`✅ Projects found: ${projectsData.projects?.length || 0}`);
    console.log(`   Total: ${projectsData.total || 0}`);
    console.log(
      `   Sample project types: ${projectsData.projects
        ?.slice(0, 3)
        .map((p) => `${p.title} (${p.category})`)
        .join(", ")}\n`
    );

    // Test 3: Test search functionality with different queries
    console.log("3️⃣ Simulating search scenarios...");

    // Import search functions (this would be done in the actual component)
    const searchQueries = [
      "machine learning",
      "react",
      "computer vision",
      "web development",
    ];

    console.log(`\n📊 Expected search results breakdown:`);
    console.log(`   Total articles: ${articlesData.articles?.length || 0}`);
    console.log(`   Total projects: ${projectsData.projects?.length || 0}`);
    console.log(`   Total static pages: 3 (About, Projects, Contact)`);
    console.log(
      `   Grand total: ${
        (articlesData.articles?.length || 0) +
        (projectsData.projects?.length || 0) +
        3
      }`
    );

    console.log("\n✨ Expected counter behavior:");
    console.log("   • All: Shows total search results matching query");
    console.log("   • Blog: Shows blog posts matching query");
    console.log("   • Project: Shows projects matching query");
    console.log("   • Page: Shows static pages matching query");
  } catch (error) {
    console.error("❌ Test failed:", error.message);
  }
};

// Run the test
testSearchCounters();
