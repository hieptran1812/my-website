// Quick test script to verify search functionality
const testSearchEndpoints = async () => {
  const baseUrl = "http://localhost:3002";

  console.log("🧪 Testing Search Optimization Implementation...\n");

  // Test 1: Projects API
  try {
    console.log("1️⃣ Testing Projects API...");
    const projectsResponse = await fetch(`${baseUrl}/api/projects`);
    const projects = await projectsResponse.json();
    console.log(`✅ Projects API: Found ${projects.length} projects`);
    console.log(
      `   Featured projects: ${projects.filter((p) => p.featured).length}`
    );
    console.log(
      `   Categories: ${[...new Set(projects.map((p) => p.category))].join(
        ", "
      )}\n`
    );
  } catch (error) {
    console.log(`❌ Projects API failed: ${error.message}\n`);
  }

  // Test 2: Articles API
  try {
    console.log("2️⃣ Testing Articles API...");
    const articlesResponse = await fetch(`${baseUrl}/api/articles`);
    const articles = await articlesResponse.json();
    console.log(`✅ Articles API: Found ${articles.length} articles`);
    console.log(
      `   Featured articles: ${articles.filter((a) => a.featured).length}`
    );
    console.log(
      `   Categories: ${[...new Set(articles.map((a) => a.category))].join(
        ", "
      )}\n`
    );
  } catch (error) {
    console.log(`❌ Articles API failed: ${error.message}\n`);
  }

  // Test 3: Search page accessibility
  try {
    console.log("3️⃣ Testing Search Page...");
    const searchResponse = await fetch(`${baseUrl}/search`);
    if (searchResponse.ok) {
      console.log("✅ Search page is accessible\n");
    } else {
      console.log(`❌ Search page returned status: ${searchResponse.status}\n`);
    }
  } catch (error) {
    console.log(`❌ Search page failed: ${error.message}\n`);
  }

  console.log("🎉 Search optimization testing complete!");
  console.log(
    "🔍 Visit http://localhost:3002/search to test search functionality manually"
  );
};

// Run if this is called directly
if (typeof window === "undefined") {
  testSearchEndpoints().catch(console.error);
}
