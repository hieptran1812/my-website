// Test script to verify subcategories
const articles = [
  // Machine Learning articles
  { category: "machine-learning", subcategory: "Reinforcement Learning" },
  { category: "machine-learning", subcategory: "Traditional ML" },
  { category: "machine-learning", subcategory: "LLM" },
  { category: "machine-learning", subcategory: "MLOps" },
  { category: "machine-learning", subcategory: "Neural Architecture" },
  { category: "machine-learning", subcategory: "Deep Learning" },
  { category: "machine-learning", subcategory: "Optimization" },
  { category: "machine-learning", subcategory: "Traditional ML" },

  // Notes articles
  { category: "notes", subcategory: "Book Summaries" },
  { category: "notes", subcategory: "Idea Dump" },
  { category: "notes", subcategory: "Self-reflection Entries" },

  // Paper Reading articles
  { category: "paper-reading", subcategory: "AI Agent" },
  { category: "paper-reading", subcategory: "Multimodal" },
  { category: "paper-reading", subcategory: "LLM" },
  { category: "paper-reading", subcategory: "Multimodal" },
  { category: "paper-reading", subcategory: "Computer Vision" },
  { category: "paper-reading", subcategory: "AI Interpretability" },
  { category: "paper-reading", subcategory: "LLM" },
  { category: "paper-reading", subcategory: "Machine Learning" },
  { category: "paper-reading", subcategory: "AI Agent" },
  { category: "paper-reading", subcategory: "Computer Vision" },
  { category: "paper-reading", subcategory: "Speech Processing" },
  { category: "paper-reading", subcategory: "Speech Processing" },

  // Software Development articles
  { category: "software-development", subcategory: "Algorithms" },
  { category: "software-development", subcategory: "Data Engineering" },
  { category: "software-development", subcategory: "Database" },
  { category: "software-development", subcategory: "Distributed Systems" },
  {
    category: "software-development",
    subcategory: "Site Reliability Engineering",
  },
  { category: "software-development", subcategory: "Coding Practices" },
  { category: "software-development", subcategory: "System Design" },
  { category: "software-development", subcategory: "Coding Practices" },
];

function getSubcategories(category) {
  const subcategoryMap = new Map();

  articles
    .filter((article) => !category || article.category === category)
    .forEach((article) => {
      if (article.subcategory) {
        subcategoryMap.set(
          article.subcategory,
          (subcategoryMap.get(article.subcategory) || 0) + 1
        );
      }
    });

  return Array.from(subcategoryMap.entries()).map(([subcategory, count]) => ({
    name: subcategory,
    slug: subcategory,
    count,
  }));
}

console.log("Machine Learning subcategories:");
console.log(getSubcategories("machine-learning"));

console.log("\nNotes subcategories:");
console.log(getSubcategories("notes"));

console.log("\nPaper Reading subcategories:");
console.log(getSubcategories("paper-reading"));

console.log("\nSoftware Development subcategories:");
console.log(getSubcategories("software-development"));
