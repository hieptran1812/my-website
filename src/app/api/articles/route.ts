import { NextRequest, NextResponse } from "next/server";
import {
  getAllArticles,
  getArticleById,
  getArticlesByCategory,
  getFeaturedArticles,
} from "@/data/articles";

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);

    // Get query parameters
    const category = searchParams.get("category");
    const featured = searchParams.get("featured");
    const limit = searchParams.get("limit");
    const id = searchParams.get("id");

    let result = getAllArticles();

    // Filter by ID if provided
    if (id) {
      const article = getArticleById(id);
      return NextResponse.json({
        data: article ? [article] : [],
        total: article ? 1 : 0,
      });
    }

    // Filter by featured status
    if (featured === "true") {
      result = getFeaturedArticles();
    }

    // Filter by category
    if (category) {
      result = getArticlesByCategory(category);
    }

    // Apply limit
    if (limit) {
      const limitNum = parseInt(limit, 10);
      if (!isNaN(limitNum) && limitNum > 0) {
        result = result.slice(0, limitNum);
      }
    }

    return NextResponse.json({
      data: result,
      total: result.length,
    });
  } catch (error) {
    console.error("Error fetching articles:", error);
    return NextResponse.json(
      { error: "Failed to fetch articles" },
      { status: 500 }
    );
  }
}
