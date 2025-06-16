import { NextRequest, NextResponse } from "next/server";
import { prisma } from "../../../lib/prisma";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { postSlug, platform } = body;

    if (!postSlug || !platform) {
      return NextResponse.json(
        {
          error: "Post slug and platform are required",
        },
        { status: 400 }
      );
    }

    // Valid platforms
    const validPlatforms = [
      "twitter",
      "facebook",
      "linkedin",
      "reddit",
      "telegram",
      "whatsapp",
      "email",
      "copy-link",
      "native",
    ];
    if (!validPlatforms.includes(platform)) {
      return NextResponse.json({ error: "Invalid platform" }, { status: 400 });
    }

    // Get client IP
    const forwarded = request.headers.get("x-forwarded-for");
    const ipAddress = forwarded
      ? forwarded.split(",")[0]
      : request.headers.get("x-real-ip") || "unknown";

    // Get or create blog post
    let post = await prisma.blogPost.findUnique({
      where: { slug: postSlug },
    });

    if (!post) {
      post = await prisma.blogPost.create({
        data: {
          slug: postSlug,
          title: postSlug,
        },
      });
    }

    // Create share record
    await prisma.share.create({
      data: {
        platform,
        postId: post.id,
        ipAddress,
      },
    });

    // Get updated share count
    const shareCount = await prisma.share.count({
      where: { postId: post.id },
    });

    return NextResponse.json({
      success: true,
      shareCount,
      message: "Share recorded successfully!",
    });
  } catch (error) {
    console.error("Error recording share:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const postSlug = searchParams.get("postSlug");

    if (!postSlug) {
      return NextResponse.json(
        { error: "Post slug is required" },
        { status: 400 }
      );
    }

    // Get or create blog post
    let post = await prisma.blogPost.findUnique({
      where: { slug: postSlug },
    });

    if (!post) {
      post = await prisma.blogPost.create({
        data: {
          slug: postSlug,
          title: postSlug,
        },
      });
    }

    // Get share count
    const shareCount = await prisma.share.count({
      where: { postId: post.id },
    });

    return NextResponse.json({
      shareCount,
    });
  } catch (error) {
    console.error("Error fetching share count:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
