import { NextRequest, NextResponse } from "next/server";
import { prisma } from "../../../lib/prisma";
import { ReactionType } from "@prisma/client";

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

    // Get reactions for this post
    const reactions = await prisma.reaction.findMany({
      where: { postId: post.id },
    });

    // Group reactions by type
    const reactionCounts = reactions.reduce(
      (acc: Record<string, number>, reaction) => {
        acc[reaction.type] = (acc[reaction.type] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );

    // Get comments count
    const commentsCount = await prisma.comment.count({
      where: { postId: post.id },
    });

    // Get shares count
    const sharesCount = await prisma.share.count({
      where: { postId: post.id },
    });

    return NextResponse.json({
      reactions: reactionCounts,
      totalReactions: reactions.length,
      totalComments: commentsCount,
      totalShares: sharesCount,
    });
  } catch (error) {
    console.error("Error fetching reactions:", error);
    return NextResponse.json({
      reactions: {},
      totalReactions: 0,
      totalComments: 0,
      totalShares: 0,
    });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { postSlug, reactionType, userEmail } = body;

    if (!postSlug || !reactionType) {
      return NextResponse.json(
        { error: "Post slug and reaction type are required" },
        { status: 400 }
      );
    }

    // Valid reaction types
    const validReactions = ["like", "love", "laugh", "wow", "sad", "angry"];
    if (!validReactions.includes(reactionType)) {
      return NextResponse.json(
        { error: "Invalid reaction type" },
        { status: 400 }
      );
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

    // Check if user already reacted
    const existingReaction = await prisma.reaction.findFirst({
      where: {
        postId: post.id,
        OR: [
          { ipAddress: ipAddress },
          ...(userEmail ? [{ userEmail: userEmail }] : []),
        ],
      },
    });

    if (existingReaction) {
      // Update existing reaction
      await prisma.reaction.update({
        where: { id: existingReaction.id },
        data: { type: reactionType as ReactionType },
      });
    } else {
      // Create new reaction
      await prisma.reaction.create({
        data: {
          type: reactionType as ReactionType,
          postId: post.id,
          ipAddress,
          userEmail: userEmail || null,
        },
      });
    }

    // Get updated reaction counts
    const reactions = await prisma.reaction.findMany({
      where: { postId: post.id },
    });

    const reactionCounts = reactions.reduce(
      (acc: Record<string, number>, reaction) => {
        acc[reaction.type] = (acc[reaction.type] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );

    return NextResponse.json({
      success: true,
      reactions: reactionCounts,
      totalReactions: reactions.length,
    });
  } catch (error) {
    console.error("Error creating reaction:", error);
    return NextResponse.json(
      { error: "Failed to create reaction" },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const postSlug = searchParams.get("postSlug");
    const userEmail = searchParams.get("userEmail");

    if (!postSlug) {
      return NextResponse.json(
        { error: "Post slug is required" },
        { status: 400 }
      );
    }

    // Get client IP
    const forwarded = request.headers.get("x-forwarded-for");
    const ipAddress = forwarded
      ? forwarded.split(",")[0]
      : request.headers.get("x-real-ip") || "unknown";

    // Find the post
    const post = await prisma.blogPost.findUnique({
      where: { slug: postSlug },
    });

    if (!post) {
      return NextResponse.json({ error: "Post not found" }, { status: 404 });
    }

    // Delete user's reaction
    await prisma.reaction.deleteMany({
      where: {
        postId: post.id,
        OR: [
          { ipAddress: ipAddress },
          ...(userEmail ? [{ userEmail: userEmail }] : []),
        ],
      },
    });

    // Get updated reaction counts
    const reactions = await prisma.reaction.findMany({
      where: { postId: post.id },
    });

    const reactionCounts = reactions.reduce(
      (acc: Record<string, number>, reaction) => {
        acc[reaction.type] = (acc[reaction.type] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );

    return NextResponse.json({
      success: true,
      reactions: reactionCounts,
      totalReactions: reactions.length,
    });
  } catch (error) {
    console.error("Error deleting reaction:", error);
    return NextResponse.json(
      { error: "Failed to delete reaction" },
      { status: 500 }
    );
  }
}
