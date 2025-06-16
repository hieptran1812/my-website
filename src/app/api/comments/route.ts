import { NextRequest, NextResponse } from "next/server";
import { prisma } from "../../../lib/prisma";

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

    // Get comments for this post
    const comments = await prisma.comment.findMany({
      where: {
        postId: post.id,
        parentId: null,
      },
      include: {
        replies: {
          orderBy: { createdAt: "asc" },
        },
      },
      orderBy: { createdAt: "desc" },
    });

    // Transform to match expected format
    const organizedComments = comments.map((comment) => ({
      id: comment.id,
      content: comment.content,
      author: comment.author,
      email: comment.email,
      website: comment.website,
      createdAt: comment.createdAt.toISOString(),
      replies: comment.replies.map((reply) => ({
        id: reply.id,
        content: reply.content,
        author: reply.author,
        email: reply.email,
        website: reply.website,
        createdAt: reply.createdAt.toISOString(),
      })),
    }));

    return NextResponse.json({ comments: organizedComments });
  } catch (error) {
    console.error("Error fetching comments:", error);
    return NextResponse.json({ comments: [] });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { postSlug, content, author, email, website, parentId } = body;

    if (!postSlug || !content || !author || !email) {
      return NextResponse.json(
        { error: "Post slug, content, author, and email are required" },
        { status: 400 }
      );
    }

    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { error: "Invalid email format" },
        { status: 400 }
      );
    }

    // Content length validation
    if (content.length > 1000) {
      return NextResponse.json(
        { error: "Comment is too long (max 1000 characters)" },
        { status: 400 }
      );
    }

    // Get client IP for tracking
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

    // If parentId is provided, verify parent comment exists
    if (parentId) {
      const parentComment = await prisma.comment.findUnique({
        where: { id: parentId },
      });

      if (!parentComment) {
        return NextResponse.json(
          { error: "Parent comment not found" },
          { status: 404 }
        );
      }
    }

    // Create new comment
    const newComment = await prisma.comment.create({
      data: {
        content: content.trim(),
        author: author.trim(),
        email: email.trim(),
        website: website?.trim() || null,
        ipAddress,
        postId: post.id,
        parentId: parentId || null,
      },
      include: {
        replies: true,
      },
    });

    return NextResponse.json(
      {
        comment: {
          id: newComment.id,
          content: newComment.content,
          author: newComment.author,
          email: newComment.email,
          website: newComment.website,
          createdAt: newComment.createdAt.toISOString(),
          replies: [],
        },
        message: "Comment added successfully!",
      },
      { status: 201 }
    );
  } catch (error) {
    console.error("Error creating comment:", error);
    return NextResponse.json(
      { error: "Failed to create comment" },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const commentId = searchParams.get("commentId");

    if (!commentId) {
      return NextResponse.json(
        { error: "Comment ID is required" },
        { status: 400 }
      );
    }

    const comment = await prisma.comment.findUnique({
      where: { id: commentId },
    });

    if (!comment) {
      return NextResponse.json({ error: "Comment not found" }, { status: 404 });
    }

    await prisma.comment.delete({
      where: { id: commentId },
    });

    return NextResponse.json({
      message: "Comment deleted successfully!",
    });
  } catch (error) {
    console.error("Error deleting comment:", error);
    return NextResponse.json(
      { error: "Failed to delete comment" },
      { status: 500 }
    );
  }
}
