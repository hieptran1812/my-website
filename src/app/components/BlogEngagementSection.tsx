"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useTheme } from "../ThemeProvider";
import type {
  ReactionType,
  BlogEngagement,
  Comment,
} from "../types/blog-engagement";
import {
  reactionEmojis,
  reactionLabels,
  shareButtons,
} from "../types/blog-engagement";

interface BlogEngagementSectionProps {
  postSlug: string;
  title: string;
  url?: string;
}

export default function BlogEngagementSection({
  postSlug,
  title,
  url,
}: BlogEngagementSectionProps) {
  const { theme, isReadingMode } = useTheme();
  const [engagement, setEngagement] = useState<BlogEngagement>({
    reactions: {} as Record<ReactionType, number>,
    totalReactions: 0,
    totalComments: 0,
    totalShares: 0,
  });
  const [selectedReaction, setSelectedReaction] = useState<ReactionType | null>(
    null
  );
  const [showReactions, setShowReactions] = useState(false);
  const [showComments, setShowComments] = useState(false);
  const [showShareMenu, setShowShareMenu] = useState(false);
  const [loading, setLoading] = useState(true);
  const [reactionLoading, setReactionLoading] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);

  const currentUrl =
    url || (typeof window !== "undefined" ? window.location.href : "");

  // Load initial engagement data
  const fetchEngagementData = useCallback(async () => {
    try {
      const response = await fetch(
        `/api/reactions?postSlug=${encodeURIComponent(postSlug)}`
      );
      if (response.ok) {
        const data = await response.json();
        setEngagement(data);
      }
    } catch {
      console.error("Error fetching engagement data");
    } finally {
      setLoading(false);
    }
  }, [postSlug]);

  useEffect(() => {
    fetchEngagementData();
  }, [fetchEngagementData]);

  const handleReaction = async (reactionType: ReactionType) => {
    if (reactionLoading) return;

    setReactionLoading(true);
    try {
      if (selectedReaction === reactionType) {
        // Remove reaction
        const response = await fetch(
          `/api/reactions?postSlug=${encodeURIComponent(postSlug)}`,
          { method: "DELETE" }
        );
        if (response.ok) {
          const data = await response.json();
          setEngagement((prev: BlogEngagement) => ({
            ...prev,
            reactions: data.reactions,
            totalReactions: data.totalReactions,
          }));
          setSelectedReaction(null);
        }
      } else {
        // Add or update reaction
        const response = await fetch("/api/reactions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            postSlug,
            reactionType,
          }),
        });
        if (response.ok) {
          const data = await response.json();
          setEngagement((prev: BlogEngagement) => ({
            ...prev,
            reactions: data.reactions,
            totalReactions: data.totalReactions,
          }));
          setSelectedReaction(reactionType);
        }
      }
    } catch {
      console.error("Error handling reaction");
    } finally {
      setReactionLoading(false);
    }
  };

  const handleShare = async (platform: string) => {
    if (platform === "copy-link") {
      try {
        await navigator.clipboard.writeText(currentUrl);
        setCopySuccess(true);
        setTimeout(() => setCopySuccess(false), 2000);
      } catch (err) {
        console.error("Error copying link:", err);
      }
    } else {
      const shareButton = shareButtons.find((btn) => btn.platform === platform);
      if (shareButton) {
        const shareUrl = shareButton.getUrl(currentUrl, title);
        window.open(shareUrl, "_blank", "width=600,height=400");
      }
    }

    // Record share
    try {
      const response = await fetch("/api/shares", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ postSlug, platform }),
      });
      if (response.ok) {
        const data = await response.json();
        setEngagement((prev: BlogEngagement) => ({
          ...prev,
          totalShares: data.shareCount,
        }));
      }
    } catch {
      console.error("Error recording share");
    }

    setShowShareMenu(false);
  };

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-16 bg-gray-200 dark:bg-gray-700 rounded-lg"></div>
      </div>
    );
  }

  return (
    <div
      className="border-t pt-8 mt-12"
      style={{
        borderColor: isReadingMode
          ? theme === "dark"
            ? "#52403d"
            : "#fef3c7"
          : "var(--border)",
      }}
    >
      {/* Engagement Stats */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span
              className="text-sm font-medium"
              style={{
                color: isReadingMode
                  ? theme === "dark"
                    ? "#e8d5b7"
                    : "#92400e"
                  : "var(--text-primary)",
              }}
            >
              {engagement.totalReactions} reactions
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className="text-sm font-medium"
              style={{
                color: isReadingMode
                  ? theme === "dark"
                    ? "#e8d5b7"
                    : "#92400e"
                  : "var(--text-primary)",
              }}
            >
              {engagement.totalComments} comments
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className="text-sm font-medium"
              style={{
                color: isReadingMode
                  ? theme === "dark"
                    ? "#e8d5b7"
                    : "#92400e"
                  : "var(--text-primary)",
              }}
            >
              {engagement.totalShares} shares
            </span>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div
        className="flex items-center gap-4 pb-6 border-b"
        style={{
          borderColor: isReadingMode
            ? theme === "dark"
              ? "#52403d"
              : "#fef3c7"
            : "var(--border)",
        }}
      >
        {/* Reactions Button */}
        <div className="relative">
          <button
            onClick={() => setShowReactions(!showReactions)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${
              selectedReaction ? "font-medium" : ""
            }`}
            style={{
              backgroundColor: selectedReaction
                ? isReadingMode
                  ? theme === "dark"
                    ? "#52403d"
                    : "#fef3c7"
                  : "var(--surface-accent)"
                : "transparent",
              color: selectedReaction
                ? isReadingMode
                  ? theme === "dark"
                    ? "#fbbf24"
                    : "#92400e"
                  : "var(--accent)"
                : isReadingMode
                ? theme === "dark"
                  ? "#e8d5b7"
                  : "#78350f"
                : "var(--text-secondary)",
            }}
            onMouseEnter={(e) => {
              if (!selectedReaction) {
                e.currentTarget.style.backgroundColor = isReadingMode
                  ? theme === "dark"
                    ? "rgba(82, 64, 61, 0.3)"
                    : "rgba(254, 243, 199, 0.5)"
                  : "var(--surface)";
                e.currentTarget.style.color = isReadingMode
                  ? theme === "dark"
                    ? "#fbbf24"
                    : "#92400e"
                  : "var(--text-primary)";
              }
            }}
            onMouseLeave={(e) => {
              if (!selectedReaction) {
                e.currentTarget.style.backgroundColor = "transparent";
                e.currentTarget.style.color = isReadingMode
                  ? theme === "dark"
                    ? "#e8d5b7"
                    : "#78350f"
                  : "var(--text-secondary)";
              }
            }}
          >
            <span className="text-lg">
              {selectedReaction ? reactionEmojis[selectedReaction] : "👍"}
            </span>
            <span className="text-sm">
              {selectedReaction ? reactionLabels[selectedReaction] : "React"}
            </span>
            {engagement.totalReactions > 0 && (
              <span className="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded-full">
                {engagement.totalReactions}
              </span>
            )}
          </button>

          {/* Reaction Picker */}
          {showReactions && (
            <div
              className="absolute bottom-full left-0 mb-2 p-2 rounded-lg shadow-lg border backdrop-blur-md z-50"
              style={{
                backgroundColor: isReadingMode
                  ? theme === "dark"
                    ? "rgba(41, 37, 36, 0.95)"
                    : "rgba(255, 251, 235, 0.95)"
                  : "var(--background)/95",
                borderColor: isReadingMode
                  ? theme === "dark"
                    ? "#52403d"
                    : "#fef3c7"
                  : "var(--border)",
              }}
            >
              <div className="flex items-center gap-1">
                {(Object.keys(reactionEmojis) as ReactionType[]).map(
                  (reactionType) => (
                    <button
                      key={reactionType}
                      onClick={() => handleReaction(reactionType)}
                      disabled={reactionLoading}
                      className={`flex flex-col items-center gap-1 p-2 rounded-lg transition-all duration-200 hover:scale-110 ${
                        selectedReaction === reactionType
                          ? "bg-blue-100 dark:bg-blue-900"
                          : ""
                      }`}
                      style={{
                        backgroundColor:
                          selectedReaction === reactionType
                            ? isReadingMode
                              ? theme === "dark"
                                ? "#52403d"
                                : "#fef3c7"
                              : "var(--surface-accent)"
                            : "transparent",
                      }}
                      title={reactionLabels[reactionType]}
                    >
                      <span className="text-lg">
                        {reactionEmojis[reactionType]}
                      </span>
                      <span
                        className="text-xs"
                        style={{
                          color: isReadingMode
                            ? theme === "dark"
                              ? "#e8d5b7"
                              : "#78350f"
                            : "var(--text-secondary)",
                        }}
                      >
                        {engagement.reactions[reactionType] || 0}
                      </span>
                    </button>
                  )
                )}
              </div>
            </div>
          )}
        </div>

        {/* Comments Button */}
        <button
          onClick={() => setShowComments(!showComments)}
          className="flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200"
          style={{
            color: isReadingMode
              ? theme === "dark"
                ? "#e8d5b7"
                : "#78350f"
              : "var(--text-secondary)",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.backgroundColor = isReadingMode
              ? theme === "dark"
                ? "rgba(82, 64, 61, 0.3)"
                : "rgba(254, 243, 199, 0.5)"
              : "var(--surface)";
            e.currentTarget.style.color = isReadingMode
              ? theme === "dark"
                ? "#fbbf24"
                : "#92400e"
              : "var(--text-primary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.backgroundColor = "transparent";
            e.currentTarget.style.color = isReadingMode
              ? theme === "dark"
                ? "#e8d5b7"
                : "#78350f"
              : "var(--text-secondary)";
          }}
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
            />
          </svg>
          <span className="text-sm">Comment</span>
          {engagement.totalComments > 0 && (
            <span className="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded-full">
              {engagement.totalComments}
            </span>
          )}
        </button>

        {/* Share Button */}
        <div className="relative">
          <button
            onClick={() => setShowShareMenu(!showShareMenu)}
            className="flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200"
            style={{
              color: isReadingMode
                ? theme === "dark"
                  ? "#e8d5b7"
                  : "#78350f"
                : "var(--text-secondary)",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = isReadingMode
                ? theme === "dark"
                  ? "rgba(82, 64, 61, 0.3)"
                  : "rgba(254, 243, 199, 0.5)"
                : "var(--surface)";
              e.currentTarget.style.color = isReadingMode
                ? theme === "dark"
                  ? "#fbbf24"
                  : "#92400e"
                : "var(--text-primary)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = "transparent";
              e.currentTarget.style.color = isReadingMode
                ? theme === "dark"
                  ? "#e8d5b7"
                  : "#78350f"
                : "var(--text-secondary)";
            }}
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z"
              />
            </svg>
            <span className="text-sm">Share</span>
            {engagement.totalShares > 0 && (
              <span className="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded-full">
                {engagement.totalShares}
              </span>
            )}
          </button>

          {/* Share Menu */}
          {showShareMenu && (
            <div
              className="absolute bottom-full left-0 mb-2 p-2 rounded-lg shadow-lg border backdrop-blur-md z-50 min-w-[200px]"
              style={{
                backgroundColor: isReadingMode
                  ? theme === "dark"
                    ? "rgba(41, 37, 36, 0.95)"
                    : "rgba(255, 251, 235, 0.95)"
                  : "var(--background)/95",
                borderColor: isReadingMode
                  ? theme === "dark"
                    ? "#52403d"
                    : "#fef3c7"
                  : "var(--border)",
              }}
            >
              {shareButtons.map((button) => (
                <button
                  key={button.platform}
                  onClick={() => handleShare(button.platform)}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200 text-left"
                  style={{
                    color: isReadingMode
                      ? theme === "dark"
                        ? "#e8d5b7"
                        : "#78350f"
                      : "var(--text-secondary)",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = isReadingMode
                      ? theme === "dark"
                        ? "rgba(82, 64, 61, 0.5)"
                        : "rgba(254, 243, 199, 0.7)"
                      : "var(--surface)";
                    e.currentTarget.style.color = isReadingMode
                      ? theme === "dark"
                        ? "#fbbf24"
                        : "#92400e"
                      : "var(--text-primary)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = "transparent";
                    e.currentTarget.style.color = isReadingMode
                      ? theme === "dark"
                        ? "#e8d5b7"
                        : "#78350f"
                      : "var(--text-secondary)";
                  }}
                >
                  <span className="text-lg">{button.icon}</span>
                  <span className="text-sm">
                    {button.platform === "copy-link" && copySuccess
                      ? "Copied!"
                      : button.label}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Comments Section */}
      {showComments && (
        <div className="mt-6">
          <CommentSection
            postSlug={postSlug}
            onCommentAdded={fetchEngagementData}
          />
        </div>
      )}

      {/* Click outside handlers */}
      {(showReactions || showShareMenu) && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => {
            setShowReactions(false);
            setShowShareMenu(false);
          }}
        />
      )}
    </div>
  );
}

// Comment Section Component
interface CommentSectionProps {
  postSlug: string;
  onCommentAdded: () => void;
}

function CommentSection({ postSlug, onCommentAdded }: CommentSectionProps) {
  const { theme, isReadingMode } = useTheme();
  const [comments, setComments] = useState<Comment[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCommentForm, setShowCommentForm] = useState(false);

  const fetchComments = useCallback(async () => {
    try {
      const response = await fetch(
        `/api/comments?postSlug=${encodeURIComponent(postSlug)}`
      );
      if (response.ok) {
        const data = await response.json();
        setComments(data.comments);
      }
    } catch (err) {
      console.error("Error fetching comments:", err);
    } finally {
      setLoading(false);
    }
  }, [postSlug]);

  useEffect(() => {
    fetchComments();
  }, [fetchComments]);

  const handleCommentAdded = () => {
    fetchComments();
    onCommentAdded();
    setShowCommentForm(false);
  };

  if (loading) {
    return (
      <div className="space-y-4">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="animate-pulse">
            <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/4 mb-2"></div>
            <div className="h-16 bg-gray-200 dark:bg-gray-700 rounded"></div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3
          className="text-lg font-semibold"
          style={{
            color: isReadingMode
              ? theme === "dark"
                ? "#f5e6d3"
                : "#92400e"
              : "var(--text-primary)",
          }}
        >
          Comments ({comments.length})
        </h3>
        <button
          onClick={() => setShowCommentForm(!showCommentForm)}
          className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
          style={{
            backgroundColor: isReadingMode
              ? theme === "dark"
                ? "#52403d"
                : "#fef3c7"
              : "var(--surface-accent)",
            color: isReadingMode
              ? theme === "dark"
                ? "#fbbf24"
                : "#92400e"
              : "var(--accent)",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = "translateY(-1px)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
          }}
        >
          {showCommentForm ? "Cancel" : "Add Comment"}
        </button>
      </div>

      {showCommentForm && (
        <CommentForm postSlug={postSlug} onCommentAdded={handleCommentAdded} />
      )}

      <div className="space-y-4">
        {comments.map((comment) => (
          <CommentItem key={comment.id} comment={comment} />
        ))}
        {comments.length === 0 && !loading && (
          <div
            className="text-center py-8"
            style={{
              color: isReadingMode
                ? theme === "dark"
                  ? "#d97706"
                  : "#78350f"
                : "var(--text-secondary)",
            }}
          >
            <p>No comments yet. Be the first to share your thoughts!</p>
          </div>
        )}
      </div>
    </div>
  );
}

// Comment Form Component
interface CommentFormProps {
  postSlug: string;
  onCommentAdded: () => void;
  parentId?: string;
}

function CommentForm({ postSlug, onCommentAdded, parentId }: CommentFormProps) {
  const { theme, isReadingMode } = useTheme();
  const [formData, setFormData] = useState({
    author: "",
    email: "",
    website: "",
    content: "",
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submitting) return;

    setSubmitting(true);
    setError("");

    try {
      const response = await fetch("/api/comments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...formData,
          postSlug,
          parentId,
        }),
      });

      if (response.ok) {
        setFormData({ author: "", email: "", website: "", content: "" });
        onCommentAdded();
      } else {
        const data = await response.json();
        setError(data.error || "Failed to post comment");
      }
    } catch {
      setError("Network error. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {error && (
        <div className="p-3 bg-red-100 border border-red-400 text-red-700 rounded-lg text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label
            className="block text-sm font-medium mb-2"
            style={{
              color: isReadingMode
                ? theme === "dark"
                  ? "#e8d5b7"
                  : "#92400e"
                : "var(--text-primary)",
            }}
          >
            Name *
          </label>
          <input
            type="text"
            required
            value={formData.author}
            onChange={(e) =>
              setFormData({ ...formData, author: e.target.value })
            }
            className="w-full px-3 py-2 rounded-lg border transition-colors duration-200"
            style={{
              backgroundColor: isReadingMode
                ? theme === "dark"
                  ? "rgba(82, 64, 61, 0.2)"
                  : "rgba(254, 243, 199, 0.3)"
                : "var(--surface)",
              borderColor: isReadingMode
                ? theme === "dark"
                  ? "#52403d"
                  : "#fef3c7"
                : "var(--border)",
              color: isReadingMode
                ? theme === "dark"
                  ? "#f5e6d3"
                  : "#92400e"
                : "var(--text-primary)",
            }}
          />
        </div>

        <div>
          <label
            className="block text-sm font-medium mb-2"
            style={{
              color: isReadingMode
                ? theme === "dark"
                  ? "#e8d5b7"
                  : "#92400e"
                : "var(--text-primary)",
            }}
          >
            Email *
          </label>
          <input
            type="email"
            required
            value={formData.email}
            onChange={(e) =>
              setFormData({ ...formData, email: e.target.value })
            }
            className="w-full px-3 py-2 rounded-lg border transition-colors duration-200"
            style={{
              backgroundColor: isReadingMode
                ? theme === "dark"
                  ? "rgba(82, 64, 61, 0.2)"
                  : "rgba(254, 243, 199, 0.3)"
                : "var(--surface)",
              borderColor: isReadingMode
                ? theme === "dark"
                  ? "#52403d"
                  : "#fef3c7"
                : "var(--border)",
              color: isReadingMode
                ? theme === "dark"
                  ? "#f5e6d3"
                  : "#92400e"
                : "var(--text-primary)",
            }}
          />
        </div>
      </div>

      <div>
        <label
          className="block text-sm font-medium mb-2"
          style={{
            color: isReadingMode
              ? theme === "dark"
                ? "#e8d5b7"
                : "#92400e"
              : "var(--text-primary)",
          }}
        >
          Website (optional)
        </label>
        <input
          type="url"
          value={formData.website}
          onChange={(e) =>
            setFormData({ ...formData, website: e.target.value })
          }
          className="w-full px-3 py-2 rounded-lg border transition-colors duration-200"
          style={{
            backgroundColor: isReadingMode
              ? theme === "dark"
                ? "rgba(82, 64, 61, 0.2)"
                : "rgba(254, 243, 199, 0.3)"
              : "var(--surface)",
            borderColor: isReadingMode
              ? theme === "dark"
                ? "#52403d"
                : "#fef3c7"
              : "var(--border)",
            color: isReadingMode
              ? theme === "dark"
                ? "#f5e6d3"
                : "#92400e"
              : "var(--text-primary)",
          }}
        />
      </div>

      <div>
        <label
          className="block text-sm font-medium mb-2"
          style={{
            color: isReadingMode
              ? theme === "dark"
                ? "#e8d5b7"
                : "#92400e"
              : "var(--text-primary)",
          }}
        >
          Comment *
        </label>
        <textarea
          required
          rows={4}
          value={formData.content}
          onChange={(e) =>
            setFormData({ ...formData, content: e.target.value })
          }
          placeholder="Share your thoughts..."
          className="w-full px-3 py-2 rounded-lg border transition-colors duration-200 resize-none"
          style={{
            backgroundColor: isReadingMode
              ? theme === "dark"
                ? "rgba(82, 64, 61, 0.2)"
                : "rgba(254, 243, 199, 0.3)"
              : "var(--surface)",
            borderColor: isReadingMode
              ? theme === "dark"
                ? "#52403d"
                : "#fef3c7"
              : "var(--border)",
            color: isReadingMode
              ? theme === "dark"
                ? "#f5e6d3"
                : "#92400e"
              : "var(--text-primary)",
          }}
        />
        <div
          className="text-xs mt-1"
          style={{
            color: isReadingMode
              ? theme === "dark"
                ? "#d97706"
                : "#78350f"
              : "var(--text-secondary)",
          }}
        >
          {formData.content.length}/1000 characters
        </div>
      </div>

      <div className="flex justify-end">
        <button
          type="submit"
          disabled={
            submitting ||
            !formData.content.trim() ||
            !formData.author.trim() ||
            !formData.email.trim()
          }
          className="px-6 py-2 rounded-lg font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          style={{
            backgroundColor: isReadingMode
              ? theme === "dark"
                ? "#fbbf24"
                : "#92400e"
              : "var(--accent)",
            color: "white",
          }}
          onMouseEnter={(e) => {
            if (!e.currentTarget.disabled) {
              e.currentTarget.style.transform = "translateY(-1px)";
            }
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
          }}
        >
          {submitting ? "Posting..." : "Post Comment"}
        </button>
      </div>
    </form>
  );
}

// Comment Item Component
interface CommentItemProps {
  comment: Comment;
}

function CommentItem({ comment }: CommentItemProps) {
  const { theme, isReadingMode } = useTheme();
  const [showReplyForm, setShowReplyForm] = useState(false);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div
      className="p-4 rounded-lg border"
      style={{
        backgroundColor: isReadingMode
          ? theme === "dark"
            ? "rgba(82, 64, 61, 0.1)"
            : "rgba(254, 243, 199, 0.2)"
          : "var(--surface)",
        borderColor: isReadingMode
          ? theme === "dark"
            ? "#52403d"
            : "#fef3c7"
          : "var(--border)",
      }}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold"
            style={{ backgroundColor: "var(--accent)" }}
          >
            {comment.author.charAt(0).toUpperCase()}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span
                className="font-medium"
                style={{
                  color: isReadingMode
                    ? theme === "dark"
                      ? "#f5e6d3"
                      : "#92400e"
                    : "var(--text-primary)",
                }}
              >
                {comment.author}
              </span>
              {comment.website && (
                <a
                  href={comment.website}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs"
                  style={{
                    color: isReadingMode
                      ? theme === "dark"
                        ? "#fbbf24"
                        : "#b45309"
                      : "var(--accent)",
                  }}
                >
                  🌐
                </a>
              )}
            </div>
            <time
              className="text-xs"
              style={{
                color: isReadingMode
                  ? theme === "dark"
                    ? "#d97706"
                    : "#78350f"
                  : "var(--text-secondary)",
              }}
            >
              {formatDate(comment.createdAt)}
            </time>
          </div>
        </div>
      </div>

      <div
        className="mb-3 leading-relaxed"
        style={{
          color: isReadingMode
            ? theme === "dark"
              ? "#e8d5b7"
              : "#92400e"
            : "var(--text-primary)",
        }}
      >
        {comment.content}
      </div>

      <div className="flex items-center gap-4">
        <button
          onClick={() => setShowReplyForm(!showReplyForm)}
          className="text-sm transition-colors duration-200"
          style={{
            color: isReadingMode
              ? theme === "dark"
                ? "#d97706"
                : "#78350f"
              : "var(--text-secondary)",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = isReadingMode
              ? theme === "dark"
                ? "#fbbf24"
                : "#92400e"
              : "var(--accent)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = isReadingMode
              ? theme === "dark"
                ? "#d97706"
                : "#78350f"
              : "var(--text-secondary)";
          }}
        >
          Reply
        </button>
      </div>

      {showReplyForm && (
        <div className="mt-4">
          <CommentForm
            postSlug="current"
            parentId={comment.id}
            onCommentAdded={() => setShowReplyForm(false)}
          />
        </div>
      )}

      {comment.replies && comment.replies.length > 0 && (
        <div className="mt-4 ml-8 space-y-4">
          {comment.replies.map((reply: Comment) => (
            <CommentItem key={reply.id} comment={reply} />
          ))}
        </div>
      )}
    </div>
  );
}
