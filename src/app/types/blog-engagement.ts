export type ReactionType = "like" | "love" | "laugh" | "wow" | "sad" | "angry";

export interface Reaction {
  type: ReactionType;
  count: number;
}

export interface Comment {
  id: string;
  content: string;
  author: string;
  email: string;
  website?: string;
  createdAt: string;
  replies?: Comment[];
}

export interface BlogEngagement {
  reactions: Record<ReactionType, number>;
  totalReactions: number;
  totalComments: number;
  totalShares: number;
}

export const reactionEmojis: Record<ReactionType, string> = {
  like: "👍",
  love: "❤️",
  laugh: "😂",
  wow: "😮",
  sad: "😢",
  angry: "😠",
};

export const reactionLabels: Record<ReactionType, string> = {
  like: "Like",
  love: "Love",
  laugh: "Laugh",
  wow: "Wow",
  sad: "Sad",
  angry: "Angry",
};

export const shareButtons = [
  {
    platform: "twitter",
    label: "Twitter",
    icon: "𝕏",
    getUrl: (url: string, title: string) =>
      `https://twitter.com/intent/tweet?url=${encodeURIComponent(
        url
      )}&text=${encodeURIComponent(title)}`,
  },
  {
    platform: "facebook",
    label: "Facebook",
    icon: "📘",
    getUrl: (url: string, _title: string) =>
      `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`,
  },
  {
    platform: "linkedin",
    label: "LinkedIn",
    icon: "💼",
    getUrl: (url: string, _title: string) =>
      `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(
        url
      )}`,
  },
  {
    platform: "copy-link",
    label: "Copy Link",
    icon: "🔗",
    getUrl: () => "", // Special case for copy functionality
  },
];
