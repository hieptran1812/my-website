import React from "react";
import { useIntersectionObserver } from "./hooks/useIntersectionObserver";

interface LoadMoreTriggerProps {
  onLoadMore: () => void;
  loading: boolean;
  hasMore: boolean;
  threshold?: number;
}

export default function LoadMoreTrigger({
  onLoadMore,
  loading,
  hasMore,
  threshold = 0.1,
}: LoadMoreTriggerProps) {
  const { elementRef, isVisible } = useIntersectionObserver<HTMLDivElement>({
    threshold,
    freezeOnceVisible: false,
  });

  React.useEffect(() => {
    if (isVisible && hasMore && !loading) {
      onLoadMore();
    }
  }, [isVisible, hasMore, loading, onLoadMore]);

  if (!hasMore) {
    return (
      <div className="text-center py-8">
        <div
          className="text-lg font-medium mb-2"
          style={{ color: "var(--text-secondary)" }}
        >
          🎉 You&apos;ve seen all the articles!
        </div>
        <p className="text-sm" style={{ color: "var(--text-muted)" }}>
          Halley is cooking up more content. Stay tuned!
        </p>
      </div>
    );
  }

  return (
    <div ref={elementRef} className="flex justify-center py-8">
      {loading ? (
        <div className="flex flex-col items-center">
          <div
            className="inline-block animate-spin rounded-full h-8 w-8 border-b-2"
            style={{ borderColor: "var(--accent)" }}
          ></div>
          <p
            className="mt-4 text-sm"
            style={{ color: "var(--text-secondary)" }}
          >
            Loading more articles...
          </p>
        </div>
      ) : (
        <button
          onClick={onLoadMore}
          className="px-6 py-3 rounded-lg font-medium transition-all duration-200 hover:scale-105 border"
          style={{
            backgroundColor: "var(--surface)",
            color: "var(--text-primary)",
            borderColor: "var(--border)",
          }}
        >
          Load More Articles
        </button>
      )}
    </div>
  );
}
