import React from "react";
import FadeInWrapper from "./FadeInWrapper";

interface ArticleGridProps {
  children: React.ReactNode;
  variant?: "default" | "compact" | "large";
}

export default function ArticleGrid({
  children,
  variant = "default",
}: ArticleGridProps) {
  const getGridClasses = () => {
    switch (variant) {
      case "compact":
        return "grid md:grid-cols-2 lg:grid-cols-4 gap-4";
      case "large":
        return "grid md:grid-cols-1 lg:grid-cols-2 gap-8";
      default:
        return "grid md:grid-cols-2 lg:grid-cols-3 gap-6";
    }
  };

  return (
    <FadeInWrapper direction="none" duration={400}>
      <div className={getGridClasses()}>{children}</div>
    </FadeInWrapper>
  );
}
