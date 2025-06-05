import React, { ReactNode } from "react";
import { useIntersectionObserver } from "./hooks/useIntersectionObserver";

interface FadeInWrapperProps {
  children: ReactNode;
  delay?: number;
  duration?: number;
  threshold?: number;
  className?: string;
  direction?: "up" | "down" | "left" | "right" | "none";
  distance?: number;
}

export default function FadeInWrapper({
  children,
  delay = 0,
  duration = 600,
  threshold = 0.1,
  className = "",
  direction = "up",
  distance = 20,
}: FadeInWrapperProps) {
  const { elementRef, isVisible } = useIntersectionObserver<HTMLDivElement>({
    threshold,
    freezeOnceVisible: true,
  });

  const getTransform = () => {
    if (direction === "none") return "none";

    const transforms = {
      up: `translateY(${distance}px)`,
      down: `translateY(-${distance}px)`,
      left: `translateX(${distance}px)`,
      right: `translateX(-${distance}px)`,
    };

    return transforms[direction];
  };

  return (
    <div
      ref={elementRef}
      className={`transition-all ease-out ${className}`}
      style={{
        opacity: isVisible ? 1 : 0,
        transform: isVisible ? "none" : getTransform(),
        transitionDuration: `${duration}ms`,
        transitionDelay: `${delay}ms`,
      }}
    >
      {children}
    </div>
  );
}
