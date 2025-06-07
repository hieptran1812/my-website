"use client";

import React from "react";

interface SkeletonLoaderProps {
  className?: string;
  width?: string | number;
  height?: string | number;
  borderRadius?: string;
}

export const SkeletonLoader: React.FC<SkeletonLoaderProps> = ({
  className = "",
  width = "100%",
  height = "20px",
  borderRadius = "4px",
}) => {
  return (
    <div
      className={`animate-pulse bg-gradient-to-r from-gray-300 via-gray-200 to-gray-300 dark:from-gray-700 dark:via-gray-600 dark:to-gray-700 ${className}`}
      style={{
        width,
        height,
        borderRadius,
        backgroundSize: "200% 100%",
        animation: "skeleton-loading 1.5s infinite ease-in-out",
      }}
    />
  );
};

// Specific skeleton components for common use cases
export const TextSkeleton: React.FC<{ lines?: number; className?: string }> = ({
  lines = 1,
  className = "",
}) => {
  return (
    <div className={`space-y-2 ${className}`}>
      {Array.from({ length: lines }).map((_, index) => (
        <SkeletonLoader
          key={index}
          height="16px"
          width={index === lines - 1 ? "75%" : "100%"}
        />
      ))}
    </div>
  );
};

export const ImageSkeleton: React.FC<{
  width?: string | number;
  height?: string | number;
  className?: string;
}> = ({ width = "100%", height = "200px", className = "" }) => {
  return (
    <SkeletonLoader
      className={className}
      width={width}
      height={height}
      borderRadius="8px"
    />
  );
};

export const CardSkeleton: React.FC<{ className?: string }> = ({
  className = "",
}) => {
  return (
    <div className={`space-y-4 p-4 ${className}`}>
      <SkeletonLoader height="20px" width="60%" />
      <TextSkeleton lines={3} />
      <SkeletonLoader height="40px" width="30%" borderRadius="8px" />
    </div>
  );
};
