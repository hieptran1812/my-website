"use client";

import React from "react";
import MathRenderer, { useMathParser } from "./MathRenderer";

interface MathContentProps {
  children: string;
  className?: string;
}

export default function MathContent({
  children,
  className = "",
}: MathContentProps) {
  const { parseLatex } = useMathParser();

  const parts = parseLatex(children);

  return (
    <span className={className}>
      {parts.map((part, index) => {
        switch (part.type) {
          case "display-math":
            return (
              <div key={index} className="my-4 text-center">
                <MathRenderer latex={part.content} displayMode={true} />
              </div>
            );
          case "inline-math":
            return (
              <MathRenderer
                key={index}
                latex={part.content}
                displayMode={false}
              />
            );
          case "text":
          default:
            return <span key={index}>{part.content}</span>;
        }
      })}
    </span>
  );
}
