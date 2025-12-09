"use client";

import React, { ReactNode, useEffect, useRef, useState } from "react";

type AnimationType =
  | "fade-up"
  | "fade-down"
  | "fade-left"
  | "fade-right"
  | "zoom-in"
  | "zoom-out"
  | "flip-up"
  | "flip-down"
  | "slide-up"
  | "slide-down"
  | "slide-left"
  | "slide-right"
  | "rotate-in"
  | "blur-in"
  | "bounce-in"
  | "scale-up";

interface ScrollRevealProps {
  children: ReactNode;
  animation?: AnimationType;
  delay?: number;
  duration?: number;
  threshold?: number;
  className?: string;
  distance?: number;
  once?: boolean;
  easing?: string;
  blur?: number;
  scale?: number;
}

export default function ScrollReveal({
  children,
  animation = "fade-up",
  delay = 0,
  duration = 700,
  threshold = 0.1,
  className = "",
  distance = 40,
  once = true,
  easing = "cubic-bezier(0.16, 1, 0.3, 1)",
  blur = 10,
  scale = 0.95,
}: ScrollRevealProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [hasAnimated, setHasAnimated] = useState(false);
  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          if (!hasAnimated) {
            setHasAnimated(true);
          }
          if (once) {
            observer.unobserve(element);
          }
        } else if (!once) {
          setIsVisible(false);
        }
      },
      { threshold, rootMargin: "0px 0px -50px 0px" }
    );

    observer.observe(element);

    return () => observer.disconnect();
  }, [threshold, once, hasAnimated]);

  const getInitialStyles = (): React.CSSProperties => {
    const baseStyles: React.CSSProperties = {
      opacity: 0,
      willChange: "transform, opacity, filter",
      backfaceVisibility: "hidden",
    };

    switch (animation) {
      case "fade-up":
        return { ...baseStyles, transform: `translateY(${distance}px)` };
      case "fade-down":
        return { ...baseStyles, transform: `translateY(-${distance}px)` };
      case "fade-left":
        return { ...baseStyles, transform: `translateX(${distance}px)` };
      case "fade-right":
        return { ...baseStyles, transform: `translateX(-${distance}px)` };
      case "zoom-in":
        return { ...baseStyles, transform: `scale(${scale})` };
      case "zoom-out":
        return { ...baseStyles, transform: `scale(${2 - scale})` };
      case "flip-up":
        return {
          ...baseStyles,
          transform: `perspective(1000px) rotateX(45deg) translateY(${distance}px)`,
        };
      case "flip-down":
        return {
          ...baseStyles,
          transform: `perspective(1000px) rotateX(-45deg) translateY(-${distance}px)`,
        };
      case "slide-up":
        return {
          ...baseStyles,
          transform: `translateY(${distance * 1.5}px)`,
          opacity: 0,
        };
      case "slide-down":
        return {
          ...baseStyles,
          transform: `translateY(-${distance * 1.5}px)`,
          opacity: 0,
        };
      case "slide-left":
        return {
          ...baseStyles,
          transform: `translateX(${distance * 1.5}px)`,
          opacity: 0,
        };
      case "slide-right":
        return {
          ...baseStyles,
          transform: `translateX(-${distance * 1.5}px)`,
          opacity: 0,
        };
      case "rotate-in":
        return {
          ...baseStyles,
          transform: `rotate(-10deg) scale(${scale})`,
        };
      case "blur-in":
        return {
          ...baseStyles,
          transform: `translateY(${distance / 2}px) scale(${scale})`,
          filter: `blur(${blur}px)`,
        };
      case "bounce-in":
        return {
          ...baseStyles,
          transform: `translateY(${distance}px) scale(${scale})`,
        };
      case "scale-up":
        return {
          ...baseStyles,
          transform: `scale(0.8)`,
        };
      default:
        return baseStyles;
    }
  };

  const getVisibleStyles = (): React.CSSProperties => {
    return {
      opacity: 1,
      transform: "translateY(0) translateX(0) scale(1) rotate(0deg)",
      filter: "blur(0px)",
      willChange: "auto",
    };
  };

  const shouldAnimate = once ? hasAnimated || isVisible : isVisible;

  return (
    <div
      ref={elementRef}
      className={className}
      style={{
        ...(shouldAnimate ? getVisibleStyles() : getInitialStyles()),
        transitionProperty: "transform, opacity, filter",
        transitionDuration: `${duration}ms`,
        transitionDelay: `${delay}ms`,
        transitionTimingFunction: easing,
      }}
    >
      {children}
    </div>
  );
}

// Stagger container for sequential animations
interface StaggerContainerProps {
  children: ReactNode;
  className?: string;
  staggerDelay?: number;
  animation?: AnimationType;
  baseDelay?: number;
  duration?: number;
}

export function StaggerContainer({
  children,
  className = "",
  staggerDelay = 100,
  animation = "fade-up",
  baseDelay = 0,
  duration = 700,
}: StaggerContainerProps) {
  return (
    <div className={className}>
      {React.Children.map(children, (child, index) => (
        <ScrollReveal
          animation={animation}
          delay={baseDelay + index * staggerDelay}
          duration={duration}
        >
          {child}
        </ScrollReveal>
      ))}
    </div>
  );
}

// Text reveal animation - reveals text character by character or word by word
interface TextRevealProps {
  text: string;
  className?: string;
  delay?: number;
  duration?: number;
  by?: "character" | "word";
  staggerDelay?: number;
}

export function TextReveal({
  text,
  className = "",
  delay = 0,
  duration = 500,
  by = "word",
  staggerDelay = 50,
}: TextRevealProps) {
  const [isVisible, setIsVisible] = useState(false);
  const elementRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.unobserve(element);
        }
      },
      { threshold: 0.1 }
    );

    observer.observe(element);

    return () => observer.disconnect();
  }, []);

  const elements = by === "character" ? text.split("") : text.split(" ");

  return (
    <span ref={elementRef} className={`inline-block ${className}`}>
      {elements.map((el, index) => (
        <span
          key={index}
          className="inline-block"
          style={{
            opacity: isVisible ? 1 : 0,
            transform: isVisible ? "translateY(0)" : "translateY(20px)",
            transitionProperty: "transform, opacity",
            transitionDuration: `${duration}ms`,
            transitionDelay: `${delay + index * staggerDelay}ms`,
            transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)",
          }}
        >
          {el}
          {by === "word" && index < elements.length - 1 ? "\u00A0" : ""}
        </span>
      ))}
    </span>
  );
}

// Counter animation
interface CounterRevealProps {
  end: number;
  start?: number;
  duration?: number;
  delay?: number;
  suffix?: string;
  prefix?: string;
  className?: string;
  decimals?: number;
}

export function CounterReveal({
  end,
  start = 0,
  duration = 2000,
  delay = 0,
  suffix = "",
  prefix = "",
  className = "",
  decimals = 0,
}: CounterRevealProps) {
  const [count, setCount] = useState(start);
  const [isVisible, setIsVisible] = useState(false);
  const elementRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.unobserve(element);
        }
      },
      { threshold: 0.1 }
    );

    observer.observe(element);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!isVisible) return;

    const timeout = setTimeout(() => {
      const startTime = Date.now();
      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out-expo)
        const easeOutExpo =
          progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
        const current = start + (end - start) * easeOutExpo;

        setCount(current);

        if (progress < 1) {
          requestAnimationFrame(animate);
        }
      };

      requestAnimationFrame(animate);
    }, delay);

    return () => clearTimeout(timeout);
  }, [isVisible, start, end, duration, delay]);

  return (
    <span ref={elementRef} className={className}>
      {prefix}
      {count.toFixed(decimals)}
      {suffix}
    </span>
  );
}

// Progress bar reveal
interface ProgressRevealProps {
  progress: number;
  className?: string;
  barClassName?: string;
  delay?: number;
  duration?: number;
  showLabel?: boolean;
  label?: string;
}

export function ProgressReveal({
  progress,
  className = "",
  barClassName = "",
  delay = 0,
  duration = 1000,
  showLabel = true,
  label = "",
}: ProgressRevealProps) {
  const [currentProgress, setCurrentProgress] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.unobserve(element);
        }
      },
      { threshold: 0.1 }
    );

    observer.observe(element);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!isVisible) return;

    const timeout = setTimeout(() => {
      setCurrentProgress(progress);
    }, delay);

    return () => clearTimeout(timeout);
  }, [isVisible, progress, delay]);

  return (
    <div ref={elementRef} className={className}>
      {showLabel && (
        <div className="flex justify-between mb-2">
          <span style={{ color: "var(--text-secondary)" }}>{label}</span>
          <span style={{ color: "var(--accent)" }}>{progress}%</span>
        </div>
      )}
      <div
        className="w-full h-2 rounded-full overflow-hidden"
        style={{ backgroundColor: "var(--surface)" }}
      >
        <div
          className={`h-full rounded-full ${barClassName}`}
          style={{
            width: `${currentProgress}%`,
            backgroundColor: "var(--accent)",
            transitionProperty: "width",
            transitionDuration: `${duration}ms`,
            transitionTimingFunction: "cubic-bezier(0.16, 1, 0.3, 1)",
          }}
        />
      </div>
    </div>
  );
}

// Parallax scroll effect
interface ParallaxRevealProps {
  children: ReactNode;
  className?: string;
  speed?: number;
  direction?: "up" | "down";
}

export function ParallaxReveal({
  children,
  className = "",
  speed = 0.5,
  direction = "up",
}: ParallaxRevealProps) {
  const [offset, setOffset] = useState(0);
  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      const element = elementRef.current;
      if (!element) return;

      const rect = element.getBoundingClientRect();
      const windowHeight = window.innerHeight;

      // Calculate how far the element is from the center of the viewport
      const elementCenter = rect.top + rect.height / 2;
      const viewportCenter = windowHeight / 2;
      const distanceFromCenter = elementCenter - viewportCenter;

      // Apply parallax based on distance from center
      const parallaxOffset = distanceFromCenter * speed * 0.1;
      setOffset(direction === "up" ? -parallaxOffset : parallaxOffset);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();

    return () => window.removeEventListener("scroll", handleScroll);
  }, [speed, direction]);

  return (
    <div
      ref={elementRef}
      className={className}
      style={{
        transform: `translateY(${offset}px)`,
        willChange: "transform",
      }}
    >
      {children}
    </div>
  );
}
