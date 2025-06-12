"use client";

import React, { useState, useEffect, useCallback } from "react";

interface ContactFormProps {
  variant?: "default" | "card" | "full";
  className?: string;
}

interface FormData {
  name: string;
  email: string;
  subject: string;
  message: string;
}

interface SubmitStatus {
  type: "success" | "error" | null;
  message: string;
}

interface FieldErrors {
  name?: string;
  email?: string;
  subject?: string;
  message?: string;
}

// Extract InputField component outside to prevent re-creation on every render
const InputField = React.memo(
  ({
    type,
    id,
    name,
    label,
    placeholder,
    required = true,
    isTextarea = false,
    value,
    onChange,
    onBlur,
    hasError,
    isTouched,
    disabled,
  }: {
    type?: string;
    id: string;
    name: string;
    label: string;
    placeholder: string;
    required?: boolean;
    isTextarea?: boolean;
    value: string;
    onChange: (
      e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
    ) => void;
    onBlur: (
      e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>
    ) => void;
    hasError?: string;
    isTouched: boolean;
    disabled: boolean;
  }) => {
    // Enhanced styling with CSS classes for better theme integration
    const baseClassName =
      "contact-form-input w-full px-4 py-3 rounded-lg transition-all duration-300 focus:outline-none focus:ring-2 border disabled:opacity-50 disabled:cursor-not-allowed";

    const inputClassName = `${baseClassName} ${
      hasError && isTouched ? "error" : ""
    }`;

    return (
      <div>
        <label
          htmlFor={id}
          className="contact-form-label block text-sm font-medium mb-2 transition-colors duration-300"
        >
          {label} {required && <span className="text-red-500">*</span>}
        </label>
        {isTextarea ? (
          <textarea
            id={id}
            name={name}
            rows={5}
            value={value}
            onChange={onChange}
            onBlur={onBlur}
            required={required}
            disabled={disabled}
            placeholder={placeholder}
            className={`${inputClassName} resize-none`}
          />
        ) : (
          <input
            type={type}
            id={id}
            name={name}
            value={value}
            onChange={onChange}
            onBlur={onBlur}
            required={required}
            disabled={disabled}
            placeholder={placeholder}
            className={inputClassName}
          />
        )}
        {hasError && isTouched && (
          <p className="contact-form-error mt-1 text-sm animate-in slide-in-from-top-1 duration-200">
            {hasError}
          </p>
        )}
      </div>
    );
  }
);

// Add display name for better debugging
InputField.displayName = "InputField";

const ContactForm: React.FC<ContactFormProps> = ({
  variant = "default",
  className = "",
}) => {
  const [formData, setFormData] = useState<FormData>({
    name: "",
    email: "",
    subject: "",
    message: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState<SubmitStatus>({
    type: null,
    message: "",
  });
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  // Auto-hide status notifications with 15s countdown
  useEffect(() => {
    if (submitStatus.type) {
      const hideDelay = 15000; // 15 seconds for both success and error

      const timer = setTimeout(() => {
        setSubmitStatus({ type: null, message: "" });
      }, hideDelay);

      return () => clearTimeout(timer);
    }
  }, [submitStatus.type]);

  // Enhanced real-time validation with more comprehensive checks
  const validateField = useCallback(
    (name: string, value: string): string | undefined => {
      switch (name) {
        case "name":
          if (!value.trim()) return "Name is required";
          if (value.trim().length < 2)
            return "Name must be at least 2 characters";
          if (value.trim().length > 100)
            return "Name must be less than 100 characters";
          // Check for invalid characters (only letters, spaces, hyphens, and apostrophes)
          const nameRegex = /^[a-zA-ZÀ-ÿ\s\-']+$/;
          if (!nameRegex.test(value.trim()))
            return "Name can only contain letters, spaces, hyphens, and apostrophes";
          break;
        case "email":
          if (!value.trim()) return "Email is required";
          // Enhanced email validation
          const emailRegex =
            /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/;
          if (!emailRegex.test(value.trim()))
            return "Please enter a valid email address";
          if (value.trim().length > 254) return "Email address is too long";
          break;
        case "subject":
          if (!value.trim()) return "Subject is required";
          if (value.trim().length < 5)
            return "Subject must be at least 5 characters";
          if (value.trim().length > 200)
            return "Subject must be less than 200 characters";
          // Check for spam-like patterns
          const spamWords = [
            "urgent",
            "free money",
            "click here",
            "limited time",
          ];
          const hasSpamContent = spamWords.some((word) =>
            value.toLowerCase().includes(word.toLowerCase())
          );
          if (hasSpamContent)
            return "Subject contains potentially inappropriate content";
          break;
        case "message":
          if (!value.trim()) return "Message is required";
          if (value.trim().length < 10)
            return "Message must be at least 10 characters";
          if (value.trim().length > 5000)
            return "Message must be less than 5000 characters";
          // Check message quality
          const words = value.trim().split(/\s+/);
          if (words.length < 3)
            return "Please provide a more detailed message (at least 3 words)";
          // Check for excessive repetition
          const uniqueWords = new Set(words.map((word) => word.toLowerCase()));
          if (words.length > 10 && uniqueWords.size / words.length < 0.5)
            return "Message appears to have excessive repetition";
          break;
      }
      return undefined;
    },
    []
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const { name, value } = e.target;
      setFormData((prev) => ({ ...prev, [name]: value }));

      // Clear submit status when user starts typing
      setSubmitStatus((prevStatus) =>
        prevStatus.type ? { type: null, message: "" } : prevStatus
      );

      // Real-time validation for touched fields
      setTouched((prevTouched) => {
        if (prevTouched[name]) {
          const error = validateField(name, value);
          setFieldErrors((prev) => ({
            ...prev,
            [name]: error,
          }));
        }
        return prevTouched;
      });
    },
    [validateField]
  );

  const handleBlur = useCallback(
    (e: React.FocusEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const { name, value } = e.target;
      setTouched((prev) => ({ ...prev, [name]: true }));

      const error = validateField(name, value);
      setFieldErrors((prev) => ({
        ...prev,
        [name]: error,
      }));
    },
    [validateField]
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitStatus({ type: null, message: "" });

    // Validate all fields
    const errors: FieldErrors = {};
    Object.keys(formData).forEach((key) => {
      const error = validateField(key, formData[key as keyof FormData]);
      if (error) errors[key as keyof FieldErrors] = error;
    });

    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      setTouched({ name: true, email: true, subject: true, message: true });
      setIsSubmitting(false);
      setSubmitStatus({
        type: "error",
        message: "Please fix the errors above before submitting.",
      });
      return;
    }

    try {
      const response = await fetch("/api/contact/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();

      if (response.ok) {
        setSubmitStatus({
          type: "success",
          message:
            result.message ||
            "🎉 Message sent successfully! I'll get back to you within 24 hours.",
        });
        setFormData({ name: "", email: "", subject: "", message: "" });
        setFieldErrors({});
        setTouched({});
      } else {
        setSubmitStatus({
          type: "error",
          message: result.error || "Failed to send message. Please try again.",
        });
      }
    } catch (err: unknown) {
      console.error("Contact form error:", err);
      setSubmitStatus({
        type: "error",
        message:
          "Network error. Please check your connection and try again, or email me directly at hieptran.jobs@gmail.com.",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const StatusNotification = () => {
    if (!submitStatus.type) return null;

    return (
      <div
        className={`mb-6 p-5 rounded-2xl border-l-4 transition-all duration-700 ease-out transform animate-in slide-in-from-top-3 fade-in-0 ${
          submitStatus.type === "success"
            ? "notification-success border-l-emerald-500 shadow-emerald-500/25"
            : "notification-error border-l-red-500 shadow-red-500/25"
        } shadow-xl backdrop-blur-sm relative overflow-hidden`}
      >
        {/* Enhanced background gradient overlay */}
        <div
          className={`absolute inset-0 opacity-30 ${
            submitStatus.type === "success"
              ? "bg-gradient-to-r from-emerald-400/10 via-green-400/5 to-emerald-400/10"
              : "bg-gradient-to-r from-red-400/10 via-rose-400/5 to-red-400/10"
          }`}
        />

        {/* Animated border shimmer */}
        <div
          className={`absolute inset-0 rounded-2xl border-2 opacity-40 ${
            submitStatus.type === "success"
              ? "border-emerald-300 dark:border-emerald-600"
              : "border-red-300 dark:border-red-600"
          } animate-pulse`}
        />

        <div className="relative flex items-start gap-4">
          <div
            className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition-all duration-500 transform shadow-lg relative overflow-hidden ${
              submitStatus.type === "success"
                ? "notification-success-icon animate-bounce shadow-emerald-400/50"
                : "notification-error-icon shadow-red-400/50"
            }`}
          >
            {/* Icon glow effect */}
            <div
              className={`absolute inset-0 rounded-full animate-ping ${
                submitStatus.type === "success"
                  ? "bg-emerald-400/30"
                  : "bg-red-400/30"
              }`}
            />

            {submitStatus.type === "success" ? (
              <svg
                className="w-5 h-5 notification-success-icon relative z-10 animate-in zoom-in-50 duration-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2.5"
                  d="M5 13l4 4L19 7"
                />
              </svg>
            ) : (
              <svg
                className="w-5 h-5 notification-error-icon relative z-10 animate-in zoom-in-50 duration-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2.5"
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            )}
          </div>

          <div className="flex-1 min-w-0">
            <h4
              className={`text-base font-bold mb-2 transition-all duration-300 ${
                submitStatus.type === "success"
                  ? "notification-success-title"
                  : "notification-error-title"
              }`}
            >
              {submitStatus.type === "success" ? (
                <span className="flex items-center gap-2">
                  ✨ Message Sent Successfully!
                  <span className="inline-block animate-bounce text-lg">
                    🎉
                  </span>
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  ⚠️ Error Sending Message
                  <span className="inline-block animate-pulse text-lg">💔</span>
                </span>
              )}
            </h4>
            <p
              className={`text-sm leading-relaxed font-medium ${
                submitStatus.type === "success"
                  ? "notification-success-text"
                  : "notification-error-text"
              }`}
            >
              {submitStatus.message}
            </p>

            {/* 15-second countdown progress bar for all notifications */}
            <div className="mt-3 w-full h-2 bg-gray-200/50 dark:bg-gray-700/30 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full notification-progress-15s ${
                  submitStatus.type === "success"
                    ? "notification-progress-success"
                    : "notification-progress-error"
                }`}
              />
            </div>
          </div>

          <button
            onClick={() => setSubmitStatus({ type: null, message: "" })}
            className={`flex-shrink-0 p-2 rounded-full transition-all duration-300 transform hover:scale-110 active:scale-95 ${
              submitStatus.type === "success"
                ? "hover:bg-emerald-200/70 dark:hover:bg-emerald-700/50 hover:shadow-emerald-400/30"
                : "hover:bg-red-200/70 dark:hover:bg-red-700/50 hover:shadow-red-400/30"
            } hover:shadow-lg relative group`}
            aria-label="Dismiss notification"
          >
            {/* Button hover effect */}
            <div className="absolute inset-0 rounded-full bg-white/20 scale-0 group-hover:scale-100 transition-transform duration-200" />

            <svg
              className={`w-4 h-4 relative z-10 transition-transform duration-200 group-hover:rotate-90 ${
                submitStatus.type === "success"
                  ? "text-emerald-600 dark:text-emerald-300"
                  : "text-red-600 dark:text-red-300"
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2.5"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>
      </div>
    );
  };

  if (variant === "full") {
    return (
      <div className={`w-full ${className}`}>
        <StatusNotification />
        <form className="space-y-5" onSubmit={handleSubmit}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <InputField
              type="text"
              id="name"
              name="name"
              label="Your Name"
              placeholder="Your full name"
              value={formData.name}
              onChange={handleInputChange}
              onBlur={handleBlur}
              hasError={fieldErrors.name}
              isTouched={touched.name || false}
              disabled={isSubmitting}
            />
            <InputField
              type="email"
              id="email"
              name="email"
              label="Your Email"
              placeholder="your.email@example.com"
              value={formData.email}
              onChange={handleInputChange}
              onBlur={handleBlur}
              hasError={fieldErrors.email}
              isTouched={touched.email || false}
              disabled={isSubmitting}
            />
          </div>
          <InputField
            type="text"
            id="subject"
            name="subject"
            label="Subject"
            placeholder="What would you like to discuss?"
            value={formData.subject}
            onChange={handleInputChange}
            onBlur={handleBlur}
            hasError={fieldErrors.subject}
            isTouched={touched.subject || false}
            disabled={isSubmitting}
          />
          <InputField
            id="message"
            name="message"
            label="Message"
            placeholder="Tell me about your project, ideas, or just say hello..."
            isTextarea={true}
            value={formData.message}
            onChange={handleInputChange}
            onBlur={handleBlur}
            hasError={fieldErrors.message}
            isTouched={touched.message || false}
            disabled={isSubmitting}
          />

          <button
            type="submit"
            disabled={
              isSubmitting ||
              Object.keys(fieldErrors).some(
                (key) => fieldErrors[key as keyof FieldErrors]
              )
            }
            className={`w-full py-4 px-6 rounded-xl font-semibold transition-all duration-500 transform hover:scale-[1.02] active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:ring-opacity-50 relative overflow-hidden group disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 shadow-lg hover:shadow-2xl contact-submit-button ${
              isSubmitting ? "loading-state-light dark:loading-state-dark" : ""
            }`}
            style={
              !isSubmitting
                ? {
                    background:
                      "linear-gradient(135deg, var(--accent), var(--accent-hover))",
                    color: "white",
                    boxShadow:
                      "0 8px 25px var(--accent)/30, inset 0 1px 0 rgba(255,255,255,0.2)",
                  }
                : undefined
            }
          >
            {/* Enhanced shimmer effect */}
            <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-white/0 via-white/25 to-white/0 transform -skew-x-12 -translate-x-full group-hover:translate-x-full ease-out duration-1000 transition-transform"></span>

            {/* Loading progress bar */}
            {isSubmitting && (
              <div className="absolute bottom-0 left-0 h-1 bg-white/30 rounded-full overflow-hidden w-full">
                <div className="h-full bg-white/60 rounded-full progress-bar animate-in slide-in-from-left-full duration-[3000ms] ease-out" />
              </div>
            )}

            <div className="relative flex items-center justify-center gap-3">
              {isSubmitting ? (
                <>
                  {/* Enhanced loading spinner with multiple rings */}
                  <div className="relative w-6 h-6">
                    <div className="absolute w-6 h-6 loading-spinner-ring-primary rounded-full animate-spin"></div>
                    <div className="absolute w-4 h-4 top-1 left-1 loading-spinner-ring-secondary rounded-full animate-spin animate-reverse"></div>
                    <div className="absolute w-2 h-2 top-2 left-2 loading-spinner-center rounded-full animate-pulse"></div>
                  </div>
                  <span className="loading-button-text animate-pulse">
                    <span
                      className="inline-block animate-bounce"
                      style={{ animationDelay: "0ms" }}
                    >
                      S
                    </span>
                    <span
                      className="inline-block animate-bounce"
                      style={{ animationDelay: "100ms" }}
                    >
                      e
                    </span>
                    <span
                      className="inline-block animate-bounce"
                      style={{ animationDelay: "200ms" }}
                    >
                      n
                    </span>
                    <span
                      className="inline-block animate-bounce"
                      style={{ animationDelay: "300ms" }}
                    >
                      d
                    </span>
                    <span
                      className="inline-block animate-bounce"
                      style={{ animationDelay: "400ms" }}
                    >
                      i
                    </span>
                    <span
                      className="inline-block animate-bounce"
                      style={{ animationDelay: "500ms" }}
                    >
                      n
                    </span>
                    <span
                      className="inline-block animate-bounce"
                      style={{ animationDelay: "600ms" }}
                    >
                      g
                    </span>
                    <span className="mx-1"></span>
                    <span
                      className="inline-block animate-bounce"
                      style={{ animationDelay: "700ms" }}
                    >
                      .
                    </span>
                    <span
                      className="inline-block animate-bounce"
                      style={{ animationDelay: "800ms" }}
                    >
                      .
                    </span>
                    <span
                      className="inline-block animate-bounce"
                      style={{ animationDelay: "900ms" }}
                    >
                      .
                    </span>
                  </span>
                </>
              ) : (
                <>
                  <svg
                    className="w-5 h-5 transition-all duration-300 group-hover:translate-x-1 group-hover:scale-110"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                    />
                  </svg>
                  <span className="group-hover:tracking-wide transition-all duration-300">
                    Send Message
                  </span>
                  {/* Success celebration dots */}
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    <div
                      className="w-1 h-1 bg-white rounded-full animate-bounce"
                      style={{ animationDelay: "0ms" }}
                    ></div>
                    <div
                      className="w-1 h-1 bg-white rounded-full animate-bounce"
                      style={{ animationDelay: "150ms" }}
                    ></div>
                    <div
                      className="w-1 h-1 bg-white rounded-full animate-bounce"
                      style={{ animationDelay: "300ms" }}
                    ></div>
                  </div>
                </>
              )}
            </div>
          </button>

          <div className="text-center">
            <p
              className="text-xs transition-colors duration-300"
              style={{ color: "var(--text-muted)" }}
            >
              🔒 Your information is secure and will only be used to respond to
              your message.
            </p>
          </div>
        </form>
      </div>
    );
  }

  // Default variant (for ContactSection)
  return (
    <div className={`w-full ${className}`}>
      <StatusNotification />
      <form className="space-y-5" onSubmit={handleSubmit}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <InputField
            type="text"
            id="contact-name"
            name="name"
            label="Full Name"
            placeholder="Your full name"
            value={formData.name}
            onChange={handleInputChange}
            onBlur={handleBlur}
            hasError={fieldErrors.name}
            isTouched={touched.name || false}
            disabled={isSubmitting}
          />
          <InputField
            type="email"
            id="contact-email"
            name="email"
            label="Email Address"
            placeholder="your.email@example.com"
            value={formData.email}
            onChange={handleInputChange}
            onBlur={handleBlur}
            hasError={fieldErrors.email}
            isTouched={touched.email || false}
            disabled={isSubmitting}
          />
        </div>

        <InputField
          type="text"
          id="contact-subject"
          name="subject"
          label="Subject"
          placeholder="What would you like to discuss?"
          value={formData.subject}
          onChange={handleInputChange}
          onBlur={handleBlur}
          hasError={fieldErrors.subject}
          isTouched={touched.subject || false}
          disabled={isSubmitting}
        />

        <InputField
          id="contact-message"
          name="message"
          label="Message"
          placeholder="Tell me about your project, ideas, or just say hello..."
          isTextarea={true}
          value={formData.message}
          onChange={handleInputChange}
          onBlur={handleBlur}
          hasError={fieldErrors.message}
          isTouched={touched.message || false}
          disabled={isSubmitting}
        />

        <button
          type="submit"
          disabled={
            isSubmitting ||
            Object.keys(fieldErrors).some(
              (key) => fieldErrors[key as keyof FieldErrors]
            )
          }
          className={`w-full px-6 py-4 rounded-xl font-semibold transition-all duration-500 focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-[1.02] active:scale-[0.98] shadow-lg hover:shadow-2xl relative overflow-hidden group contact-submit-button ${
            isSubmitting ? "loading-state-light dark:loading-state-dark" : ""
          }`}
          style={
            !isSubmitting
              ? {
                  background:
                    "linear-gradient(135deg, var(--accent), var(--accent-hover))",
                  color: "white",
                  boxShadow:
                    "0 8px 25px var(--accent)/30, inset 0 1px 0 rgba(255,255,255,0.2)",
                }
              : undefined
          }
        >
          {/* Enhanced shimmer effect */}
          <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-white/0 via-white/25 to-white/0 transform -skew-x-12 -translate-x-full group-hover:translate-x-full ease-out duration-1000 transition-transform"></span>

          {/* Loading progress bar */}
          {isSubmitting && (
            <div className="absolute bottom-0 left-0 h-1 bg-white/30 rounded-full overflow-hidden w-full">
              <div className="h-full bg-white/60 rounded-full progress-bar animate-in slide-in-from-left-full duration-[3000ms] ease-out" />
            </div>
          )}

          <div className="relative flex items-center justify-center gap-3">
            {isSubmitting ? (
              <>
                {/* Enhanced loading spinner with multiple rings */}
                <div className="relative w-6 h-6">
                  <div className="absolute w-6 h-6 loading-spinner-ring-primary rounded-full animate-spin"></div>
                  <div className="absolute w-4 h-4 top-1 left-1 loading-spinner-ring-secondary rounded-full animate-spin animate-reverse"></div>
                  <div className="absolute w-2 h-2 top-2 left-2 loading-spinner-center rounded-full animate-pulse"></div>
                </div>
                <span className="loading-button-text animate-pulse">
                  <span
                    className="inline-block animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  >
                    S
                  </span>
                  <span
                    className="inline-block animate-bounce"
                    style={{ animationDelay: "100ms" }}
                  >
                    e
                  </span>
                  <span
                    className="inline-block animate-bounce"
                    style={{ animationDelay: "200ms" }}
                  >
                    n
                  </span>
                  <span
                    className="inline-block animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  >
                    d
                  </span>
                  <span
                    className="inline-block animate-bounce"
                    style={{ animationDelay: "400ms" }}
                  >
                    i
                  </span>
                  <span
                    className="inline-block animate-bounce"
                    style={{ animationDelay: "500ms" }}
                  >
                    n
                  </span>
                  <span
                    className="inline-block animate-bounce"
                    style={{ animationDelay: "600ms" }}
                  >
                    g
                  </span>
                  <span className="mx-1"></span>
                  <span
                    className="inline-block animate-bounce"
                    style={{ animationDelay: "700ms" }}
                  >
                    .
                  </span>
                  <span
                    className="inline-block animate-bounce"
                    style={{ animationDelay: "800ms" }}
                  >
                    .
                  </span>
                  <span
                    className="inline-block animate-bounce"
                    style={{ animationDelay: "900ms" }}
                  >
                    .
                  </span>
                </span>
              </>
            ) : (
              <>
                <svg
                  className="w-5 h-5 transition-all duration-300 group-hover:translate-x-1 group-hover:scale-110"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                  />
                </svg>
                <span className="group-hover:tracking-wide transition-all duration-300">
                  Send Message
                </span>
                {/* Success celebration dots */}
                <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  <div
                    className="w-1 h-1 bg-white rounded-full animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  ></div>
                  <div
                    className="w-1 h-1 bg-white rounded-full animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  ></div>
                  <div
                    className="w-1 h-1 bg-white rounded-full animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  ></div>
                </div>
              </>
            )}
          </div>
        </button>

        <div className="text-center">
          <p
            className="text-xs transition-colors duration-300"
            style={{ color: "var(--text-muted)" }}
          >
            🔒 Your information is secure and will only be used to respond to
            your message.
          </p>
        </div>
      </form>
    </div>
  );
};

export default ContactForm;
