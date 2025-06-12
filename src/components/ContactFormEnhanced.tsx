"use client";

import React, { useState, useCallback, useEffect } from "react";

interface ContactFormProps {
  className?: string;
}

interface FormData {
  name: string;
  email: string;
  subject: string;
  message: string;
}

interface SubmitStatus {
  type: "success" | "error" | "info" | null;
  message: string;
  progress?: number;
}

interface LoadingStage {
  stage: "validating" | "sending" | "processing" | "confirming" | "complete";
  message: string;
}

interface FieldErrors {
  name?: string;
  email?: string;
  subject?: string;
  message?: string;
}

// Success animation component
const SuccessAnimation: React.FC = () => {
  const [showCelebration, setShowCelebration] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setShowCelebration(false), 3000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="flex items-center justify-center space-x-2">
      <div className="relative">
        <svg
          className="w-6 h-6 text-green-600 dark:text-green-400"
          fill="currentColor"
          viewBox="0 0 20 20"
        >
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
            clipRule="evenodd"
          />
        </svg>
        {showCelebration && (
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-ping"></div>
        )}
      </div>
      {showCelebration && (
        <div className="flex space-x-1">
          {["🎉", "✨", "🎊"].map((emoji, i) => (
            <span
              key={i}
              className="animate-bounce text-lg"
              style={{ animationDelay: `${i * 100}ms` }}
            >
              {emoji}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

// Enhanced loading spinner with stages
const EnhancedLoadingSpinner: React.FC<{ stage: LoadingStage }> = ({
  stage,
}) => {
  const stageProgress = {
    validating: 25,
    sending: 50,
    processing: 75,
    confirming: 90,
    complete: 100,
  };

  return (
    <div className="flex items-center space-x-3">
      <div className="relative w-6 h-6">
        {/* Background circle */}
        <svg className="w-6 h-6 transform -rotate-90" viewBox="0 0 24 24">
          <circle
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="2"
            fill="none"
            className="loading-spinner-ring-secondary"
          />
          {/* Progress circle */}
          <circle
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="2"
            fill="none"
            strokeDasharray={62.83}
            strokeDashoffset={
              62.83 - (62.83 * stageProgress[stage.stage]) / 100
            }
            className="loading-spinner-ring-primary transition-all duration-500 ease-in-out"
            strokeLinecap="round"
          />
        </svg>
        {/* Center dot */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="loading-spinner-center w-2 h-2 rounded-full animate-pulse"></div>
        </div>
      </div>
      <div className="flex flex-col">
        <span className="text-sm font-medium loading-button-text">
          {stage.message}
        </span>
        <div className="flex space-x-1 mt-1">
          {[...Array(3)].map((_, i) => (
            <div
              key={i}
              className="w-1 h-1 loading-spinner-ring-primary rounded-full animate-pulse"
              style={{ animationDelay: `${i * 200}ms` }}
            ></div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Enhanced notification component with 15-second countdown
const EnhancedNotification: React.FC<{
  status: SubmitStatus;
  onDismiss: () => void;
}> = ({ status, onDismiss }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [timeLeft, setTimeLeft] = useState(15);

  useEffect(() => {
    if (status.type) {
      setIsVisible(true);
      setTimeLeft(15);

      // Auto-dismiss timer with countdown
      const dismissTimer = setTimeout(() => {
        setIsVisible(false);
        setTimeout(onDismiss, 300);
      }, 15000);

      // Countdown timer
      const countdownTimer = setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 1) {
            clearInterval(countdownTimer);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);

      return () => {
        clearTimeout(dismissTimer);
        clearInterval(countdownTimer);
      };
    }
  }, [status.type, onDismiss]);

  const handleDismiss = useCallback(() => {
    setIsVisible(false);
    setTimeout(onDismiss, 300);
  }, [onDismiss]);

  if (!status.type) return null;

  const isSuccess = status.type === "success";
  const isError = status.type === "error";

  return (
    <div
      className={`mb-6 relative overflow-hidden rounded-lg transform transition-all duration-500 ease-in-out ${
        isVisible
          ? "translate-x-0 opacity-100 scale-100"
          : "translate-x-4 opacity-0 scale-95"
      } ${
        isSuccess
          ? "notification-success"
          : isError
          ? "notification-error"
          : "bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800"
      }`}
    >
      {/* Progress bar */}
      <div
        className={`absolute bottom-0 left-0 h-1 transition-all duration-1000 ease-linear ${
          isSuccess
            ? "notification-progress-success notification-progress-15s"
            : isError
            ? "notification-progress-error notification-progress-15s"
            : "bg-blue-500"
        }`}
        style={{ width: `${(timeLeft / 15) * 100}%` }}
      />

      <div className="flex items-start gap-3 p-4">
        <div
          className={`flex-shrink-0 ${
            isSuccess
              ? "notification-success-icon"
              : isError
              ? "notification-error-icon"
              : "text-blue-600 dark:text-blue-400"
          }`}
        >
          {isSuccess ? (
            <SuccessAnimation />
          ) : isError ? (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
          ) : (
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                clipRule="evenodd"
              />
            </svg>
          )}
        </div>
        <div className="flex-1">
          <h4
            className={`text-sm font-semibold mb-1 ${
              isSuccess
                ? "notification-success-title"
                : isError
                ? "notification-error-title"
                : "text-blue-800 dark:text-blue-200"
            }`}
          >
            {isSuccess ? "Success!" : isError ? "Error" : "Info"}
          </h4>
          <p
            className={`text-sm ${
              isSuccess
                ? "notification-success-text"
                : isError
                ? "notification-error-text"
                : "text-blue-700 dark:text-blue-300"
            }`}
          >
            {status.message}
          </p>
          <div className="flex items-center justify-between mt-2">
            <span
              className={`text-xs ${
                isSuccess
                  ? "notification-success-text"
                  : isError
                  ? "notification-error-text"
                  : "text-blue-600 dark:text-blue-400"
              } opacity-75`}
            >
              Auto-dismiss in {timeLeft}s
            </span>
          </div>
        </div>
        <button
          onClick={handleDismiss}
          className={`flex-shrink-0 p-1 rounded-full transition-all duration-200 hover:scale-110 ${
            isSuccess
              ? "hover:bg-green-200 dark:hover:bg-green-800/30 text-green-600 dark:text-green-400"
              : isError
              ? "hover:bg-red-200 dark:hover:bg-red-800/30 text-red-600 dark:text-red-400"
              : "hover:bg-blue-200 dark:hover:bg-blue-800/30 text-blue-600 dark:text-blue-400"
          }`}
        >
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
        </button>
      </div>
    </div>
  );
};

const ContactFormEnhanced: React.FC<ContactFormProps> = ({
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
  const [loadingStage, setLoadingStage] = useState<LoadingStage>({
    stage: "validating",
    message: "Preparing...",
  });
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  // Real-time validation with comprehensive rules
  const validateField = useCallback(
    (name: string, value: string): string | undefined => {
      switch (name) {
        case "name":
          if (!value.trim()) return "Name is required";
          if (value.trim().length < 2)
            return "Name must be at least 2 characters";
          if (value.trim().length > 100)
            return "Name must be less than 100 characters";
          if (!/^[a-zA-ZÀ-ÿ\s'-]+$/.test(value.trim()))
            return "Name can only contain letters, spaces, hyphens, and apostrophes";
          break;
        case "email":
          if (!value.trim()) return "Email is required";
          const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
          if (!emailRegex.test(value.trim()))
            return "Please enter a valid email address";
          if (value.length > 254) return "Email address is too long";
          // Check for common typos
          const commonDomains = [
            "gmail.com",
            "yahoo.com",
            "hotmail.com",
            "outlook.com",
          ];
          const domain = value.split("@")[1]?.toLowerCase();
          if (domain && !commonDomains.includes(domain) && domain.length < 4) {
            return "Please check your email domain";
          }
          break;
        case "subject":
          if (!value.trim()) return "Subject is required";
          if (value.trim().length < 5)
            return "Subject must be at least 5 characters";
          if (value.trim().length > 200)
            return "Subject must be less than 200 characters";
          if (value.trim().toLowerCase().includes("test"))
            return "Please provide a meaningful subject";
          break;
        case "message":
          if (!value.trim()) return "Message is required";
          if (value.trim().length < 10)
            return "Message must be at least 10 characters";
          if (value.trim().length > 5000)
            return "Message must be less than 5000 characters";
          // Check for meaningful content
          const words = value.trim().split(/\s+/);
          if (words.length < 3) return "Please provide a more detailed message";
          if (words.every((word) => word.length <= 2))
            return "Please use meaningful words in your message";
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

  const simulateLoadingStages = useCallback(async () => {
    const stages: LoadingStage[] = [
      { stage: "validating", message: "Validating your message..." },
      { stage: "sending", message: "Sending your message..." },
      { stage: "processing", message: "Processing request..." },
      { stage: "confirming", message: "Confirming delivery..." },
      { stage: "complete", message: "Complete!" },
    ];

    for (const stage of stages) {
      setLoadingStage(stage);
      await new Promise((resolve) => setTimeout(resolve, 800));
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitStatus({ type: null, message: "" });

    // Comprehensive validation
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

    // Start loading stages
    const loadingPromise = simulateLoadingStages();

    try {
      const response = await fetch("/api/contact/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      // Wait for loading stages to complete for better UX
      await loadingPromise;

      const result = await response.json();

      if (response.ok) {
        setSubmitStatus({
          type: "success",
          message:
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
    } catch (err) {
      console.error("Contact form error:", err);
      setSubmitStatus({
        type: "error",
        message: "Network error. Please check your connection and try again.",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const isFormValid =
    formData.name.trim() &&
    formData.email.trim() &&
    formData.subject.trim() &&
    formData.message.trim() &&
    !Object.values(fieldErrors).some((error) => error);

  return (
    <div className={`w-full ${className}`}>
      {/* Enhanced Status notification */}
      <EnhancedNotification
        status={submitStatus}
        onDismiss={() => setSubmitStatus({ type: null, message: "" })}
      />

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label
              htmlFor="name"
              className="contact-form-label block text-sm font-medium mb-2 transition-colors duration-300"
            >
              Full Name *
            </label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleInputChange}
              onBlur={handleBlur}
              required
              disabled={isSubmitting}
              placeholder="Your full name"
              className={`contact-form-input w-full px-4 py-3 border rounded-lg focus:ring-2 focus:outline-none transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed ${
                fieldErrors.name && touched.name ? "error" : ""
              }`}
            />
            {fieldErrors.name && touched.name && (
              <p className="contact-form-error mt-1 text-sm animate-in slide-in-from-top-1 duration-200">
                {fieldErrors.name}
              </p>
            )}
          </div>

          <div>
            <label
              htmlFor="email"
              className="contact-form-label block text-sm font-medium mb-2 transition-colors duration-300"
            >
              Email Address *
            </label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              onBlur={handleBlur}
              required
              disabled={isSubmitting}
              placeholder="your.email@example.com"
              className={`contact-form-input w-full px-4 py-3 border rounded-lg focus:ring-2 focus:outline-none transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed ${
                fieldErrors.email && touched.email ? "error" : ""
              }`}
            />
            {fieldErrors.email && touched.email && (
              <p className="contact-form-error mt-1 text-sm animate-in slide-in-from-top-1 duration-200">
                {fieldErrors.email}
              </p>
            )}
          </div>
        </div>

        <div>
          <label
            htmlFor="subject"
            className="contact-form-label block text-sm font-medium mb-2 transition-colors duration-300"
          >
            Subject *
          </label>
          <input
            type="text"
            id="subject"
            name="subject"
            value={formData.subject}
            onChange={handleInputChange}
            onBlur={handleBlur}
            required
            disabled={isSubmitting}
            placeholder="What would you like to discuss?"
            className={`contact-form-input w-full px-4 py-3 border rounded-lg focus:ring-2 focus:outline-none transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed ${
              fieldErrors.subject && touched.subject ? "error" : ""
            }`}
          />
          {fieldErrors.subject && touched.subject && (
            <p className="contact-form-error mt-1 text-sm animate-in slide-in-from-top-1 duration-200">
              {fieldErrors.subject}
            </p>
          )}
        </div>

        <div>
          <label
            htmlFor="message"
            className="contact-form-label block text-sm font-medium mb-2 transition-colors duration-300"
          >
            Message *
          </label>
          <textarea
            id="message"
            name="message"
            rows={5}
            value={formData.message}
            onChange={handleInputChange}
            onBlur={handleBlur}
            required
            disabled={isSubmitting}
            placeholder="Tell me about your project, ideas, or just say hello..."
            className={`contact-form-input w-full px-4 py-3 border rounded-lg focus:ring-2 focus:outline-none transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed resize-none ${
              fieldErrors.message && touched.message ? "error" : ""
            }`}
          />
          {fieldErrors.message && touched.message && (
            <p className="contact-form-error mt-1 text-sm animate-in slide-in-from-top-1 duration-200">
              {fieldErrors.message}
            </p>
          )}
          <div className="flex justify-between items-center mt-2">
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {formData.message.length}/5000 characters
            </span>
            {formData.message.length > 0 && (
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {formData.message.trim().split(/\s+/).length} words
              </span>
            )}
          </div>
        </div>

        {/* Enhanced Submit Button with ContactSection styling */}
        <button
          type="submit"
          disabled={
            isSubmitting ||
            !isFormValid ||
            Object.keys(fieldErrors).some(
              (key) => fieldErrors[key as keyof FieldErrors]
            )
          }
          className={`w-full py-4 px-6 rounded-xl font-semibold transition-all duration-500 transform hover:scale-[1.02] active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:ring-opacity-50 relative overflow-hidden group disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 shadow-lg hover:shadow-2xl contact-submit-button ${
            isSubmitting ? "loading-state-light dark:loading-state-dark" : ""
          }`}
          style={
            !isSubmitting && isFormValid
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
              <div className="loading-button-text">
                <EnhancedLoadingSpinner stage={loadingStage} />
              </div>
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

        {/* Security notice */}
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

export default ContactFormEnhanced;
