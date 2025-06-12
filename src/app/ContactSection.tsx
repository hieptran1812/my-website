"use client";

import React, { useState, useEffect } from "react";

const ContactSection = () => {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState<{
    type: "success" | "error" | null;
    message: string;
  }>({ type: null, message: "" });

  // Auto-hide status notifications
  useEffect(() => {
    if (submitStatus.type) {
      const hideDelay = submitStatus.type === "success" ? 5000 : 8000;

      const timer = setTimeout(() => {
        setSubmitStatus({ type: null, message: "" });
      }, hideDelay);

      return () => clearTimeout(timer);
    }
  }, [submitStatus.type]);

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitStatus({ type: null, message: "" });

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
          message: result.message || "Message sent successfully!",
        });
        setFormData({ name: "", email: "", subject: "", message: "" });
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
        message: "Network error. Please check your connection and try again.",
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  return (
    <section
      className="py-16 md:py-24 transition-colors duration-300"
      style={{ backgroundColor: "var(--surface-secondary)" }}
    >
      <div className="container mx-auto px-4 sm:px-6 max-w-4xl">
        <div className="max-w-2xl mx-auto mb-12 text-center">
          <h2 className="section-heading text-3xl md:text-4xl lg:text-5xl font-bold mb-4 transition-colors duration-300 relative">
            <span className="bg-gradient-to-r from-rose-600 via-pink-600 to-rose-600 bg-clip-text text-transparent">
              Get in Touch
            </span>
            <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-24 h-1 bg-gradient-to-r from-rose-600 via-pink-600 to-rose-600 rounded-full"></div>
          </h2>
          <p
            className="text-base max-w-lg mx-auto transition-colors duration-300 mt-6"
            style={{ color: "var(--text-secondary)" }}
          >
            Have a project in mind or just want to chat? Feel free to reach out
            through any of these channels.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Contact Info Cards */}
          <div className="space-y-4">
            {/* Email */}
            <div
              className="flex items-center gap-4 p-4 rounded-xl border transition-all duration-200 hover:shadow-sm"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center"
                style={{ backgroundColor: "var(--accent-subtle)" }}
              >
                <svg
                  className="w-5 h-5"
                  style={{ color: "var(--accent)" }}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M3 8l7.89 4.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                  />
                </svg>
              </div>
              <div className="flex-1">
                <h3
                  className="text-sm font-medium mb-1 transition-colors duration-300"
                  style={{ color: "var(--text-primary)" }}
                >
                  Email
                </h3>
                <a
                  href="mailto:hieptran.jobs@gmail.com"
                  className="text-sm transition-colors duration-300 hover:underline"
                  style={{ color: "var(--text-secondary)" }}
                >
                  hieptran.jobs@gmail.com
                </a>
              </div>
            </div>

            {/* LinkedIn */}
            <div
              className="flex items-center gap-4 p-4 rounded-xl border transition-all duration-200 hover:shadow-sm"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center"
                style={{ backgroundColor: "var(--accent-subtle)" }}
              >
                <svg
                  className="w-5 h-5"
                  style={{ color: "var(--accent)" }}
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3
                  className="text-sm font-medium mb-1 transition-colors duration-300"
                  style={{ color: "var(--text-primary)" }}
                >
                  LinkedIn
                </h3>
                <a
                  href="https://www.linkedin.com/in/hieptran01"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm transition-colors duration-300 hover:underline"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Connect with me
                </a>
              </div>
            </div>

            {/* GitHub */}
            <div
              className="flex items-center gap-4 p-4 rounded-xl border transition-all duration-200 hover:shadow-sm"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center"
                style={{ backgroundColor: "var(--accent-subtle)" }}
              >
                <svg
                  className="w-5 h-5"
                  style={{ color: "var(--accent)" }}
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M12 0C5.374 0 0 5.373 0 12 0 17.302 3.438 21.8 8.207 23.387c.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3
                  className="text-sm font-medium mb-1 transition-colors duration-300"
                  style={{ color: "var(--text-primary)" }}
                >
                  GitHub
                </h3>
                <a
                  href="https://github.com/hieptran1812"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm transition-colors duration-300 hover:underline"
                  style={{ color: "var(--text-secondary)" }}
                >
                  View my projects
                </a>
              </div>
            </div>

            {/* X (Twitter) */}
            <div
              className="flex items-center gap-4 p-4 rounded-xl border transition-all duration-200 hover:shadow-sm"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center group"
                style={{ backgroundColor: "var(--accent-subtle)" }}
              >
                <svg
                  className="w-5 h-5 transition-all duration-200 group-hover:scale-110"
                  viewBox="0 0 120 120"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  style={{ display: "block" }}
                >
                  <rect width="120" height="120" rx="24" fill="#fff" />
                  <path
                    d="M86.4 33.6H99L72.6 62.1L103.2 99H80.7L61.8 76.2L40.8 99H28.2L56.1 68.1L27 33.6H50.1L66.3 53.7L86.4 33.6ZM82.2 92.1H88.8L48.3 40.2H41.1L82.2 92.1Z"
                    fill="#000"
                  />
                </svg>
              </div>
              <div className="flex-1">
                <h3
                  className="text-sm font-medium mb-1 transition-colors duration-300"
                  style={{ color: "var(--text-primary)" }}
                >
                  X (Twitter)
                </h3>
                <a
                  href="https://x.com/halleytran01"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm transition-colors duration-300 hover:underline"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Follow me on X
                </a>
              </div>
            </div>
          </div>

          {/* Contact Form */}
          <div
            className="p-6 rounded-xl border"
            style={{
              backgroundColor: "var(--card-bg)",
              borderColor: "var(--border)",
            }}
          >
            <h3
              className="text-lg font-semibold mb-4 transition-colors duration-300"
              style={{ color: "var(--text-primary)" }}
            >
              Send me a message
            </h3>

            <form className="space-y-4" onSubmit={handleSubmit}>
              <div>
                <label
                  htmlFor="name"
                  className="block text-sm font-medium mb-2 transition-colors duration-300"
                  style={{ color: "var(--text-primary)" }}
                >
                  Name *
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  required
                  disabled={isSubmitting}
                  className="w-full px-3 py-2 border rounded-lg transition-colors duration-300 focus:outline-none focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    backgroundColor: "var(--background)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>
              <div>
                <label
                  htmlFor="email"
                  className="block text-sm font-medium mb-2 transition-colors duration-300"
                  style={{ color: "var(--text-primary)" }}
                >
                  Email *
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleInputChange}
                  required
                  disabled={isSubmitting}
                  className="w-full px-3 py-2 border rounded-lg transition-colors duration-300 focus:outline-none focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    backgroundColor: "var(--background)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>
              <div>
                <label
                  htmlFor="subject"
                  className="block text-sm font-medium mb-2 transition-colors duration-300"
                  style={{ color: "var(--text-primary)" }}
                >
                  Subject *
                </label>
                <input
                  type="text"
                  id="subject"
                  name="subject"
                  value={formData.subject}
                  onChange={handleInputChange}
                  required
                  disabled={isSubmitting}
                  className="w-full px-3 py-2 border rounded-lg transition-colors duration-300 focus:outline-none focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    backgroundColor: "var(--background)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>
              <div>
                <label
                  htmlFor="message"
                  className="block text-sm font-medium mb-2 transition-colors duration-300"
                  style={{ color: "var(--text-primary)" }}
                >
                  Message *
                </label>
                <textarea
                  id="message"
                  name="message"
                  rows={4}
                  value={formData.message}
                  onChange={handleInputChange}
                  required
                  disabled={isSubmitting}
                  className="w-full px-3 py-2 border rounded-lg transition-colors duration-300 focus:outline-none focus:ring-2 resize-vertical disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{
                    backgroundColor: "var(--background)",
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                  }}
                />
              </div>
              <button
                type="submit"
                disabled={isSubmitting}
                className="w-full px-4 py-2 rounded-lg font-medium transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed"
                style={{
                  backgroundColor: "var(--accent)",
                  color: "white",
                }}
              >
                {isSubmitting ? "Sending..." : "Send Message"}
              </button>

              {/* Status Notification */}
              {submitStatus.type && (
                <div
                  className={`mt-4 p-4 rounded-lg border transition-all duration-300 ${
                    submitStatus.type === "success"
                      ? "bg-green-50 border-green-200"
                      : "bg-red-50 border-red-200"
                  }`}
                  style={{
                    backgroundColor:
                      submitStatus.type === "success"
                        ? "rgba(16, 185, 129, 0.1)"
                        : "rgba(239, 68, 68, 0.1)",
                    borderColor:
                      submitStatus.type === "success" ? "#10b981" : "#ef4444",
                  }}
                >
                  <div className="flex items-start gap-3">
                    <div
                      className={`flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center ${
                        submitStatus.type === "success"
                          ? "bg-green-100"
                          : "bg-red-100"
                      }`}
                    >
                      {submitStatus.type === "success" ? (
                        <svg
                          className="w-3 h-3 text-green-600"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M5 13l4 4L19 7"
                          />
                        </svg>
                      ) : (
                        <svg
                          className="w-3 h-3 text-red-600"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M6 18L18 6M6 6l12 12"
                          />
                        </svg>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4
                        className={`text-sm font-semibold mb-1 ${
                          submitStatus.type === "success"
                            ? "text-green-800"
                            : "text-red-800"
                        }`}
                      >
                        {submitStatus.type === "success"
                          ? "Message Sent Successfully!"
                          : "Error Sending Message"}
                      </h4>
                      <p
                        className={`text-xs ${
                          submitStatus.type === "success"
                            ? "text-green-700"
                            : "text-red-700"
                        }`}
                      >
                        {submitStatus.message}
                      </p>
                    </div>
                    <button
                      onClick={() =>
                        setSubmitStatus({ type: null, message: "" })
                      }
                      className={`flex-shrink-0 p-1 rounded-full transition-colors duration-200 ${
                        submitStatus.type === "success"
                          ? "hover:bg-green-200"
                          : "hover:bg-red-200"
                      }`}
                      aria-label="Dismiss notification"
                    >
                      <svg
                        className={`w-3 h-3 ${
                          submitStatus.type === "success"
                            ? "text-green-600"
                            : "text-red-600"
                        }`}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M6 18L18 6M6 6l12 12"
                        />
                      </svg>
                    </button>
                  </div>
                </div>
              )}
            </form>
          </div>
        </div>

        <div className="mt-12 text-center">
          <p
            className="text-sm transition-colors duration-300"
            style={{ color: "var(--text-muted)" }}
          >
            Prefer email? You can also reach me directly at{" "}
            <a
              href="mailto:hieptran.jobs@gmail.com"
              className="transition-colors duration-300 hover:underline"
              style={{ color: "var(--accent)" }}
            >
              hieptran.jobs@gmail.com
            </a>
          </p>
        </div>
      </div>
    </section>
  );
};

export default ContactSection;
