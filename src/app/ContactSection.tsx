"use client";

import React from "react";
import ContactForm from "@/components/ContactForm";

const ContactSection = () => {
  return (
    <section className="py-20 md:py-28 section-accent">
      <div className="container mx-auto px-4 sm:px-6 max-w-4xl">
        <div className="max-w-2xl mx-auto mb-12 text-center">
          <h2 className="section-heading text-4xl md:text-5xl lg:text-6xl font-bold mb-6 transition-colors duration-300 relative">
            <span className="bg-gradient-to-r from-rose-600 via-pink-600 to-rose-600 bg-clip-text text-transparent">
              Get in Touch
            </span>
            <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-24 h-1 bg-gradient-to-r from-rose-600 via-pink-600 to-rose-600 rounded-full"></div>
          </h2>
          <p
            className="text-lg md:text-xl max-w-3xl mx-auto transition-colors duration-300 leading-relaxed mt-6"
            style={{ color: "var(--text-secondary)" }}
          >
            Have a project in mind or just want to chat? Feel free to reach out
            through any of these channels.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Contact Info Cards */}
          <div className="space-y-4">
            {/* Email */}
            <div
              className="flex items-center gap-4 p-4 rounded-xl border transition-all duration-200 hover:shadow-md hover:scale-[1.02]"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="w-12 h-12 rounded-lg flex items-center justify-center"
                style={{ backgroundColor: "var(--accent-subtle)" }}
              >
                <svg
                  className="w-6 h-6"
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
                  className="text-base font-semibold mb-1 transition-colors duration-300"
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
              className="flex items-center gap-4 p-4 rounded-xl border transition-all duration-200 hover:shadow-md hover:scale-[1.02]"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="w-12 h-12 rounded-lg flex items-center justify-center"
                style={{ backgroundColor: "var(--accent-subtle)" }}
              >
                <svg
                  className="w-6 h-6"
                  style={{ color: "var(--accent)" }}
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3
                  className="text-base font-semibold mb-1 transition-colors duration-300"
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
              className="flex items-center gap-4 p-4 rounded-xl border transition-all duration-200 hover:shadow-md hover:scale-[1.02]"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="w-12 h-12 rounded-lg flex items-center justify-center"
                style={{ backgroundColor: "var(--accent-subtle)" }}
              >
                <svg
                  className="w-6 h-6"
                  style={{ color: "var(--accent)" }}
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M12 0C5.374 0 0 5.373 0 12 0 17.302 3.438 21.8 8.207 23.387c.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z" />
                </svg>
              </div>
              <div className="flex-1">
                <h3
                  className="text-base font-semibold mb-1 transition-colors duration-300"
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
              className="flex items-center gap-4 p-4 rounded-xl border transition-all duration-200 hover:shadow-md hover:scale-[1.02]"
              style={{
                backgroundColor: "var(--card-bg)",
                borderColor: "var(--border)",
              }}
            >
              <div
                className="w-12 h-12 rounded-lg flex items-center justify-center group"
                style={{ backgroundColor: "var(--accent-subtle)" }}
              >
                <svg
                  className="w-6 h-6 transition-all duration-200 group-hover:scale-110"
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
                  className="text-base font-semibold mb-1 transition-colors duration-300"
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
            className="p-6 md:p-8 rounded-xl border shadow-lg backdrop-blur-sm"
            style={{
              backgroundColor: "var(--card-bg)",
              borderColor: "var(--border)",
            }}
          >
            <div className="flex items-center gap-3 mb-6">
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center"
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
                    d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                  />
                </svg>
              </div>
              <h3
                className="text-xl font-bold transition-colors duration-300"
                style={{ color: "var(--text-primary)" }}
              >
                Send me a message
              </h3>
            </div>

            <ContactForm />
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
              className="transition-colors duration-300 hover:underline font-semibold"
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
