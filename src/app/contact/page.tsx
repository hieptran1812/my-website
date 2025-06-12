import React from "react";
import { Metadata } from "next";
import ContactFormEnhanced from "@/components/ContactFormEnhanced";

export const metadata: Metadata = {
  title: "Contact Me - Hiep Tran",
  description:
    "Get in touch with Hiep Tran for collaborations, opportunities, or just a friendly chat about technology and innovation.",
};

export default function ContactPage() {
  return (
    <div
      className="flex flex-col min-h-screen will-change-auto"
      style={{
        backgroundColor: "var(--background)",
        color: "var(--text-primary)",
      }}
    >
      <main className="flex-1 flex flex-col">
        <div className="container mx-auto py-16 px-4 sm:px-6 lg:px-8">
          <div className="max-w-5xl mx-auto">
            {/* Page Header */}
            <div className="text-center mb-12 animate-in fade-in duration-700">
              <h1 className="text-4xl md:text-5xl font-bold mb-4 transition-colors duration-300">
                <span className="bg-gradient-to-r from-rose-600 via-pink-600 to-rose-600 bg-clip-text text-transparent">
                  Let&apos;s Work Together
                </span>
              </h1>
              <p
                className="text-lg max-w-2xl mx-auto transition-colors duration-300"
                style={{ color: "var(--text-secondary)" }}
              >
                I&apos;m always excited to hear about new projects and
                opportunities. Whether you have a specific project in mind or
                just want to connect, I&apos;d love to hear from you.
              </p>

              {/* Quick stats */}
              <div className="flex flex-wrap justify-center gap-6 mt-8">
                <div
                  className="flex items-center gap-2 text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span>Usually responds within 24 hours</span>
                </div>
                <div
                  className="flex items-center gap-2 text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <span>Available for new projects</span>
                </div>
                <div
                  className="flex items-center gap-2 text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  <span>Open to collaborations</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-5 gap-12">
              <div
                className="lg:col-span-3 order-2 lg:order-1 animate-in slide-in-from-left duration-700"
                style={{ contain: "layout" }}
              >
                <div
                  className="rounded-2xl p-6 md:p-8 shadow-lg border transition-all duration-300 hover:shadow-xl"
                  style={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                  }}
                >
                  <h2
                    className="text-xl font-semibold mb-6 flex items-center gap-2"
                    style={{ color: "var(--text-primary)" }}
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
                        d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                      />
                    </svg>
                    Send Me a Message
                  </h2>

                  <ContactFormEnhanced />
                </div>
              </div>

              {/* Contact Info Sidebar */}
              <div className="lg:col-span-2 order-1 lg:order-2 animate-in slide-in-from-right duration-700">
                <div className="space-y-6">
                  {/* Contact Details */}
                  <div
                    className="rounded-2xl p-6 shadow-lg border transition-all duration-300 hover:shadow-xl"
                    style={{
                      backgroundColor: "var(--surface)",
                      borderColor: "var(--border)",
                    }}
                  >
                    <h3
                      className="text-lg font-semibold mb-4"
                      style={{ color: "var(--text-primary)" }}
                    >
                      Get in Touch
                    </h3>
                    <div className="space-y-4">
                      <div className="flex items-center gap-3 group">
                        <div
                          className="w-10 h-10 rounded-lg flex items-center justify-center transition-transform duration-200 group-hover:scale-110"
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
                        <div>
                          <p
                            className="text-sm font-medium"
                            style={{ color: "var(--text-primary)" }}
                          >
                            Email
                          </p>
                          <a
                            href="mailto:hieptran.jobs@gmail.com"
                            className="text-sm hover:underline transition-colors duration-300"
                            style={{ color: "var(--text-secondary)" }}
                          >
                            hieptran.jobs@gmail.com
                          </a>
                        </div>
                      </div>

                      <div className="flex items-center gap-3 group">
                        <div
                          className="w-10 h-10 rounded-lg flex items-center justify-center transition-transform duration-200 group-hover:scale-110"
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
                        <div>
                          <p
                            className="text-sm font-medium"
                            style={{ color: "var(--text-primary)" }}
                          >
                            LinkedIn
                          </p>
                          <a
                            href="https://www.linkedin.com/in/hieptran01"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sm hover:underline transition-colors duration-300"
                            style={{ color: "var(--text-secondary)" }}
                          >
                            Connect with me
                          </a>
                        </div>
                      </div>

                      <div className="flex items-center gap-3 group">
                        <div
                          className="w-10 h-10 rounded-lg flex items-center justify-center transition-transform duration-200 group-hover:scale-110"
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
                        <div>
                          <p
                            className="text-sm font-medium"
                            style={{ color: "var(--text-primary)" }}
                          >
                            GitHub
                          </p>
                          <a
                            href="https://github.com/hieptran1812"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sm hover:underline transition-colors duration-300"
                            style={{ color: "var(--text-secondary)" }}
                          >
                            View my projects
                          </a>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Response Time */}
                  <div
                    className="rounded-2xl p-6 shadow-lg border transition-all duration-300 hover:shadow-xl"
                    style={{
                      backgroundColor: "var(--surface)",
                      borderColor: "var(--border)",
                    }}
                  >
                    <h3
                      className="text-lg font-semibold mb-4"
                      style={{ color: "var(--text-primary)" }}
                    >
                      Response Time
                    </h3>
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <span
                          className="text-sm"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          Usually responds within 24 hours
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        <span
                          className="text-sm"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          Available for new projects
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* What to Expect */}
                  <div
                    className="rounded-2xl p-6 shadow-lg border transition-all duration-300 hover:shadow-xl"
                    style={{
                      backgroundColor: "var(--surface)",
                      borderColor: "var(--border)",
                    }}
                  >
                    <h3
                      className="text-lg font-semibold mb-4"
                      style={{ color: "var(--text-primary)" }}
                    >
                      What to Expect
                    </h3>
                    <ul
                      className="space-y-2 text-sm"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      <li className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-green-500 rounded-full"></div>
                        Detailed project discussion
                      </li>
                      <li className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-blue-500 rounded-full"></div>
                        Timeline and scope clarification
                      </li>
                      <li className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-purple-500 rounded-full"></div>
                        Technology recommendations
                      </li>
                      <li className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-orange-500 rounded-full"></div>
                        Clear communication throughout
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Footer Note */}
            <div
              className="mt-16 text-center animate-in fade-in duration-700"
              style={{ animationDelay: "400ms" }}
            >
              <div
                className="inline-flex items-center gap-2 px-6 py-3 rounded-full"
                style={{
                  backgroundColor: "var(--surface)",
                  borderColor: "var(--border)",
                  border: "1px solid",
                }}
              >
                <svg
                  className="w-4 h-4"
                  style={{ color: "var(--accent)" }}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <p
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
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
          </div>
        </div>
      </main>
    </div>
  );
}
