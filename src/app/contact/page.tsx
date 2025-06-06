import React from "react";
import Image from "next/image";

export default function ContactPage() {
  return (
    <div
      className="flex flex-col min-h-screen transition-colors duration-300"
      style={{
        backgroundColor: "var(--background)",
        color: "var(--text-primary)",
      }}
    >
      <main className="flex-1 flex flex-col transition-colors duration-300">
        <div className="container mx-auto py-16 px-4 sm:px-6 lg:px-8">
          <div className="max-w-5xl mx-auto">
            {/* Header section */}
            <div className="text-center mb-16 relative">
              <div className="relative inline-block">
                <h1
                  className="text-5xl md:text-7xl font-extrabold mb-6 relative z-10 tracking-tight"
                  style={{
                    background:
                      "linear-gradient(135deg, var(--accent), var(--accent-hover), #4f46e5)",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    backgroundClip: "text",
                    textShadow: "0 4px 20px var(--accent)/20",
                  }}
                >
                  Get in Touch
                </h1>
                {/* Enhanced decorative elements */}
                <div
                  className="absolute -top-4 -left-4 w-12 h-12 rounded-full opacity-20 animate-pulse"
                  style={{
                    backgroundColor: "var(--accent)",
                    boxShadow: "0 0 20px var(--accent)/30",
                  }}
                />
                <div
                  className="absolute -bottom-4 -right-4 w-8 h-8 rounded-full opacity-30 animate-pulse"
                  style={{
                    backgroundColor: "var(--accent-hover)",
                    boxShadow: "0 0 15px var(--accent-hover)/30",
                  }}
                />
                <div
                  className="absolute top-1/2 -left-8 w-2 h-16 rounded-full opacity-10 rotate-12"
                  style={{ backgroundColor: "var(--accent)" }}
                />
                <div
                  className="absolute top-1/4 -right-6 w-3 h-12 rounded-full opacity-15 -rotate-12"
                  style={{ backgroundColor: "var(--accent-hover)" }}
                />
              </div>
              <p
                className="text-xl max-w-2xl mx-auto leading-relaxed"
                style={{ color: "var(--text-secondary)" }}
              >
                I&apos;m always open to new opportunities, collaborations, or
                just a friendly chat about technology and innovation.
              </p>
              {/* Subtle background gradient */}
              <div
                className="absolute inset-0 -z-10 opacity-5"
                style={{
                  background: `radial-gradient(circle at center, var(--accent), transparent 70%)`,
                }}
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-5 gap-12">
              {/* Contact Form Section */}
              <div className="lg:col-span-3 order-2 lg:order-1">
                <div
                  className="rounded-2xl p-6 md:p-8 shadow-lg border transition-all duration-300"
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
                  <form className="space-y-5">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                      <div>
                        <label
                          htmlFor="name"
                          className="block text-sm font-medium mb-2"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          Your Name
                        </label>
                        <input
                          type="text"
                          id="name"
                          placeholder="Halley"
                          className="w-full px-4 py-3 rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-[var(--accent)] border"
                          style={{
                            backgroundColor: "var(--surface-hover)",
                            color: "var(--text-primary)",
                            borderColor: "var(--border)",
                          }}
                        />
                      </div>
                      <div>
                        <label
                          htmlFor="email"
                          className="block text-sm font-medium mb-2"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          Your Email
                        </label>
                        <input
                          type="email"
                          id="email"
                          placeholder="halley@example.com"
                          className="w-full px-4 py-3 rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-[var(--accent)] border"
                          style={{
                            backgroundColor: "var(--surface-hover)",
                            color: "var(--text-primary)",
                            borderColor: "var(--border)",
                          }}
                        />
                      </div>
                    </div>
                    <div>
                      <label
                        htmlFor="subject"
                        className="block text-sm font-medium mb-2"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        Subject
                      </label>
                      <input
                        type="text"
                        id="subject"
                        placeholder="How can I help you?"
                        className="w-full px-4 py-3 rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-[var(--accent)] border"
                        style={{
                          backgroundColor: "var(--surface-hover)",
                          color: "var(--text-primary)",
                          borderColor: "var(--border)",
                        }}
                      />
                    </div>
                    <div>
                      <label
                        htmlFor="message"
                        className="block text-sm font-medium mb-2"
                        style={{ color: "var(--text-secondary)" }}
                      >
                        Message
                      </label>
                      <textarea
                        id="message"
                        rows={5}
                        placeholder="Your message here..."
                        className="w-full px-4 py-3 rounded-lg resize-none transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-[var(--accent)] border"
                        style={{
                          backgroundColor: "var(--surface-hover)",
                          color: "var(--text-primary)",
                          borderColor: "var(--border)",
                        }}
                      ></textarea>
                    </div>
                    <button
                      type="submit"
                      className="w-full py-3 px-6 rounded-lg font-medium transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] focus:ring-opacity-50 relative overflow-hidden group"
                      style={{
                        background:
                          "linear-gradient(135deg, var(--accent), var(--accent-hover))",
                        color: "white",
                        boxShadow: "0 4px 12px var(--accent)/25",
                      }}
                    >
                      <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-white/0 via-white/20 to-white/0 transform -skew-x-12 -translate-x-full group-hover:translate-x-full ease-out duration-700 transition-transform"></span>
                      Send Message
                      <svg
                        className="w-5 h-5 inline-block ml-2"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M13 7l5 5m0 0l-5 5m5-5H6"
                        />
                      </svg>
                    </button>
                  </form>
                </div>
              </div>

              {/* Contact Info Section */}
              <div className="lg:col-span-2 order-1 lg:order-2">
                <div
                  className="rounded-2xl p-6 md:p-8 h-full shadow-lg flex flex-col border transition-all duration-300"
                  style={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                    background:
                      "linear-gradient(145deg, var(--surface), var(--surface-hover))",
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
                        d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    Contact Information
                  </h2>

                  <div className="space-y-6 mb-8 flex-grow">
                    <div className="flex items-start gap-4 group transition-all duration-300 hover:translate-x-1">
                      <div
                        className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 transition-all duration-300 group-hover:scale-110"
                        style={{ backgroundColor: "var(--surface-accent)" }}
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
                        <h3
                          className="text-sm font-medium transition-all duration-300"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          Email
                        </h3>
                        <a
                          href="mailto:hieptran.jobs@gmail.com"
                          className="text-base hover:underline transition-all duration-200"
                          style={{ color: "var(--accent)" }}
                        >
                          hieptran.jobs@gmail.com
                        </a>
                      </div>
                    </div>

                    <div className="flex items-start gap-4 group transition-all duration-300 hover:translate-x-1">
                      <div
                        className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 transition-all duration-300 group-hover:scale-110"
                        style={{ backgroundColor: "var(--surface-accent)" }}
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
                            d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                          />
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth="2"
                            d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                          />
                        </svg>
                      </div>
                      <div>
                        <h3
                          className="text-sm font-medium transition-all duration-300"
                          style={{ color: "var(--text-secondary)" }}
                        >
                          Location
                        </h3>
                        <p
                          className="text-base"
                          style={{ color: "var(--text-primary)" }}
                        >
                          Hanoi, Vietnam
                        </p>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3
                      className="text-sm font-medium mb-4 flex items-center gap-2"
                      style={{ color: "var(--text-secondary)" }}
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
                          d="M13 10V3L4 14h7v7l9-11h-7z"
                        />
                      </svg>
                      Connect with me
                    </h3>
                    <div className="flex gap-4">
                      <a
                        href="https://github.com/hieptran1812"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 hover:scale-110 hover:shadow-md"
                        style={{
                          backgroundColor: "var(--surface-hover)",
                          boxShadow: "0 2px 8px var(--border)",
                        }}
                      >
                        <Image
                          src="/github.svg"
                          alt="GitHub"
                          width={20}
                          height={20}
                        />
                      </a>
                      <a
                        href="https://www.linkedin.com/in/hieptran01"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 hover:scale-110 hover:shadow-md"
                        style={{
                          backgroundColor: "var(--surface-hover)",
                          boxShadow: "0 2px 8px var(--border)",
                        }}
                      >
                        <Image
                          src="/linkedin.svg"
                          alt="LinkedIn"
                          width={20}
                          height={20}
                        />
                      </a>
                      <a
                        href="https://x.com/halleytran01"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 hover:scale-110 hover:shadow-md"
                        style={{
                          backgroundColor: "var(--surface-hover)",
                          boxShadow: "0 2px 8px var(--border)",
                        }}
                      >
                        <Image
                          src="/twitter.svg"
                          alt="Twitter"
                          width={20}
                          height={20}
                        />
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* FAQ Section */}
            <div className="mt-16">
              <h2
                className="text-2xl font-bold mb-8 text-center flex items-center justify-center gap-2"
                style={{ color: "var(--text-primary)" }}
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
                    d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Frequently Asked Questions
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {[
                  {
                    question: "What services do you offer?",
                    answer:
                      "I specialize in web development, machine learning solutions, and software engineering consulting. My expertise includes React, Next.js, TypeScript, and various ML frameworks.",
                  },
                  {
                    question: "What is your typical response time?",
                    answer:
                      "I usually respond to all inquiries within 24-48 hours. For urgent matters, please indicate so in your message subject.",
                  },
                  {
                    question: "Are you available for freelance work?",
                    answer:
                      "Yes, I'm open to freelance opportunities that align with my expertise and schedule. Feel free to reach out with project details.",
                  },
                  {
                    question: "Do you offer mentoring services?",
                    answer:
                      "Yes, I offer mentoring for developers looking to advance their skills in web development and machine learning. Contact me for availability and rates.",
                  },
                ].map((faq, index) => (
                  <div
                    key={index}
                    className="p-6 rounded-xl border transition-all duration-300 hover:shadow-lg hover:border-[var(--accent-subtle)]"
                    style={{
                      backgroundColor: "var(--surface)",
                      borderColor: "var(--border)",
                    }}
                  >
                    <h3
                      className="text-lg font-medium mb-3 flex items-center gap-2"
                      style={{ color: "var(--text-primary)" }}
                    >
                      <span
                        className="w-6 h-6 rounded-full text-xs flex items-center justify-center flex-shrink-0"
                        style={{
                          backgroundColor: "var(--accent-subtle)",
                          color: "var(--accent)",
                        }}
                      >
                        {index + 1}
                      </span>
                      {faq.question}
                    </h3>
                    <p
                      className="text-base pl-8"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {faq.answer}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
