import React from "react";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Privacy Policy - Hiep Tran Developer Portfolio",
  description:
    "Privacy policy for Hiep Tran's personal website and portfolio. Learn how we collect, use, and protect your personal information.",
  robots: {
    index: true,
    follow: true,
  },
  alternates: {
    canonical: "https://halleyverse.dev/privacy",
  },
};

export default function PrivacyPage() {
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "WebPage",
    name: "Privacy Policy",
    description:
      "Privacy policy for Hiep Tran's personal website and portfolio",
    url: "https://halleyverse.dev/privacy",
    mainEntity: {
      "@type": "PrivacyPolicy",
      name: "Hiep Tran Portfolio Privacy Policy",
      url: "https://halleyverse.dev/privacy",
      datePublished: "2025-05-27",
      dateModified: "2025-05-27",
      publisher: {
        "@type": "Person",
        name: "Hiep Tran",
        url: "https://halleyverse.dev",
      },
    },
  };

  return (
    <div
      className="flex flex-col min-h-screen transition-colors duration-300"
      style={{
        backgroundColor: "var(--background)",
        color: "var(--text-primary)",
      }}
    >
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
      />
      <main className="flex-1 max-w-4xl mx-auto py-12 px-4">
        <article>
          <header className="mb-8">
            <h1 className="text-4xl font-bold mb-4">Privacy Policy</h1>
            <p className="text-lg" style={{ color: "var(--text-secondary)" }}>
              Last updated: May 27, 2025
            </p>
          </header>

          <div className="prose dark:prose-invert max-w-none space-y-8">
            <section>
              <h2 className="text-2xl font-semibold mb-4">Introduction</h2>
              <p>
                This Privacy Policy describes how Hiep Tran (&ldquo;I&rdquo;,
                &ldquo;me&rdquo;, or &ldquo;my&rdquo;) collects, uses, and
                shares information when you visit my personal website and
                portfolio at halleyverse.dev (the &ldquo;Service&rdquo;).
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">
                Information I Collect
              </h2>
              <h3 className="text-xl font-medium mb-3">
                Automatically Collected Information
              </h3>
              <p>
                When you visit my website, I may automatically collect certain
                information, including:
              </p>
              <ul className="list-disc pl-6 space-y-2">
                <li>Your IP address and general location information</li>
                <li>Browser type and version</li>
                <li>Operating system</li>
                <li>Pages you visit and time spent on each page</li>
                <li>Referring website</li>
                <li>Device information</li>
              </ul>

              <h3 className="text-xl font-medium mb-3 mt-6">
                Information You Provide
              </h3>
              <p>I may collect information you voluntarily provide, such as:</p>
              <ul className="list-disc pl-6 space-y-2">
                <li>
                  Contact information when you reach out via the contact form
                </li>
                <li>Comments or feedback you submit</li>
                <li>Email address if you subscribe to updates</li>
              </ul>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">
                How I Use Your Information
              </h2>
              <p>I use the collected information for the following purposes:</p>
              <ul className="list-disc pl-6 space-y-2">
                <li>To provide and maintain the website</li>
                <li>To improve user experience and website functionality</li>
                <li>To respond to your inquiries and requests</li>
                <li>To analyze website usage and performance</li>
                <li>To comply with legal obligations</li>
              </ul>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">
                Analytics and Cookies
              </h2>
              <p>
                I may use analytics services like Google Analytics to understand
                how visitors interact with my website. These services may use
                cookies and similar technologies to collect and analyze
                information. You can opt out of Google Analytics by installing
                the Google Analytics opt-out browser add-on.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">
                Information Sharing
              </h2>
              <p>
                I do not sell, trade, or otherwise transfer your personal
                information to third parties without your consent, except in the
                following circumstances:
              </p>
              <ul className="list-disc pl-6 space-y-2">
                <li>To comply with legal requirements</li>
                <li>To protect my rights and safety</li>
                <li>
                  With service providers who assist in website operations (under
                  strict confidentiality)
                </li>
              </ul>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">Data Security</h2>
              <p>
                I implement appropriate security measures to protect your
                personal information against unauthorized access, alteration,
                disclosure, or destruction. However, no method of transmission
                over the internet is 100% secure.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">Your Rights</h2>
              <p>You have the right to:</p>
              <ul className="list-disc pl-6 space-y-2">
                <li>Access the personal information I hold about you</li>
                <li>Request correction of inaccurate information</li>
                <li>Request deletion of your personal information</li>
                <li>Object to processing of your personal information</li>
                <li>Request data portability</li>
              </ul>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">Third-Party Links</h2>
              <p>
                My website may contain links to third-party websites. I am not
                responsible for the privacy practices of these external sites. I
                encourage you to review their privacy policies.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">
                Children&apos;s Privacy
              </h2>
              <p>
                My website is not intended for children under 13 years of age. I
                do not knowingly collect personal information from children
                under 13.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">
                Changes to This Policy
              </h2>
              <p>
                I may update this Privacy Policy from time to time. Any changes
                will be posted on this page with an updated revision date. I
                encourage you to review this policy periodically.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">
                Contact Information
              </h2>
              <p>
                If you have any questions about this Privacy Policy or your
                personal information, please contact me at:
              </p>
              <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg mt-4">
                <p>
                  <strong>Email:</strong> hieptran.jobs@gmail.com
                </p>
                <p>
                  <strong>Website:</strong>{" "}
                  <a
                    href="https://halleyverse.dev"
                    className="text-blue-600 dark:text-blue-400"
                  >
                    halleyverse.dev
                  </a>
                </p>
              </div>
            </section>
          </div>
        </article>
      </main>
    </div>
  );
}
