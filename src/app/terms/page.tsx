import React from "react";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Terms of Service - Hiep Tran Developer Portfolio",
  description:
    "Terms of service for Hiep Tran's personal website and portfolio. Learn about the terms and conditions for using this website.",
  robots: {
    index: true,
    follow: true,
  },
  alternates: {
    canonical: "https://halleyverse.dev/terms",
  },
};

export default function TermsPage() {
  const structuredData = {
    "@context": "https://schema.org",
    "@type": "WebPage",
    name: "Terms of Service",
    description:
      "Terms of service for Hiep Tran's personal website and portfolio",
    url: "https://halleyverse.dev/terms",
    mainEntity: {
      "@type": "TermsOfService",
      name: "Hiep Tran Portfolio Terms of Service",
      url: "https://halleyverse.dev/terms",
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
            <h1 className="text-4xl font-bold mb-4">Terms of Service</h1>
            <p className="text-lg" style={{ color: "var(--text-secondary)" }}>
              Last updated: May 27, 2025
            </p>
          </header>

          <div className="prose dark:prose-invert max-w-none space-y-8">
            <section>
              <h2 className="text-2xl font-semibold mb-4">Introduction</h2>
              <p>
                Welcome to Hiep Tran&apos;s personal website and portfolio
                (halleyverse.dev). These Terms of Service (&ldquo;Terms&rdquo;)
                govern your use of my website and services. By accessing or
                using my website, you agree to be bound by these Terms.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">
                Acceptance of Terms
              </h2>
              <p>
                By accessing and using this website, you accept and agree to be
                bound by the terms and provision of this agreement. If you do
                not agree to abide by the above, please do not use this service.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">Use License</h2>
              <p>
                Permission is granted to temporarily download one copy of the
                materials on Hiep Tran&apos;s website for personal,
                non-commercial transitory viewing only. This is the grant of a
                license, not a transfer of title, and under this license you may
                not:
              </p>
              <ul className="list-disc pl-6 space-y-2">
                <li>Modify or copy the materials</li>
                <li>
                  Use the materials for any commercial purpose or for any public
                  display
                </li>
                <li>
                  Attempt to reverse engineer any software contained on the
                  website
                </li>
                <li>
                  Remove any copyright or other proprietary notations from the
                  materials
                </li>
              </ul>
              <p>
                This license shall automatically terminate if you violate any of
                these restrictions and may be terminated by me at any time.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">
                Content and Intellectual Property
              </h2>
              <p>
                The content on this website, including but not limited to text,
                graphics, images, code, and other material, is owned by Hiep
                Tran and is protected by copyright and other intellectual
                property laws.
              </p>
              <h3 className="text-xl font-medium mb-3 mt-6">
                Blog Posts and Articles
              </h3>
              <p>
                Blog posts and articles are original content created by me. You
                may share links to articles, but please do not reproduce entire
                articles without permission.
              </p>
              <h3 className="text-xl font-medium mb-3 mt-6">Code Examples</h3>
              <p>
                Code examples and snippets shared on this website are provided
                for educational purposes. You may use them in your own projects,
                but please provide attribution when appropriate.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">User Conduct</h2>
              <p>When using this website, you agree not to:</p>
              <ul className="list-disc pl-6 space-y-2">
                <li>Use the website for any unlawful purpose</li>
                <li>
                  Attempt to gain unauthorized access to any portion of the
                  website
                </li>
                <li>
                  Interfere with or disrupt the website&apos;s functionality
                </li>
                <li>Upload or transmit viruses or malicious code</li>
                <li>
                  Engage in any activity that could harm the website or its
                  users
                </li>
              </ul>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">
                Contact and Communication
              </h2>
              <p>
                When you contact me through the website&apos;s contact form or
                email, you agree that:
              </p>
              <ul className="list-disc pl-6 space-y-2">
                <li>Your communication is truthful and accurate</li>
                <li>
                  You will not spam or send unsolicited commercial messages
                </li>
                <li>
                  You understand that I may not be able to respond to all
                  messages
                </li>
                <li>
                  Any ideas or suggestions you share may be used without
                  compensation
                </li>
              </ul>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">Disclaimer</h2>
              <p>
                The materials on this website are provided on an &lsquo;as
                is&rsquo; basis. I make no warranties, expressed or implied, and
                hereby disclaim and negate all other warranties including
                without limitation, implied warranties or conditions of
                merchantability, fitness for a particular purpose, or
                non-infringement of intellectual property or other violation of
                rights.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">Limitations</h2>
              <p>
                In no event shall Hiep Tran or its suppliers be liable for any
                damages (including, without limitation, damages for loss of data
                or profit, or due to business interruption) arising out of the
                use or inability to use the materials on this website, even if I
                or my authorized representative has been notified orally or in
                writing of the possibility of such damage.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">External Links</h2>
              <p>
                This website may contain links to external websites. I am not
                responsible for the content, privacy policies, or practices of
                these external sites. Use of external websites is at your own
                risk.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">Modifications</h2>
              <p>
                I may revise these Terms of Service at any time without notice.
                By using this website, you are agreeing to be bound by the then
                current version of these Terms of Service.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">Governing Law</h2>
              <p>
                These Terms of Service and any separate agreements whereby I
                provide you services shall be governed by and construed in
                accordance with the laws of the jurisdiction where I reside.
              </p>
            </section>

            <section>
              <h2 className="text-2xl font-semibold mb-4">
                Contact Information
              </h2>
              <p>
                If you have any questions about these Terms of Service, please
                contact me at:
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
