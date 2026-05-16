import { getPopularPosts } from "@/lib/getRelatedPosts";
import RelatedPosts from "@/components/RelatedPosts";
import NotFoundHero from "./_components/NotFoundHero";

export default async function NotFound() {
  const popular = await getPopularPosts(6);
  const surprise = popular.length
    ? `/blog/${popular[Math.floor(Math.random() * popular.length)].slug}`
    : undefined;

  return (
    <main
      className="min-h-screen flex flex-col items-center px-4 sm:px-6 lg:px-8 pt-24 pb-16"
      style={{
        backgroundColor: "var(--background)",
        color: "var(--text-primary)",
      }}
    >
      <div className="max-w-4xl w-full">
        <NotFoundHero surpriseHref={surprise} />

        <div className="mt-16">
          <RelatedPosts posts={popular} heading="Recent posts" />
        </div>
      </div>
    </main>
  );
}
