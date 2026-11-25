/**
 * halleywiki-r2: authenticated gateway to the site's R2 buckets.
 *
 *   PUT    /media/<key>     upload an image (served publicly at img.halleywiki.com/<key>)
 *   PUT    /content/<key>   upload a markdown post (private bucket)
 *   GET    /content/<key>   read a markdown post
 *   HEAD   /content/<key>   check a post exists
 *   DELETE /media|content/<key>
 *   GET    /list/media|content?cursor=…   one page (≤1000) of {key, etag}; etag is
 *                                         the MD5 for single-part uploads, which
 *                                         lets the sync script skip unchanged files
 *
 * Every request needs `Authorization: Bearer <R2_GATEWAY_SECRET>`. The content
 * bucket is private because it holds paywalled posts; images are public through
 * the bucket's custom domain, not through this Worker.
 */
export default {
  async fetch(request, env) {
    const auth = request.headers.get("authorization") || "";
    if (!env.R2_GATEWAY_SECRET || !timingSafeEqual(auth, `Bearer ${env.R2_GATEWAY_SECRET}`)) {
      return new Response("unauthorized", { status: 401 });
    }
    const url = new URL(request.url);
    const l = url.pathname.match(/^\/list\/(media|content)$/);
    if (l && request.method === "GET") {
      const page = await (l[1] === "media" ? env.MEDIA : env.CONTENT).list({
        limit: 1000,
        cursor: url.searchParams.get("cursor") || undefined,
      });
      return Response.json({
        objects: page.objects.map((o) => ({ key: o.key, etag: o.etag })),
        cursor: page.truncated ? page.cursor : null,
      });
    }
    const m = url.pathname.match(/^\/(media|content)\/(.+)$/);
    if (!m) return new Response("not found", { status: 404 });
    const bucket = m[1] === "media" ? env.MEDIA : env.CONTENT;
    const key = decodeURIComponent(m[2]);
    if (key.includes("..")) return new Response("bad key", { status: 400 });

    switch (request.method) {
      case "PUT": {
        const type = request.headers.get("content-type") || undefined;
        await bucket.put(key, request.body, {
          httpMetadata: {
            contentType: type,
            cacheControl: m[1] === "media" ? "public, max-age=31536000, immutable" : undefined,
          },
        });
        return new Response("ok");
      }
      case "GET":
      case "HEAD": {
        const obj = await bucket.get(key);
        if (!obj) return new Response("not found", { status: 404 });
        const headers = new Headers();
        obj.writeHttpMetadata(headers);
        headers.set("etag", obj.httpEtag);
        return new Response(request.method === "HEAD" ? null : obj.body, { headers });
      }
      case "DELETE":
        await bucket.delete(key);
        return new Response("ok");
      default:
        return new Response("method not allowed", { status: 405 });
    }
  },
};

function timingSafeEqual(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
