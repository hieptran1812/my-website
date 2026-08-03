/**
 * Client-safe half of the paywall config.
 *
 * `src/lib/paywall.ts` imports `node:path`, so the client component that draws
 * the gate cannot import from it. Only shared constants live here.
 */

/** Where the "keep reading" button points. */
export const PAYWALL_SUBSCRIBE_URL = "https://halleytech.substack.com/";
