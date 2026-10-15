---
name: x-thread
description: Turn one blog post into a paste-ready X thread with the post's own figures. Drafts the tweets, then builds a copy deck where each click copies the next tweet or its image in order. Triggers on /x-thread, "tweet this post", "viết tweet cho bài", "promote this on X", "make a thread from".
argument-hint: [slug-or-title] [--publish]
---

# x-thread

One post in, one thread out, plus a deck that makes posting it a sequence of click-paste-click-paste.

## When to use

- `/x-thread <slug>`, "tweet this post", "promote X on Twitter", "viết tweet quảng bá bài này"
- After a post ships to Substack and you want the traffic

Not for: the long-form X Article version of a post (that is `/x-article`), or copy for LinkedIn.

## Pipeline

### 1. Brief, not the post

```bash
npm run x -- extract <slug>
```

Prints frontmatter, the Substack permalink, the TL;DR bullets, the section list, every figure with its number and alt text, and the sentences that carry hard numbers. That is everything a thread needs.

**Do not read the post.** An 11k-word post is ~20k tokens re-billed on every remaining turn of the session, and the brief already holds the parts a thread is built from. Read one section of the source only if a specific claim needs checking.

### 2. Write the thread

Write `.x/<slug>.thread.md`. One section per tweet:

```markdown
---
slug: <slug>
url: <substack permalink from the brief>
---

## 1 | img: 1
Tweet text. Blank lines inside a tweet are fine.

## 2
No image on this one.

## 3 | img: 5
Tweet text. "img: 5" means the post's figure 5, matched by the -5 suffix on
the file. A filename or a path works too.
```

Shape that works:

| Tweet | Job |
| --- | --- |
| 1 | The hook (a challenge to answer, the sharpest number, or the contradiction the post resolves), a figure, **and the Substack link**. The link goes in the first post, always, on its own last line. Most readers never reach the end of a thread, so a link that only lives there reaches nobody. Keep the hook first and the link last inside the tweet, and remember X counts any URL as 23 characters. |
| 2 | The payoff of the hook. If tweet 1 asked something, answer it here, and make the answer the thesis. |
| 3 to n-1 | One idea per tweet, each with the post's figure that actually carries it. Skip figures that need the surrounding prose to make sense. Land a question or a dry line every second or third post. |
| n | One real question to the reader, then the link again. Repeating it is right, the people who read to the end are the ones most likely to click. |

6 to 10 tweets. More than that and people bail before the link.

### 3. Build the deck

```bash
npm run x -- thread <slug>
```

Writes `.x/<slug>.thread/`, converts each figure to PNG, serves it on 127.0.0.1 and opens it. The step button walks the thread one tweet per click: each write puts `text/plain` and `image/png` on the clipboard together, so a single paste fills the composer with the tweet and its figure. If a composer only reads one flavour, the first failed write flips the walker into text-then-image mode for the rest of the thread, and every card also has **Text only** and **Image only**. Keyboard: `c`.

The server is a foreground process that idles out after 45 minutes. Run it with `run_in_background: true` and tell the user the URL, or hand them the command to run themselves. `--no-serve` writes the deck for `file://` instead, where text copy still works but one-click image copy does not.

Flags worth knowing: `--url <permalink>` when the Substack slug differs, `--ttl <min>`, `--no-open`.

## Voice

**Every tweet ships in English**, no matter which language the conversation is in. The account publishes in English, so a Vietnamese request still produces English copy.

The rules that decide whether anyone reads past tweet 1:

- **No em dashes.** Comma, period, or colon. This is the most common tell and the user will notice.
- **Open on the collision, not the topic.** "The backtest said +60R. Twelve weeks later the account was down." A first tweet that describes what the post covers is a table of contents, and nobody clicks a table of contents.
- **One idea per tweet, and it has to land on its own.** If a tweet only makes sense after the previous one, merge them or cut it.
- **Concrete beats abstract every time.** Numbers, dollars, a specific sequence of trades. "Nine trades to build it, two to give it back" survives; "risk management is important" does not.
- **Second person.** "You skipped the winner" outperforms "traders skip winners".
- **Talk to the reader, do not brief them.** A thread that only states facts reads like documentation. Ask something they can answer in their head, then answer it a post later. "Be honest, how many of your rules were found this way?" earns a reply; "most rules are curve-fit" does not. Two or three of these across a thread, not one per post.
- **Open with a challenge when the material allows one.** A quiz whose answer is counterintuitive ("one of these charts is a real trend, the other is noise, pick one") buys the whole thread, because people scroll back for the answer.
- **Close by asking, not by announcing.** The last post gets the link and one genuine question worth replying to.
- **Vary the rhythm.** A four-word sentence after a long one is what makes prose sound spoken. Three tweets with identical shape read as a template.
- **Dry, deadpan, a little mean about the right target.** Aim the irony at the industry, the sales pitch, the $997 course, the guy with three arrows on a chart, and at the reader's own reflexes ("great pattern engine, terrible risk manager"). Never at a named person, and never at the reader's intelligence. The joke rides on the truth already in the post, it never bends it. Roughly one landed line every second or third post. Cheerful jokes in every tweet read as try-hard; sarcasm with nothing behind it reads as bitter.
- **Nothing gets invented.** Every number traces to the post. Same honesty gate as the article itself.
- No hashtags, no emoji bullets, no "🧵", no "a thread:", no "Let me explain", no "Here's the thing". The first tweet earns attention by being interesting, not by announcing itself.

**Reject and rewrite** anything that sounds like: "In my latest post", "Ever wondered why", "Most traders don't realize", "Let that sink in", "The truth is", or a tweet that is three clauses joined by semicolons. Those are the machine voice.

Length: keep every tweet at or under 280 characters. The dry run and the deck both flag anything longer, and a free-tier account cannot post it at all.

## Publishing

**The normal path is the deck.** The served page shows the whole thread for review, with a red **Publish thread** button. Clicking it opens a confirm bar that first calls `/whoami` and names the account and access level it would post as, and only the second click publishes. The server refuses a second run, so a double click cannot double post.

That is the flow to hand the user: build the deck, tell them what is in it, let them read it and press the button themselves. **You do not press it for them, and there is no way for you to press it, since the click happens in their browser.**

The CLI path exists for a user who asks for it explicitly:

```bash
npm run x -- verify               # read-only: which account, and is the token read-write
npm run x -- post <slug>          # dry run: prints every tweet, char count, image
npm run x -- post <slug> --yes    # publishes, each tweet replying to the last
```

Credentials come from `.env` (already gitignored), the environment, or `.x/credentials.json`: `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_SECRET`. A user-context `X_OAUTH2_TOKEN` also works; a Bearer Token does not, it is app-only and cannot post. **Never write a key into a file yourself and never ask for one in chat** — the user fills `.env` themselves, and `verify` tells them whether it worked.

Two failure modes worth knowing, both common: an access token generated before App permissions were set to Read and write stays read-only forever, so it has to be regenerated after saving the permission; and the OAuth 2.0 Client ID and Secret sit right next to the access token pair in the portal and get pasted into the wrong slots. `verify` catches the first, and the token shape catches the second, since a real access token starts with the numeric user id and a hyphen.

**The gate: you may run `verify` and the dry run freely, and you may never run `--yes` without the user saying yes to that specific thread in this session.** Publishing is public and irreversible. Approval on one thread is not approval for the next one.

**Posting through the API costs X credits, and this account has none.** `POST /2/tweets` currently returns `402 credits-depleted`, so publishing from the deck or the CLI will fail on tweet 1 and post nothing. Reads and media uploads on the same token still work. Unless the user says they have bought credits, treat the copy path as the real workflow and mention the API only if they ask why the button is there.

If a tweet fails mid-thread, the script prints the URLs of everything already live and the `--reply-to <id>` command to resume from there. Relay that verbatim, do not re-run the whole thread on top of a half-posted one.

`npm run x -- verify [image]` checks the account read-only, and with a file argument also probes media upload. Uploading media publishes nothing, so that probe is safe to run whenever a tier question comes up.

## Handing it back

Give the user the deck URL, the tweet count, the character counts, and paste the thread text into the reply so it reads without opening anything. If a figure is missing, say which and why (usually the post references a `.png` that was never rendered). Then ask whether to publish.
