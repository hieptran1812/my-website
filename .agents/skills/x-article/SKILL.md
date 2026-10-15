---
name: x-article
description: Republish a Substack post as an X Article — adapted, not reprinted. Cuts the post to a feed-length read of about 20 minutes that keeps the thesis, the mechanism, the numbers and the figures, rewrites it into X's voice with no em dashes anywhere, strips what the X editor cannot hold, and builds a deck that copies the body in one click and walks the figures into place one at a time. Triggers on /x-article, "post this as an X article", "đăng bài này lên X", "republish on X", "X long form".
argument-hint: [slug-or-title] [--minutes 20] [--code flatten|drop] [--full]
---

# x-article

The long-form counterpart to `/x-thread`. Takes a post that already exists on Substack and gets it into X's Article editor with the formatting intact and the canonical link pointing home.

It is an **adaptation, not a reprint**. The post is built to be read in a tab, with a sidebar, by someone who came looking for it. An Article is read in a feed, on a phone, by someone who did not plan to read anything today. Same argument, same numbers, same figures, a third of the words.

Requires X Premium+ on the user's account, since that is who can write Articles.

## When to use

- `/x-article <slug>`, "republish this on X", "đăng bài này thành X article"
- Usually right after the Substack version is live, so the backlink resolves

Not for: the thread version (`/x-thread`), or a post that has not been exported to Substack yet and never will be.

## The target

**Twenty minutes, which X reads as about 4,400 words.** The working band is 3,700 to 4,800. The deck measures it on every build and says so in the header, red when it is over.

A 12,000-word post therefore keeps about a third of its words and **all** of its argument. That is only possible by dropping whole sections. It is not possible by thinning every paragraph: thinning is what turns analysis into summary, and a summary is the thing nobody finishes. Cut six sections and write the remaining five at full strength.

Posts already under ~5,000 words need no adaptation. Build them with `--full` and move on.

## Pipeline

Three steps, and only the middle one is yours.

```bash
npm run x -- extract <slug> --article     # the cut plan
#   write .x/<slug>.article.md
npm run x -- article <slug>               # the deck
```

**1. The cut brief.** `extract --article` prints the post's word count and read time, the budget arithmetic ("keep about 51% of the words"), the TL;DR, **every section and subsection with its own word count and share of the post**, the figure list, and the sentences carrying hard numbers. That is what makes the cut decidable: the map shows that "common misconceptions" plus the five-case survey is 1,400 words of restatement, and that the Monte Carlo section is 777 words of actual mechanism.

**2. The adaptation.** Write `.x/<slug>.article.md`. Rules below.

**3. The build.** `article <slug>` renders the adaptation through the Substack converter, so LaTeX to Unicode, callouts, figure URLs and cross-post links are solved in exactly one place and never re-litigated here. Then it:

1. Flattens tables to lists, because X Articles have no table node.
2. Remaps heading levels. `##` becomes Heading and `###` becomes Subheading, which is all X has; anything deeper becomes a bold line. The deck labels each heading in the preview so the spine is checkable at a glance.
3. Handles code blocks. Default `--code flatten` keeps every line as its own paragraph. `--code drop` swaps the block for a pointer at the Substack post, which reads better for anything longer than a screenful.
4. Gives every image slot a `[ IMAGE n ]` marker, with the file staged in a side rail. **Copy body, markers only** is the mode for X: the whole piece in one paste, with a marker line where each figure goes, which you then fill via **Step**. **Copy body + linked images** keeps live `https://halleywiki.com/...` tags instead, for pasting somewhere that fetches them, like Notion or a Google Doc. An image that fails to load in the deck turns red, which means it would not have pasted anywhere either, usually a post that has not deployed yet.

   **Figures cannot ride along with the text into an X Article. Do not try to build that button.** It has been measured against the live editor and it fails in both available forms:

   - A live `<img src="https://halleywiki.com/…">` arrives as a real image node with a broken-camera placeholder in it. X keeps the tag and discards the picture.
   - A base64 `data:` URI does exactly the same, *even though* `x.com`'s CSP lists `data:` in its `img-src`, so the browser would happily have painted it. The editor is stripping the `src` itself.

   X mints its own `twimg` URL and media id for every picture in an Article, so the only way in is its upload path: the toolbar, a dropped file, or a clipboard `image/png` binary. The clipboard holds exactly one of those per copy. **One paste therefore carries at most one figure, by construction**, and a one-click-everything button is not a feature this deck is missing, it is a thing that cannot exist.

   If a user asks for it, say that plainly and point them at the two flows that do work: **Step** (copy, switch, paste, repeat, keyboard `c`) or **Show file** on a rail card, which reveals the staged PNG in Finder so it can be dragged into the editor. With the windows side by side, dragging is usually the faster of the two.
5. Turns every display formula into a `[ MATH n ]` marker plus a rail card. X renders LaTeX, but only through **Insert > LaTeX** in the editor; pasted `$$…$$` stays literal on the page. So each formula gets a **Copy LaTeX** button and a rendered preview to check against before inserting. Inline math is already Unicode. Escaped currency (`\$100`) is never touched.
6. Reports **words, read time and the budget** in the header and the notes bar, plus a nudge when the piece is over, and a different one when it is so far under that the analysis probably did not survive.
7. Appends the backlink line: *"This piece first ran on my Substack, where the full formatting, the code and the math live."*
8. Generates `cover.png` at **1500×600, the 5:2 ratio X asks for**: the kicker and headline on the left, the opening figure trimmed of white margin and fitted on the right, publication name at the foot. Upload it as the Article cover before pasting the body. Needs `sharp`; without it the deck says so instead of shipping a wrong-ratio image.
9. Picks up `.x/<slug>.caption.md` if it exists and puts each variant at the top of the rail with a character count and a copy button.
10. Serves the deck and opens it.

With no adaptation on disk, step 3 still builds — from the whole post — and says so in red, in the console and the notes bar. `--full` makes that deliberate and silences it. `--minutes <n>` moves the budget for a piece that genuinely earns 30. A `--full` build is the post's own published prose, so its punctuation stays as written and `--dashes comma` is how you strip its em dashes; on an adaptation that strip is automatic and cannot be switched off.

The deck has: **Copy title**, **Copy link**, **Copy body, markers only**, **Copy body + linked images** (both rich HTML, so headings and bold survive), and a **Step** button that walks every image and formula in document order, copying the right thing for each. Keyboard: `c`. Step is the normal path for figures, not a fallback.

## Writing the adaptation

```markdown
---
title: The X headline
description: One line under it, usually the sharpest sentence in the piece
---

The opening, no heading above it.

## A section, which is a Heading in X

### A subsection, which is a Subheading

![alt text](/imgs/blogs/<slug>-3.webp)

$$E = p \cdot W - (1 - p) \cdot L$$
```

The frontmatter `title` **is** the X headline, and it overrides the post's title everywhere in the deck and on the cover. Post titles are built for search; headlines are read in a feed. Reference figures by their real path so the build finds the file on disk. Inline `$…$` becomes Unicode, `$$…$$` becomes a marker, `> [!NOTE]` callouts convert, `##` and `###` are the only heading levels worth using.

### What survives the cut, in this order

1. **The thesis.** The uncomfortable thing the post proves. It belongs in the first two paragraphs, not after a warm-up.
2. **The mechanism.** *Why* the thing is true: the chain of cause and effect, the model, the accounting identity, the failure mode. This is what gets dropped by accident when someone compresses by skimming headings, and it is the entire reason the piece is worth reading rather than a tweet.
3. **The numbers**, every one traceable to the post. 65% in sample against 48% out of sample is the argument; "overfitting is dangerous" is a mood.
4. **One worked example**, the best one. The post's second and third example of the same point are the first thing to go.
5. **Four to eight figures.** They cost the reader no time and carry the argument on a phone better than any paragraph. Keep the ones that show a mechanism, drop decorative ones.
6. **The honest caveat** that makes the piece trustworthy: what this does not prove, where it breaks. One paragraph, kept.

### What goes

- **Repetition.** Long posts restate the thesis at every section boundary because the reader has been away. A 20-minute read does not need that.
- **The reference-manual sections.** "Common misconceptions", glossaries, the five-case survey, further reading. Pick the single case that lands and write it properly; cut the other four.
- **Series scaffolding.** "As we saw in the previous section", "in the next post". Keep at most two or three cross-post links, inline in a sentence, never as a paragraph of links.
- **Long code.** Over about twelve lines, point at the original instead. The X editor has no code node and a flattened 60-line listing is 60 paragraphs of noise.
- **Tables.** They flatten to bullet lists. If a table was carrying the argument, write the two rows that matter into prose.
- **Every hedge that is not load-bearing.** The post can afford five caveats. This cannot.

### The voice

- **Short paragraphs**, one to three sentences. A wall of text on a phone is a scroll-past.
- **Subheads make a claim, not a label.** "Overfitting has a number" beats "On overfitting".
- **The first three paragraphs carry the thesis and one hard number.** No throat-clearing, no "in this article we will explore".
- **Keep the post's own dryness.** Deadpan, aimed at the practice and the industry, never at a named person and never at the reader. The joke has to be true.
- **No em dashes.** See below, it is a gate, not a preference. No hashtags, no emoji, no "thread below", no brand-account cadence.
- **Everything ships in English**, whatever language the conversation is in.
- **Nothing new.** Compression must not introduce a claim, a number, a mechanism or a hedge the post does not make. If a tightened sentence would need a fact the post does not have, cut the sentence rather than invent the fact. No new figures either: reference only the post's own images.

Then build, read the notes bar, and fix what it flags. Over budget means cut another section, not another round of thinning.

### No em dashes, and the build enforces it

Nothing that leaves this deck carries an em dash: not the body, not the headline, not a caption. On an adaptation the strip is **unconditional**, and `--dashes keep` does not override it.

Write the prose without them in the first place. A dash is nearly always a comma, a full stop or a colon doing a better job, and the automatic repair is always a comma, which is right about two thirds of the time. The build tells you how many it touched so those sentences get read once; a converted dash is a sentence you still owe a rewrite.

What the strip covers, so you know what you cannot smuggle through:

- `—` anywhere, spaced or not.
- A **spaced** en dash (`a – b`), which is the same mark in disguise. An unspaced one is a range and survives: `2018–2022`, `3–5 bps`.
- ` -- ` in the source. This one matters even when you never type a dash: the Substack converter turns ` -- ` into an em dash, so the build renders the adaptation with that conversion switched off and then treats the literal `--` as a dash to replace.
- The headline in your frontmatter, and every caption variant.

Code is exempt, both fenced blocks and inline spans, because ` -- ` in `npm run x -- article <slug>` is a flag separator and not punctuation.

If the notes bar ever says **"N em dashes survived the strip"**, do not paste. That is a bug in the build, not a style question.

### Long posts: read the post once

A 15,000-word post is ~20k tokens, and every token in a session is re-billed on every later turn of it. For anything past ~10,000 words, hand the whole of step 2 to **one Opus subagent**: give it the post path, the `extract --article` brief, the budget, and the rules in this section, and have it write `.x/<slug>.article.md` and return only the word count and the final section list. The post's prose then never enters the session that gates, captions and hands over.

## The caption

An Article does not travel on its own. It travels on the post that carries it, and that post is the only piece of copy in this whole pipeline you write from scratch. Put **two or three variants** in `.x/<slug>.caption.md`, same shape as a thread file:

```markdown
---
slug: <slug>
---

## 1
First variant.

## 2
Second variant, a different angle.
```

Rules for it, in order of how much they matter:

- **The first line has to earn the scroll.** State the uncomfortable thing the post proves, not the topic. "A backtest is a machine that tells you what you want to hear, and it is very good at its job" works. "New post about backtesting pitfalls" does not.
- **Be funny the way the post is honest.** Dry, deadpan, from a real finding in the piece. "If your backtest has never disappointed you, it is not a backtest, it is a mirror."
- **Numbers travel.** One hard pair beats any adjective: 65% in sample against 48% out of sample says everything the word "overfitting" does not.
- **No em dashes here either.** The build strips them out of caption variants too, but a comma patched into 280 crafted characters is visible. Write the line without one.
- **No link in the caption.** The Article card is already the link, and the Substack backlink lives in the body. A second URL just competes with itself.
- Under 280 characters, English, **no em dashes**, no hashtags, no "thread below", no "I wrote a thing".

Write the file, rebuild, and let the user pick a variant off the rail. Do not pick for them, but say which one you would ship and why in one line.

## What you own, beyond running the script

- **The cut.** Which sections live, which fold into a paragraph, which go. Say what you dropped when you hand over, in one line each, so the user can veto.
- **The headline.** Offer one alternative to the post's title, do not silently rename.
- **Which figure goes on the cover.** It defaults to the adaptation's first figure, which is usually right since opening figures carry the thesis. If it is a small diagram or a table, say so and suggest a better one, then reorder.
- **Read the deck's notes bar** and relay it: words against the budget, headings remapped, code blocks flattened, images missing from disk, formulas needing the LaTeX dialog.
- **The paste order:** cover, then **Copy body, markers only** into the editor, then walk the figures and formulas with **Step**. Doing it the other way means hunting for cursor positions.
- **A pairing tweet.** An Article with no thread pointing at it gets little reach. Offer `/x-thread` for the same slug afterward.

## The rules that do not bend

**The Substack link stays.** The backlink line is on by default and `--no-backlink` exists only for a user who explicitly asks. X Articles are a distribution surface, not the canonical home of the writing, and the canonical link is what makes the republication worth doing.

**The adaptation cannot say more than the post.** Cutting is allowed, sharpening is allowed, inventing is not. If a section cannot be shortened without becoming false, keep it long or drop it whole and say which you did.

**No em dash ships.** Body, headline, caption, every build. The strip is automatic on an adaptation and has no override; writing prose that needed it is the actual failure, and the deck's count of what it rewrote is a list of sentences to go back and fix.

**You never publish an Article yourself.** X has no public API for Articles, so there is no `--yes` path here. The deck's copy buttons are the whole mechanism and the user does the final click. If they want a promo thread afterward, `/x-thread` has a direct-publish path with its own confirmation gate.

## Handing it back

Four lines: the deck URL, words and read time against the budget, the image and formula count, and the paste sequence. Then one line naming what the cut dropped. If the Substack export had to be generated on the fly, say so, since that means the Substack post itself may not be published yet and the backlink may 404 until it is.
