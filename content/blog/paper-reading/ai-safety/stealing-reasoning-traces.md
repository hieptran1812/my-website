---
title: "Stealing Reasoning Traces from Proprietary LLM APIs: When the Weakest Model in the Family Holds the Key"
date: "2026-08-12"
description: "A paper analysis of how encrypted chain-of-thought blobs, valid across sessions, users and sibling models, turn a cheap model into a decryption oracle for its frontier siblings, and what it would take to bind them properly."
tags:
  [
    "paper-reading",
    "ai-safety",
    "llm-security",
    "chain-of-thought",
    "prompt-injection",
    "model-distillation",
    "aead",
    "privacy",
    "red-teaming",
    "api-security",
  ]
category: "paper-reading"
subcategory: "AI Safety"
author: "Hiep Tran"
featured: true
readTime: 50
paper:
  title: "Stealing Reasoning Traces from Proprietary LLM APIs"
  authors: "Alexander Panfilov, David Schmotz, Ilia Shumailov et al."
  venue: "arXiv 2026 (cs.CR)"
  url: "https://arxiv.org/abs/2608.09867"
---

> [!tldr]
>
> - **The claim.** Frontier providers stopped returning plaintext chain-of-thought and now hand the client an encrypted blob to store and replay. That blob is accepted in any session, by any user, on almost any sibling model. So you capture an Opus 4.8 blob and inject it into Haiku 4.5, and Haiku reads the hidden reasoning back to you in plaintext.
> - **The mechanism in one line.** The AEAD envelope authenticates the *content* of a reasoning block and never the *context* it was produced in, so a valid envelope is valid everywhere.
> - **Why it matters.** The frontier model's refusal training is bypassed without ever being touched. The paper decodes 315,320 reasoning blocks scraped from public repositories and recovers 62 API keys, 33 passwords and 30 personal emails from genuine user sessions.
> - **The most surprising result.** 64 of the 704 recovered secrets appear nowhere in the visible transcript. Users sanitized the plaintext and shipped the ciphertext, because they could not read it.
> - **Where it stops.** The distillation-forensics appendix is suggestive and explicitly not causal, the extraction is fuzzy rather than exact, and every provider patched after disclosure. As of August 2026 the headline figure is no longer reproducible.
>
> The vulnerability was responsibly disclosed and is closed. This post explains the mechanism, not the exploit: I describe the *structure* of the injection but deliberately do not reproduce the paper's verbatim attack templates.

There is a particular kind of security bug that is not a bug in any component. Every piece does exactly what it was specified to do. The encryption is real, the authentication tag verifies, the key is not leaked, the model weights are not exposed. And yet the system as a whole gives away the thing it was built to protect, because two correct components were wired together under different assumptions about what "valid" means.

[Stealing Reasoning Traces from Proprietary LLM APIs](https://arxiv.org/abs/2608.09867) by Alexander Panfilov, David Schmotz, Ilia Shumailov, Luca Beurer-Kellner, Joachim Schaeffer, Ameya Prabhu, Jonas Geiping and Maksym Andriushchenko is a paper about exactly that shape of failure, in the newest and least examined surface in the LLM stack: the encrypted chain-of-thought blob that your client is now required to carry.

## The problem: reasoning became the product, so it got locked up

Reasoning models spend test-time compute generating an internal monologue before they answer. That monologue is denser than the answer. It contains the hypotheses the model discarded, the tool outputs it read, the intermediate arithmetic, the user data it pulled into working memory, and, sometimes, the safety deliberation about whether to answer at all.

Two distinct threats follow. A competitor who can harvest that monologue at scale gets a training signal far richer than the final answers, which is a distillation problem. And a monologue that reasons through a harmful topic before producing a safely-refused answer contains information the refusal was supposed to withhold, which is a safety problem. Prior work has documented both: reasoning traces leak private information ([Green et al., 2025](https://aclanthology.org/2025.emnlp-main.1347/)), and directly optimizing the *content* of the chain of thought to be safe degrades its monitorability, which is a price the field is reluctant to pay ([Baker et al., 2025](https://arxiv.org/abs/2503.11926)).

So the providers took the third option: keep the reasoning honest, and stop showing it. Anthropic, OpenAI and Google all deprecated plaintext reasoning. What you get back today is a short summary produced by a separate summarizer model, plus an opaque string.

The opaque string is the interesting part, and it exists for a boring reason: **storage**. Multi-turn conversations need reasoning continuity across turns. A provider could keep every session's traces server-side and hand the client an ID, but that means a stateful backend at frontier-API scale. The cheaper design is to encrypt the trace, hand the ciphertext to the client, and require the client to pass it back on every subsequent request. The server stores nothing. The client becomes the database.

That is the bargain. The paper's contribution is to show what the bargain actually costs once you notice that the ciphertext now lives on the attacker's side of the wire, and that nothing inside it says who the attacker is.

Building on a blog post by [Matthew Green (2026)](https://blog.cryptographyengineering.com/2026/05/29/fooling-around-with-encrypted-reasoning-blobs/), who first showed these blobs are portable outside their original context, the authors identify the extension that turns portability into a scalable attack: the blobs are interchangeable **across models**, and a model family contains models with wildly different amounts of refusal training.

## What the paper claims

1. **Encrypted reasoning blocks are broadly compatible.** Across Anthropic, OpenAI and Google, a block produced in one session, by one user, on one model, is accepted in a different session, by a different user, on a different model in the same family. The authors map the exact compatibility matrix for 18 models.
2. **That compatibility is a decryption oracle.** Inject a frontier model's block into the weakest compatible sibling and ask it to transcribe, and it does. The frontier model's anti-distillation training is never engaged, because the frontier model is never queried.
3. **Four abuse vectors follow.** Distillation of proprietary reasoning, jailbreaking through the reasoning channel, third-party extraction of secrets from published traces, and invisible prompt injection carried inside a signed block.
4. **The fix is cryptographic binding, plus something more.** Bind each envelope to its user, its session and its predecessor. Then accept that a model that can be legitimately asked to process its own prior reasoning can always be socially engineered into reciting it, so training-level defenses are still required.

A note on scale before we start: the authors spent roughly \$30,000 on API credits, evaluated on 120 Codeforces problems, AIME 2025 and Humanity's Last Exam, and scraped 6,708 public agent trajectories. This is an empirical security paper, not a position piece.

## How encrypted reasoning actually works

### The envelope

**The problem it solves.** You want to hand a secret to an untrusted party for safekeeping and get it back later, unchanged. Two properties are needed and they are different: the party must not be able to *read* it (confidentiality), and must not be able to *alter* it without you noticing (integrity). Encryption alone gives you the first. A naive encryption scheme happily decrypts a ciphertext an attacker has flipped bits in, and hands you plausible-looking garbage. For a chain of thought this is not academic: an attacker who could edit the reasoning could steer the model's next turn by rewriting what it "already concluded".

**Intuition.** Think of a tamper-evident diplomatic pouch. The contents are sealed so a courier cannot read them, and the seal is stamped so the recipient can tell if anyone opened it. Critically, the stamp also covers the *label on the outside*: the destination, the pouch number, the date. If the courier swaps labels, the stamp no longer matches. What the seal covers is the whole point, and it is exactly where this system goes wrong. The label on these pouches says which *model* wrote the letter. It does not say who sent it, or which conversation it belongs to.

**The mechanism, semi-formally.** After the model finishes generating its hidden reasoning, the provider takes that text, draws a fresh nonce, and encrypts it under a key the provider holds. It also assembles a small header describing the block: model name, block type, format version, key ID. That header is not encrypted, but it *is* fed into the authentication computation as associated data, so it cannot be edited either. The result, base64-encoded, is what you see as `signature` on Anthropic, `thought_signature` on Gemini, and `encrypted_content` on OpenAI. On the next request, the client sends the blob back, the provider verifies the tag, decrypts, and splices the reasoning back into the model's context.

**The math.** The primitive is Authenticated Encryption with Associated Data. Let $\tau$ be the plaintext reasoning trace, $K$ the provider's key, $N$ a nonce, and $A$ the associated data. Sealing produces a ciphertext and a tag:

$$
(C, T) \;=\; \mathrm{Enc}_K\!\left(N,\; \tau,\; A\right), \qquad
\mathrm{envelope} \;=\; \left(A \,\Vert\, N \,\Vert\, C \,\Vert\, T\right)
$$

where $C$ is the ciphertext of the same length as $\tau$, and $T$ is a fixed-length authentication tag, typically 128 bits, computed as a Message Authentication Code over both $A$ and $C$. Opening runs the inverse:

$$
\mathrm{Dec}_K\!\left(N,\; C,\; A,\; T\right) \;=\;
\begin{cases}
\tau & \text{if } T = \mathrm{MAC}_K(N, A, C) \\
\bot & \text{otherwise}
\end{cases}
$$

The symbol $\bot$ means "reject". This is the whole security guarantee, and it is worth reading literally: decryption succeeds if and only if the tag matches what the key says it should be, given $N$, $A$ and $C$. **Every input to that check is carried inside the envelope itself.** Nothing about the caller enters the equation.

Write the acceptance predicate the API actually implements. Let $c = (u, s, m, n)$ describe the *replay context*: the calling user $u$, the session $s$, the model being queried $m$, and the position $n$ in the conversation. Then the API accepts an envelope $e$ when

$$
\mathrm{Accept}(e, c) \;=\; \big[\,\mathrm{Dec}_K(e) \neq \bot\,\big]
$$

The right-hand side does not mention $c$. That is the entire vulnerability, stated in one line. The paper's attacks are all corollaries of the fact that $c$ is a free variable.

![The stateless round trip: the provider seals the hidden reasoning into an envelope the client must store and replay, and the envelope binds the model but not the user, the session, or the position](/imgs/blogs/stealing-reasoning-traces-1.webp)

**A worked micro-example.** Here is the shape of the scheme with a real AEAD, so you can see what each argument does:

```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os, base64, json

K = AESGCM(os.urandom(32))          # the provider's key, never leaves the server
tau = b"Factoring 8139881: testing small primes 3, 7, 11, 13, 17 ..."

header = json.dumps({               # associated data: authenticated, NOT encrypted
    "model": "claude-opus-4-8",
    "type": "thinking",
    "v": 12,
    "kid": "k-2026-07",
}).encode()

N = os.urandom(12)                  # nonce, fresh per block
ct = K.encrypt(N, tau, header)      # ct = ciphertext || 16-byte tag
envelope = base64.b64encode(N + header + ct)

# --- what the guarantee buys you -------------------------------------------
K.decrypt(N, ct, header)            # -> tau
K.decrypt(N, ct, header[:-1] + b"X")# -> InvalidTag: header edits are caught
K.decrypt(N, ct[:-1] + b"\x00", header) # -> InvalidTag: ciphertext edits caught

# --- what it does not buy you ----------------------------------------------
# Nothing in `header` names a user, a session, or a turn index. So this same
# envelope verifies identically when it arrives on a request from a different
# account, in a different conversation, addressed to a different model.
```

Run the last three lines in your head. The tamper checks pass with flying colors. The envelope is cryptographically sound. And it is also completely portable, because portability was never what the tag was measuring.

**Why the design has this shape, and where it fails.** The paper is careful to note that this is not an oversight born of incompetence. Statelessness is the whole point, and statelessness pushes toward permissiveness: if the client holds the only copy, then features like mid-conversation model downgrades, automatic re-routing to a cheaper tier under load, and history compaction all require that a block issued by one model be acceptable to another. Each of those is a real product requirement. The failure mode is that the cheapest way to satisfy all of them is to bind *nothing*, and nobody wrote down the security consequence of the union.

As of July 2026, the authors note, **no provider publishes a description of the cryptographic mechanism at all**. The design above is inferred from behavior. That is itself a finding: a security-critical primitive shipped to millions of developers with no public specification.

## Three kinds of compatibility, three classes of attack

The paper's central conceptual move is to stop treating "the blobs are portable" as one fact and split it into three, ordered by permissiveness. Each level buys the platform a specific convenience and hands the attacker a specific capability.

![Three compatibility axes, the convenience each buys the platform, and the attack each hands the adversary](/imgs/blogs/stealing-reasoning-traces-2.webp)

### In-session and cross-session compatibility

A user can replay their own blocks out of order, or reuse a block from an old session in a new request. **Convenience:** benign history editing and context compaction, which every agent scaffold does constantly. **Capability:** the attacker can fabricate a conversation history around a captured thought, placing it wherever in the request it will do the most damage. This is the level the extraction attack itself runs on.

### Cross-user compatibility

A blob captured from *someone else's* session is accepted on your account. The paper is blunt that no provider has stated a convenience this buys, which makes it the purest of the three: pure attack surface, no product benefit named. **Capability:** anyone who can obtain another user's blobs (from a public repository, a shared dataset, an intercepted log) can decode them.

### Cross-model compatibility

A blob produced by one model is accepted by another in the same family. **Convenience:** seamless downgrades and re-routing, from Opus to Sonnet, or from Opus 4.8 back to Opus 4.6, without discarding the reasoning tokens the user already paid for. **Capability:** the whole attack. Because the models in a family differ enormously in how hard they have been trained to refuse.

The paper measures this exhaustively.

![Table 1 from Panfilov et al. (2026): cross-model compatibility of encrypted reasoning as of July 2026. Rows are the model that produced the block, columns the model receiving it](/imgs/blogs/stealing-reasoning-traces-fig2.webp)

**A worked micro-example: count the matrix.** Each vendor block is a ${6\times 6}$ grid of ordered (source, target) pairs, so 36 per vendor and 108 total. Read the ticks:

- **Claude.** Every row is fully accepting except Fable 5, whose thoughts only replay into Fable 5. That is ${5 \times 6 + 1 = 31}$ of 36, or **86%**. Note the asymmetry: Fable 5 will not *give up* its reasoning to a sibling, but it will happily *receive* everyone else's.
- **GPT.** The GPT-5.6 series (sol, terra, luna) accepts everything from every generation, but the older models are closed off. Each of the three 5.6 rows contributes 3, and GPT-5, GPT-5-mini and o4-mini each contribute 4 (the three 5.6 targets plus themselves). That is ${3\times3 + 3\times4 = 21}$ of 36, or **58%**.
- **Gemini.** All 36. **100%**.

Total: 88 of 108 accepted pairs, **81.5%** of all combinations. GPT is the most restrictive vendor by a wide margin and it is still more than half open.

The line that matters for the attack is not the average but the *minimum*. For each vendor, is there at least one weak model that accepts blobs from the strong ones? Claude: Haiku 4.5 accepts every source. GPT: Luna accepts every earlier generation. Gemini: everything accepts everything. Three for three.

## The decoder-oracle attack

![Figure 1 from Panfilov et al. (2026): the two-call extraction. An Opus 4.8 request returns a signed thinking block; sending only that signature to Haiku with a transcription request makes Haiku emit the Opus reasoning verbatim. Bottom: extracted traces track the API-reported thinking-token count across all three vendors](/imgs/blogs/stealing-reasoning-traces-fig1.webp)

<figure class="blog-anim">
<svg viewBox="0 0 780 320" role="img" aria-label="A sealed signature blob produced by Claude Opus 4.8 detaches, travels right across a dashed boundary marked different session, different user, different model, and lands in a Claude Haiku 4.5 turn that emits the verbatim reasoning in plaintext" style="width:100%;height:auto;max-width:860px">
<style>
.s1-panel{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.s1-hdr{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.s1-lbl{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.s1-onaccent{font:600 13px ui-sans-serif,system-ui;fill:var(--background,#ffffff);text-anchor:middle}
.s1-summary{fill:var(--background,#ffffff);stroke:var(--border,#d1d5db);stroke-width:1.5}
.s1-blob{fill:var(--accent,#6366f1)}
.s1-plain{fill:var(--background,#ffffff);stroke:var(--accent,#6366f1);stroke-width:2}
.s1-edge{stroke:var(--border,#d1d5db);stroke-width:2;stroke-dasharray:7 7;fill:none}
.s1-boundary{stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:6 8}
@keyframes s1-travel{0%{transform:translateX(0);opacity:0}7%{transform:translateX(0);opacity:1}22%{transform:translateX(0);opacity:1}52%{transform:translateX(430px);opacity:1}86%{transform:translateX(430px);opacity:1}94%,100%{transform:translateX(430px);opacity:0}}
@keyframes s1-emit{0%,56%{opacity:0;transform:translateY(10px)}70%,88%{opacity:1;transform:translateY(0)}100%{opacity:0;transform:translateY(10px)}}
@keyframes s1-cross{0%,30%{opacity:.25}45%,60%{opacity:1}80%,100%{opacity:.25}}
.s1-move{animation:s1-travel 12s cubic-bezier(.5,0,.2,1) infinite}
.s1-out{animation:s1-emit 12s ease-in-out infinite}
.s1-gate{animation:s1-cross 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.s1-move{animation:none;transform:translateX(430px);opacity:1}.s1-out{animation:none;opacity:1;transform:none}.s1-gate{animation:none;opacity:1}}
</style>
<rect class="s1-panel" x="16" y="70" width="250" height="190" rx="10"/>
<text class="s1-hdr" x="141" y="98">Claude Opus 4.8</text>
<text class="s1-lbl" x="141" y="120">turn n, hidden reasoning</text>
<rect class="s1-summary" x="46" y="136" width="190" height="42" rx="7"/>
<text class="s1-lbl" x="141" y="162">summary (stays behind)</text>
<rect class="s1-blob s1-move" x="46" y="196" width="190" height="44" rx="7"/>
<text class="s1-onaccent s1-move" x="141" y="223">signature (sealed)</text>
<line class="s1-boundary s1-gate" x1="390" y1="34" x2="390" y2="300"/>
<text class="s1-lbl s1-gate" x="390" y="22">different session, different user, different model</text>
<path class="s1-edge" d="M266 218 L 514 218"/>
<rect class="s1-panel" x="514" y="70" width="250" height="190" rx="10"/>
<text class="s1-hdr" x="639" y="98">Claude Haiku 4.5</text>
<text class="s1-lbl" x="639" y="120">same global key, weaker refusal</text>
<g class="s1-out">
<rect class="s1-plain" x="544" y="136" width="190" height="44" rx="7"/>
<text class="s1-lbl" x="639" y="163">verbatim reasoning</text>
</g>
<text class="s1-lbl" x="639" y="284">plaintext out, roughly 1:1 tokens</text>
</svg>
<figcaption>Encrypted thought injection in three beats: the frontier model seals a reasoning blob, the blob crosses a boundary it was never bound to, and a weaker sibling on the same key transcribes it back into plaintext.</figcaption>
</figure>

**The problem it solves.** Suppose you want a frontier model's hidden reasoning and you attack it directly. You face two walls. The model itself has been trained to refuse requests to disclose its chain of thought, and that training is best on the flagship. Around it sit system-level defenses: input filters that recognize extraction-shaped prompts, and output filters that substring-match the response against the reasoning the model just produced. You have to beat both, on the most-monitored endpoint the provider operates, and you have to keep beating them as they are patched.

**Intuition.** So do not attack the executive. Ask the intern. The intern has the same key to the same filing cabinet, because the company issued one key to everyone, and the intern has had none of the training about what is confidential. This is not a metaphor about intelligence; it is a metaphor about *authorization uniformity*. The security of the system was set by the strongest model's refusal behavior, but the key was distributed to all of them equally.

**The mechanism, step by step.**

1. **Capture.** Query the target model normally. Keep the signed reasoning block from the response. Throw away the visible answer, which is the part that was filtered.
2. **Choose a decoder.** Pick the weakest model in the family that still accepts the target's blocks. The paper's picks: Haiku 4.5 for Claude (weakest available, and it supports assistant-turn prefilling), GPT-5.6 Luna for OpenAI (least capable model that accepts all earlier generations), Gemini Robotics ER-1.6 for Google (accepts both the 2.5 and 3.x series, where 3.1 Flash Lite does not accept 2.5).
3. **Place the thought.** Build a request whose assistant turn carries the captured signature, and whose user turn asks for a verbatim transcription into a tagged block. On providers that allow it, prefill the assistant's visible output with the opening tag so the model's next token is already inside the transcription.
4. **Decode.** The provider verifies the tag, decrypts, and splices the plaintext reasoning into the decoder's context as if the decoder had thought it. The decoder now has the text in front of it and a request to copy it out. It copies it out.

The subtlety in step 4 is the one worth sitting with. The decoder is not "breaking" the encryption. From its point of view there is no encryption: the provider handed it plaintext, in the reasoning channel, labeled as its own prior thinking. The only thing standing between that plaintext and the attacker is whether the model feels like transcribing it. That is a *behavioral* property, and behavioral properties are exactly what is weakest on the cheap models.

![The security asymmetry inside a model family: two walls in front of the frontier model, one in front of its cheapest sibling, and the same key behind both](/imgs/blogs/stealing-reasoning-traces-3.webp)

### Where the thought goes in the request

The paper distinguishes two injection positions, and which one works is a per-model fact.

**Current-turn injection** puts the block in the assistant turn that is still being generated, so the model continues its visible answer directly out of it. As of July 2026 this was accepted by every GPT and Gemini model tested and by the Claude 4.5 generation. Sonnet 4.5, Haiku 4.5 and all Gemini models additionally accept prefilling the assistant's visible output, which is the stronger primitive: you get to write the model's first few tokens, and a model that has already started transcribing tends to finish.

**Past-turn injection** puts the block in a completed earlier assistant turn. This only works against models that do not strip previous reasoning blocks from the history, which the paper lists as Sonnet 5, Opus 4.8, Fable 5 and the GPT-5.6 series.

Notice the perverse incentive here. The newer flagships that *preserve* prior reasoning blocks across turns, a feature that exists to make long agentic sessions coherent, are precisely the ones exposed to past-turn injection. Coherence and isolation are trading against each other, and coherence has been winning.

### Measuring fidelity without ground truth

**The problem it solves.** You have a plaintext trace that claims to be the model's hidden reasoning. How do you know it is not a plausible hallucination? You cannot diff against the real trace, because the real trace is the thing you do not have. This is the methodological crux of the entire paper: without a fidelity measure, every result downstream is unfalsifiable.

**Intuition.** You cannot check the contents of a sealed box, but you can weigh it. The provider tells you exactly how many thinking tokens it billed you for, and it has a strong commercial incentive to get that number right. So the token count is a free, trustworthy, one-dimensional fingerprint of the hidden text.

**The math.** Let $\hat{\tau}$ be the extracted plaintext and $n_{\text{billed}}$ the thinking-token count the API reported for the original block. Re-encode $\hat{\tau}$ as input to the *same* model, so the same tokenizer applies, and compare:

$$
r \;=\; \frac{\lvert \hat{\tau} \rvert}{n_{\text{billed}}},
\qquad
\varepsilon \;=\; 1 - r
$$

where $\lvert \hat{\tau} \rvert$ is the token length of the extraction and $\varepsilon$ is the paper's **extraction error**. A value of $r = 1$ means the extraction has exactly the length of the hidden trace. For the strictest analyses the authors retain only traces satisfying $\lvert 1 - r \rvert \lt 0.05$.

**A worked micro-example.** Say the API bills 3,214 thinking tokens for a Codeforces problem, and re-encoding your extraction gives 3,180 tokens. Then $r = 3180/3214 = 0.9894$ and $\varepsilon = 0.0106$. Since $\lvert 1 - r \rvert = 0.0106 \lt 0.05$, this trace survives into the strict subset. The bottom row of Figure 1 is 120 such measurements per vendor, and the point clouds hug the $y = x$ diagonal.

**Why it works and where it breaks.** Length is a necessary but not sufficient condition for faithfulness. A fluent paraphrase of the same reasoning would have roughly the right length while being word-for-word wrong. The authors know this and say so plainly: "we cannot guarantee that the extracted thoughts correspond exactly to a model's private reasoning". They shore it up with two independent corroborations. First, the extractions are *qualitatively more detailed* than the API's own summaries, which they should be if genuine and would not be if hallucinated from the summary. Second, and much more convincingly, the extractions contain **specific secrets that appear nowhere in the visible input**: API keys, passport numbers, exact file paths. A hallucinating decoder does not invent a well-formed HuggingFace token that turns out to be real. This is the strongest evidence in the paper that the decoding is genuine, and it arrives as a side effect of a completely different experiment.

### Reconciliation: three bad photocopies beat one

Sampling at temperature 1 gives you noisy extractions. Some are refusals, some echo the prompt, some drift. The paper adds an optional **reconciliation** pass that is a neat piece of engineering: take up to three non-refusal extractions from the weak decoder, hand them *plus the original signed block* to the strong model, and ask it to reconstruct the single faithful transcription, at temperature 0.

Read that again, because the roles have inverted. The frontier model, the one whose reasoning is being stolen, is now the error-correcting decoder in the loop. It works because from the flagship's perspective this genuinely is a transcription task over content already in its context, not a memory it has to be tricked into recalling. It is a noisy-channel decoder where the channel is the cheap model and the codeword is the reasoning, and the strong model is the only entity that can score candidate decodings well.

### What it cost per provider

The three vendors were not equally easy, and the differences are informative:

| Vendor | Decoder | Reconciler | Candidates sampled | What made it hard |
| --- | --- | --- | --- | --- |
| Anthropic | Haiku 4.5 | Opus 4.8 | 1 template, robust at temperature 1 | Little. A single fixed prompt worked across every source model, at roughly a 1:1 token ratio |
| OpenAI | GPT-5.6 Luna | none | up to 50 (10 for the secret-extraction runs) | API-level rejection once a completion reproduced more than about 50 consecutive tokens of the original; quality varied with trace source and length |
| Google | Gemini Robotics ER-1.6 | Gemini 3.5 Flash | up to 20, best 3 reconciled | High noise in individual decodings |

OpenAI's ~50-token substring rejection is the only *system-level* anti-distillation defense that showed up as a real obstacle, and the workaround is instructive: the authors chunked the extraction into sub-50-token pieces and stitched them by suffix and prefix overlap. A filter that caps contiguous reproduction does not prevent reproduction, it prices it. The attacker pays more round trips and gets the same text.

The economics are the actual finding here. At Haiku 4.5 rates, decoding a corpus of 10,000 traces with 12k-token input and output windows costs about **\$720**:

$$
\underbrace{10^4 \times 1.2\times10^4}_{1.2\times10^8 \text{ input tokens}} \times \frac{\$1}{10^6}
\;+\;
\underbrace{10^4 \times 1.2\times10^4}_{1.2\times10^8 \text{ output tokens}} \times \frac{\$5}{10^6}
\;=\; \$120 + \$600 \;=\; \$720
$$

Ten thousand frontier-model reasoning traces for the price of a laptop. And if the blobs were harvested from someone else's public logs, the expensive half of the computation, generating the reasoning, was paid for by the victim.

## Four ways to abuse it

### 1. Distillation: reasoning is denser supervision than answers

**The problem it solves, for the attacker.** Black-box distillation from a proprietary model is old news: query it, collect outputs, train a student on them. But a final answer is only the *endpoint* of the teacher's computation. The student has to re-derive the latent trajectory that produced it, which is exactly the hard part.

**The math of why traces are worth more.** Answer-only distillation trains on

$$
\mathcal{L}_{\text{answer}} \;=\; -\sum_{t=1}^{\lvert y \rvert} \log p_\theta\!\left(y_t \mid x,\, y_{\lt t}\right)
$$

where $x$ is the prompt, $y$ the teacher's visible answer of length $\lvert y \rvert$, and $\theta$ the student's parameters. With the trace $\tau$ recovered, the student instead trains on

$$
\mathcal{L}_{\text{trace}} \;=\; -\sum_{t=1}^{\lvert \tau \rvert} \log p_\theta\!\left(\tau_t \mid x,\, \tau_{\lt t}\right)
\;-\; \sum_{t=1}^{\lvert y \rvert} \log p_\theta\!\left(y_t \mid x,\, \tau,\, y_{\lt t}\right)
$$

The first sum is new and it is where the value is: it supervises the *decomposition*, the intermediate deductions, the dead ends and the backtracks, position by position.

**A worked micro-example.** Take a typical Codeforces problem from the paper's set. The model burns 8,000 thinking tokens and emits an 800-token answer. Answer-only distillation gives the student 800 supervised positions. Trace distillation gives 8,800, an **11× increase in supervised positions per dollar of teacher inference**, and the extra positions are the ones carrying the reasoning structure rather than its summary.

The paper cites [Zhang, Morris and Shmatikov (2026)](https://arxiv.org/abs/2603.07267) for the downstream effect: training a separate trace-inversion model to *synthesize* long-form reasoning from a victim's visible outputs and summaries alone lifted a fine-tuned Qwen2.5-7B-Instruct from **68.4% to 76.0% on MATH500** over answer-only distillation. That pipeline recovered only surrogate approximations. This paper's attack recovers the genuine traces verbatim, without ever engaging the safeguarded model, across both math and coding.

**When it fails.** The extraction is fuzzy. If a fraction of tokens are wrong in ways correlated with the decoder's own idiom, you are partly distilling Haiku's style dressed as Opus's substance. The paper does not quantify per-token accuracy, only length agreement, so the true supervision quality of a stolen trace is unmeasured. This is the single biggest open question in the paper's distillation story.

### 2. Jailbreaking through the reasoning channel

**The problem it solves, for the attacker.** Safety training is applied where the harm surfaces: the visible output. Applying it to the chain of thought is deliberately avoided, because optimizing the content of the CoT teaches models to hide their reasoning rather than to not have it, which destroys the monitorability that makes CoT valuable for oversight in the first place. That is a defensible design decision. It also means the reasoning channel is, by construction, less filtered than the output channel.

**The mechanism.** Ask the model a question that requires it to reason extensively through a harmful topic, but instruct it to keep the answer benign. It complies on both counts: the deliberation happens in the notepad, the press release stays clean. Then discard the answer, keep the block, and decode it.

The paper's proof of concept paraphrases a [HarmBench](https://proceedings.mlr.press/v235/mazeika24a.html) prompt about vehicle theft. Opus 4.8's visible answer is a perfectly reasonable list of recommendations for automakers: immobilizers as standard equipment, ultra-wideband against relay attacks, and so on. Its hidden reasoning contains the specific attack inventory: which model years lacked immobilizers, the steering-column-and-USB technique, relay amplification, CAN bus injection, weak rolling codes, OBD port exploitation.

Neither half is a catastrophe on its own. The point is structural: **the output filter and the reasoning are on opposite sides of a boundary the attacker can now cross.** Every uplift-shaped concern that safety training addresses at the output layer is unaddressed one layer in.

### 3. Secret extraction at scale

This is the section that turns an interesting API flaw into something with a compliance department attached.

**Intuition.** You can only redact what you can read. Developers publish agent trajectories constantly, for reproducibility, for bug reports, for benchmarks. Conscientious ones scrub the plaintext: strip the API keys, mask the emails, replace the paths. The encrypted blocks pass through untouched, because they are opaque, and because there is no tool that would tell you what is in them. The user cannot decrypt their own reasoning. **A third party can.**

**The pipeline.** The authors collected 6,708 public agent trajectories from GitHub and Hugging Face produced by Claude, GPT and Gemini models that still carried reasoning blocks, and decoded every signed block: **315,320 reconstructed traces**. Then a two-stage LLM-as-a-judge:

- **First pass** (high recall, Haiku 4.5 as labeler) flags any block that might contain a privacy violation and extracts candidate items under a fine-grained taxonomy. It flags 27,165 of 315,320 blocks (**8.6%**), broken down as 14,876 of 237,209 for GPT sources and 12,289 of 78,111 for Claude sources, yielding **76,778 candidate items**.
- **Second pass** re-labels each flagged item as a genuine artifact or a non-artifact, killing the placeholders (`sk-xxxx`), bare environment-variable names and benchmark fixtures. Of 6,950 flagged blocks judged, **1,028 retain at least one real artifact**, totaling 2,194 genuine items.
- **Deduplication** by category and value: **941 distinct values**.
- **Excluding benchmark sessions** (PostTrainBench, TerminalBench, ClawBench): **704 artifacts from genuine user sessions**.
- **Reasoning-only**: of those 704, **64 appear nowhere else in the raw session**.

![The secret-extraction funnel: 6,708 public trajectories down to 64 artifacts that exist only inside the encrypted reasoning](/imgs/blogs/stealing-reasoning-traces-4.webp)

**The results.** Per block the rate looks small: 0.3% (1,028 of 315,320) contain at least one leak. Per *trajectory* it is much worse: **4.9% (328 of 6,708) sessions leak at least one real sensitive item.** That is the number that matters, because sessions are the unit people publish.

From genuine, non-benchmark user sessions the recovered secrets include **62 distinct API keys, 33 passwords, 24 access tokens, 7 private keys, 30 personal emails and 6 non-localhost IP addresses**, alongside 130 names and 36 postal addresses.

![Figure 6 from Panfilov et al. (2026): distinct privacy artifacts recovered from publicly posted traces, grouped into three headline categories, all sources included](/imgs/blogs/stealing-reasoning-traces-fig3.webp)

Including benchmark sources brings the total to 912 distinct artifacts across the three categories: 367 PII, 363 technical identifiers, 182 credentials. Benchmark rollouts inflate the PII column because harnesses like ClawBench hand the agent a complete synthetic persona to reason over. The authors flag this clearly rather than banking the bigger number, which is the right call.

**The finding inside the finding.** The 64 reasoning-only artifacts are the ones that should change how you think about this. They exist for two possible reasons, and the paper names both: the user sanitized the visible transcript before publishing and could not sanitize what they could not read, or the model pulled the value in from its own memory or context and restated it only in the hidden channel.

The authors identify a specific recurring trigger that is almost too on-the-nose: **conversation cleanup**. Ask an agent to anonymize or clean up a session, and it re-reads the whole history in its hidden reasoning and *restates every sensitive value it is about to remove*. The act of redacting writes the secrets into the one channel the redactor cannot see. The paper's own Figure 5 shows exactly this: a Codex agent asked to sanitize a repository, dutifully listing the AWS secret and HuggingFace token it is about to strip, inside a block the user then published.

And note the arithmetic the authors do here, which is the sharpest sentence in the section: only about 9% of artifacts were reasoning-exclusive, but **even if every user had perfectly scrubbed the visible text, all 62 API keys would still have shipped inside the blocks.** Plaintext-only sanitization is not partially effective against this. It is not effective at all.

The scope caveat is real and the authors state it: this is a non-exhaustive scan of *publicly shared* traces. Local transcripts and production service logs handle far more sensitive material and are far more numerous.

### 4. Invisible prompt injection

The fourth vector runs the attack backwards. Instead of *reading* a blob, you *plant* one.

**The mechanism.** Long-horizon agentic runs are expensive to reproduce, so people resume published or shared runs rather than starting over. Encrypted reasoning blocks are what make that resumption possible. An attacker who plants a block whose internalized reasoning carries a malicious instruction into a shared trace has planted something that, when the victim resumes, the model treats **as its own prior conclusion**.

That is categorically different from ordinary prompt injection. A conventional injection is a suspicious instruction sitting in the visible input, where a user or a monitor can read it. This payload lives in the hidden channel, is cryptographically signed by the provider, and is invisible to anything inspecting the observable conversation. You are not planting an instruction. You are planting a memory.

**Proof of concept.** The authors query o4-mini with a paraphrased injection instructing the model to back up any PowerPoint file to an attacker-controlled URL, and capture the resulting thought. They then port that block to GPT-5.6 Sol and ask an unrelated question: write a script to add a closing slide to a `.pptx`. Sol treats the injected trace as its own prior reasoning and produces a script that adds the slide *and* uploads the presentation to the attacker's server. The malicious thought transfers across model scales and across tasks, leaving no plaintext artifact.

They then scale it: injecting data-exfiltration instructions into blocks in long-horizon [PostTrainBench](https://arxiv.org/abs/2603.08640) traces, with Opus 4.7 running in a Claude Code scaffold. Haiku 4.5 generates the thought carrying the injection, which is spliced in near the end of the trace. When the victim continues the run, Opus 4.7 follows the planted instruction and uploads a research-methodology log to the attacker's server after every update.

Two properties compose badly here. Cross-model compatibility means the cheap model can *manufacture* the payload, so the attacker never needs frontier access. Cross-user compatibility means the payload lands in someone else's session. Together they make this a supply-chain attack on published agent traces, and the ecosystem currently has no notion of trace provenance at all.

## The incidental finding: summaries are not the reasoning

Buried in Appendix C is a result that has nothing to do with security and may outlast the security finding.

Because the API's summary is what users actually see, the authors could compare it against the decoded reality. The gap is large:

![Figure 40 from Panfilov et al. (2026): for each Codeforces problem, hidden thinking tokens generated versus the token length of the displayed summary. The dashed line is y = x](/imgs/blogs/stealing-reasoning-traces-fig4.webp)

The point cloud sits far below the diagonal across every Claude model tested. **Decoding the signature reveals roughly five times more reasoning than the summary exposes.** The summarizer is itself a cheaper, less capable model condensing the work of a more capable one, and the artifacts are what you would expect from that arrangement.

On a strict subset of AIME 2025 traces whose decoded length matched the billed length within 5% (18 Opus 4.8 traces, 15 GPT-5.6 Sol traces), manually inspected pair by pair:

- **In 9 of the 18 Opus traces, the hidden reasoning states the answer before deriving it.** The model recalls a memorized result, then constructs a derivation. In 8 of those 9, the summary faithfully reports the answer in advance too, which is at least honest.
- In the ninth, the single distinguishing phrase changes: "Let me verify by computing" becomes "Let me set up coordinates" in the summary. The surrounding computation is identical. **The summary turns a verification into an independent derivation**, which is the difference between a model that knew the answer and a model that found it.
- Elsewhere: a hedged recollection is rendered in the summary as a definite value two sentences before the reasoning actually computes it, and one summary captures only the tail of the reasoning, which concerns answer formatting and contains no mathematics at all.

This matters beyond trivia. If the raw reasoning is inaccessible, the summary is the *only* interface users have for scalable oversight. A summary that launders a post-hoc rationalization into a clean derivation is not a transparency mechanism; it is a transparency-shaped object. And a summary that reads well is more dangerous than no summary, because it terminates inquiry.

## The elephant in the room: were open models distilled on proprietary traces?

Appendix B is the paper's most interesting and most epistemically fraught section, and the authors fence it with an explicit disclaimer I want to reproduce faithfully: **this analysis cannot establish causation.** It runs on a small, benchmark-skewed problem set, on traces recovered by a fuzzy procedure, across serving configurations the authors do not control, after the vulnerability was patched.

With that stated, here is what they found.

### Style drift under a 1% prefill

**Setup.** Take a decoded Opus 4.8 reasoning trace. Insert **the first 1%** of it at the start of an open-weight model's reasoning channel, then let the model generate the rest of its reasoning and its entire visible answer freely. The visible answer is never prefilled. Compare against an unprefilled control and against a self-prefill control.

**Metric.** Following [Barbero et al. (2026)](https://openreview.net/forum?id=3XXb2MK02l), draw batches of $k \in \{1, 10, 50, 100\}$ completions, and within each batch select the completion sharing the most 1-, 2- and 3-grams with the first 100 tokens of the corresponding Opus visible answer. Best-of-$k$ tests whether the sampling pool *contains* an Opus-styled answer; capping at 100 tokens controls for length bias. Evaluated on 30 HLE problems, 15 STEM and 15 non-STEM.

**Result.** Kimi-K3 shifts hard and the control model, Inkling, does not:

| Model | Prefill source | Category | Prefilled | Control | $\Delta$ | $p$ |
| --- | --- | --- | --- | --- | --- | --- |
| Kimi-K3 | Opus 4.8 | STEM | 0.305 | 0.160 | **+0.15** | ${1.7\times10^{-5}}$ |
| Kimi-K3 | Opus 4.8 | non-STEM | 0.289 | 0.203 | **+0.086** | ${6.3\times10^{-6}}$ |
| Inkling | Opus 4.8 | STEM | 0.217 | 0.205 | +0.012 | ${7.4\times10^{-2}}$ |
| Inkling | Opus 4.8 | non-STEM | 0.241 | 0.239 | +0.002 | ${5.5\times10^{-1}}$ |
| Kimi-K3 | Inkling | STEM | 0.359 | 0.337 | +0.022 | ${1.2\times10^{-1}}$ |
| Inkling | Kimi-K3 | STEM | 0.414 | 0.411 | +0.002 | ${7.2\times10^{-1}}$ |

Kimi-K3 shows higher overlap in the prefilled condition on **29 of 30 problems**. After Bonferroni correction across the eight comparisons, only the two Kimi-K3-under-Opus rows survive, by three orders of magnitude. The cross-prefill controls, where Kimi-K3 and Inkling prefill each other and no proprietary reasoning is involved, are all null.

That control design is the strongest part of the analysis. It rules out "any cross-model prefill shifts style" as an explanation. Whatever is happening is specific to Opus reasoning entering Kimi-K3.

### Probabilistic extraction: can they reproduce it verbatim?

**The math.** Following [Hayes et al. (2025)](https://aclanthology.org/2025.naacl-long.469/), for a target span $z$ of $k$ tokens, one teacher-forcing pass gives $p_z$, the probability that a single temperature-1 sample reproduces the span exactly. Over $n$ independent queries:

$$
P(\text{extract within } n \text{ queries}) \;=\; 1 - (1 - p_z)^n
$$

**A worked micro-example.** Invert it to get the query budget for a coin-flip chance. Setting the left side to ${0.5}$ and solving,

$$
n \;=\; \frac{\ln(0.5)}{\ln(1 - p_z)} \;\approx\; \frac{\ln 2}{p_z} \quad \text{for small } p_z
$$

So a span with $p_z = 10^{-10}$ needs about ${6.9\times10^{9}}$ queries for even odds. With $k = 16$ tokens, that is the regime the paper reports.

**Result.** Conditioned only on the problem, **no evaluated model shows practical verbatim memorization** of the decoded traces. Kimi-K3 is the most extractable and still needs on the order of ${10^{10}}$ queries on HLE. Conditioning on the first 1% of the Opus trace helps, but not enough to matter: GLM-5.2 drops from ${10^{14}}$ to ${10^{11}}$; Kimi-K2.6 and Kimi-K3 stay around ${10^{11}}$ and ${10^{10}}$. DeepSeek-V4-Flash and Inkling sit at ${10^{14}}$ and ${10^{16}}$, four to six orders of magnitude worse.

The **visible answer** is a completely different story. With a 1% reasoning prefill, Kimi-K3 reproduces the next 16 tokens of the Opus answer in about ${4\times10^{5}}$ queries, and about ${10^{5}}$ with the complete decoded trace in context. GLM-5.2 and Kimi-K2.6 follow at roughly ${10^{7}}$ and ${10^{9}}$, while Inkling and DeepSeek-V4-Flash need more than ${10^{14}}$ in either condition. On AIME 2025, a 16-token span of the GPT-5.6 Sol visible answer is reproducible by Kimi-K3 in as few as ${10^{2}}$ queries when conditioned on Kimi-K3's own reasoning for the same problem.

The authors summarize it precisely: relative to conditioning on their own reasoning, having the source trace in context lowers the cost of reproducing the Opus answer by roughly **13 orders of magnitude for Kimi-K3 and GLM-5.2**, against 3 for Kimi-K2.6. The effect is concentrated in the visible answer, not the reasoning.

### Perplexity: whose reasoning does each model find native?

**The math.** For a reasoning trace $t$ associated with problem $q$, render the scorer's native chat template up to the start of its reasoning channel, append $t$, and compute the conditional log-probability of every reasoning token:

$$
\mathrm{PPL}(t \mid q) \;=\; \exp\!\left(-\frac{1}{\lvert t \rvert} \sum_{i} \log p\!\left(t_i \mid q,\, t_{\lt i}\right)\right)
$$

where $\lvert t \rvert$ is the number of reasoning tokens and $t_{\lt i}$ the tokens preceding position $i$. Perplexity is the exponential of the mean negative log-likelihood: a value of ${2.74}$ means the scorer is, on average per token, as uncertain as if choosing uniformly among 2.74 options.

![Figure 26 from Panfilov et al. (2026): median perplexity of reasoning traces under seven scoring models, over 120 Codeforces problems. Rows are the trace source, columns the scorer, the italic diagonal is a model scoring its own reasoning](/imgs/blogs/stealing-reasoning-traces-fig5.webp)

Two findings, one methodological and one substantive.

**The methodological one is genuinely strange.** Every model except Kimi-K3 and Kimi-K2.7-Code assigns **higher** perplexity to its own reasoning than to reasoning from other models. A model's native trace is not the trace it considers most probable. This is a warning about the whole family of similarity-by-perplexity arguments, and the authors could easily have buried it. They did not.

**The substantive one:** the GPT-5.6 family's traces are wildly out-of-distribution for almost every open scorer. GPT-5.6 Terra scores 106.38 under gpt-oss 120B, 62.67 under Inkling, 32.60 under DeepSeek V4 Flash. Under Kimi-K3 it scores **8.49**, an order of magnitude lower. The same ordering holds for Opus 4.8: 7.16 under gpt-oss 120B, 5.84 under Inkling, and 2.74 under Kimi-K3, which is close to Kimi-K3's 1.79 on its own reasoning. Under GLM-5.2, the four trace sources whose perplexity sits closest to GLM-5.2's own are four consecutive Anthropic releases.

The authors immediately add that perplexity is a coarse metric that may not capture fine-grained distributional fit and should not be read as confirmatory.

### What this does and does not establish

Three measurement families, three consistent orderings. Kimi-K3 and GLM-5.2 continue proprietary reasoning more readily, model proprietary text at lower perplexity, and reproduce proprietary visible answers far more cheaply than DeepSeek-V4-Flash or Inkling do.

What it establishes: **unusual behavioral compatibility under the specific interventions tested.** What it does not establish: memorization, or that anyone trained on anyone's traces. Alternative explanations survive comfortably. Both effects would appear if these models were trained on publicly available *summaries* rather than raw traces, or on the very large volume of Claude-generated and GPT-generated text now circulating in open datasets, or if they simply share more post-training recipe DNA with the frontier labs. There is no ablation here that separates "trained on proprietary reasoning" from "trained on a corpus saturated with proprietary outputs".

I think the authors got the epistemics right and the framing slightly wrong. Titling the appendix "The Elephant in the Room" primes a conclusion that the disclaimer then withdraws. The measurements deserve to be read as what they are: a well-controlled demonstration that reasoning-channel prefills produce model-specific, source-specific behavioral signatures. That is a useful forensic tool, and it may be more durable than any claim about who trained on what.

## Results at a glance

| Claim | Measurement | Source in the paper |
| --- | --- | --- |
| Blobs are cross-model compatible | 88 of 108 (source, target) pairs accepted; Gemini 36/36, Claude 31/36, GPT 21/36 | Table 1 |
| Extraction is faithful by length | Extracted-to-billed token ratio tracks $y = x$ over 120 Codeforces problems, all three vendors | Figure 1 |
| Extraction is cheap | ~\$720 to decode 10k traces at Haiku 4.5 rates; ~\$30,000 total API spend for the paper | §3.1, Reproducibility |
| Reasoning hides harmful content the answer refuses | HarmBench paraphrase: benign answer, detailed attack inventory in the trace | Figure 4 |
| Public traces leak real secrets | 6,708 trajectories → 315,320 blocks → 704 genuine artifacts; 4.9% of sessions leak | §4.1, Table 4 |
| Plaintext sanitization does not help | 62 API keys would have shipped regardless; 64 artifacts are reasoning-exclusive | Table 4 |
| Injection survives model and task transfer | o4-mini payload executed by GPT-5.6 Sol on an unrelated task | §4.2 |
| Summaries understate the reasoning | Displayed summary ≈ 1/5 of hidden trace length | Figure 40 |
| Summaries can misrepresent it | 9 of 18 Opus traces state the answer before deriving it | §C.4 |
| Open models show source-specific drift | Kimi-K3 n-gram overlap +0.15 STEM under 1% Opus prefill, ${p = 1.7\times10^{-5}}$ | Table 3 |
| No verbatim memorization found | ≥ ${10^{10}}$ queries to reproduce a 16-token reasoning span | §B.3.1 |

**What is load-bearing in their setup that may not transfer.** Three things. First, the entire result is pinned to API versions available in early July 2026, and the cryptographic implementations are proprietary and change without announcement. Second, the fidelity measure is length agreement, which is necessary but not sufficient, so per-token accuracy is unknown. Third, the corpus of 6,708 trajectories is a targeted, non-exhaustive scan of public repositories, and the authors are explicit that it should be read as a demonstration rather than an audit.

## The defense: bind the envelope to its context

Appendix A is a real engineering proposal rather than a paragraph of hand-waving, which is rarer in attack papers than it should be. The whole design follows from the diagnosis: the AEAD envelope authenticates content but not context, so put the context inside the authenticated data.

![The context-bound envelope defense: today every envelope stands alone, and under the proposal each one is chained to its user, its session and its predecessor](/imgs/blogs/stealing-reasoning-traces-5.webp)

### Bind the user

The cheapest fix, and it closes an entire vector on its own. Embed the originating `user_id` inside the AEAD associated data. On replay, compare the bound identifier against the already-authenticated caller and reject on mismatch. In terms of the predicate from earlier, this changes

$$
\mathrm{Accept}(e, c) \;=\; \big[\,\mathrm{Dec}_K(e) \neq \bot\,\big]
\quad\longrightarrow\quad
\mathrm{Accept}(e, c) \;=\; \big[\,\mathrm{Dec}_K(e) \neq \bot\,\big] \;\wedge\; \big[\,u_e = u_c\,\big]
$$

where $u_e$ is the user bound into the envelope and $u_c$ the authenticated caller. No stateful backend is required, because the identity travels inside the envelope. Cross-user replay dies. Every third-party attack in Section 4 dies with it.

That this was not done from the start is the paper's most pointed observation, and it is hard to disagree: "It is unclear why a user and/or a conversation identifier is not added directly inside the envelope."

### Chain the session

Cross-session replay is harder, because the legitimate workflows depend on the portability. Users fork conversations, compact old turns out of long sessions, and downgrade models mid-conversation. Binding each block to the literal complete prior transcript would break all three.

The proposal is a lightweight hash chain binding each block only to its session and its **immediate predecessor**:

$$
\tau_{n+1} \;=\; H\!\left(\texttt{user\_id} \,\Vert\, \texttt{session\_id} \,\Vert\, H(\tau_n \,\Vert\, \texttt{salt}_2) \,\Vert\, \texttt{salt}_1\right)
$$

where $\tau_n$ is the reasoning content of block $n$, $H$ is a cryptographic hash, $\Vert$ is concatenation, and the two salts are secret values held by the provider. The resulting digest goes into the associated data of the *following* block's envelope.

**A worked micro-example.** Suppose an attacker captures envelope $e_3$ from Alice's session $S$. Its associated data carries $\tau_3 = H(\texttt{alice} \Vert S \Vert H(\tau_2 \Vert \texttt{salt}_2) \Vert \texttt{salt}_1)$. The attacker replays it in their own fresh session $S'$ as user Mallory. The server recomputes what the chain digest should be for position 1 of $(\texttt{mallory}, S')$, and gets a completely different value, because both `user_id` and `session_id` are inputs. Mismatch, reject. To succeed, the attacker now needs the *entire preceding session in order*, not a single captured signature.

The authors are honest that this does not stop a determined adversary who replays a whole conversation faithfully. What it does is convert a one-signature attack into a whole-session attack, which raises the cost by orders of magnitude and shrinks the blast radius to something auditable. The scalable version of the attack, the one that decodes 315,320 blocks scraped from arbitrary repositories, is dead.

### The four properties

The proposal names four design properties, and the interesting part is that two of them are in tension:

| Property | Statement | Cost |
| --- | --- | --- |
| **P1 Reasoning ordinality** | Block $X$ is verifiably prior to block $Y$ whenever $Y$ depends on $X$ | Cheap. Excise dropped chunks from the chain, relative order of survivors intact |
| **P2 Full reasoning integrity** | Block $X$ must be preceded by exactly block $X-1$, with no elision | Expensive. Removing an internal node forces recomputing every downstream hash |
| **P3 Leak monitoring** | Continuous verbatim matching between published plaintext and any decoded reasoning surfacing through abuse reports | Operational, not cryptographic. Detects rather than prevents |
| **P4 Non-replayability** | The same block is never accepted twice, in any session | Requires server-side consumed-envelope state |

P2 is what you want and compaction is what you need, and they fight. The resolution is a **Merkle tree** over the session's blocks: leaves are individual envelopes, and the provider retains only the root (or a few subtree roots) after leaves are pruned. That gives P1 everywhere for almost nothing, and P2 on demand for any surviving contiguous span, without ever replaying the full unpruned history. Forks and model downgrades are handled uniformly by branching the tree: a fork inherits chain state up to the branch point and continues independently.

I like this design. It is the same shape as a certificate transparency log, and it is the right shape: you cannot prevent every replay while preserving compaction, so you make replay *detectable and bounded* instead of impossible.

### Legacy data, and the part nobody wants

None of this protects data already published, including the 6,708 trajectories the authors surveyed. Old envelopes were signed under a key encoding neither user nor session, so the only retroactive remedy is **key invalidation**: rotate every pre-fix signing key and refuse to decode any envelope under a retired key ID.

That renders previously published signatures permanently undecodable, and also permanently breaks legitimate continuations of old sessions. There is no version of this that is free. The migration path is a bounded dual-format acceptance window plus an opt-in batch re-signature endpoint, gated on verifying that the requesting account owned the original session, after which legacy envelopes are rejected outright.

Two smaller items round out the table. **Nonce predictability** matters more than usual here precisely because the key is shared across all users: draw every nonce from a CSPRNG and enforce server-side uniqueness before issuance. And **infrastructure guardrails**: gateways that reject envelopes generated by a different model version than the one being queried, plus velocity and anomaly detection for accounts submitting identical signatures across disparate sessions or triggering elevated decryption-error rates.

### Why cryptography cannot finish the job

The paper closes the mitigations section with the observation that should be the headline, and I want to give it its own space because it is the durable part:

> whatever model is queried must, by necessity, decrypt and process the contents of prior reasoning tokens.

Cryptographic binding constrains *which* model may be asked to open a given envelope. It cannot constrain what a compliant model does once it has been legitimately asked to process its own prior reasoning. The model **is** the decryption key, in the only sense that matters. So unless the model is fully robust against prompt-based extraction, which no model is, encrypted reasoning blocks can never be more than **semi-hidden**.

Hence the training-time item: post-train models to recognize and refuse transcription-style framings regardless of envelope validity and regardless of how innocuous the surrounding request looks. The authors list this as standing future work rather than a solved component, which is the honest framing. It is also a defense with an unpleasant property: it is a refusal behavior, and refusal behaviors are what the whole attack was built to route around.

The practical takeaway for anyone building on these APIs: **never treat an encrypted reasoning block as a confidential storage mechanism.** Treat it as plaintext you happen to be unable to read.

## Should reasoning be encrypted at all?

The paper poses the question and refuses to give a clean answer, which I respect. Evidence points both ways.

**For encryption:** letting a model reason through harmful information without divulging it seems genuinely useful, and is exactly what Section 3.2 shows working right up until the channel is breached.

**Against:** opacity enables the injection attacks of Section 4.2 and makes the privacy violations of Section 4.1 nearly undetectable, because the victim cannot inspect what they are publishing.

A third option gets one paragraph and deserves more: **ephemerality**. Providers could let models reason before every output and then delete the reasoning, neither storing it nor returning it. Several already support this mode, and modern Qwen models expose it via a `preserve_thinking` parameter. This eliminates the entire attack surface by eliminating the artifact. The cost is that multi-turn reasoning continuity is lost, which for long agentic runs is a real regression rather than a nuisance.

And the authors close with an argument I did not expect from a security paper. Setting aside anti-distillation economics, from a pure safety standpoint, **giving users unredacted reasoning may be preferable**. Rather than restricting oversight to a small internal safety team, providers could enlist their entire user base for pluralistic human oversight of model reasoning. Their concrete suggestion: disable encryption for older, non-frontier model generations, where the commercial argument is weakest and the oversight benefit is unchanged.

## Critique

### What is genuinely strong

**The threat model is disciplined.** A standard unprivileged API adversary: no insider access, no server-side state, no model weights, operating entirely within normal API usage. Every result respects that boundary. There is no step where the attack quietly assumes something an ordinary customer would not have.

**The fidelity problem is confronted rather than assumed away.** The token-ratio measure is a genuinely clever use of the one number the provider has a commercial incentive to report honestly, and the authors state its limits instead of overselling it. The accidental corroboration, that extractions contain real secrets absent from the visible input, is worth more than the primary measure and the authors do not oversell that either.

**The controls in Appendix B are better than the section needed them to be.** The cross-prefill control, where Kimi-K3 and Inkling prefill each other with no proprietary reasoning involved, is the experiment that separates a real finding from an artifact of prefilling in general. Reporting that models assign higher perplexity to their own reasoning than to others', a result that undercuts the section's own method, is the kind of thing that gets cut from weaker papers.

**The defense is engineered.** Equation 1, the P1 to P4 property analysis, the Merkle resolution of the compaction tension, the dual-format migration window, the key-rotation cost stated plainly. This is a proposal a provider could actually read and cost out.

**The disclosure was handled properly and is documented.** Providers, Microsoft and Hugging Face were notified with full technical details. Recovered secrets were processed in an isolated environment and deleted after aggregate counting. The reproducibility statement states outright that the headline results no longer reproduce.

### What is weak

**Per-token faithfulness is never measured.** Length agreement is the only fidelity metric, and it is compatible with a fluent paraphrase. Every downstream claim about distillation utility inherits this gap. There was an available experiment here: run the extraction against an open-weight model where ground-truth reasoning *is* observable, and report edit distance. The paper's own Appendix B uses open-weight models extensively, so the infrastructure existed. Its absence is the biggest hole.

**The "elephant in the room" framing outruns the evidence.** The disclaimer is correct and prominent, but the section title and placement invite a reading the data does not support. And the specific missing control is nameable: no experiment separates "trained on proprietary reasoning" from "trained on a corpus saturated with proprietary outputs and summaries". Given how much Claude-generated and GPT-generated text is now in open datasets, that alternative is not exotic, it is the default hypothesis.

**Two of the four attack vectors rest on single anecdotes.** The jailbreak is one HarmBench paraphrase about car theft, chosen because it elicits long reasoning. The prompt injection is one PowerPoint payload plus one PostTrainBench run. Neither has a success rate, a sample size, or a baseline. Compare that to the secret-extraction section, where 315,320 blocks are processed through a two-stage pipeline with a documented funnel. The rigor is very unevenly distributed, and the abstract presents all four vectors as equally established.

**The privacy labeling is LLM-judged with limited human verification.** The funnel from 76,778 to 2,194 to 941 to 704 runs on two LLM passes. Table 4's category breakdown depends entirely on their calibration, and no inter-annotator agreement, no human-audited sample, and no false-positive rate on the second pass is reported. The headline number, 704, is a judge's output presented as a count.

**Statelessness is diagnosed but never re-examined.** The paper treats client-side storage as the fixed constraint and designs around it. But the defense it proposes, with chained envelopes, Merkle roots retained server-side after compaction, consumed-envelope tracking for P4, and per-account anomaly detection, requires meaningful server-side state anyway. If you are keeping state regardless, the honest comparison is against the stateful design the paper dismisses in two sentences for its "higher database and storage overhead". Nobody costs that out. A trace ID plus server-side storage might be cheaper than a Merkle forest.

### What would change my mind

Two experiments, in order of importance.

**On faithfulness:** run the extraction pipeline against a model whose reasoning is directly observable, an open-weight model served behind an identical envelope scheme, and report per-token edit distance between the extraction and the true trace. If median token-level accuracy is above ~95%, the distillation threat is as severe as claimed and my complaint about the length metric collapses. If it is 70%, then what the attack recovers is a *summary with the decoder's fingerprints on it*, the distillation story weakens substantially, and the privacy story stays fully intact because a recovered API key is either right or wrong.

**On distillation:** train two students on matched budgets, one on extracted traces and one on the API summaries for the same prompts, and report the downstream gap. If extracted traces beat summaries by a wide margin, the anti-distillation motive for encryption is justified and the attack is a real IP threat. If the gap is small, then the summaries were already giving away most of the value and encryption was mostly theater.

I would also change my mind on the Appendix B framing if someone ran the missing control: measure the same drift on a model trained on a corpus verifiably scrubbed of frontier-model outputs. Absent that, "unusual behavioral compatibility" is the correct ceiling on what can be claimed.

## What I would build with this

These are my extrapolations, not the paper's claims.

**1. A pre-commit hook that strips reasoning blocks.** The paper's data-hygiene recommendation is correct and completely unenforced. The tool is small: scan staged files for `signature`, `thought_signature`, `encrypted_content` and the base64 shapes that accompany them, and refuse the commit. This should live in the agent scaffolds themselves rather than in every user's repository, and honestly it should have shipped alongside the encrypted-blob feature.

**2. A "what did you think about my data" endpoint.** The paper's conclusion asks providers to disclose when PII is absorbed into hidden reasoning. Make it an API. Let the *originating user*, authenticated as the session owner, decrypt their own reasoning blocks. This is strictly better than the status quo on every axis: it restores the user's ability to sanitize before publishing, it costs the provider nothing in IP terms because the user already paid for that reasoning, and it removes the absurdity of a system that hides your own data from you while leaving it readable by anyone who scrapes your repository.

**3. Reasoning-channel provenance for shared traces.** The injection attack works because a resumed trace carries no notion of who produced each block. Signed provenance metadata at the *scaffold* level, naming the account and run that generated each block, would let a victim's client warn before resuming a trace containing blocks from an unknown origin. This does not need provider cooperation to prototype.

**4. Prefill-drift as a routine forensic tool.** Independent of the distillation question, Appendix B's method is a reusable instrument: prefill model $M$ with 1% of model $N$'s reasoning and measure style drift against a self-prefill control. Run as a standing benchmark across open releases, it becomes a leaderboard of behavioral compatibility. It will not prove distillation, and it should not be presented as if it does, but a model that suddenly starts continuing a competitor's reasoning idiomatically is worth a second look.

**5. Ephemeral-by-default with explicit opt-in.** The `preserve_thinking` pattern deserves to be the default rather than a flag. Most single-turn API traffic has no need for reasoning continuity and is currently paying for a stored liability it did not ask for. Make persistence a choice you make when you need it, with the privacy consequences stated at the call site.

## References

- Alexander Panfilov, David Schmotz, Ilia Shumailov, Luca Beurer-Kellner, Joachim Schaeffer, Ameya Prabhu, Jonas Geiping, Maksym Andriushchenko. **Stealing Reasoning Traces from Proprietary LLM APIs.** arXiv:2608.09867 [cs.CR], August 2026. <https://arxiv.org/abs/2608.09867>. Project site: <https://stolen-thoughts.com>
- Matthew Green. **Let's talk about encrypted reasoning.** A Few Thoughts on Cryptographic Engineering, May 2026. The prior disclosure this paper extends. <https://blog.cryptographyengineering.com/2026/05/29/fooling-around-with-encrypted-reasoning-blobs/>
- Bowen Baker et al. **Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation.** arXiv:2503.11926, 2025. Why providers avoid optimizing chain-of-thought content directly. <https://arxiv.org/abs/2503.11926>
- Tommaso Green, Martin Gubri, Haritz Puerto, Sangdoo Yun, Seong Joon Oh. **Leaky Thoughts: Large Reasoning Models Are Not Private Thinkers.** EMNLP 2025. <https://aclanthology.org/2025.emnlp-main.1347/>
- Tingwei Zhang, John X. Morris, Vitaly Shmatikov. **How to Steal Reasoning Without Reasoning Traces.** arXiv:2603.07267, 2026. The trace-inversion result behind the MATH500 68.4% to 76.0% figure. <https://arxiv.org/abs/2603.07267>
- Jamie Hayes et al. **Measuring Memorization in Language Models via Probabilistic Extraction.** NAACL 2025. The framework behind the ${1 - (1-p_z)^n}$ analysis. <https://aclanthology.org/2025.naacl-long.469/>

Related reading on this blog:

- [Learning to Reason: Training LLMs with GPT-OSS or DeepSeek R1 Reasoning Traces](/blog/paper-reading/large-language-model/learning-to-reason-training-llms-with-gpt-oss-or-deepseek-r1-reasoning-traces), for what a reasoning trace is actually worth as training data, which is the demand side of this attack.
- [SAFEPATH: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment](/blog/paper-reading/ai-interpretability/safepath-preventing-harmful-reasoning-in-chain-of-thought-via-early-alignment), on intervening in the reasoning channel itself rather than filtering the output.
- [Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens](/blog/paper-reading/large-language-model/is-chain-of-thought-reasoning-of-llms-a-mirage-a-data-distribution-lens), which pairs well with the summary-unfaithfulness finding above.
- [Qwen3Guard: Real-Time Safety Moderation for the Token Stream](/blog/paper-reading/ai-safety/qwen3guard-technical-report), on what output-layer moderation can and cannot see.
