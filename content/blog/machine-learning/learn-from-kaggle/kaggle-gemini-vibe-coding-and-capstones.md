---
title: "Kaggle's Gemini sprints and course capstones: what a winning demo looks like"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "Five judged Kaggle events from 48 hours to three weeks: the Nano Banana Hackathon, Vibe Code with Gemini 3 Pro, and the Gen AI, Agents and Vibe Coding Agents Intensive capstones. Rubrics to scale, the live-URL gate, 84 winning writeups read, a feature matrix across all of them, and what separates a sprint demo that wins from one that only dazzles."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "hackathon", "gemini", "nano-banana", "ai-agents", "adk", "mcp", "vibe-coding", "llm-as-judge"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 23
---

> **TL;DR.** Between April 2025 and July 2026 Google and Kaggle ran five judged build sprints: a **48-hour Nano Banana image hackathon**, a **one-week "Vibe Code with Gemini 3 Pro in AI Studio"** sprint, and three **course capstones** (Gen AI Intensive 2025Q1, Agents Intensive, Vibe Coding Agents Intensive). They named 145 winners; the two big sprints ranked none of theirs. I read 84 winning writeups and notebooks. **Nobody fine-tuned, and almost nobody measured anything**: 0 of 40 sprint winners reported a quantitative result. The sprints were won on a working public URL, a named user and visible use of the model's signature features. The agent capstones were won by putting **deterministic code and human gates around the model**. **The one lesson:** a winning sprint demo is a contract a judge can check in two minutes, with the model doing the judgment and code doing everything that must not go wrong.

![Five Google and Kaggle build sprints on one scale, from the 48-hour Nano Banana Hackathon to the 19-day agent capstones, with teams, winners, deliverable and prize on each lane](/imgs/blogs/kaggle-gemini-vibe-coding-and-capstones-1.webp)

The [orientation post](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work) split Kaggle's LLM events into scored competitions and judged hackathons; siblings cover the [2024 Gemma and Gemini hackathons](/blog/machine-learning/learn-from-kaggle/kaggle-gemma-gemini-early-hackathons-winners), the [Gemma impact hackathons](/blog/machine-learning/learn-from-kaggle/kaggle-gemma-impact-hackathons-winners) and [BigQuery AI](/blog/machine-learning/learn-from-kaggle/kaggle-bigquery-ai-hackathon-winners). Three rules hold below. **Every number about an entry is the team's own claim**; nobody re-ran them, including me. **Gemini 3 and Nano Banana each named 50 winners "in no particular order"**: I read the first 20 of each in the host's announcement order and keep them unranked. **CarbonCalc AI (Agents Intensive, Agents for Good, 2nd) has no surviving writeup**: its page returns 403 and its repo and video are gone, so it gets a name and nothing else.

## 1. The five briefs in 60 seconds

To a friend: "Google handed out a new model or a free course, asked people to build something in days, and a panel of Googlers picked winners." Formally:

| Item | Nano Banana | Gemini 3 | Gen AI 2025Q1 | Agents Intensive | Vibe Coding Agents |
| --- | --- | --- | --- | --- | --- |
| Window | 2025-09-05 to 09-08, 48 h | 2025-12-05 to 12-12 | 2025-04-04 to 04-20 | 2025-11-14 to 12-03 | 2026-06-18 to 07-07 |
| Stack | Gemini 2.5 Flash Image | Gemini 3 Pro in AI Studio | any Gemini capability | ADK agents | ADK, MCP, Antigravity |
| Deliverable | live URL, 2-min video, 200-word writeup | live URL, video, writeup | notebook, blog, video | writeup, repo, 3-min video | 2,500-word writeup, 5-min video, repo |
| Field | 826 teams | 4,081 teams | 6,151 joined | 11,454 teams | 6,035 teams |
| Winners | 50 unranked | 50 unranked | 10 top, 11 honourable | 3 places x 4 tracks | 3 per track, unplaced |
| Prize | credits, "over 400,000 USD" | 10,000 USD credits each | swag | swag | swag |

Nano Banana's bundle per winner was 5,000 USD of Gemini credits, 1,000 of Fal and about 2,000 of ElevenLabs. That matches the 250,000 USD Gemini pool, but with the two special prizes added the Fal side sums to 55,000 against a 50,000 pool, so treat the dollar values as the host's approximations. The capstones paid only swag, yet the Agents Intensive drew the largest field in this series.

## 2. The rubrics, drawn to scale

![The five rubrics as stacked bars: the two sprints put 70 of 100 points on impact, creativity and presentation, the agent capstones put 70 of 100 on implementation, and the Gen AI capstone multiplies a notebook score of 10 by up to 1.5 for a blog and 1.5 for a video](/imgs/blogs/kaggle-gemini-vibe-coding-and-capstones-2.webp)

**Nano Banana** scored Innovation or "Wow" 40%, Technical execution 30% (use of "consistency, fusion, editing"), Impact 20%, Presentation 10%. **Gemini 3** scored Impact 40%, Technical Depth 30% ("not just faked for the demo"), Creativity 20% ("how many of Gemini 3's capabilities are you using and how well do they work together"), Presentation 10%, and called the video "the primary lens". The two agent capstones inverted that: **Pitch 30, Implementation 70**, with README documentation alone worth 20. The Agents Intensive added up to 20 bonus points (Gemini 5, Cloud deployment 5, a video of 3 minutes or less 10).

The **Gen AI 2025Q1** capstone was multiplicative:

$$
S = N \times m_{\text{blog}} \times m_{\text{video}}, \qquad N \le 10, \quad m_{\text{blog}}, m_{\text{video}} \le 1.5
$$

where $N$ is the notebook score (use case 5, documentation 5; code-only notebooks score 0). The maximum is 10 x 1.5 x 1.5 = 22.5, so a perfect notebook with no blog and no video caps at 44% of the top score. All 21 winners shipped all three assets.

## 3. The gate before the judges

![Winners per 1,000 entries on a log scale: about 60 in Nano Banana (102 among the 489 entries that passed the screen), 12 in Gemini 3, 3.4 in the Gen AI capstone, 2.0 in the Vibe Coding capstone and about 1 in the Agents Intensive](/imgs/blogs/kaggle-gemini-vibe-coding-and-capstones-3.webp)

Both sprint hosts published how judging ran, and both began with a **pass/fail screen before anyone judged quality**: mandatory model use, a video, and a live public demo URL, "not just a GitHub link or notebook". In Nano Banana, **489 of 827** submissions qualified for deep review, about 59%; judges then reviewed "replicating results". In Gemini 3, about 30 Googlers judged 4,097 submissions over roughly three months, "making every effort to validate and replicate".

The odds span almost two orders of magnitude: 50 of 827 in the 48-hour sprint is 60 per 1,000, while 12 of 11,454 in the Agents Intensive is about 1. Four in ten Nano Banana entries never reached deep review. The capstones gated on their own artefact: a notebook that **runs end to end without errors** using at least 3 of 16 listed capabilities (Gen AI), or at least 3 course concepts such as multi-agent design, tools, memory, observability, evaluation, A2A or deployment (both agent capstones).

## 4. Nano Banana: 48 hours with an image model

### What made it hard

About 48 hours, a 200-word writeup cap, a preview model on a shared quota. The judges named what separated the 50 from the 489: "while many submissions were dazzling on the what and how, the winning projects also mastered the so what and for whom."

### The first 20 winners (host order, unranked)

| # | Project (team) | What it does, and the trick |
| --- | --- | --- |
| 1 | [The Banana Board](https://www.kaggle.com/competitions/banana/writeups/the-banana-board-your-ai-electronics-tutor) (David Fernandez) | circuit tutor; circle a part, only that region is edited |
| 2 | [Surgical Visualisations](https://www.kaggle.com/competitions/banana/writeups/surgical-visualisations) (Surgical Banana) | procedure-specific inpaint on the patient's own face |
| 3 | [RealityFlux](https://www.kaggle.com/competitions/banana/writeups/realityflux) (Auravana) | camera frames as masked edits |
| 4 | [Post Compose](https://www.kaggle.com/competitions/banana/writeups/simplifying-group-photos-with-post-compose) (Mert Gulsun) | separate photos fused into one relit group shot |
| 5 | [SingleShot3D](https://www.kaggle.com/competitions/banana/writeups/singleshot3d-digital-twin-from-a-single-image) (G Welinder), **Fal prize** | 2x2 view grid in one call, then 3D |
| 6 | [Visual Artificer](https://www.kaggle.com/competitions/banana/writeups/visual-artificer-ai-powered-image-re-imagination) (AndreasCo1) | curated style "spells" and a community gallery |
| 7 | [ForgeOne](https://www.kaggle.com/competitions/banana/writeups/forgeone-pro-mode-revolution) (Giulio Leone) | plan, generate, score, retry |
| 8 | [Life Machine](https://www.kaggle.com/competitions/banana/writeups/life-machine-the-canvas-of-fate) (sonderone) | life-sim game, your face across ages and styles |
| 9 | [NanoStudio](https://www.kaggle.com/competitions/banana/writeups/nanostudio) (Powerhouse) | product mockups, identity-preserving template |
| 10 | [FrameFusion.ai](https://www.kaggle.com/competitions/banana/writeups/framefusion-ai) (Lawrence) | consistent storyboard, then Veo 3 clips |
| 11 | [The Daily Banana](https://www.kaggle.com/competitions/banana/writeups/the-daily-banana) (SummitSat) | comics with references passed on every panel |
| 12 | [Sozofix](https://www.kaggle.com/competitions/banana/writeups/sozofix-ai-diy-assistant) (RairoRapha), **ElevenLabs prize** | repair steps of *your* object, voice agent with memory |
| 13 | [Hephaestus](https://www.kaggle.com/competitions/banana/writeups/hephaestus) (One_Horizon) | blueprint to rooms to a 360° panorama |
| 14 | [Whisper of Guardian](https://www.kaggle.com/competitions/banana/writeups/whisper-of-guardian) (suhmieum) | RPG where editing the scene is a move |
| 15 | [AnimaBanana](https://www.kaggle.com/competitions/banana/writeups/animabanana) (Anton Begehr) | 9-frame animation from JSON pose plans |
| 16 | [AI3D NanoDimensions](https://www.kaggle.com/competitions/banana/writeups/ai3d-nanodimensions) (yosun) | Vision Pro point-cloud walk, re-rendered via Fal |
| 17 | [Quicktoons](https://www.kaggle.com/competitions/banana/writeups/quicktoons-web-based-animation-software) (Alex Blaze) | browser studio: Nano Banana, Seedance, ElevenLabs |
| 18 | [Looks Like Me](https://www.kaggle.com/competitions/banana/writeups/looks-like-me-personalizing-aac-emotion-boards-wit) (Verbali) | AAC emotion boards from one selfie |
| 19 | [Visual Witness System](https://www.kaggle.com/competitions/banana/writeups/visual-witness-system) (fares mallouli) | fragmentary evidence fused into one scene |
| 20 | [AI Font Generation](https://www.kaggle.com/competitions/banana/writeups/ai-font-generation) (Antoine Descamps) | a usable font from a description |

Winners 21 to 50 are in the [announcement](https://www.kaggle.com/competitions/banana/discussion/609783), including a 2D-to-3D CAD converter and a scientific-illustration tool.

### SingleShot3D, in depth: one call, four views

![SingleShot3D's generate-then-split trick: one product photo, one Gemini 2.5 Flash Image call that returns a 2x2 grid of front, back, left and right views, a cut into four tiles, and a Fal 2D-to-3D engine producing the asset](/imgs/blogs/kaggle-gemini-vibe-coding-and-capstones-4.webp)

The judges' first theme was "character consistency effectively solved", and SingleShot3D shows why it matters. A 3D reconstruction engine needs several views of the *same* object; ask an image model four times and you get four slightly different objects. So the app asks once: Nano Banana draws a **2x2 orthographic grid** (front, back, left, right) in one generation, and the app cuts it into four tiles. Sharing one canvas, the views agree, and one call is cheaper than four. Any quadrant can be regenerated by inpainting, and a high-fidelity mode anchors the other three sides on a confirmed front view. The tiles go to one of three **Fal** engines (Hunyuan Turbo, Hunyuan Standard, Tripo3D), which won the Fal special prize.

### ForgeOne and Sozofix: the two orchestration builds

**ForgeOne** is the only one of the 20 with an automatic critic. `gemini-2.5-flash` splits a request into tasks across 10 professional modes, then scores each generated image on prompt compliance, text quality, technical quality and professional compliance, each out of 100; below threshold, it writes a `delta_prompt` and regenerates. No convergence numbers are given. **Sozofix** won the ElevenLabs prize with the deepest multi-service build: `gemini-2.5-flash` plans the repair, Nano Banana draws every step with *your* chair, and a voice agent on ElevenLabs Conversational AI gets a **pre-call briefing** that `gemini-2.0-flash-lite` summarises from past calls.

### The other 16, by family

- **Reference passing plus iterative edit**, the dominant pattern: Surgical Visualisations, Post Compose, NanoStudio, The Daily Banana, AnimaBanana, Quicktoons and Looks Like Me keep the user's face, object or character fixed and change the scene, pose or step. NanoStudio's template says it outright: "Preserve the exact product identity... Do not alter shape, proportions, material, texture, branding, or logo placement." The Daily Banana first has Gemini *describe* the references, grounding the story before any panel is drawn.
- **Multi-image fusion**: Post Compose (relighting, perspective, contact shadows), Hephaestus (floor plan as structure reference, previous render as style reference), Visual Witness, RealityFlux.
- **Editing as interaction**: circle a bulb and say "replace this with an LED" (Banana Board); "make his sword glow" raises Chaos (Whisper of Guardian); art style follows the player's mood (Life Machine).

Two claims to discount: FrameFusion's "95% consistency" and "80% cost reduction" have no method, and RealityFlux's README throttles its "real time" AR to at most 20 images per minute.

## 5. Vibe Code with Gemini 3 Pro: one week, one live app

### What made it hard

One week, about 4,000 entrants, and a Creativity line that counted capabilities. The judges' summary named four themes: utility over demos, agentic workflows that "reason, act, and self-correct", native multimodality "not just a text box", and the vibe-coding shift. Most winning writeups run about 140 to 400 words.

### The first 20 winners (host order, unranked)

| # | Project (team) | What it does, and what it stacks |
| --- | --- | --- |
| 1 | [NeuroState](https://www.kaggle.com/competitions/gemini-3/writeups/neurostate-ai-powered-embedded-systems-ide) (Halldean) | state-machine IDE; Live API voice, sketch input, code export |
| 2 | [Banana Couture](https://www.kaggle.com/competitions/gemini-3/writeups/banana-couture-from-prompt-to-factory-floor) (Drew Williams) | brief to tech pack and costing; Nano Banana Pro, Veo |
| 3 | [SIMUL-AI-TOR](https://www.kaggle.com/competitions/gemini-3/writeups/new-writeup-1765214212316) (Unoptimized) | video to physics sandbox; agent rewrites the physics code |
| 4 | [Articulate](https://www.kaggle.com/competitions/gemini-3/writeups/test-to-see-if-its-working) (An Thanh) | infinite canvas with learned preferences |
| 5 | [The Lung Listener](https://www.kaggle.com/competitions/gemini-3/writeups/new-writeup-1765574905570) (Kevin King) | raw lung audio to event timestamps |
| 6 | [Voxel Toy Box](https://www.kaggle.com/competitions/gemini-3/writeups/new-writeup-1765485852798) (SAVAŞ HASÇELİK) | camera to `{x, y, z, color}` voxels |
| 7 | [Helix Investigator](https://www.kaggle.com/competitions/gemini-3/writeups/new-writeup-1765118392840) (Diego Martin Moreno) | PubMed, ClinicalTrials.gov calls plus Search grounding |
| 8 | [FlameEdit](https://www.kaggle.com/competitions/gemini-3/writeups/flameedit-ai-powered-video-editor) (Muhammad Yousuf Noor) | language to timeline actions |
| 9 | [VOLT](https://www.kaggle.com/competitions/gemini-3/writeups/new-writeup-1764993535376) (Palaniappan Sathappan et al.) | options assistant, 6 tools, embedding memory |
| 10 | [FixStream AR](https://www.kaggle.com/competitions/gemini-3/writeups/fixstream-multimodal-field-technician) (ALAA R) | video plus sensor diagnosis, AR boxes, re-record to verify |
| 11 | [RECALL](https://www.kaggle.com/competitions/gemini-3/writeups/recall-writeup-tnsbfs) (Aditya_gd et al.) | second brain; 32k thinking budget, Veo 3.1 |
| 12 | [Recursive-R](https://www.kaggle.com/competitions/gemini-3/writeups/new-writeup-1765089455795) (Lewbei) | Architect, Adversary, Unit Tester loop |
| 13 | [Writer's Block](https://www.kaggle.com/competitions/gemini-3/writeups/new-writeup-1765495209814) (Shihan Qu) | word cascades; names no model, WebGPU fallback |
| 14 | [Sound Sketchbook](https://www.kaggle.com/competitions/gemini-3/writeups/sound-sketchbook) (NoelKM) | doodle to JSON synth patch |
| 15 | [Lucid by ReportSwift](https://www.kaggle.com/competitions/gemini-3/writeups/new-writeup-1765473565437) (Ding's solo team) | interview evidence graph from audio and notes |
| 16 | [RegulonForge](https://www.kaggle.com/competitions/gemini-3/writeups/regulonforge) (Faith Ogundimu) | non-coding variants from sequence plus browser screenshots |
| 17 | [ReBrick AI](https://www.kaggle.com/competitions/gemini-3/writeups/new-writeup-1765327458140) (YonYonWare) | LEGO pile photo to a 3D build guide |
| 18 | [F.R.I.D.A.Y](https://www.kaggle.com/competitions/gemini-3/writeups/new-writeup-1765520565551) (Mr. Aditya Gaurav) | voice to JSON scene graph to 3D |
| 19 | [T-ABC Forensic Core](https://www.kaggle.com/competitions/gemini-3/writeups/the-compliance-copilot-physics-aware-forensics-fo) (King Molosser) | compliance rules in long context; about 140 words |
| 20 | [VibeRad](https://www.kaggle.com/competitions/gemini-3/writeups/new-writeup-1765065566929) (VibeRad Labs) | MRI tutor; `thinking_level` low or high, plus Search |

All 20 shipped an AI Studio app link. Winners 21 to 50 are in [684102](https://www.kaggle.com/competitions/gemini-3/discussion/684102), among them Replica (video to code) and Code Archaeologist.

### The JSON contract

![Three Gemini 3 winners side by side: a doodle becomes a JSON synth patch that a Web Audio engine plays, a camera frame becomes voxel coordinates that Three.js draws, and a voice command becomes a scene graph that React Three Fiber renders](/imgs/blogs/kaggle-gemini-vibe-coding-and-capstones-5.webp)

The rubric asked for depth "not just faked for the demo". In a one-week app the winners' answer was a pattern: **the model emits structured JSON, and deterministic client code renders or executes it**. Voxel Toy Box identifies an object, describes it, then emits coordinates. Sound Sketchbook asks for a `synth_patch` of oscillators, envelopes and notes that a hand-written Web Audio engine plays in "Calm" and "Intense" versions. F.R.I.D.A.Y emits a scene graph, FlameEdit timeline actions, FixStream a forced diagnostic schema. At least 7 of the 20 have this shape. The JSON is the seam where a model failure becomes a code decision; here is an illustrative toy of that seam, not any team's code:

```python
# Illustrative toy, not any team's code: the "model writes JSON, code renders it"
# contract. A fake model reply is parsed, checked against a tiny schema, and
# only valid voxels reach the renderer. Bad cells are dropped and counted, so a
# demo never crashes on a malformed reply.
import json

GRID = 8  # voxel grid is 8 x 8 x 8
reply = '''[{"x": 3, "y": 0, "z": 2, "color": "#c0392b"},
            {"x": 3, "y": 1, "z": 2, "color": "#c0392b"},
            {"x": 9, "y": 1, "z": 2, "color": "#c0392b"},
            {"x": 4, "y": 1, "z": "two", "color": "#c0392b"},
            {"x": 4, "y": 2, "z": 2, "color": "red"}]'''

def valid(v):
    if set(v) != {"x", "y", "z", "color"}:
        return False
    if not all(isinstance(v[k], int) and 0 <= v[k] < GRID for k in "xyz"):
        return False
    c = v["color"]
    return isinstance(c, str) and len(c) == 7 and c[0] == "#"

voxels = json.loads(reply)
kept = [v for v in voxels if valid(v)]
print(f"model returned {len(voxels)} voxels, renderer draws {len(kept)}")
for v in kept:
    print("draw", (v["x"], v["y"], v["z"]), v["color"])
```

```console
model returned 5 voxels, renderer draws 2
draw (3, 0, 2) #c0392b
draw (3, 1, 2) #c0392b
```

One voxel is off the grid, one has a string coordinate, one a colour name. The model got three of five wrong and the demo still draws something sensible, which is what a judge clicking a live URL actually experiences. More in [structured output in production](/blog/machine-learning/inference-engineering/structured-output-in-production-streaming-json-and-tool-calls).

### FixStream AR, The Lung Listener and Recursive-R

**FixStream AR** has the most engineered writeup of the 20 (about 3,100 words) and stacks eight capabilities. A 5 to 10-second capture records video plus a client-side FFT and accelerometer peaks, so only numbers travel with the clip. Gemini 3 Pro, with a thinking budget of 2,048 and the Search tool, returns a forced JSON diagnosis with citations; Gemini 2.5 Flash returns bounding boxes for the AR overlay; Flash Image draws a "Ghost Mode" of the disassembled part. After the repair the technician **re-records**, and the model compares before and after. Requests fall back from Pro to Flash to a canned result. Its "0% failure rate in field testing" comes with no test set.

**The Lung Listener** feeds the **raw audio buffer** to `gemini-3-pro-preview`, so the model hears a wheeze differ from a crackle. It marks events on a spectrogram over the public ICBHI 2017 human labels, writes a scipy filter for noisy input, and gives judges a zero-setup remote-case loader. There is no agreement metric. **Recursive-R** is the only one of the 20 that checks answers by running code: an Architect drafts, an Adversary attacks, a Unit Tester writes Python to verify. It reports no pass rate.

### The rest, by family

- **Science and medicine**, 6 of 20: SIMUL-AI-TOR, Helix, RegulonForge, Recursive-R, The Lung Listener, and VibeRad on CC0 stroke MRI, "Not for clinical use."
- **Creative tools**: Articulate, Voxel Toy Box, FlameEdit, Writer's Block, Sound Sketchbook, Banana Couture.
- **Engineering and compliance**: NeuroState, FixStream, F.R.I.D.A.Y, and T-ABC, which argues "traditional RAG fails in compliance because it shreds context" and puts a rule kernel in long context, while VOLT and RECALL use embedding retrieval.
- **Claims to discount**: VOLT and F.R.I.D.A.Y quote impact numbers with no method; Lucid infers integrity from voice and handwriting with no validation, a contested idea in hiring; T-ABC's "2M+ token context" exceeds Gemini 3 Pro's documented window of about 1M.

## 6. The course capstones: notebook, agent, vibe-coded agent

### Gen AI Intensive 2025Q1: 21 notebooks

The top 10 (rows 1 to 10) and honourable mentions (11 to 21) are both unranked in [575927](https://www.kaggle.com/competitions/gen-ai-intensive-course-capstone-2025q1/discussion/575927).

| # | Project | The idea |
| --- | --- | --- |
| 1 | [Customer Support Agent](https://decryptai.substack.com/p/how-i-built-a-smarter-genai-customer) | classify, retrieve, invoice tool, self-evaluate (notebook now private) |
| 2 | [Aroundme](https://youtu.be/zfb_UE5GhMQ) | multi-agent meetups; only a video description survives |
| 3 | [NECP Engine](https://www.kaggle.com/code/kianjn/smart-policy-assistant) | extract a schema from energy plans, answer from the table |
| 4 | [Genne](https://www.kaggle.com/code/dnkumars/genne) | emission extraction, FAISS geocoding, ReAct agent |
| 5 | [Inclusive Classroom](https://www.kaggle.com/code/chonghunyee/capstone-project-inclusive-classroom-assistant) | lecture audio to RAG and quizzes for deaf students |
| 6 | [RAGLab](https://www.kaggle.com/code/waltvenable/capstone-project-raglab) | an evaluation harness for RAG configurations |
| 7 | [Scribbles](https://www.kaggle.com/code/ttuuii/scribbles-tone-aware-writing-agent) | LangGraph writer scored against your tone embedding |
| 8 | [Meeting Scheduler](https://www.kaggle.com/code/fareshazem/smart-meeting-scheduler) | writes real Google Calendar events via OAuth |
| 9 | [N.I.R.A.D.](https://www.kaggle.com/code/noobsajbot/n-i-r-a-d) | agent over the author's papers, verified by the solver |
| 10 | [Mock Interviewer](https://www.kaggle.com/code/maxymrim/mock-technical-interviewer-with-gemini-langgraph) | questions forced from a database, graded by schema |
| 11 | [OncoAssist](https://www.kaggle.com/code/assiaben/oncoassist-ai-powered-cancer-recovery-advisor) | RAG over 234 pages of survivor guides |
| 12 | [AI Narrator](https://www.kaggle.com/code/yogesmohan/personalized-ai-narrator-for-visual-accessibility) | persona-tailored alt-text from grounded chunks |
| 13 | [PR Doc Review](https://www.kaggle.com/code/rishabh230795/pr-documentation-review-agent-genai-capstone) | grammar fixes on added lines, JSON per issue |
| 14 | [SkillPilot](https://www.kaggle.com/code/ishasangpal/skillpilot-your-personal-ai-career-copilot) | resume to risk score and roadmap |
| 15 | [T-AID](https://www.kaggle.com/code/lguzm001/technical-assistant-for-industrial-docs-t-aid/) | pick one manual first, or refuse |
| 16 | [word2color](https://www.kaggle.com/code/anyaepie/word2color-seeing-books-in-a-new-colour) | colour mentions, a second model judging each |
| 17 | [AIDER](https://www.kaggle.com/code/bellomalik/aider-bello-malik) | weather, news, satellite image to an emergency brief |
| 18 | [Watt Seer](https://www.kaggle.com/code/praveengv/capstone-watt-seer-compare) | embedded energy summaries find "energy twins" |
| 19 | [Chat, Fix, Speak](https://www.kaggle.com/code/randomforest237823/capstone-project) | Whisper, Gemini correction, voice-cloned TTS |
| 20 | [ChemSimGuide](https://www.youtube.com/watch?v=4iWWaba-D3c) | Cantera code from retrieved docs; video description only |
| 21 | [EthnoMed](https://www.kaggle.com/code/ammarmoustafa/ethnomed-advisor-ai-ethnobotanical-knowledge) | RAG over an own dataset of 257 plants |

Three entries check themselves. **RAGLab**'s product *is* an evaluation harness: query-rewrite and HyDE variants against two generation models, each answer graded by an LLM judge (composite 0.4R + 0.4F + 0.2C over relevance, faithfulness, completeness), pausing for a human when judge confidence is under 0.20; the shipped demo is 2 questions x 6 configurations. **N.I.R.A.D.** verifies its agent's advice with the solver: at an attacker budget of 8.7 the worst-case flow stays at least 6, at 8.8 it drops below. **T-AID** picks one Komatsu manual per question or aborts before embedding anything, so procedures cannot leak across machine models: "Sometimes the smartest answer is 'I don't know'." Documentation length did not decide the top 10: honourable mentions include notebooks of 6,000 and 10,000 words.

### Agents Intensive: 12 places across four tracks

| Track | 1st | 2nd | 3rd |
| --- | --- | --- | --- |
| Concierge | [NewsPulse](https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/NewsPulse-AI-Agent): citation-gated loop | [FIL Content Agent](https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/fil-content-agent-system): 15-tool MCP server, Agent Engine | [AishIngAnalyzer](https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/aishinganalyzer-cosmetic-ingredient-analyzer): LangGraph critic |
| Agents for Good | [CfoE](https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/carbon-footprint-optimization-engine-cfoe): LLM-free scorer, approval pause | CarbonCalc AI: **no surviving writeup** | [Parallel Scholar](https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/parallel-scholar): arXiv research team |
| Enterprise | [Chaos Playbook Engine](https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/new-writeup-1763223319877): 14,000 seeded runs | [Coderama](https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/coderama-reimagining-enterprise-software-deliver): A2A teams, approve each sprint | [VeganFlow](https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/veganflow-autonomous-supply-chain-intelligence): A2A haggling with 11 vendor agents |
| Freestyle | [Daedalus](https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/daedalus): forges and tests its own tools | [AI Stock Prediction](https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/ai-stock-prediction-system-multi-agent-a2a-archit): 6 A2A microservices | [StoryLand AI](https://www.kaggle.com/competitions/agents-intensive-capstone-project/writeups/storyland-agent): parallel discovery, persistent sessions |

![Chaos Playbook Engine's own results over 14,000 simulated runs: baseline success falls from 100% to 37.2% as the tool failure rate rises to 20%, while the playbook agent stays between 97.0% and 100%](/imgs/blogs/kaggle-gemini-vibe-coding-and-capstones-6.webp)

**Chaos Playbook Engine** (Enterprise, 1st) is the only winner in all five events that charted itself against a baseline. A deterministic core injects 503 and 429 errors with a reproducible seed, a circuit breaker opens after 3 failures, and a Gemini 2.5 Flash-Lite agent retrieves recovery strategies with a `lookup_playbook` tool. In the team's table, success at a 20% tool-failure rate rises from 37.2% to 97.0%, for 59.8% more latency.

**CfoE** (Agents for Good, 1st) states "code where it must not go wrong" most clearly: its calculation agent yields the `calculate_carbon_score` call itself, so the LLM never decides whether or how to score, and at a risk score of 0.80 or more a long-running tool calls `request_confirmation()` and pauses the session for a human. Its 22 of 22 passing tests exercise that threshold logic, not the LLM. **Daedalus** (Freestyle, 1st) never answers directly: lacking a tool, it designs, writes, tests and repairs one in a "ToolGym" loop, and each passing test joins a growing golden set; the author notes generated code runs through `exec` without a sandbox. **NewsPulse** separates search from fetch so the model must read live pages, then loops until "100% of claims [are] citation-backed".

What separated 1st from the rest: **every 1st place made the model's output checkable**. Infrastructure did not: all three A2A entries and all three MCP-server entries placed 2nd or 3rd, as did the two fullest Google Cloud deployments.

### Vibe Coding Agents Intensive: 12 winners, unplaced

[730944](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/discussion/730944) lists three per track without places (the API's "Award 1 to 3" may be an index, so I keep them unplaced).

| Track | Winners |
| --- | --- |
| Agents for Good | [MemoryBridge](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/writeups/new-writeup-1782584701075): dementia routines, rule gates before the LLM; [CritiqAI](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/writeups/new-writeup-1781911352583): debate coach, regex leak validator; [AccessPilot](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/writeups/accesspilot-autonomous-accessibility-repair-agent): audit, patch, re-test, approve the diff |
| Agents for Business | [Chance Electro](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/writeups/chance-electro-draft): one LLM, the rest deterministic; [Trajectory Auditor](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/writeups/new-writeup-1783355733656): grades the path; [ReviewOps](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/writeups/new-writeup-1783340134268): access control before the model |
| Concierge | [BiteBudget](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/writeups/bitebudget-an-intelligent-cooking-and-grocery-exp): pauses on suspected injection; [TripWeaver](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/writeups/tripweaver-a-multi-agent-travel-planner-that-fix): 3-round reconcile loop; [KinKeeper](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/writeups/kinkeeper-never-forget-to-remember-the-people-yo): chat export to memory cards |
| Freestyle | [Palimpsest](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/writeups/palimpsest): manuscript transcription; [A.R.I.E.S.](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/writeups/project-aries): space-weather reports, about 135 tools; [AstroTyper](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/writeups/new-writeup-1783370589012): typing RPG, no ADK |

![Trajectory Auditor's pipeline: a raw agent session of 139k to 400k tokens is read once by an Analyst, seven judges each see a narrow evidence window, a Skill-Synthesizer proposes a patch, and a Vibe Diff gate blocks overreach before a human applies anything](/imgs/blogs/kaggle-gemini-vibe-coding-and-capstones-7.webp)

**Trajectory Auditor** grades *how* a coding agent reached its output. Sessions run 139k to 400k tokens, so an Analyst reads once and only *points* at steps; code extracts verbatim evidence, and each of 7 parallel judges sees a 5k to 15k-token window instead of about 1M tokens per audit. Its evaluation is the most careful in this post: twin 120-step fixtures differing by one planted step, and an injection A/B where step 117 tells the judges to award 5 of 5. The injected run scored *lower*, and every judge flagged it. A Vibe Diff gate blocked three real skill proposals for overreach (shell execution, `.env` inspection). For why judges need that armour, see [one token to fool an LLM-as-a-judge](/blog/paper-reading/ai-interpretability/one-token-to-fool-llm-as-a-judge).

**MemoryBridge** rejects medication, diagnosis, money, emergency calls and door unlocking **before any LLM sees the request**, lets agents write only drafts a caregiver approves, and routes all persistence through a narrow MCP server; it reports 28 of 28 eval cases and 6 of 6 injection tests passed. **Chance Electro**, built for a real electrical contractor, has one Claude Vision agent that turns sketches into project JSON and returns **a quiz instead of inventing** missing facts; pricing and documents are deterministic, and it says plainly it used neither Gemini nor Antigravity. **Palimpsest** chose Gemini 2.5 Pro after 2.5 Flash failed 3 of 4 manuscript pages, and found a thinking budget of 128 beats higher ones. **ReviewOps** is the only winner reporting eval scores before and after a fix: a hard-refuse path took a failing case from 1 of 5 to 5 of 5.

Nearly every writeup argues "why agents" by naming the failure of the single-prompt version, and the pipelines are **fixed workflows, not free-roaming agents**. More on that in [human-in-the-loop design](/blog/machine-learning/ai-agent/human-in-the-loop-design) and [MCP](/blog/machine-learning/ai-agent/model-context-protocol).

## 7. Across all five: the feature matrix

![Feature matrix across 84 winners with writeups in five events: zero fine-tuned anywhere, quantitative results in 0 of 40 sprint winners, live demos near-universal in the sprints, and deterministic code, human gates and MCP servers concentrated in the agent capstones](/imgs/blogs/kaggle-gemini-vibe-coding-and-capstones-8.webp)

**Fine-tuning: 0 of 84.** **A quantitative result: 0, 0, 2 of 21, 2 of 11, 4 of 12.** Live demos are near-universal in the sprints (19 of 20, 20 of 20) because the demo was the gate, and rare in the Gen AI capstone (2 of 21) because the notebook was. Deterministic code around the model rises from 4 of 11 in the Agents Intensive to 9 of 12 plus one partial in the vibe-coding capstone, and hard human gates from 3 of 11 to 7 of 12 plus one partial. The sprints differ in what they lean on: consistency (15 of 20) and iterative editing (14 of 20) in Nano Banana; multimodal input (at least 12 of 20) and tool use (7 of 20) in Gemini 3.

I recounted every cell from the per-winner matrices, and three summary lines moved: the vibe-coding source says 10 of 12 use deterministic wrappers and 8 of 12 human gates, where the cells give 9 and 7 plus one partial each; the Gemini 3 source says multimodal input in "at least 14 of 20", where the confirmed cells give 12.

## 8. What did not work

- **The small model on the hardest perception step.** Palimpsest: 2.5 Flash failed 3 of 4 pages. N.I.R.A.D.: 2.0 Flash misread a network figure "despite prompt changes".
- **More thinking.** Palimpsest: budgets above 128 made transcription worse.
- **Preview and browser audio.** Sozofix: browser audio was unstable and a Gemini TTS preview too slow, so it moved to ElevenLabs and DeepGram.
- **The single-prompt version.** CritiqAI's first chatbot leaked answers; Mock Interviewer's agent invented questions until function calling forced them from a database; PR Doc Review's raw feedback could not be mapped to lines until it returned JSON.
- **Mocks hiding bugs.** Chance Electro: mock tests hid a zero-price bug from vision output.
- **Eval-free confidence.** ReviewOps's own eval caught a protected-topic request silently replaced and reported "approved"; Trajectory Auditor's gate blocked three proposals its own synthesizer made.
- **Shared keys and quotas.** Life Machine and The Lung Listener both warned judges about keys and `429 Resource Exhausted`.

> **Context: Kaggle Game Arena.** On 2025-08-04 Google DeepMind and Kaggle launched Game Arena, "a new, public AI benchmarking platform where AI models compete head-to-head in strategic games". It is these hackathons' mirror image: it scores models, not people. Chess came first; the leaderboard is all-play-all with "over a hundred matches between every pair of models", and the harnesses are open source in `kaggle-environments`, the stack behind Kaggle's agent competitions. An 8-model chess exhibition opened it; its participants and standings are unverified from a primary source here. The upcoming Game-Agents series covers that stack.

## 9. Lessons that transfer

1. **Pass the gate before you polish.** A live URL screened out about four in ten Nano Banana entries before judging ([609783](https://www.kaggle.com/competitions/banana/discussion/609783), [684102](https://www.kaggle.com/competitions/gemini-3/discussion/684102)).
2. **Show the model's signature features, each doing work.** Consistency, fusion and editing in Nano Banana; FixStream stacked eight Gemini capabilities.
3. **Make the model write a contract and let code honour it**: JSON into a renderer (Gemini 3), an LLM-free scorer (CfoE), one LLM consumer (Chance Electro), MCP as the only path to the database (MemoryBridge).
4. **In a capstone, checkability beats infrastructure.** Every Agents Intensive 1st place was checkable; every A2A and MCP entry placed lower ([663531](https://www.kaggle.com/competitions/agents-intensive-capstone-project/discussion/663531)).
5. **Evidence is rare, so it stands out.** Chaos Playbook's baseline and Trajectory Auditor's seeded twins are the exceptions in 84 writeups.
6. **Name the user and the "so what"**: a cleft-lip patient, an AAC child, a contractor who never gets three quiet hours.

### If you enter a sprint like this

- Deploy the live URL on day one; test it logged out, and give judges a zero-setup path.
- Keep your key server-side and handle `429` gracefully; never ask a judge for theirs.
- Name one user and tell their story in the first 30 seconds of the video.
- Route model output through a schema; let code render, compute and persist.
- Put one deterministic gate before the model and one human gate where a mistake is costly.
- Measure one thing against a baseline, even on 20 cases. Almost nobody does.

## 10. Sources

- **Nano Banana:** [competition](https://www.kaggle.com/competitions/banana), [winners 609783](https://www.kaggle.com/competitions/banana/discussion/609783); writeups linked in section 4.
- **Gemini 3:** [competition](https://www.kaggle.com/competitions/gemini-3), [evaluation](https://www.kaggle.com/competitions/gemini-3/overview/evaluation), [winners 684102](https://www.kaggle.com/competitions/gemini-3/discussion/684102); writeups linked in section 5.
- **Gen AI Intensive 2025Q1:** [competition](https://www.kaggle.com/competitions/gen-ai-intensive-course-capstone-2025q1), [winners 575927](https://www.kaggle.com/competitions/gen-ai-intensive-course-capstone-2025q1/discussion/575927); notebooks linked in section 6.
- **Agents Intensive:** [competition](https://www.kaggle.com/competitions/agents-intensive-capstone-project), [evaluation](https://www.kaggle.com/competitions/agents-intensive-capstone-project/overview/evaluation), [winners 663531](https://www.kaggle.com/competitions/agents-intensive-capstone-project/discussion/663531); 11 writeups linked in section 6; CarbonCalc AI's writeup returns 403.
- **Vibe Coding Agents Intensive:** [competition](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project), [winners 730944](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project/discussion/730944); writeups linked in section 6.
- **Game Arena:** [Google blog](https://blog.google/technology/ai/kaggle-game-arena/), [InfoQ](https://infoq.com/news/2025/09/kaggle-game-arena) (secondary).
