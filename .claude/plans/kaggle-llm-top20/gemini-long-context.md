# Google: Gemini Long Context (2024): every winner and honorable mention

- **Slug:** `gemini-long-context`
- **URL:** https://www.kaggle.com/competitions/gemini-long-context
- **Dates:** 2024-10-17 to 2024-12-01 (competition API). Winners: https://www.kaggle.com/competitions/gemini-long-context/discussion/552419. Get-started thread: /discussion/541152 (not re-read; nothing about winners).
- **Format:** notebook-judged, no Writeup objects. Submission = public Kaggle Notebook + YouTube video using Gemini 1.5's long context (up to 2M tokens) and context caching.
- **Entrants:** team/submission counts not reported by the API for this format; 5,682 joined users.
- **Prize pool:** $100,000 as 4 x $25,000 (Prizes page per bank). The 8 Honorable Mentions carry no stated prize.
- **Rubric:** Notebook 50 (useful 10, informative on how/why long context enabled it 10, interesting 10, documented 10, efficient context caching 5, novel 5); Video 40 (accurate 10, informative 10, instructional 10, entertaining/professional 10).
- **Bank coverage:** `kaggle-llm-writeup-bank-part4.md` H2 "## 6. Google: Gemini Long Context (2024)". Already in depth there: the 4 winners (one paragraph each, token counts, caching). Honorable mentions are only listed. This file adds benchmark tables and design details the bank lacks for the 4 winners, and covers all 8 honorable mentions from their pulled notebooks.
- **Source used:** each notebook pulled via Kaggle `kernels/pull` (markdown + truncated code cells), saved under `/private/tmp/claude-501/kaggle-top20/hack/nb_glc/`.

## Winners table (host order; the host gives no ranking among the four)

| Award | Author (handle) | Project | Notebook | Video |
|---|---|---|---|---|
| Winner ($25,000) | kyle1373 (FrameCut team) | FrameCut: A Natural Language Video Editor | [nb](https://www.kaggle.com/code/kyle1373/framecut-a-natural-language-video-editor) | [YouTube](https://www.youtube.com/watch?v=D2i8AudrMQw) |
| Winner ($25,000) | oleksandrarsentiev | Exploring Sports Sponsorship: Ads Exposure Analysis | [nb](https://www.kaggle.com/code/oleksandrarsentiev/exploring-sports-sponsorship-ad-exposure-analysis) | [YouTube](https://www.youtube.com/watch?v=Gh5WFUoU3nY) |
| Winner ($25,000) | mrdbourke (Daniel Bourke) | KeepTrack: Use Gemini to Keep Track of Anything | [nb](https://www.kaggle.com/code/mrdbourke/keeptrack-use-gemini-to-keep-track-of-anything) | [YouTube](https://www.youtube.com/watch?v=dbDgcFnlZlE) |
| Winner ($25,000) | futureofworkchannel | Building Process Documentation from Video with AI | [nb](https://www.kaggle.com/code/futureofworkchannel/building-process-documentation-from-video-with-ai) | [YouTube](https://www.youtube.com/watch?v=z1j2cn2_OdI) |
| Honorable Mention | feuera1225 | AI VTuber as Game Master: Unleashing Gemini's TRPG | [nb](https://www.kaggle.com/code/feuera1225/ai-vtuber-as-game-master-unleashing-gemini-s-trpg) | see section |
| Honorable Mention | blackbigswan | GitHub Profile Chat, Analysis and Insights | [nb](https://www.kaggle.com/code/blackbigswan/github-profile-chat-analysis-and-insights) | see section |
| Honorable Mention | abikalmukhiya | PODcast PROfessor: Enhancing Podcast Comprehension | [nb](https://www.kaggle.com/code/abikalmukhiya/podcast-professor-enhancing-podcast-comprehension) | see section |
| Honorable Mention | mechallenge | YouTube Actionable Insight Generator | [nb](https://www.kaggle.com/code/mechallenge/youtube-actionable-insight-generator) | see section |
| Honorable Mention | jobro1234 | Wikipedia Video Director | [nb](https://www.kaggle.com/code/jobro1234/wikipedia-video-director) | see section |
| Honorable Mention | kavisekar | Storyboard Sculptor | [nb](https://www.kaggle.com/code/kavisekar/storyboard-sculptor-gemini/notebook) | see section |
| Honorable Mention | karnikakapoor | Gemini 1.5 Powered Patent Analysis | [nb](https://www.kaggle.com/code/karnikakapoor/gemini-1-5-powered-patent-analysis) | see section |
| Honorable Mention | spitfire2nd | From Prompt to Pull Request: Benchmarking Gemini | [nb](https://www.kaggle.com/code/spitfire2nd/from-prompt-to-pull-request-benchmarking-gemini/notebook) | see section |

## Per-entry synthesis

### W1. FrameCut: A Natural Language Video Editor (Winner, kyle1373)
Product site framecut.ai (contact founders@framecut.ai). Bank covers the Flash/Pro routing and Pydantic OperationBlocks; new detail below.

**Problem and user.** Beginners and founders who find DaVinci Resolve / Premiere menus impenetrable and want to edit by chat (quotes from early-access sign-ups). Traction: ~2,000 unique visitors from Product Hunt / Hacker News and **118 waitlist sign-ups** for the alpha (notebook).

**Data.** Own small video set, 23 s to 6+ min; the long example is a Garry Tan / Sam Altman interview of **106,799 tokens**.

**Models and context.** `models/gemini-1.5-pro-001` and `models/gemini-1.5-flash-001` via the `google.generativeai` SDK; videos uploaded through the Gemini File API (20 GB per project, files expire after 48 h, so the service re-uploads expired files and waits for ACTIVE state). Videos under the **32,768-token minimum for caching** go in directly; larger ones are context-cached. The long window carries a very detailed system prompt and full chat history. Edit requests pass canvas range and time range; Gemini returns structured edit operations (text overlays, transcription captions, timing) rendered with ffmpeg (rendering disabled on Kaggle for the long video because it crashes the notebook).

**Evaluation: a latency benchmark with a counter-intuitive result.** Short `cars.mp4` (no cache): Flash 5.19 / 21.00 / 16.21 s vs Pro 9.05 / 31.26 / 23.11 s for first request / edit / correction. Long 106,799-token video: Pro no-cache 35.86 / 56.55 / 28.31 s vs **with cache 35.95 / 86.27 / 27.63 s**; Flash no-cache 12.56 / 20.37 / 7.12 s vs **with cache 22.54 / 51.73 / 22.19 s**. The team reports caching made things slower and concludes "Gemini 1.5 Flash without context cache provides the most efficient performance" (notebook). Edit accuracy is judged qualitatively (Flash "as accurate as Pro" on short videos, less accurate in general).

**Judges:** "comprehensive approach to video editing through natural language ... a compelling vision for a complete product in just a single month" ([552419](https://www.kaggle.com/competitions/gemini-long-context/discussion/552419)).

**Matrix facts:** Gemini 1.5 Pro-001 + Flash-001; max tokens shown ~107k; caching used and measured (slower); modality video (+ audio track for transcription); structured output Y; live product Y; no fine-tuning.

### W2. Exploring Sports Sponsorship: Ads Exposure Analysis (Winner, oleksandrarsentiev)
Bank covers the idea and the verification pass; the notebook adds an important correction.

**Problem and user.** Sponsors and agencies in a sports-sponsorship market of ~$55.1B (2023), projected $121.7B by 2032 (notebook), who want to know how many seconds each brand is actually on screen.

**Data.** Three public YouTube broadcasts uploaded as a Kaggle dataset (`sample-videos-gemini-competition`): 2024 Las Vegas F1 Grand Prix race highlights, Warriors vs Hawks NBA full-game highlights, and the 2018 World Cup Portugal v Spain match.

**Model and prompting.** `gemini-1.5-flash` via `google.generativeai`, File API upload, system instruction as an "expert video reviewer"; **JSON-schema constrained output** (`response_mime_type = application/json`, `response_schema = list[AdDisplayInfo]` with brand_name, total_sec_displayed, ad_type in {on-site, player, broadcast, other}, industry enum of 11 values, country), temperature 0.2, top_p 0.95, top_k 64, max_output_tokens 8192, 600 s timeout. The FIFA prompt adds "pay additional attention to summing up seconds correctly". Results plotted by brand, ad type, industry and country.

**Context caching.** A verification stage caches the F1 video once (`genai.caching.CachedContent.create` on `gemini-1.5-flash-002`) and re-queries brand by brand (Oracle, Red Bull, T-Mobile, CrowdStrike). The notebook does not tabulate how the per-brand answers compare with the first pass.

**Correction to the bank.** The 1.7M-token figure is the *full* 1 h 40 min Spain v Portugal match, which the author **intended** to process but abandoned after repeated `503 upstream request timeout` and `504 Deadline Exceeded` errors in Kaggle; the analysis shifted to shorter clips. Exact token counts of the clips are printed (`usage_metadata`) but not stated in markdown (UNVERIFIED). The conclusion credits "Gemini 1.5 Pro" while all code calls Flash.

**Evaluation and limits (author-reported).** No ground truth; the honest finding is **inconsistency**: the same brand's exposure came back as 15 s in one run and 10 s in another, and lowering temperature only helped slightly. Proposed fixes: chain-of-thought, first list brands then ask for each brand's time intervals (e.g. 1:30 to 1:34) and sum outside the model, move industry/country enrichment to a separate call, join with viewership peaks.

**Judges:** "We really loved the novel approach to extracting marketing insights from video content" ([552419](https://www.kaggle.com/competitions/gemini-long-context/discussion/552419)).

**Matrix facts:** Gemini 1.5 Flash (+ Flash-002 for cache); tokens in context: highlight clips (count not stated), full 1.7M-token match attempted and failed; caching Y (per-brand re-query); modality video; structured JSON output Y; eval = self-consistency observation only.

### W3. KeepTrack: Use Gemini to Keep Track of Anything (Winner, mrdbourke / Daniel Bourke)
Bank covers the CSV schema and Flash-8B / Flash / Pro comparison; the notebook (very long, tutorial style) adds the pipeline and cost findings below.

**Problem and user.** Households cataloguing belongings for home-and-contents insurance or records; input is one ~10-minute smartphone house-tour video narrated by the owner.

**Model and context.** `models/gemini-1.5-flash-002` chosen as the "goldilocks" option after tabulating Flash-8B (1,048,576 in), Flash (1,048,576) and Pro (2,097,152), all 8,192 output tokens. The flowchart puts the video at **~165,000 tokens** (bank says ~170k; the notebook prints the exact count). The notebook explains the arithmetic: 258 tokens per frame + 32 tokens/s of audio, about 300 tokens per second of video, so 1M tokens holds a bit under an hour.

**Pipeline: three prompt passes over one cached video, each followed by a validator.** Step 1 (initial prompt ~10,000 tokens with few-shot examples and a `<video_inventory_analysis>` think-first block: rooms, timestamps, occlusion and lighting challenges) extracts items; Step 2 (2,000 to 5,000 token prompt) continues if the output stopped early; Step 3 reviews and finalises. Each pass is checked by a separate Gemini call that repairs malformed CSV (`quick_check_csv` + a fix-CSV prompt). Schema: item_number, item_name, item_type, description, brand, condition, number_of_items, estimated_worth, mentioned_worth, room, timestamp, summary, flag.

**Handling the 8k output limit (the real bottleneck).** Output CSV instead of JSON because it uses far fewer tokens (measured with a helper); ask the model to emit a sentinel `MORETIMESTAMPSTODO` if unfinished; compare the last extracted timestamp against the true video length from `cv2`; re-prompt. Author notes the model reliably extracts 50+ items from ~10 minutes but items near the end get missed, and suggests a reverse-order pass (untried).

**Context caching, with a cost result.** Cache created on the Flash-002 version (version suffix required, 32,768-token minimum) with TTL 15 min. Cached input is 4x cheaper ($0.01875 vs $0.075 per 1M for Flash), but a pricing calculator shows **the cached responses cost more than uncached ones, especially with a 2-hour TTL**, because storage cost dominates. Rule of thumb from the author: caching pays when you run inference on the same input more than twice, with TTL set just above the job length.

**Evaluation.** Qualitative (inspection of extracted inventory on one video); no ground-truth item list.

**Judges:** "offers a streamlined, accurate solution for cataloging personal belongings ... I would definitely use this myself!" ([552419](https://www.kaggle.com/competitions/gemini-long-context/discussion/552419)).

**Matrix facts:** Gemini 1.5 Flash-002 (Flash-8B, Pro compared); ~165 to 170k tokens; caching Y (3 passes, cost measured); modality video + spoken audio; structured CSV output + LLM validator; few-shot ICL; no fine-tuning.

### W4. Building Process Documentation from Video with AI (Winner, futureofworkchannel)
Bank gives two lines (500k tokens, caching, Flash); added detail below.

**Problem and user.** Operations / RPA / process-improvement teams: process-mining tools are costly and poor for small processes, so processes stay undocumented. Proposal: a subject-matter expert records a narrated show-and-tell screen recording, AI writes the documentation.

**Data.** A Kaggle dataset of RPA-style business-process recordings for a fictional credit bank (`rpa-credit-bank-business-process-videos`), e.g. "E2E Process - Offer Acceptation.mp4" (used in the code) and "E2E Process - Full.mp4", a half-hour end-to-end customer-service recording cited at **over 500k tokens**. Which video the PDD demo actually ran on: the shorter Offer Acceptation clip per the code cell (UNVERIFIED for the 500k run).

**Pipeline (three outputs).** (1) **Process Definition Document:** `gemini-1.5-flash` with a "Business Analyst reviewing an SME recording" system prompt that asks for ordered steps, business exceptions and steps the SME only references; JSON output (process_name, list_of_steps) rendered into a formatted **Word .docx** with python-docx. (2) **Process map:** the step list is sent to `gemini-1.5-flash-002` (temperature 1, top_p 0.8, top_k 40) to emit **Draw.io MX XML** (rules: start/end nodes, rectangles for actions, diamonds for decisions, one cell per step), extracted by regex into a `.drawio` file. The author admits "the quality is far from what is desired" and suggests fine-tuning. (3) **Onboarding Q&A:** the recording is context-cached (`CachedContent.create`, `gemini-1.5-flash-002`, TTL 60 min) with a one-sentence-answer system prompt, then simulated new-employee questions are answered against the cache.

**Caching and cost evaluation.** Token usage per query (prompt, output, total, cached) is tabulated and a `calculate_costs` helper compares cached vs uncached cost with Pro-002 and Flash price tables, concluding caching "significantly reduce[s] the cost of querying video data". Headline claim: Gemini Flash can analyse **14 hours of process recordings for $1** (notebook). No accuracy evaluation of the generated PDD against a reference.

**Judges:** "novel and efficient approach to solving the pervasive problem of inadequate process documentation" ([552419](https://www.kaggle.com/competitions/gemini-long-context/discussion/552419)).

**Matrix facts:** Gemini 1.5 Flash / Flash-002; up to 500k+ tokens claimed (30-min video); caching Y (TTL 60 min, multi-query Q&A, cost compared); modality video (screen recording + narration); structured JSON -> .docx and Draw.io XML; no fine-tuning.

### HM1. AI VTuber as Game Master: Unleashing Gemini's TRPG (Honorable Mention, feuera1225 / team "A1 Lab")
Video: [YouTube](https://www.youtube.com/watch?v=J27tycW7yhw). Not in the bank beyond its name.

**Problem and user.** Players and stream audiences of tabletop role-playing games (TRPG): a fully AI-run game hosted by "Feuera", described by the team (Wei and William) as Taiwan's most popular AI VTuber, with multilingual voice, emotion detection and simulated "drinking" reactions.

**Design.** A four-step game loop on **Vertex AI `GenerativeModel`** (model ID not visible in the pulled cells, UNVERIFIED): (1) a long system instruction holding the full game bible ("Cosmic Adventure: Feuera's Galactic Quest": theme, rules framework, check mechanics) generates a story outline, difficulty, main and sub-missions, a round limit (20 in the example) and player characters with stats, tools and skills; (2) each round generates a story level with NPCs (e.g. Crystal Guardian) and executable checks; (3) the player inputs a dice roll and the model judges success and narrates the outcome, looping until the round limit; (4) at any time the full storyline and progress are handed back to the model to judge whether objectives are met and the game has ended. The save-state (`game_save_progress`) is passed back each round, which is where long context is used: retaining plot points, backstories and player choices across rounds.

**Long context and caching.** Claimed in italic prose ("By leveraging Gemini's context-caching capabilities ... player histories and intricate world-building"), but **no caching code and no token counts appear** in the notebook. Thin on technical evidence; strongest asset is the entertainment / product angle and video.

**Evaluation.** None.

**Matrix facts:** Gemini via Vertex AI (version ?); tokens not reported; caching claimed, not shown; modality text (VTuber voice in the video layer); multi-turn stateful agent loop; no fine-tuning.

### HM2. GitHub Profile Chat, Analysis and Insights (Honorable Mention, blackbigswan)
Links: [CLI code shortdoom/kaggle-stylometry](https://github.com/shortdoom/kaggle-stylometry), [video (with Google TTS)](https://youtu.be/lu9uK_3wTIQ). Not in the bank.

**Problem and user.** Hiring managers, security researchers and developers who want a deep profile of a GitHub account: expertise, code patterns, preferred tools, how style evolves, and targeted interview questions. Grew out of the author's `gh-fake-analyzer` package for reconnaissance of suspicious GitHub profiles (threat-intel work).

**Model and context.** `gemini-1.5-flash-001` in a `PromptAnalyzer` class with an "experienced software engineer" system prompt, a token-bucket rate limiter (5 requests / 60 s), and per-prompt tracking of tokens, time and JSON validity. Whole repositories' source samples go into the prompt: **largest single prompt 841,280 tokens** (identity-confidence analysis), several >500k on one repo, ~300k average per repository; ~50 prompts and ~15 minutes for a full account; Kaggle demo replays the author's own profile (6.2 min processing). No context caching used.

**Key finding on long-context structured output (most transferable lesson in the HMs).** Above ~100k input tokens with nested JSON, Gemini struggled to traverse inputs and emit valid output. Chunking hurt because re-assembling JSON failed. What worked: process each repository separately, then combine in one big final prompt; and Gemini is **far more stable building "a single big analysis" than "a list of analyses"**, staying stable at 300k+ tokens. Also: hard-coded "Critical requirements" blocks (ONLY VALID JSON, exact keys, arrays max 10 items, "Unknown" for undeterminable), with per-module tuning of repetition, CAPS and style; a strict file-selection rule to stop the model from pulling huge files or looping recursively on source code. Five analysis dimensions: expertise, work patterns, behavioural traits, knowledge breadth, confidence scores; then chat over the generated report.

**Evaluation.** Token/time tables per repository and phase; no accuracy check of the profiles (author says results are "probabilistic indicators rather than definitive assessments").

**Matrix facts:** Gemini 1.5 Flash-001; up to 841k tokens in one prompt; caching N; modality code; structured JSON; no fine-tuning; open code Y.

### HM3. PODcast PROfessor: Enhancing Podcast Comprehension (Honorable Mention, abikalmukhiya)
**Thin source:** the notebook is ~740 words with no GitHub or video link inside it (the required YouTube video was not found in the pulled source; UNVERIFIED). Not in the bank.

**Problem and user.** Curious podcast listeners who lose the thread on terminology and context and end up doing many searches.

**Design.** Deliberately keeps **raw audio** rather than transcribing ("turning it into a big document"): podcast audio files from a Kaggle dataset (`astro-podcast`, astronomy topic) are uploaded through the File API; up to **two audio files are context-cached** with `gemini-1.5-flash-002`, TTL 30 minutes, under a system instruction "You are an expert professor, PhD holder in all subjects ... explain ... as if the user is your pupil ... a teenager with basic knowledge". An interactive `input()` loop answers questions from the cached model and times each query and its token usage.

**Limits reported.** Works for most questions, but processing failed for audio over ~2 hours (single or multiple files, ~300,000 tokens) because of request time limits the author could not extend.

**Evaluation.** Qualitative ("insightful, accurate, and relevant answers"); per-query latency and token counts printed, not summarised.

**Matrix facts:** Gemini 1.5 Flash-002; ~300k-token audio attempted (failed beyond ~2 h); caching Y (TTL 30 min, multi-turn Q&A); modality audio; no structured output; no fine-tuning.

### HM4. YouTube Actionable Insight Generator (Honorable Mention, mechallenge)
**Thin source:** ~630 words; most explanation lives in images from a Kaggle dataset (`youtube-insight`), and no video/GitHub link appears in the pulled source (UNVERIFIED). Not in the bank.

**Problem and user.** YouTube creators who want to mine their own channel history (titles, descriptions, all comments) for what viewers ask for and what to reply to.

**Data.** Video and comment tables for one channel (dataframes named `youtube_yc_video` / `youtube_yc_comments`, apparently a Y Combinator channel dump); every video is rendered to markdown (title, description, "author: comment" lines) and concatenated into a whole-channel prompt plus a last-month prompt. Token count of the channel prompt not stated in markdown.

**Design.** Two **context caches on `gemini-1.5-pro-002`** (TTL 60 min): full channel history and last month, with a "YouTube expert" system instruction. Canned questions against the cache: next-video ideas with three supporting comments each, top 50 positively engaged users, what content viewers find most useful, areas to improve, and (on the last-month cache) which comments need a genuine reply. Reading the code: the last-month string is built by appending to the full-channel string, so the "last month" cache likely contains the whole history too (our observation, not the author's).

**Evaluation.** None; outputs printed.

**Matrix facts:** Gemini 1.5 Pro-002; tokens ? (whole-channel comments); caching Y (2 caches, many questions); modality text (comments); no structured output; no fine-tuning.

### HM5. Wikipedia Video Director (Honorable Mention, jobro1234)
Not in the bank. No video URL found inside the pulled notebook (UNVERIFIED); the bonus section says the entry video itself was scripted by the same pipeline.

**Problem and user.** Turning a Wikipedia article and all its media into a narrated explainer video automatically; generalises to "any collection of text and media", e.g. auto-cutting large video/audio archives.

**Data and token accounting (the most careful in the field).** Apollo 11 article, downloaded with a proper User-Agent as a 53-page PDF plus its media. Table of predicted vs `count_tokens()` results: 34 images 8,772 (34 x 258, exact match); 403 s video with audio predicted 118,885 (403 x (263 + 32)), measured 116,230; 79 s audio predicted 2,528, measured 2,496; PDF 70,390 (author infers text tokenised as text, ~55,057, plus per-page image tokens). **Total 197,888 tokens** in one prompt.

**Model and prompting.** `models/gemini-1.5-pro-002` receives every media file plus a detailed instruction prompt with an example output structure, and writes a timed video script placing each image/clip/audio at a fitting time. Practical finding: Gemini extracts timestamps from audio worse than from video (sometimes beyond the clip's length), mitigated by adding each audio clip's duration to its description. Script parsed, narration produced with Google Cloud Text-to-Speech (gTTS optional), video assembled with MoviePy (pinned to a git commit because 1.0.3 broke writing).

**Context caching, three uses.** (1) generate several candidate scripts or refine one without resending media; (2) prompt iteration during development, since the media dominate tokens; (3) a workaround for **Kaggle timeout errors**: only streaming generation worked, still not finishing the script in one request, so partial outputs are continued against the cached media. Cache TTL 30 min, deleted at the end; caching behind a flag.

**Evaluation.** None quantitative; final video shown.

**Matrix facts:** Gemini 1.5 Pro-002; 197,888 tokens measured; caching Y (multi-generation, continuation); modality docs (PDF) + images + video + audio (fully multimodal); structured script output; TTS + MoviePy; no fine-tuning.

### HM6. Storyboard Sculptor (Honorable Mention, kavisekar)
Video: [YouTube](https://youtu.be/Syqe3-TXy1A). Not in the bank.

**Problem and user.** Animators and filmmakers: a Pixar film runs 50,000 to 75,000 storyboards (WALL-E ~125,000), reviewed in loops. Idea: give an LLM the storyboard images, plot, scene scripting and research material and ask it to continue the board, insert a character, plan style, music and casting.

**Data (own, small).** No copyright-free film storyboard existed, so the author drew a ~54-scene storyboard of "the life of Alice" (office job, holiday, an open-ended mysterious gift) with camera-angle and action text, plus frames from Steamboat Willie and A. A. Milne's Winnie-the-Pooh (public-domain works); a scripting/plan markdown file; and the storyboard assembled into a video (`storyboardimages.mp4`), all as Kaggle datasets.

**Model and use.** `gemini-1.5-flash` with images, the script file and the storyboard video uploaded; token counting per image (`count_tokens`) and the 258-tokens-per-frame / 32-per-second-audio rule are shown, but **total context size is not stated in markdown** (UNVERIFIED). Prompt sequence: understand the inputs, scene detail, compare plot vs storyboard, templated responses, a trailer for the storyboard, a storyboard assistant, "Tooniverse" crossover (Alice in Winnie-the-Pooh plot), and a holiday-special casting assistant.

**Context caching.** Described in prose only ("super-organized assistant who never forgets a detail"); **no `CachedContent` call** in the pulled code.

**Evaluation.** None; creative outputs shown.

**Matrix facts:** Gemini 1.5 Flash; tokens ? ; caching claimed, not used; modality images + video + docs; own hand-made dataset; no fine-tuning.

### HM7. Gemini 1.5 Powered Patent Analysis (Honorable Mention, karnikakapoor)
Video: [YouTube](https://youtu.be/YpmvsByIICc). Not in the bank.

**Problem and user.** Patent examiners ("assist a patent-granting officer in making informed decisions about granting or rejecting") and IP teams who must compare claims, abstracts and descriptions across patents and judge novelty against priority dates.

**Data.** A Google Patents export focused on machine learning and healthcare (patent ID, title, assignee, priority and publication dates, abstract, claims, full description); rows with null abstract or claims dropped. Size not stated in markdown.

**Pipeline.** (1) Classic retrieval first: **TF-IDF + cosine similarity** (scikit-learn) over combined text, overlap threshold **0.9** (author discusses 0.95 for near-duplicates, 0.7 for thematic), clusters visualised as a network graph and a similarity histogram. (2) For each main patent, up to 3 similar patents (word limit 100,000, max 10 successful analyses) are packed into a structured prompt (main patent full text, similar patents, similarity scores, analysis tasks: overlaps, novelty, priority, recommendations to refine claims). (3) Each prompt is **context-cached** on `gemini-1.5-flash-001` with a patent-analyst system instruction so follow-up questions reuse it. Notable hack: prompts under 1,000 characters are padded with repeated filler sentences to reach the cache minimum.

**Token accounting.** ~322,662 prompt tokens per analysis, 940 to 1,393 output tokens, ~161,339 cached tokens per analysis; 3 high-similarity patents analysed. (That prompt tokens are almost exactly 2x the cached count suggests the full text was both cached and re-sent; our reading, not the author's.)

**Evaluation.** None of the novelty judgements is checked; outputs and a token-usage bar chart shown.

**Matrix facts:** Gemini 1.5 Flash-001; ~323k tokens per call; caching Y (per-patent cache, follow-ups; padding hack); modality docs (patent text); TF-IDF pre-filter (a mini-RAG step before long context); no fine-tuning.

### HM8. From Prompt to Pull Request: Benchmarking Gemini (Honorable Mention, spitfire2nd)
Links: test project [schubert-da/svelte-gemini-demo](https://github.com/schubert-da/svelte-gemini-demo), companion notebooks [Melt UI HumanEval testing](https://www.kaggle.com/code/spitfire2nd/melt-ui-humaneval-testing) and [needle-in-a-haystack pipeline](https://www.kaggle.com/code/spitfire2nd/melt-ui-needle-in-a-haystack-pipeline-notebook). No entry-video URL found in the pulled source (UNVERIFIED). Not in the bank.

**Problem and user.** Open-source maintainers and users of a niche library (**Melt UI**, a Svelte headless-UI package) where LLMs hallucinate APIs because the library is under-represented in training data. Question: does putting the whole codebase in context fix that?

**Data and context.** The Melt UI repo concatenated into one text file plus a file index; the raw concatenation was "several million tokens", so the author excluded `pnpm-lock` and all node dependencies (LLMs can infer common JS deps), giving **~900,000 tokens**. Approach adapted from Google's "analyze codebase with Gemini 1.5 Pro" sample. Prompts use chain-of-thought and a mandatory "table of relevant files first" step.

**Two benchmarks (the only entry that evaluates long context itself).**
1. *HumanEval-style, Melt-specific:* **12 tests in 4 categories** (component generation, e.g. radio group / accordion; bug fixing; real open issues from Melt UI's GitHub tracker; and recall/explanation), run against a **context cache** of the codebase, responses frozen to disk because of non-determinism, then **evaluated manually** by pasting generated code into a real Svelte project (only 6 of 12 shown). Finding: the model is excellent at locating the right files, pinpointing bug locations and reproducing source sections with edits, but larger generated code bodies fail more often, and TypeScript type checks are unforgiving.
2. *Multi-needle in a haystack:* **10 needles** at different depths inside the ~900k-token codebase, repeated across iterations in a separate pipeline notebook (no caching possible, since each run inserts new needles). Gemini 1.5 Pro recalled needles with near-perfect accuracy at all depths (chart shows perfect accuracy; exact printed % not in markdown). When the author hit the Pro daily limit and switched to **Flash, accuracy fell to 46.67%**. Debugging lesson: early runs found only 1 or 2 needles because the wrong document was being passed, not because of the model.

**Matrix facts:** Gemini 1.5 Pro (Flash for comparison); ~900k tokens; caching Y (code benchmark), N (needle test, by design); modality code; evaluation = custom benchmark + NIAH with model comparison (most rigorous in this hackathon); no fine-tuning.

## Feature matrix

Columns: FC = FrameCut, SP = Sports Sponsorship, KT = KeepTrack, PD = Process Documentation (winners); then HM1 VTuber TRPG, HM2 GitHub Profile, HM3 PODcast, HM4 YouTube Insight, HM5 Wikipedia Director, HM6 Storyboard, HM7 Patent Analysis, HM8 Prompt-to-PR.

| Feature | FC | SP | KT | PD | HM1 | HM2 | HM3 | HM4 | HM5 | HM6 | HM7 | HM8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Model | 1.5 Pro-001 + Flash-001 | 1.5 Flash (+Flash-002) | 1.5 Flash-002 | 1.5 Flash / Flash-002 | Gemini via Vertex (ver ?) | 1.5 Flash-001 | 1.5 Flash-002 | 1.5 Pro-002 | 1.5 Pro-002 | 1.5 Flash | 1.5 Flash-001 | 1.5 Pro (+Flash) |
| Tokens actually in context (as reported) | 106,799 | clips (n/s); 1.7M attempted, failed | ~165-170k | >500k claimed | not reported | up to 841,280 | ~300k attempted, failed >2 h | not reported | 197,888 | not reported | ~323k | ~900k |
| Context caching used in code | Y (measured slower) | Y (per-brand verify) | Y (3 passes; cost measured) | Y (Q&A, TTL 60) | N (claimed only) | N | Y (TTL 30) | Y (2 caches, TTL 60) | Y (multi-gen + continuation) | N (claimed only) | Y (per patent) | Y (code bench) |
| Modality | video | video | video + speech | screen video + narration | text | code | audio | text (comments) | PDF + images + video + audio | images + video + docs | docs | code |
| Fine-tuned | N | N | N | N | N | N | N | N | N | N | N | N |
| On-device / offline | N | N | N | N | N | N | N | N | N | N | N | N |
| Agentic (tools / multi-step loop) | Y (edit ops -> ffmpeg) | N | Y (3-pass + validator LLM) | N | Y (stateful game loop) | Y (multi-prompt pipeline) | N | N | Y (script -> TTS -> MoviePy) | N | N | N |
| Structured output | Y (Pydantic ops) | Y (JSON schema) | Y (CSV) | Y (JSON -> docx, drawio XML) | N | Y (JSON) | N | N | Y (script) | N | N | N |
| Own dataset built | Y (own videos) | N (public broadcasts) | Y (own house video) | ? (Kaggle RPA videos) | Y (game bible) | Y (own GitHub) | N (podcast) | N (channel dump) | N (Wikipedia) | Y (hand-drawn boards) | N (Google Patents) | Y (curated repo + tests) |
| Eval rigour | small quant. (latency; Pro vs Flash, cache vs none) | none (self-consistency noted) | qualitative + cost calc | cost calc only | none | token/time stats | none | none | token prediction vs count | none | token stats | benchmark w/ baseline (12 tests; NIAH Pro vs Flash) |
| Video | Y | Y | Y | Y | Y | Y | ? | ? | ? | Y | Y | ? |
| Live demo / product | Y (framecut.ai alpha) | N | N | N | Y (live VTuber) | N | N | N | N | N | N | N |
| Open code | Y (nb) | Y (nb) | Y (nb) | Y (nb) | Y (nb) | Y (nb + GitHub) | Y (nb) | Y (nb) | Y (nb) | Y (nb) | Y (nb) | Y (nb + GitHub) |

**Counted summary (M = 12; 4 winners + 8 HMs):**
- 12 of 12 used Gemini 1.5 (Flash or Pro) through the API; 0 fine-tuned; 0 on-device.
- **Video:** 4 of 4 winners used video as the primary input; only 2 of 8 HMs did (Wikipedia Director, Storyboard). HMs spread over code (2), audio (1), docs (1), text/comments (1), text game (1).
- **Caching in code:** 9 of 12 actually call `CachedContent` (all 4 winners + HM3, HM4, HM5, HM7, HM8). HM1 and HM6 only claim caching in prose; HM2 does not use it.
- **Token counts:** 8 of 12 state a concrete number; the largest successful inputs were HM8 (~900k) and HM2 (841k); two entries (SP at 1.7M, HM3 at ~300k audio) report failing at their largest attempt because of request timeouts.
- **Structured output:** 4 of 4 winners vs 3 of 8 HMs.
- Only 1 of 12 (HM8) ran a real benchmark with a baseline model comparison; FrameCut ran a latency comparison. Accuracy of the end task was never measured against ground truth by any winner.

## Consensus vs divergence

**Table stakes.** Gemini 1.5 via the Python SDK and File API; a token count shown to justify long context (the rubric's "informative" line); a context-cache demo for the 5-point "efficient caching" criterion; a YouTube video (40 of 90 points); an everyday user story. Nobody fine-tuned or used RAG infrastructure instead of long context (HM7's TF-IDF pre-filter is the only retrieval step).

**What separated the 4 winners from the 8 HMs.** (1) Video input: all 4 winners vs 2 of 8 HMs, matching the host's "video understanding and processing" theme. (2) A concrete output artifact a non-expert would use: editable video (FC), brand screen-time chart (SP), insurance CSV (KT), Word PDD + flowchart (PD); 4 of 4 winners use structured output vs 3 of 8 HMs. (3) Caching actually exercised in code by every winner, while two HMs only described it. Technical rigour did **not** decide placement: the most rigorous entry (HM8's benchmark) and the most careful token accounting (HM5) were HMs, while one winner (SP) abandoned its headline long input.

**Contradictory findings on caching.**
- FrameCut measured caching as **slower** on a 107k-token video (Pro edit 56.6 s -> 86.3 s; Flash 20.4 s -> 51.7 s) ([FrameCut](https://www.kaggle.com/code/kyle1373/framecut-a-natural-language-video-editor)); KeepTrack found cached calls **cost more** once storage TTL is priced in, paying off only after more than two reuses ([KeepTrack](https://www.kaggle.com/code/mrdbourke/keeptrack-use-gemini-to-keep-track-of-anything)); Process Documentation concluded caching "significantly reduce[s] the cost" for repeated onboarding questions ([Process Docs](https://www.kaggle.com/code/futureofworkchannel/building-process-documentation-from-video-with-ai)). Reconciliation: caching saves money only with many queries over a short TTL, and did not save latency in late-2024 measurements.
- Flash vs Pro: FrameCut found Flash as accurate as Pro on short videos and faster everywhere; HM8 found Flash's multi-needle recall fell to 46.67% vs near-perfect for Pro at ~900k tokens ([Prompt to PR](https://www.kaggle.com/code/spitfire2nd/from-prompt-to-pull-request-benchmarking-gemini)). Length is the moderator.
- Output limit, not input limit, was the practical bottleneck for KeepTrack (8,192 output tokens; CSV over JSON, continuation sentinel) and for Wikipedia Director (streaming + continuation from cache), whereas GitHub Profile's issue was JSON validity above ~100k input tokens (fixed by one big analysis instead of a list).

## UNVERIFIED
- Per-entry scores; whether HMs received any prize (host thread names none; Prizes page per bank lists 4 x $25k only).
- Exact token counts printed in code outputs but not in markdown: Sports Sponsorship clips, KeepTrack (165k in flowchart vs ~170k in bank), HM4, HM6.
- Which recording Process Documentation actually ran for the PDD (code uses "Offer Acceptation"; the 500k-token "Full" video is cited as an example).
- Entry videos for HM3, HM4, HM5, HM8: not linked inside the pulled notebooks (the competition required one; links may be on the Kaggle page itself).
- HM1 model version; HM8's printed Pro needle accuracy (chart alt text says perfect).
- Gemini 1.5 get-started thread 541152 was not re-read (no winner content expected).
