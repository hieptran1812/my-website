# Drawing with LLMs: top-20 writeup synthesis

- Slug: `drawing-with-llms` (competition id 89659, forum id 6478774)
- Existing deep coverage: `.claude/plans/kaggle-llm-writeup-bank-part3.md`, H2 `## Drawing with LLMs` (metric versions, ranks 1 to 3 in depth, short 4th/5th/12th, key threads). Not repeated here.
- Leaderboard from `competitions.LeaderboardService/GetLeaderboard`, pulled 2026-10-28. Public LB = 52% of test. No ties in the top 20.

## Top-20 table

| Priv rank | Team | Members | Private | Public (rank) | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | new team name | conormacamhlaoibh, huanligong, kenyeungtech | 0.79058 | 0.79038 (1) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581027 (bank) | SSD-1B x64 → 6-colour polygons → hidden prompt text + OCR decoy + searched aesthetic artifacts |
| 2 | move on | takuji, arsenypoyda, samson8, brianwilliamzhao, guoooooooss | 0.77776 | 0.77187 (3) | 581023 (bank) | Pure OCR attack: dashed mask hides text from OCR, visible after preprocessing |
| 3 | CP GMs | cnumber, daiwakun | 0.76947 | 0.77523 (2) | 581024 (bank) | FLUX.1-schnell → vtracer → SigLIP rerank → diffvg through differentiable preprocessing |
| 4 | Kevin Clark | kevlark | 0.74314 | 0.74108 (6) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581108 | SD3.5M LoRA trained with DRaFT-LV on 5 rewards → heuristic SVG → diffvg; best-of-n by VQA proxy |
| 5 | tomirol | tomirol | 0.73168 | 0.74255 (4) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581128 | SDXL-turbo x~28 with beam-searched prompt set → VTracer; gradient-found ~3.5 KB aesthetic patch; PaliGemma yes/no proxy |
| 6 | Grand Painting Artists | tascj0, liushuzhi, kapenon | 0.72474 | 0.74194 (5) | no Kaggle writeup; slide deck https://speakerdeck.com/rist/drawing-with-llms (Rist Inc., 6th place) | FLUX.1-schnell 9 candidates, EasyOCR text removal, vtracer + K-means layering, frame/colour search, yes/no proxy |
| 7 | JJ | jerry4083 | 0.72123 | 0.71321 (15) | no writeup found | - |
| 8 | Psiilu | philippsinger, ilu000 | 0.71961 | 0.73005 (11) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581095 | SDXL fine-tuned on best images of 16 Flux/SDXL runs; 14-18 candidates; "+" OCR decoy; yes/no proxy |
| 9 | linrock | linrock | 0.71473 | 0.71924 (13) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581061 | SDXL LoRA on quantised-colour, score-filtered images; frame + gradient background found by differential evolution |
| 10 | BERA | benbla, raki21 | 0.71389 | 0.73824 (7) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581213 | Flux.1-dev + HyperSD 8-step + Nunchaku, 4-bit on one T4 → vtracer + SVGO; select by VQA+aesthetic+OCR |
| 11 | 💸...🏅🙏 | diana2713, juwonkwon, youngjunh, hygseo | 0.71309 | 0.73044 (10) | no writeup found | - |
| 12 | Rock | rock139 | 0.71292 | 0.72911 (12) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581034 (not linked in API) | SD3.5M trained with Flow-GRPO on the competition metric |
| 13 | Howard（ホワルド） | howarudo | 0.71250 | 0.73544 (8) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581032 | FLUX.1-schnell + Nunchaku, binary-searched VTracer size, dummy VQA + linear calibration |
| 14 | phoenix^2 | sitaberete | 0.70560 | 0.73059 (9) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581063 ("Best outputs solution?") | Lumina-Image-2.0 (2B DiT) fine-tuned on 8.5k rasterised SVGs; LLaVE-2B + aesthetic selection |
| 15 | Mikhail Golubchik | mikhailgolubchik | 0.70528 | 0.69037 (38) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581051 | SD3.5M + vtracer; PaliGemma2-10b-mix-448 picks best of 3 with description split into 3 questions |
| 16 | Coconut Tree on an Island | xyzdivergence, yeoyunsianggeremie, evgeniimaslov2 | 0.70428 | 0.71543 (14) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581094 | SDXL-Flash 512px x~18 → GPU-KMeans vectoriser; aesthetic x SigLIP rerank, then VQA proxy |
| 17 | Magi | shiqiwang123456, chowstanley, wuxu20, wangjinghan2147, tonyhyj704 | 0.70143 | 0.69168 (34) | no writeup found | - |
| 18 | chsh2 | chaosinism | 0.70013 | 0.68724 (45) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/582286 | Quantised Flux.1-schnell → homemade vectoriser; ImageReward over NLTK noun sub-prompts |
| 19 | WXYZ123 | woosungyoon, zackm123 | 0.69744 | 0.71058 (16) | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581012 (not linked in API) | Diffusion → compressed SVG; two VQA questions (shape, colour); early-stop time banking |
| 20 | Learning not so deep | xinyulin1116, losist, fushuaikang, bueyounglaw, hu1zer | 0.69705 | 0.69430 (27) | no writeup found | - |

## Per-team synthesis

### Ranks 1 to 3
The bank covers them in depth. Two facts it lacks, from 6th place's slide deck summary of the field: the deck notes **LLM approaches were rare; Qwen zero-shot SVG scored ~0.5 to 0.6, RL improved it, but diffusion won**. No new primary facts on 1st to 3rd were read for this file.

### 4th - Kevin Clark (kevlark, solo)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581108 ; code https://github.com/clarkkev/svg-diffusion (repo README says it was the only top-5 entry that did not attack the metric).
- **Generator:** Stable Diffusion 3.5 Medium **without the T5 encoder** (small memory, fine-tunable).
- **Training:** LoRA with **DRaFT-LV** (backprop the reward through the sampler; unlike RL/DPO it uses differentiability). Data: 1,000 prompts + Q/A generated by **GPT-4o**, two-stage (50 categories, then prompts; known categories landscapes/abstract/fashion oversampled). Rewards: **HPSv2, PickScore**, the competition aesthetic score, a **VQA proxy** ("Does the image show {prompt}?", faster and apparently better than real question sets), and **-LPIPS(image, convert_to_svg(image))** so outputs vectorise well.
- **Result:** ~+10% aesthetic, ~+5% VQA, ~+0.03 LB; removed the need for prompt engineering.
- **SVG:** simple heuristic converter after Rich Olson, then **diffvg** minimising L1 to the diffusion image over path positions/colours ("worked amazingly well"); Rich Olson's OCR decoy.
- **Inference:** keep generating until ~60 s per prompt (adaptive to schedule), typically 5-6 candidates; return highest VQA proxy among those with OCR score 1.
- Distinguishing idea: train the generator on the judge (and on vectorisability) instead of hacking the judge.

### 5th - tomirol (solo)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581128 ; code https://github.com/matheuspf/draw ; final notebook linked in the post.
- **Generation:** SDXL-turbo with a set of prompts chosen by **beam search over ~200 candidate prompts** on a generated dataset (more prompts kept helping, 20+). Negative prompts "no text", "no other colors" helped slightly. Tried and dropped: fine-tuning SDXL-turbo on preference data, REFL with ImageReward, DPO with VQA reward.
- **Vectorise:** `primitive` looked better but took >30 s on Kaggle CPUs; **VTracer** <2 s single-core with polygon simplification and size clipping (~6 KB per image); small white OCR decoy on black.
- **Selection:** VQA proxy per arXiv 2404.01291 = P("Yes") vs P("No") for "Does this image contain {description}?"; beat generated Q/A from Qwen-2.5. The 10B-448 PaliGemma was always better than 3B-224/10B-224/3B-448 despite being slower.
- **Pipelining:** three background processes (diffusion on cuda:0, VTracer on CPU, proxy on cuda:1) to saturate slow Kaggle CPUs; **28 SVGs per prompt** in ~55 s.
- **Aesthetic patch:** built a fully differentiable copy of preprocessing + aesthetic + VQA; gradient ascent on a ~100x100 px path patch over 150 background images, a few hours; positioned away from edges because of random crops; adds ~3.5 KB and lifts **average aesthetic to ~0.7**; patches >0.8 possible but cost too much VQA.
- **Did not ship (worked but too slow / memory / >10 KB / unstable):** direct DiffVG on full score, hiding text from OCR, ReNO-style optimisation of SDXL-turbo latents against the full loss (nicer images, only 3B-224 VQA fit on a T4 with gradients), image-reward-style fine-tuning with differentiable VQA.

### 6th - Grand Painting Artists (tascj0, liushuzhi, kapenon)
No Kaggle writeup in the API or forum list. Source: Rist Inc. slide deck (Japanese) https://speakerdeck.com/rist/drawing-with-llms (6th place; member attribution from the LB only).
- **Generation:** FLUX.1-schnell at 384 px, chosen for prompt adherence ("flat design", "pastel color" hard for SD models). 9 candidates per description: 5 from "Stylized vector art of {description}, up to 4 colors, max 5 objects, bold, flat, clean lines", 4 from "Modern flat illustration with cartoon-style outlines…". Text regions detected with **EasyOCR** filled with the average colour. Text encoder run once per prompt: 9 images 49.5 s → 24.9 s.
- **SVG:** vtracer; RDP simplification if >15 KB after 2 loops; **K-means into 16 colour clusters, largest cluster replaced by a background rectangle**; "Z" OCR decoy bottom right; **rounded-corner frame with searched border/decoy colours** for aesthetic; scour; accept 8.5 to 10 KB else retune vtracer; fallback adds polygons by region.
- **Selection:** pseudo-score = aesthetic + OCR fixed at 1.0 + VQA proxy (P(yes) incl. capitalised variants for one generic question).
- **Validation:** only 15 public descriptions, so **DeepSeek V3** generated descriptions (few-shot), filtered to SigLIP similarity ≤0.7 to every existing one, then Q/A pairs: 100 cases, strongly LB-correlated until late.

### 7th - JJ (jerry4083, solo)
No writeup found (forum titles, comments and web search checked). Public 15 → private 7.

### 8th - Psiilu (philippsinger, ilu000)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581095
- **Validation:** a few hundred generated prompt/question samples; good correlation with LB but high variance on both, so they selected two high-scoring subs differing only by seed.
- **Model:** SDXL **fine-tuned on the best images from 16 prior runs** of prompted FLUX-schnell and SDXL models, ranked by their local validation score (self-distillation of the best-of-many).
- **Infra:** both GPUs generate and score in parallel; Kaggle Packages context errors forced an API-based setup. 14 to 18 images per sample.
- **SVG:** optimised VTracer settings + `scour` minification + resizing to fit 10 KB. OCR: a small "+" in a corner (robust and cheap but, in hindsight, worse for aesthetic than others' decoys).
- **Selection:** proxy question "Does this look like {prompt}?" yes/no, preprocessing reduced to MedianBlur for speed.
- **Did not ship:** diffvg + differentiable preprocessing (promising locally, killed by "Package Prediction Error"); differentiable aesthetic hacking (limited success under the defences while keeping VQA). Self-assessed missing pieces: diffvg and prompt-image similarity scores.

### 9th - linrock (solo)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581061 ; notebook https://www.kaggle.com/code/linrock/9th-place-sdxl-lora-vqa-boost-aesthetics
- **Diffusion:** SDXL base 1.0 + TinyVAE, DPMSolver++ Karras 2M, guidance 5, 8 steps, 768x768.
- **LoRA data:** 1.5k LLM-generated descriptions; images from FLUX dev and Hyper SDXL; **quantised to 24 colours**, median + gaussian filtered; keep the top-scoring image per base prompt by PaliGemma2 VQA and aesthetic. Trained with diffusers `train_text_to_image_lora_sdxl.py`, **offset noise 0.0357** to allow high-contrast images. Beat every base model/community LoRA/prompting combination tried.
- **Selection:** combos of basic yes/no questions ("Does the image show…", "Are there…", "Does … exist in the image?"); Qwen-tailored questions were not better.
- **Aesthetic tweaks:** **differential evolution** over image modifications; frames on green backgrounds did well; gradient direction and frame colour varied. Highest post-blur aesthetic images were vivid landscapes.
- PNG→SVG built on Rich Olson's notebook; OCR decoy over his earlier signature. No useful role for LLMs beyond data generation.

### 10th - BERA (benbla, raki21)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581213
- **Generation:** Flux.1-dev + **HyperSD LoRA (8 steps instead of 50)** + **Nunchaku** for the transformer; whole pipeline **4-bit bitsandbytes** on one T4; ~10 images in ~20 s.
- **SVG:** vtracer, geometric simplification (drop a point within 2 px of its neighbours' line), **SVGO** (wrapped in Python, installed from a private GitHub repo with a PAT), **binary search over vtracer parameters** under 10,000 bytes on 4 processes; two small "M" watermarks (black and white) as OCR decoys.
- **Selection:** score all candidates with VQA and aesthetic evaluators, then walk down the ranking to the first that **passes OCR**; on the top candidate try a variant with a **gradient silver border**.
- Category notes: landscapes near perfect, fashion loses detail in conversion, geometry (trapezoids) ~10% success across 12+ models.
- **Unselected better private sub:** prompt rewriting with **different formats per encoder** (comma keywords for CLIP, sentences for T5).
- Pain: Nunchaku install, SVGO packaging, multi-stage multiprocessing that ran in notebooks but not on the submission server.

### 11th - 💸...🏅🙏 (diana2713, juwonkwon, youngjunh, hygseo)
No writeup found; members only left congratulatory comments on others' posts.

### 12th - Rock (rock139, solo)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581034 ; code https://github.com/yanyanhuang/12th-Place-Solution-for-Kaggle-Drawing-with-LLMs (team writeup not linked in the API).
- **Data:** 4,000 training + 200 test samples (description + Q/A) generated with **Gemini 2.5 Flash**.
- **Model/training:** SD3.5 Medium (Flux too heavy for GRPO VRAM) trained with **Flow-GRPO**, reward = the competition metric (harmonic mean of VQA and aesthetic). 4x H800, 512x512, 8 images per GPU per step, 16 images per description, **KL weight 4e-4**.
- **Results:** base SD3.5M ~0.65 public / 0.72 local; after 500 steps local ~0.77, private ~0.73 (missed submitting the best checkpoint). Surprise: GRPO lifted **VQA more than aesthetic** (local VQA 0.82 → 0.91).
- **Inference:** 3 images → vtracer + SVG command merging/relative coords/segment merging → filter with PaliGemma 10B on a dummy question.

### 13th - Howard (howarudo, solo)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581032 ; notebook https://www.kaggle.com/code/howarudo/13th-place-solution-flux-svg-scoring
- FLUX.1-schnell + **Nunchaku** quantisation (better prompt adherence and faster than SD bases or an SDXL LoRA). 10 to 12 images per prompt.
- **SVG:** binary search for the largest bitmap size that fits 10 KB; compression (absolute → relative coords, drop `translate`, longest path → fill rect) lets bitmaps be ~50 px larger. A lowercase "a" in the corner as OCR decoy.
- **Selection:** two dummy questions with graded choices (definitely/somewhat/barely/no; yes/somewhat/no), plus a **linear regression mapping dummy scores to the real VQA score** fitted on an extended train set. Judged that more images x fewer questions beats fewer images x more questions (VQA <2 s per question, image ~1 s).
- Regrets not trying pydiffvg or SigLIP.

### 14th - phoenix^2 (sitaberete, solo)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581063 (titled "Best outputs solution?", linked as the team writeup in the API).
- **Final:** fine-tuned **Lumina-Image-2.0** (2B DiT) on **8.5k rasterised SVGs** of decent aesthetics (≤10 KB), retrained for 256 px and up; 5 images per GPU at 320x320 on both GPUs (worked around the Packages multithreading bug by defining methods in `__init__`); iterative VTracer + Node.js SVG optimiser to <10 KB; **LLaVE-2B** for prompt alignment + competition aesthetic for selection; circle OCR decoy.
- Earlier: SSD-1B LoRA and full fine-tune (poor), Flex.1-alpha 8B (good but slow), fine-tuned Flux-dev (best-looking but lower score).
- **LLM route that failed:** fine-tuning **Qwen2.5-VL-3B** to write SVG from starvector/text2svg-stack (2M+ noisy, mostly black-and-white icons), then a curated 8.5k set; two weeks without good results; planned GRPO "thinking" refinement never reached.
- Metric observation: deliberately degraded, barely recognisable outputs still scored similarly or higher locally with OCR 1.0; critical of the metric.

### 15th - Mikhail Golubchik (solo)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581051. SD3.5M with suffix "Flat illustration, stereotypical"; vtracer with binary search over parameters; **PaliGemma2-10b-mix-448** selects the best of 3. The main boost: **splitting the description into three questions** (TF-IDF-like bigrams for short descriptions, e.g. "Is purple forest?", "Is at dusk?"; three chunks for long ones). Successful submission came near the end; biggest riser in the top 20 (public 38 → 15).

### 16th - Coconut Tree on an Island (xyzdivergence, yeoyunsianggeremie, evgeniimaslov2)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581094 ; code https://github.com/bogoconic1/16th-place-kaggle-drawing-with-llms
- Budget: 36 s generation + aesthetic, 27 s selection per prompt.
- **SDXL-Flash** (tried Sana, SD3.5) at 512x512: up to 18 images in 30 s (~80% faster than 1024); prompt `Cartoon style of "{description}".` (inspired by DOODL's cover image; converts with less loss).
- **Vectoriser:** public bitmap_to_svg_layered sped up 15x by replacing cv2 KMeans with **cuML GPU KMeans** (~0.3 s); integer coords, polygon→path, colour grouping in `<g>`, scour: 60 → 168 elements on average in 10 KB.
- **Selection:** faster metric preprocessing (cv2 median, CuPy FFT). Rerank all candidates by **aesthetic x SigLIP-so400m alignment**, then run the expensive VQA+OCR proxy (one question, ~3 s) down the ranking until 63 s (~7 evaluations).
- **CV:** 500-example validation set; stable 0.705 to 0.715 public / 0.700 to 0.705 private; CV correlates up to ~0.7 LB. An unselected 0.717 private sub was a fluke (resubmission 0.709).
- **Failed:** dedup by similarity, LLM-generated questions at inference, backgrounds to lift aesthetic, reranking without VQA (best 0.701/0.698), PaliGemma 3B for VQA/OCR.

### 17th - Magi (shiqiwang123456, chowstanley, wuxu20, wangjinghan2147, tonyhyj704)
No writeup found.

### 18th - chsh2 (chaosinism, solo)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/582286
- Based on "Try, Try Again" (Karthik et al. 2023): ranking candidates by **ImageReward** raises TIFA.
- Quantised Flux.1-schnell (Q4 slightly better than nf4), prompt "{description}, stylized proportions, exaggerated form, children illustration".
- **Homemade vectoriser** (from the author's nijiGPen project): LAB colour quantisation with K-means + Felzenszwalb, a multi-colour marching-squares contour pass run once, Douglas-Peucker simplification preserving 3-colour junctions; 4 parameter sets under the size limit scored.
- **NLTK:** one ImageReward sub-prompt per noun averaged with a concave function (beat full-prompt ImageReward; ImageReward beat CLIP and SigLIP as a score predictor); appending WordNet-style **definitions** for abstract words ("trapezoids (a quadrilateral with two parallel sides)") as an extra candidate prompt.
- First Kaggle competition; public 45 → private 18.

### 19th - WXYZ123 (woosungyoon, zackm123)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581012 (not linked in API). Diffusion → SVG; the **bitmap→SVG converter was the most impactful factor** (relative paths, colour grouping). Two independent VQA questions (shape and colour). **Time banking:** fixed 63 s budget per prediction, batches of 3 images, early stop above a score threshold, leftover time carried to the next prompt. Tried PixArt and Sana.

### 20th - Learning not so deep (xinyulin1116, losist, fushuaikang, bueyounglaw, hu1zer)
No writeup found.

## Technique frequency matrix

Y / N / ? per private rank. Writeups (or equivalent) exist for 16 of 20: ranks 1-6, 8-10, 12-16, 18, 19 (6th via slide deck). No writeup: 7, 11, 17, 20.

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Final SVG comes from diffusion raster → vectorise | Y | N | Y | Y | Y | Y | ? | Y | Y | Y | ? | Y | Y | Y | Y | Y | ? | Y | Y | ? |
| LLM writes the SVG directly in the final sub | N | N | N | N | N | N | ? | N | N | N | ? | N | N | N | N | N | ? | N | N | ? |
| SDXL family (SSD-1B, SDXL, -turbo, -Flash) | Y | N | N | N | Y | N | ? | Y | Y | N | ? | N | N | N | N | Y | ? | N | ? | ? |
| FLUX family | N | N | Y | N | N | Y | ? | Y | N | Y | ? | N | Y | N | N | N | ? | Y | ? | ? |
| SD3.5 Medium | N | N | N | Y | N | N | ? | N | N | N | ? | Y | N | N | Y | N | ? | N | ? | ? |
| Generator fine-tuned (LoRA / full / RL) | N | N | N | Y | N | N | ? | Y | Y | N | ? | Y | N | Y | N | N | ? | N | ? | ? |
| ...of which reward-gradient or RL on the judge (DRaFT, Flow-GRPO) | N | N | N | Y | N | N | ? | N | N | N | ? | Y | N | N | N | N | ? | N | ? | ? |
| Synthetic prompts/Q-A from an LLM (train or validation) | ? | ? | ? | Y | Y | Y | ? | Y | Y | ? | ? | Y | Y | N | ? | Y | ? | N | ? | ? |
| Fast-sampling tricks (TAESD/turbo/Hyper/Flash/Nunchaku/4-bit) | Y | N | Y | ? | Y | Y | ? | ? | Y | Y | ? | ? | Y | N | ? | Y | ? | Y | ? | ? |
| Many candidates per prompt (≥8) | Y | N | Y | N | Y | Y | ? | Y | ? | Y | ? | N | Y | Y | N | Y | ? | ? | N | ? |
| VQA proxy (PaliGemma yes/no or dummy Q) for selection | N | N | N | Y | Y | Y | ? | Y | Y | Y | ? | Y | Y | N | Y | Y | ? | N | Y | ? |
| Embedding/CLIP-style rerank (SigLIP, LLaVE, ImageReward) | N | N | Y | N | N | N | ? | N | N | N | ? | N | N | Y | N | Y | ? | Y | N | ? |
| Aesthetic score used in selection | Y | ? | Y | N | N | Y | ? | N | Y | Y | ? | Y | N | Y | N | Y | ? | Y | N | ? |
| diffvg (differentiable rasteriser) refinement shipped | N | N | Y | Y | N | N | ? | N | N | N | ? | N | N | N | N | N | ? | N | N | ? |
| Re-implemented metric preprocessing locally | Y | Y | Y | ? | Y | ? | ? | Y | ? | ? | ? | ? | ? | ? | ? | Y | ? | ? | ? | ? |
| OCR decoy (letters/shape in a corner) | Y | Y | Y | Y | Y | Y | ? | Y | Y | Y | ? | ? | Y | Y | ? | Y | ? | Y | ? | ? |
| Hidden text exploiting the OCR-vs-VQA preprocessing gap | Y | Y | N | N | N | N | ? | N | N | N | ? | N | N | N | N | N | ? | N | N | ? |
| Searched aesthetic artifact/patch/frame/border | Y | Y | N | N | Y | Y | ? | N | Y | Y | ? | N | N | N | N | N | ? | N | N | ? |
| Size-fitting search (binary search on bitmap size or vectoriser params) | ? | N | Y | ? | ? | Y | ? | Y | ? | Y | ? | ? | Y | Y | Y | N | ? | Y | ? | ? |
| SVG minification (relative coords, colour groups, scour/SVGO) | Y | ? | Y | ? | Y | Y | ? | Y | ? | Y | ? | Y | Y | Y | ? | Y | ? | Y | Y | ? |

Cells for 1 to 3 come from the bank's text. "Many candidates" for 4 = 5-6 (N), 12 and 15 = 3, 19 = batches of 3 with early stop.

**Counted summary (16 teams with writeups):**
- 15 of 16 shipped diffusion → raster → vectorise; the only exception (2nd) shipped pure text plus OCR masking. **0 of 16** shipped an LLM writing SVG; 14th tried Qwen2.5-VL-3B for two weeks and gave up.
- 11 of 16 used a PaliGemma yes/no-style **VQA proxy** to pick among candidates; 4 used an embedding/reward reranker instead or in addition (SigLIP 3rd/16th, LLaVE 14th, ImageReward 18th); 1st selected by aesthetic score only.
- 13 of 16 added an OCR decoy (a letter, "+", circle, "Z", "M", "a", "ZOK").
- 6 of 16 added a searched aesthetic artifact, patch, frame or border; **4 of the top 6** did (all but 3rd and 4th).
- 5 of 16 fine-tuned the generator (4th, 8th, 9th, 12th, 14th); 2 of them trained against the judge's own reward (4th DRaFT, 12th Flow-GRPO).
- Only 2 of 16 shipped diffvg refinement (3rd, 4th), both in the top 4; 3 more tried and dropped it (2nd, 5th, 8th).
- FLUX family 6, SDXL family 5, SD3.5M 3, Lumina 1.

## Consensus vs divergence

**Table stakes:** diffusion image → vectorise (VTracer or custom K-means/contour) → aggressive SVG compression to use the 10 KB budget; best-of-n under a ~60 s per-prompt budget; a PaliGemma VQA proxy; an OCR decoy; LLMs used only for synthetic prompts/questions. Squeezing more candidates into the time budget (pipelining across 2 GPUs and CPU, faster samplers, GPU K-means) is the most common engineering theme.

**What separated the top 5 from 6 to 20:** each of the top 5 attacked the *judge* in one deliberate way beyond best-of-n. 1st and 2nd exploited the OCR-before / VQA-after preprocessing gap with hidden text; 3rd optimised the SVG through a differentiable copy of the preprocessing (diffvg + aesthetic + SigLIP); 4th backpropagated judge-like rewards into the generator (DRaFT) and used diffvg; 5th gradient-searched a universal aesthetic patch. Ranks 6 to 20 mostly stopped at best-of-n selection with cosmetic aesthetic tweaks (frames, borders) and better vectorisers. The private-score gap is large: 0.743 (4th) vs 0.725 (6th) vs 0.697 (20th).

**Contradictory findings:**
- *Question design for the VQA proxy:* one generic yes/no beat generated question sets (4th 581108, 5th 581128, 9th 581061, 16th 581094 "training LLM to generate real-time questions" failed) vs **splitting the description into three questions was the main boost** (15th, 581051) and two shape/colour questions (19th, 581012).
- *Fine-tuning the generator:* +0.03 LB from DRaFT (4th), Flow-GRPO 0.65 → ~0.73 (12th 581034), SDXL LoRA beat all bases (9th) vs preference fine-tuning, REFL and DPO not better than prompt-searched SDXL-turbo (5th), SDXL LoRA worse than FLUX-schnell (13th 581032), SSD fine-tuning poor (14th 581063).
- *diffvg:* "worked amazingly well" (4th) and core of 3rd vs failed because the augmentations were non-differentiable (2nd 581023), too unstable/slow (5th), blocked by package errors (8th 581095).
- *Backgrounds and frames for aesthetic:* frames on green backgrounds help (9th), rounded frame (6th), silver border variant (10th 581213) vs "adding backgrounds" failed (16th).
- *Prompt engineering:* unnecessary after DRaFT (4th) vs beam-searched prompt ensembles of 20+ prompts (5th) and encoder-specific rewriting that improved private LB (10th's unselected sub).
- *Bigger is better for the proxy:* PaliGemma 10B-448 always better than smaller variants (5th); 3B failed as proxy (16th). Consistent, not contradictory.

## Aggregated "what did not work"
- Direct LLM SVG generation (Qwen2.5-VL-3B fine-tune on text2svg-stack and curated 8.5k) - 14th; field-level: LLM zero-shot ~0.5 to 0.6 (6th deck).
- diffvg / differentiable preprocessing in the final pipeline - 2nd, 5th, 8th.
- ReNO latent optimisation against the full loss (good images, too slow, only 3B-224 VQA fits) - 5th.
- Fine-tuning SDXL-turbo on preference data, REFL with ImageReward, DPO with VQA reward - 5th.
- SSD-1B LoRA/full fine-tune; Flex.1-alpha (slow); fine-tuned Flux-dev (scored lower) - 14th.
- `primitive` vectoriser (too slow on Kaggle CPUs) - 5th.
- Generated question sets at inference (Qwen, trained LLM) - 5th, 9th, 16th.
- PaliGemma 3B as proxy - 16th; smaller PaliGemma variants - 5th.
- Similarity dedup of candidates; added backgrounds; reranking without a VQA estimate - 16th.
- Differentiable aesthetic hacking under the final defences - 8th (limited), 5th (patch worked, other variants did not ship).
- Multi-process pipelines that ran in notebooks but failed on the submission server - 10th; Kaggle Packages multithreading bug - 6th, 14th, 1st (bank).
- Adding CLIP/VQA to 1st's aesthetic-only selection; >64 images - 1st (bank).
- Removing or moving the OCR decoy per image - 14th.

## Shake-up note
Small overall (Ravi Ramakrishnan, https://www.kaggle.com/competitions/drawing-with-llms/discussion/581017: top 10% nearly stable; largest top-10% rise 120 → 45, fall 26 → 111). Inside the top 20:
- **Risers:** 15th Mikhail Golubchik (38 → 15; his successful submission came "near the end of the competition at night", 581051), 18th chsh2 (45 → 18), 17th Magi (34 → 17), 20th (27 → 20), 7th JJ (15 → 7).
- **Fallers:** 14th phoenix^2 (9 → 14), 13th Howard (8 → 13), 10th BERA (7 → 10; they had an unselected sub that "scored much higher on private LB", 581213).
- Explanation offered by writeups: high per-submission variance from random seeds and OCR luck. 8th selected two subs differing only in seed to hedge (581095); 16th's 0.717 private outlier resubmitted at 0.709 (581094). With ~500 test prompts split 52/48, one OCR hallucination per prompt moves the score visibly.

## UNVERIFIED
- 6th place details come from a third-party summary of a Japanese slide deck (WebFetch of speakerdeck.com/rist); the deck does not list members, so attribution to tascj0/liushuzhi/kapenon is from the LB only, and exact numbers should be re-checked against the deck.
- Which generator 19th shipped (PixArt and Sana "experimented with"; final model not named).
- Whether 1st, 2nd, 3rd used LLM-generated validation prompts (not in the bank's notes).
- No writeup located for ranks 7, 11, 17, 20 (forum titles of all 319 topics, comment authors, one web search).
