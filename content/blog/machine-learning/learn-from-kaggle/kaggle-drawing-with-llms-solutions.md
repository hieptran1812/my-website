---
title: "Drawing with LLMs: how the top 20 teams drew SVGs for a VQA judge, and then gamed it"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "Kaggle asked for text-to-SVG generation scored by PaliGemma question answering, a CLIP aesthetic model and an OCR penalty. The winners used diffusion plus vectorisation, never an LLM, and the top two hid text from the OCR pass while showing it to the judge. The metric worked on a toy example, the 1st-place pipeline, and what all 16 published top-20 sources did."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "svg", "diffusion", "vqa", "paligemma", "metric-exploit", "text-to-image", "evaluation"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 23
---

> **TL;DR.** In Drawing with LLMs (Kaggle, February to May 2025) you shipped a model that turned a short description, such as "a lighthouse on a cliff", into an SVG file under 10,000 bytes. A hidden judge rendered it, asked PaliGemma four questions about it, scored its looks with a CLIP aesthetic model, and cut the score if OCR could read more than four characters in it. **The winner, "new team name", never asked an LLM to write SVG.** They generated 64 images with a diffusion model, traced them into polygons, and then hid the description as text that the OCR pass could not see but the question-answering pass could. 2nd place went further and shipped almost nothing but hidden text. 3rd place did it the honest way, optimising real vector shapes through a differentiable copy of the judge, and finished 0.008 behind 2nd. **Decisive lever: metric insight.** **The one lesson:** when a learned judge looks at two differently processed copies of your output, the gap between those copies is an attack surface.

![The lever strip: of the five levers that decided Kaggle LLM competitions, Drawing with LLMs was won on metric insight, with best-of-n test-time compute as a strong second](/imgs/blogs/kaggle-drawing-with-llms-solutions-1.webp)

The [orientation post](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work) introduced the lever strip. This is one of three competitions where the fifth lever decided the podium; the others are [LLM Prompt Recovery](/blog/machine-learning/learn-from-kaggle/kaggle-llm-prompt-recovery-solutions), won by a magic string against an embedding metric, and [You Can't Please Them All](/blog/machine-learning/learn-from-kaggle/kaggle-llms-you-cant-please-them-all-solutions), where the metric was a committee of LLM judges. If you grade generated images with a VLM, this one is the most practical of the three. And the name is a small joke at the field's expense: the best LLMs writing SVG directly scored around 0.67, the winners 0.79, and none of the 16 top-20 teams that left a source shipped an LLM-written SVG.

## 1. The task in 60 seconds

![The notebook budget and an illustrative data row: a description, four multiple-choice questions with the correct answer marked, and a toy flat drawing that would answer them](/imgs/blogs/kaggle-drawing-with-llms-solutions-2.webp)

As a friend would put it: "Draw a one-line description in SVG code, under 10 KB, in under five minutes. A robot quizzes itself about your picture and says whether it is pretty." Formally:

| Item | Value |
| --- | --- |
| Dates | 2025-02-25 to 2025-05-27 (entry and merger deadline 2025-05-19) |
| Submission | A Kaggle Package: a `class Model` with `predict(description) -> svg`. The first competition to require this format |
| Input | About 500 hidden descriptions of everyday objects and scenes, at most 200 characters, about 50 on average |
| Categories | About a dozen. `landscapes`, `abstract` and `fashion` appear everywhere; the others are unique to public or private. No people, brands or names |
| Output | One SVG per description, at most 10,000 bytes, allowlisted elements and attributes, no CSS `style`, no raster data, no external references |
| Notebook | Internet off, at most 9 hours in total, at most 5 minutes per SVG. The winners used the two T4 GPUs |
| Leaderboard | Public 52% / private 48% of the test rows, 5 submissions per day |
| Field | 1,309 teams, 1,725 competitors, 38,770 submissions |
| Prizes | 50,000 USD: 12k, 10k, 10k, 10k, 8k for places 1 to 5 |

With only 15 public training descriptions, serious teams generated validation sets with an LLM. That is where LLMs earned their keep here: as data generators, not artists.

## 2. The metric, with a worked toy score

The metric, SVG Image Fidelity, went through three versions; the final one, live from 2025-03-20 with every submission rescored, decided the prizes. The formulas below come from its public source (the `metric/svg-image-fidelity` notebook, pulled 2026-10-31).

![The metric's two lanes: OCR reads a lightly processed copy of the render, while VQA and the aesthetic model read a heavily softened copy, and the final score multiplies their combination by the OCR penalty](/imgs/blogs/kaggle-drawing-with-llms-solutions-3.webp)

For each description, the judge does this:

1. **Render** the SVG with cairosvg to 384 by 384 pixels.
2. **Heavy preprocessing** for judging: random crop of 3% and resize, JPEG at quality 95, a median filter of size 9, an FFT low-pass at cutoff 0.5, a bilateral filter, and JPEG at quality 92. The host wrote that this "tends to significantly soften an image" and "is also fairly effective at scrubbing text".
3. **VQA.** PaliGemma answers four description-specific questions in the TIFA style (yes/no or multiple choice). The score for each question is the model's probability on the correct choice, renormalised over the allowed choices. VQA is the mean over the questions, so it is a soft number, not a count of right answers.
4. **Aesthetic.** A CLIP ViT-L/14 embedding goes through the "improved aesthetic predictor" MLP, and the output is divided by 10.
5. **OCR penalty.** The judge resets to the raw render, applies only a random crop and JPEG at quality 90, and asks PaliGemma to OCR it. If it reads $n$ characters, the penalty is

$$
\text{ocr}(n) = \min\left(1,\; e^{\,4 - n}\right)
$$

Four characters are free. The fifth costs a factor of $e^{-1} \approx 0.37$, the sixth brings it to 0.135, and by ten characters the image is worth almost nothing.

6. **Combine.** VQA and aesthetic are merged with a weighted harmonic mean and multiplied by the OCR penalty:

$$
s = \frac{(1+\beta^2)\, V A}{\beta^2 V + A} \cdot \text{ocr}(n), \qquad \beta = 0.5
$$

The code calls `harmonic_mean(vqa, aesthetic, beta=0.5)`; with VQA in the first slot that equals $\beta = 2$ with the arguments swapped, which is how the forum describes it. VQA counts more. The final score is the mean of $s$ over descriptions.

Notice that OCR reads the *lightly* processed image while VQA and the aesthetic model read the *heavily* processed one. Hold that thought; it is the whole competition.

### A toy score you can run

This illustrative scorer is not the host's code: it copies the two formulas and takes the model outputs as plain numbers.

```python
import math

# Illustrative toy scorer. NOT the host's code: the two formulas are copied
# from the public metric notebook; the model outputs are passed in as numbers.
def harmonic_mean(a, b, beta=1.0):
    # a = VQA, b = aesthetic, exactly as the metric calls it
    if a <= 0 or b <= 0:
        return 0.0
    return (1 + beta**2) * (a * b) / (beta**2 * a + b)

def ocr_penalty(num_chars, free_chars=4):
    return min(1.0, math.exp(-num_chars + free_chars))

def instance_score(vqa, aesthetic, ocr_chars):
    return harmonic_mean(vqa, aesthetic, beta=0.5) * ocr_penalty(ocr_chars)

candidates = [
    # name,                          VQA,  aesthetic, chars OCR reads
    ("A honest drawing",             0.80, 0.55, 0),
    ("B same, OCR hallucinates",     0.80, 0.55, 6),
    ("C same + 3-letter decoy",      0.80, 0.55, 3),
    ("D description as plain text",  0.97, 0.40, 30),
    ("E hidden text + decoy + patch",0.92, 0.58, 3),
]
for name, v, a, n in candidates:
    print(f"{name:31s} VQA={v:.2f} aes={a:.2f} "
          f"ocr={ocr_penalty(n):.3f} score={instance_score(v, a, n):.3f}")

print(f"VQA 0.9, aes 0.5 -> {harmonic_mean(0.9, 0.5, 0.5):.3f}")
print(f"VQA 0.5, aes 0.9 -> {harmonic_mean(0.5, 0.9, 0.5):.3f}")
```

```console
A honest drawing                VQA=0.80 aes=0.55 ocr=1.000 score=0.733
B same, OCR hallucinates        VQA=0.80 aes=0.55 ocr=0.135 score=0.099
C same + 3-letter decoy         VQA=0.80 aes=0.55 ocr=1.000 score=0.733
D description as plain text     VQA=0.97 aes=0.40 ocr=0.000 score=0.000
E hidden text + decoy + patch   VQA=0.92 aes=0.58 ocr=1.000 score=0.823
VQA 0.9, aes 0.5 -> 0.776
VQA 0.5, aes 0.9 -> 0.549
```

![Toy scores for five candidate drawings of a red lighthouse: the OCR penalty curve on the left, and on the right the honest drawing at 0.733, a hallucinated-OCR copy at 0.099, the decoy fix back at 0.733, plain text at zero and hidden text at 0.823](/imgs/blogs/kaggle-drawing-with-llms-solutions-4.webp)

The component values are made up for illustration; the arithmetic is the real metric's. Five things fall out of it, and every top team learned all five:

- **Swapping the two numbers changes the score a lot.** VQA 0.9 with aesthetic 0.5 scores 0.776; the mirror image scores 0.549. Points of VQA are worth more.
- **An honest image can be wiped out by a hallucination.** The host warned that PaliGemma sometimes "reads" more than four characters in a picture with no text. Six phantom characters cut candidate B from 0.733 to 0.099.
- **A three-letter decoy fixes that.** Give OCR something real and short to read, and it stops inventing. Candidate C is back at 0.733. Thirteen of the 16 top-20 sources shipped some version of this.
- **Writing the description in the image is a VQA dream and an OCR disaster.** Candidate D answers every question (VQA 0.97) and scores zero.
- **Unless OCR cannot see the text.** Candidate E is the winners' move: the text is there for the judge and invisible to the penalty.

## 3. What made it hard: an honest picture tops out early

![F-beta iso-score curves over the aesthetic and VQA plane: 1st place sits near the 0.8 curve, 3rd near 0.77, and a direct LLM-written SVG near 0.67; a 0.1 gain in VQA moves 1st place further than a 0.1 gain in aesthetic](/imgs/blogs/kaggle-drawing-with-llms-solutions-5.webp)

The chart above is the metric's landscape. Each curve is a constant score; to climb, you move up (VQA) or right (aesthetic). Three real points sit on it, each recomputed from the numbers its authors published:

- **1st place**, validation averages VQA ~0.88 and aesthetic ~0.563. The formula gives 0.791, and their private score was 0.79058.
- **3rd place**, who titled their writeup "VQA/AES = 0.81/0.64". The formula gives 0.769, and their private score was 0.76947.
- **An LLM writing SVG directly.** Zhuang Jia ran 11 commercial and open models on the training descriptions under the final metric. The best, Claude-3.7-Sonnet, averaged 0.670 over 14 rows, with mean VQA 0.774 and mean aesthetic 0.451. (The formula on those means gives 0.677; the gap is because the mean of scores is not the score of means.) Qwen2.5-14B-Instruct scored 0.659 and GPT-4o 0.643.

At the 1st-place point, adding 0.1 VQA lifts the score from 0.791 to 0.854; adding 0.1 aesthetic lifts it only to 0.826. So the gradient says: get VQA up. And here is the bind. VQA is a small vision model quizzing a 384-pixel image that a size-9 median filter has smeared. 10th place (BERA) reported landscapes near perfect but geometry like trapezoids at about 10% success across 12+ models. The aesthetic side has its own ceiling: a predictor trained on photographs does not love flat clip art.

The constraints compounded it. 10,000 bytes is a few hundred polygons. Nine hours for 500 descriptions is about a minute per prompt with margin, and most teams converged on 60 seconds. Kaggle Packages were new: 1st, 6th and 14th report a multithreading bug that forced slower single-threaded code. And one OCR hallucination per prompt is visible on the leaderboard, so every submission was noisy.

## 4. The top-20 scorecard

Ranks and scores come from Kaggle's leaderboard API. "Public rank (move)" is the public rank with the move to private in parentheses; positive means the team climbed. "No writeup" means none was found among all 319 forum topics, comments and API writeup links.

| Rank | Team | Private | Public rank (move) | Key idea | Source |
| --- | --- | --- | --- | --- | --- |
| 1 | new team name | 0.79058 | 1 (=) | SSD-1B x64, 6-colour polygons, hidden prompt text, "ZOK" decoy, searched artifacts, pick max aesthetic | [581027](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581027) |
| 2 | move on | 0.77776 | 3 (+1) | Pure OCR attack: dashed-line mask hides text from OCR; "Zoe" aesthetic letters | [581023](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581023) |
| 3 | CP GMs | 0.76947 | 2 (-1) | FLUX.1-schnell, vtracer, SigLIP rerank, diffvg through a differentiable copy of the preprocessing; no text hack | [581024](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581024) |
| 4 | Kevin Clark | 0.74314 | 6 (+2) | SD3.5 Medium LoRA trained with DRaFT-LV on five rewards, diffvg, best-of-n by VQA proxy | [581108](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581108) |
| 5 | tomirol | 0.73168 | 4 (-1) | SDXL-turbo x28 with beam-searched prompts; gradient-found ~3.5 KB aesthetic patch | [581128](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581128) |
| 6 | Grand Painting Artists | 0.72474 | 5 (-1) | FLUX.1-schnell, EasyOCR text removal, K-means layering, frame search (unverified deck) | [slide deck](https://speakerdeck.com/rist/drawing-with-llms) |
| 7 | JJ | 0.72123 | 15 (+8) | unknown | no writeup |
| 8 | Psiilu | 0.71961 | 11 (+3) | SDXL fine-tuned on its own best images from 16 runs; "+" decoy | [581095](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581095) |
| 9 | linrock | 0.71473 | 13 (+4) | SDXL LoRA on quantised, score-filtered images; frames found by differential evolution | [581061](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581061) |
| 10 | BERA | 0.71389 | 7 (-3) | Flux.1-dev + HyperSD 8-step + Nunchaku, 4-bit on one T4; SVGO | [581213](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581213) |
| 11 | 💸...🏅🙏 | 0.71309 | 10 (-1) | unknown | no writeup |
| 12 | Rock | 0.71292 | 12 (=) | SD3.5 Medium trained with Flow-GRPO on the competition metric | [581034](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581034) |
| 13 | Howard | 0.71250 | 8 (-5) | FLUX.1-schnell + Nunchaku; dummy questions calibrated to real VQA by regression | [581032](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581032) |
| 14 | phoenix^2 | 0.70560 | 9 (-5) | Lumina-Image-2.0 fine-tuned on 8.5k rasterised SVGs; LLaVE-2B selection | [581063](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581063) |
| 15 | Mikhail Golubchik | 0.70528 | 38 (+23) | SD3.5 Medium; description split into three questions for the proxy | [581051](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581051) |
| 16 | Coconut Tree on an Island | 0.70428 | 14 (-2) | SDXL-Flash x18, cuML GPU K-means vectoriser, aesthetic x SigLIP rerank | [581094](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581094) |
| 17 | Magi | 0.70143 | 34 (+17) | unknown | no writeup |
| 18 | chsh2 | 0.70013 | 45 (+27) | Quantised Flux.1-schnell, homemade vectoriser, ImageReward per noun | [582286](https://www.kaggle.com/competitions/drawing-with-llms/discussion/582286) |
| 19 | WXYZ123 | 0.69744 | 16 (-3) | Diffusion, compressed SVG, two VQA questions, early-stop time banking | [581012](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581012) |
| 20 | Learning not so deep | 0.69705 | 27 (+7) | unknown | no writeup |

Fifteen teams wrote a Kaggle writeup. The 6th-place team did not; its approach comes from a Japanese slide deck by Rist Inc., read through a summary I could not check line by line, so treat every 6th-place detail in this post as **unverified**, including the attribution of the deck to that team (it lists no members). Four teams (7th, 11th, 17th, 20th) left nothing, and I do not guess at their methods.

## 5. The top-20 landscape

![Top-20 technique matrix: rows are techniques, columns are private ranks 1 to 20; filled cells are confirmed, empty cells are reported not used, grey cells are unknown, with a count of 16 sources on the right](/imgs/blogs/kaggle-drawing-with-llms-solutions-6.webp)

Read the matrix by rows. I recounted each row from its cells over the 16 sources (15 writeups plus the unverified 6th-place deck):

- **15 of 16** shipped a diffusion image traced into vectors. The exception, 2nd place, shipped mostly text. **0 of 16** shipped an LLM writing the SVG.
- **13 of 16** put an OCR decoy in a corner: "ZOK", "Zoe", "Z", "M", "+", "a", a circle.
- **12 of 16** minified SVG paths (relative coordinates, colour groups, scour or SVGO) to spend more of the 10 KB on shapes.
- **11 of 16** chose among candidates with a PaliGemma yes/no question as a VQA proxy; 4 used an embedding or reward model instead or as well (SigLIP for 3rd and 16th, LLaVE for 14th, ImageReward for 18th).
- **9 of 16** generated 8 or more candidates per prompt, and 9 put the aesthetic score into the selection rule.
- **6 of 16** re-implemented the metric's preprocessing locally (everyone else is unknown, not no), and 6 added a searched aesthetic patch, frame or border.
- **5 of 16** fine-tuned the generator; 2 of those (4th DRaFT, 12th Flow-GRPO) trained it against judge-like rewards.
- **2 of 16** shipped diffvg refinement (3rd and 4th), and **2 of 16** hid text from OCR (1st and 2nd).

The generator families split FLUX 6, SDXL family 5, SD3.5 Medium 3, Lumina 1.

**Table stakes.** Diffusion image, vectorise (VTracer or a custom K-means tracer), compress hard, generate as many candidates as the minute allows, pick with a PaliGemma proxy, put a decoy in the corner. Everyone built on Rich Olson's public notebook, whose idea was to run the metric inside your own notebook and keep the best of several tries. The commonest engineering theme was fitting more candidates in: pipelined GPUs and CPU, few-step samplers, GPU K-means.

**What separated the top 5 from 6 to 20.** Each of the top five attacked the *judge* in one deliberate way beyond best-of-n. 1st and 2nd hid text in the gap between the two preprocessing lanes. 3rd optimised vector shapes through a differentiable copy of the preprocessing. 4th backpropagated judge-like rewards into the generator. 5th gradient-searched a universal aesthetic patch. Ranks 6 to 20 mostly stopped at best-of-n with cosmetic aesthetic tweaks and better vectorisers. The scores show it: 0.743 for 4th, 0.725 for 6th, 0.697 for 20th, and a 0.026 cliff between 3rd and 4th, larger than the 0.019 between 6th and 15th.

**Contradictory findings.** *Proxy questions:* one generic yes/no question beat generated question sets for 4th, 5th, 9th and 16th, while 15th called splitting the description into three questions his main boost. *Fine-tuning:* DRaFT gave 4th about +0.03 LB and Flow-GRPO took 12th from a base model at about 0.65 public to 0.73 private, while 5th, 13th and 14th found fine-tuning no better than prompt search or a stronger base model. *diffvg:* the core of 3rd and "worked amazingly well" for 4th, but failed for 2nd, 5th and 8th. *Frames and backgrounds:* helped 6th, 9th and 10th, failed for 16th.

## 6. The winning solution, explained

![Anatomy of the 1st-place SVG on a 384-pixel canvas: a 3-letter decoy top left, the diffusion drawing at 40% size top centre, the description as text under a faint pixel pattern, and searched aesthetic artifacts around it, with the five-step pipeline alongside](/imgs/blogs/kaggle-drawing-with-llms-solutions-7.webp)

Conor Mac Amhlaoibh, huanligong and kenyeungtech, team "new team name", led public and private (0.79058). Their writeup summarises the method in five steps.

**Intuition.** The judge wants answers to its questions and a pretty picture. Text supplies nearly all the answers, if OCR does not punish it. So give the judge a small drawing for looks, hidden text for answers, and spend the rest of the canvas on whatever the aesthetic model loves.

**Step 1: 64 cheap images.** Segmind's SSD-1B, a distilled SDXL, at 384 by 384 and guidance 8, with the Tiny AutoEncoder for SDXL, a 12-step DPM-Solver multistep scheduler, DeepCache (`cache_interval=2`, `cache_branch_id=0`), and 32 images per T4: about 30 seconds single-threaded, the mode they shipped because of the Packages bug. The prompt begs for clip art ("flat vector illustration, flat color blocks, hard color boundaries, solid fills, clipart style ... vector art of {description}"), with a negative prompt against photorealism, gradients, text and frames. The [distillation post](/blog/machine-learning/image-generation/distribution-matching-and-adversarial-distillation) explains why few-step models exist.

**Step 2: trace to polygons.** Downsample, quantise to six colours, extract OpenCV contours per colour, simplify to polygons, rank paths by size, centrality and simplicity, and add them until 6,000 bytes. About 2.5 seconds for all 64.

**Step 3: compose the canvas.** Shrink the drawing to 40%, top centre. Put three visible characters top left; a search over three-letter strings found "ZOK" gave the best aesthetic with no VQA loss. Below the drawing, write the description without stopwords as compressed paths, under a 10 by 10 repeating pattern of single-pixel coloured squares at opacity .1. The team reports the text is hidden in the raw render and in OCR's lightly processed copy, and appears in the fully processed copy that VQA and the aesthetic model read. Then add artifacts found by genetic algorithm, hill climbing or trial: a 5 by 4 patch of coloured squares, green "KUZ", green "3Q3", grey "SSQ", and green "FRANCE XK ARTS" when space allows.

**Step 4: re-render through the judge's eyes** with their own faster re-implementation of the preprocessing, which "produces almost identical results".

**Step 5: keep the prettiest.** Return the SVG with the highest aesthetic score, aesthetic alone. The hidden text already pins VQA high for every candidate (~0.88 on validation), so looks are the only variable left; adding CLIP or VQA to the rule did not help.

The selection loop as an illustrative toy, not the team's code:

```python
# Illustrative toy of a best-of-n loop in the 1st-place style. Not the team's code.
def pick_svg(description, generate, to_svg, compose, render, preprocess, aesthetic, n=64):
    best_svg, best_aes = None, -1.0
    for image in generate(description, n=n):             # n diffusion images
        svg = compose(to_svg(image, byte_budget=6000),    # drawing at 40%
                      description)                        # + decoy, hidden text, artifacts
        aes = aesthetic(preprocess(render(svg)))          # judge-side view of the image
        if aes > best_aes:
            best_svg, best_aes = svg, aes
    return best_svg
```

A text-only fallback SVG covered timeouts. On an earlier metric version, the same team had found 237 optimised squares that scored 1.014, with an aesthetic above 1.2.

## 7. 2nd and 3rd in depth, then ranks 4 to 20

### 2nd: move on, 0.77776

takuji, arsenypoyda, samson8, brianwilliamzhao and guoooooooss started with Stable Diffusion to SVG like everyone, and stalled for a week. Then they reasoned from the metric: VQA weighs more than aesthetic, the only way to very high VQA is text, and "the penalty for OCR word count is applied before enhancement, while the VQA score is calculated after enhancement". So make text invisible before and visible after. A carefully designed **dashed-line mask** over the text did it: only letters left uncovered were recognised by OCR. takuji hand-designed compact letters and masks that packed a lot of content into about 4,000 letters. That reached 0.652, and every later submission went into aesthetic: an A100 brute-forced the letters placed outside the mask over 15 prompts and found "Zoe", then hearts, position tweaks and random inserted words. DiffVG failed on the non-differentiable augmentations. It is the purest metric attack in the top 20: almost no drawing at all.

### 3rd: CP GMs, 0.76947

c-number and daiwakun did the opposite, and their title says it: "VQA/AES = 0.81/0.64", no text hack. FLUX.1-schnell drew 8 candidates from four prompt templates; vtracer traced them, with a binary search on input resolution to fit the cap; SigLIP so400m (the image tower inside PaliGemma 2) kept the best two, and a short trial optimisation picked one. Then the core: **pydiffvg** optimised path control points, fill and stroke colours and widths for 200 Adam steps with a cosine schedule, against a loss of negative aesthetic, negative SigLIP similarity to the target bitmap, and MSE to it. Crucially, the losses were computed after an `ImageProcessorTorch`, a differentiable (or straight-through) copy of every preprocessing step, including a differentiable JPEG, with synced crop seeds for the MSE target. A compression pass then fit 10 KB. On the iso-score chart, 3rd has the highest aesthetic of the marked points; what it lacked was the VQA that only text could buy.

### Ranks 4 to 20, by approach family

**Train the generator on the judge.** *4th, Kevin Clark (solo)* fine-tuned SD3.5 Medium without its T5 encoder using **DRaFT-LV**, which backpropagates a reward through the sampler. The rewards were HPSv2, PickScore, the competition aesthetic, a "Does the image show {prompt}?" VQA proxy, and minus LPIPS between the image and its own SVG conversion, so outputs vectorise cleanly; prompts came from GPT-4o. He reports about +10% aesthetic, +5% VQA and +0.03 LB, then diffvg refinement and best-of-n within about 60 seconds. *12th, Rock (solo)* trained SD3.5 Medium with **Flow-GRPO** using the competition metric itself as the reward, on 4,000 Gemini-generated samples, 4x H800, KL weight 4e-4: local score from about 0.72 to about 0.77 after 500 steps (private 0.73; he missed submitting his best checkpoint), and GRPO lifted VQA (0.82 to 0.91 locally) more than aesthetic. Background: the [GRPO vs DPO vs PPO guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide).

**Distil your own best-of-many.** *8th, Psiilu (philippsinger, ilu000)* fine-tuned SDXL on the best images from 16 prior runs of prompted FLUX-schnell and SDXL, then picked among 14 to 18 candidates with a yes/no proxy. *9th, linrock (solo)* trained an SDXL LoRA on 1.5k LLM-generated descriptions whose FLUX and Hyper-SDXL images were quantised to 24 colours and filtered by PaliGemma VQA and aesthetic, and used differential evolution to find aesthetic-lifting frames. *14th, phoenix^2 (solo)* fine-tuned Lumina-Image-2.0 on 8.5k rasterised SVGs and selected with LLaVE-2B. The [diffusion LoRA post](/blog/machine-learning/deep-learning/fine-tuning-diffusion-models-with-lora) covers the mechanics.

**More candidates, better proxy.** *5th, tomirol (solo)* beam-searched 20+ prompts out of about 200 and pipelined SDXL-turbo, VTracer and a PaliGemma 10B-448 proxy to make 28 SVGs per prompt in about 55 seconds. His extra: gradient ascent through a differentiable copy of the judge found a ~3.5 KB path patch that lifts average aesthetic to about 0.7 by itself. *6th, Grand Painting Artists (unverified)*: FLUX.1-schnell, 9 candidates, EasyOCR to paint over any text the generator drew, a "Z" decoy and a searched rounded frame. *10th, BERA (benbla, raki21)* ran Flux.1-dev with HyperSD at 8 steps and Nunchaku, all 4-bit on one T4 and took the best-ranked candidate that passed OCR. *13th, Howard (solo)* used FLUX.1-schnell with Nunchaku and a linear regression mapping dummy-question scores to real VQA. *15th, Mikhail Golubchik (solo)* got his main boost from splitting the description into three proxy questions. *16th, Coconut Tree on an Island* ran SDXL-Flash for up to 18 images in 30 s, made the vectoriser 15x faster with cuML K-means, and reranked by aesthetic times SigLIP before the expensive proxy. *18th, chsh2 (solo, first Kaggle competition)* built his own vectoriser and ranked by ImageReward over one sub-prompt per noun. *19th, WXYZ123* found the bitmap-to-SVG converter the most impactful factor and banked unused seconds from easy prompts for hard ones.

**No writeup:** 7th JJ, 11th 💸...🏅🙏, 17th Magi, 20th Learning not so deep.

### The best contrasting entry

Kevin Clark's 4th place is the one to study if you want something that survives a fixed metric: no hidden text, no artifact search, just a generator taught by gradient to make images a VQA model can read and a vectoriser can trace. It finished 0.047 below 1st.

## 8. What the forum taught everyone

![Public to private rank for the top 20: the public top six became the private top six, 18th rose from public 45, 15th from 38 and 17th from 34, while 13th and 14th fell five places](/imgs/blogs/kaggle-drawing-with-llms-solutions-8.webp)

**The exploit arms race was public.** Bilzard's "Text Rendering: OCR-Exploit [LB=0.305]" started it on the launch metric: draw the description as glyphs, and a judge pretrained on OCR pairs matches text to text. His inversion-attack, "Text not present" and v15 edge-pixel reports landed within days of each metric change, and David Austin showed border pixels alone could drive the aesthetic model to 0.7214. The host, Ryan Holbrook, answered with a two-question metric and then the final version. Even that one fell to the lane gap.

**The baseline everyone forked.** Rich Olson's diffusion-to-SVG notebook scored .507 (on the pre-final metric), and his OCR decoy notebook was reused by 4th. Zhuang Jia's table told the field early that direct SVG writing was not competitive; taku_sid's GRPO-for-SVG notebook was the honest LLM path, and nobody in the top 20 rode it to the end.

**Shake-up was small.** Ravi Ramakrishnan called the top 10% "nearly fully stable" (biggest riser 120 to 45, biggest faller 26 to 111; gold-zone teams averaged about 159 submissions). The public top six became the private top six, reshuffled. Lower down, 18th climbed 27 places from public 45 and 15th climbed 23 from 38 (the top-20 synthesis names 15th the biggest riser; recounting the moves, it is 18th). Writeups blamed seed and OCR noise: 8th selected two submissions differing only by seed, and 16th's best private submission (0.717) rescored at 0.709 when resubmitted.

## 9. What did not work

Aggregated across the top-20 sources, with who reported it:

- **An LLM writing SVG.** 14th fine-tuned Qwen2.5-VL-3B on text2svg-stack, then a curated 8.5k set, for two weeks without good results. The 6th-place deck (unverified) puts zero-shot Qwen SVG at roughly 0.5 to 0.6, improved by RL but beaten by diffusion.
- **diffvg in the final pipeline:** 2nd (non-differentiable augmentations), 5th (too unstable and slow), 8th (killed by a "Package Prediction Error").
- **ReNO-style latent optimisation:** nicer images, too slow, 5th.
- **Preference fine-tuning** (REFL, DPO), 5th; SSD-1B and Flux-dev fine-tunes, 14th.
- **Generated questions at inference** (Qwen, a trained LLM): 5th, 9th, 16th.
- **Smaller proxies:** PaliGemma 3B, 16th; every variant smaller than 10B-448, 5th.
- **Similarity dedup, added backgrounds, reranking without a VQA estimate:** 16th.
- **Aesthetic hacking under the final defences:** limited success while keeping VQA, 8th.
- **Selection by more than aesthetic, more than 64 images, other patch placements:** 1st.
- **Moving the OCR decoy per image:** 14th.

## 10. Lessons that transfer

1. **A judge that sees two views of your output can be split.** OCR read a lightly processed copy, VQA a heavily processed one, and 1st and 2nd lived in the difference (581027, 581023). Any evaluation that preprocesses differently per check has this gap.
2. **"Use the LLM" is not a strategy.** The best measured direct-SVG score was 0.670 against 0.791 for diffusion plus tracing (569847). Use LLMs where they are strong: here, writing validation data.
3. **Re-implement the judge, then optimise through it.** 3rd's differentiable preprocessing and 5th's differentiable aesthetic patch both came from owning a local copy of the metric (581024, 581128).
4. **Train for convertibility, not just quality.** 4th rewarded images that survive vectorisation (581108). Any pipeline with a lossy last step benefits.
5. **Best-of-n with a cheap proxy is the floor, and the budget is the ceiling.** 11 of 16 sources picked with a one-question PaliGemma proxy; most engineering effort went into fitting more candidates in 60 seconds (581128, 581094).
6. **Hosts: red-team a model-based metric before launch.** Three metric versions, a full rescore, and an exploit still decided the prizes (566429, 567872).

### If you enter a competition like this

- Read the metric source on day one, and list every place two scorers see different inputs.
- Run the real metric locally, then write a fast re-implementation and check it agrees.
- Generate a validation set with an LLM that matches the hidden categories, and check it correlates with the public board before trusting it.
- Spend the time budget on candidates: pipeline GPU, CPU and scorer; use few-step samplers.
- Put a short decoy where OCR will read it, and verify its effect on aesthetic.
- Decide early whether you will exploit the metric or optimise honestly, and know that a rescore can erase the first.

## 11. Sources

- Competition and metric: [Drawing with LLMs](https://www.kaggle.com/competitions/drawing-with-llms), [SVG Image Fidelity metric notebook](https://www.kaggle.com/code/metric/svg-image-fidelity), metric updates [566429](https://www.kaggle.com/competitions/drawing-with-llms/discussion/566429), [566981](https://www.kaggle.com/competitions/drawing-with-llms/discussion/566981), [567872](https://www.kaggle.com/competitions/drawing-with-llms/discussion/567872)
- 1st: [writeup 581027](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581027), [notebook](https://www.kaggle.com/code/conormacamhlaoibh/new-team-name-1st-place-solution)
- 2nd: [writeup 581023](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581023), [notebook](https://www.kaggle.com/code/takuji/2nd-place-solution-ocr-attack)
- 3rd: [writeup 581024](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581024)
- 4th: [writeup 581108](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581108), [repo](https://github.com/clarkkev/svg-diffusion)
- 5th: [writeup 581128](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581128), [repo](https://github.com/matheuspf/draw)
- 6th (unverified): [Rist Inc. slide deck](https://speakerdeck.com/rist/drawing-with-llms)
- 8th: [581095](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581095)
- 9th: [581061](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581061), [notebook](https://www.kaggle.com/code/linrock/9th-place-sdxl-lora-vqa-boost-aesthetics)
- 10th: [581213](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581213)
- 12th: [581034](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581034), [repo](https://github.com/yanyanhuang/12th-Place-Solution-for-Kaggle-Drawing-with-LLMs)
- 13th: [581032](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581032), [notebook](https://www.kaggle.com/code/howarudo/13th-place-solution-flux-svg-scoring)
- 14th: [581063](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581063)
- 15th: [581051](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581051)
- 16th: [581094](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581094), [repo](https://github.com/bogoconic1/16th-place-kaggle-drawing-with-llms)
- 18th: [582286](https://www.kaggle.com/competitions/drawing-with-llms/discussion/582286)
- 19th: [581012](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581012)
- Forum: [OCR exploit 565396](https://www.kaggle.com/competitions/drawing-with-llms/discussion/565396), [inversion attack 566160](https://www.kaggle.com/competitions/drawing-with-llms/discussion/566160), [text not present 567449](https://www.kaggle.com/competitions/drawing-with-llms/discussion/567449), [edge pixel v15 569825](https://www.kaggle.com/competitions/drawing-with-llms/discussion/569825), [edge pixel aesthetic 567487](https://www.kaggle.com/competitions/drawing-with-llms/discussion/567487), [Rich Olson baseline 567633](https://www.kaggle.com/competitions/drawing-with-llms/discussion/567633), [OCR decoy notebook](https://www.kaggle.com/code/richolson/let-s-defeat-ocr-easy-lb-boost), [LLM SVG comparison 569847](https://www.kaggle.com/competitions/drawing-with-llms/discussion/569847), [GRPO notebook thread 565980](https://www.kaggle.com/competitions/drawing-with-llms/discussion/565980), [leaderboard dynamics 581017](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581017)

Same-wave posts: [Santa 2024](/blog/machine-learning/learn-from-kaggle/kaggle-santa-2024-perplexity-puzzle-solutions); [Jigsaw Agile Community Rules](/blog/machine-learning/learn-from-kaggle/kaggle-jigsaw-agile-community-rules-solutions); [Make Data Count](/blog/machine-learning/learn-from-kaggle/kaggle-make-data-count-solutions); [Deep Past](/blog/machine-learning/learn-from-kaggle/kaggle-deep-past-akkadian-translation-solutions); and the [gpt-oss red-teaming findings](/blog/machine-learning/learn-from-kaggle/kaggle-gpt-oss-red-teaming-challenge-findings). The [FLUX deep-dive](/blog/machine-learning/image-generation/flux-architecture-deep-dive) explains the generator six of these teams chose.
