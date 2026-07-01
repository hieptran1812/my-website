---
title: "Recaptioning and VLM Instruction Data: When Model-Written Captions Beat the Web"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Alt-text was written to help browsers, not to teach models. This is a practitioner's guide to fixing it: recaptioning web pairs with a VLM, fusing synthetic captions with raw alt-text CapsFusion-style, synthesizing LLaVA-style visual instruction data from a text-only LLM, and building interleaved corpora like MMC4 and OBELICS — with runnable code, a three-way before/after, and a symptom-to-fix troubleshooting table for hallucination and homogeneity."
tags:
  - training-data
  - vision-language-models
  - recaptioning
  - visual-instruction-tuning
  - llava
  - capsfusion
  - interleaved-data
  - caption-hallucination
  - multimodal
readTime: 30
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
---

The alt-text on the web was never written to train a model. It was written so a screen reader could say *something* when the image failed to load, or so an SEO tool could stuff a keyword. When you scrape a billion image-text pairs off the open web, what you actually get is a billion strings like `IMG_2043.jpg`, `product hero shot`, `click here`, and `Nike Air Max 90 (buy now, free shipping)`. A handful are gold. Most are noise. Almost none of them *describe the picture* the way a model needs to learn to see.

For a decade we trained vision-language models on exactly this signal anyway, because it was free and it was large. CLIP, ALIGN, and the first wave of captioners all drank from the alt-text firehose. And it worked well enough to be miraculous — until people started measuring what the captions actually contained and discovered that the single biggest lever on a VLM's quality was not the architecture, not the vision encoder, not the LLM backbone, but *the words on the right side of the pair*. Swap noisy alt-text for dense, faithful, model-written descriptions and downstream accuracy jumps by double digits with the same compute. That is the entire thesis of this post: for training, model-written captions usually beat the web, and the best captions of all are the ones that *fuse* the model's density with the web's world knowledge.

![The four families of VLM training data](/imgs/blogs/recaptioning-and-vlm-instruction-data-1.webp)

The diagram above is the mental model for everything that follows. Vision-language data splits into four families, and they teach four different things. **Captions** (image to text) teach perception — what is in the picture. **Instruction and VQA** data (image plus a question to an answer) teach the model to hold a conversation grounded in pixels. **Grounding** data ties specific words to specific regions, so the model can point. **Interleaved** documents — whole web pages with images inline — teach in-context multimodal reasoning, the skill that lets a model do few-shot learning from a mixed prompt. This post is a tour of that diagram, with the heaviest weight on the first two families, because that is where recaptioning and instruction synthesis live and where the biggest quality wins are hiding.

If you have read the sibling posts in this series, this is where several threads converge. [Image-text pairs at scale](/blog/machine-learning/training-data/image-text-pairs-at-scale) covers how you *acquire* and clean the raw pairs; this post covers what to do once you have them and have realized the captions are terrible. [Synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation) covers the text-only version of the same trick — manufacturing training data with models — and the model-collapse trap that haunts both.

## Why alt-text is the wrong training signal

Here is the senior rule of thumb: **alt-text optimizes for a completely different objective than the one you are training on, so treating it as ground truth is a category error.** The person who wrote `dog.jpg` was not trying to describe the dog. They were naming a file. The person who wrote `Golden retriever puppy — Petco.com` was trying to sell you a puppy. Neither string tells your model that the dog is mid-stride on grass, wearing a red collar, in warm afternoon light. The description your model needs to learn to see is precisely the description no human bothered to write, because to a human looking at the image it is *obvious*.

![What each caption source captures](/imgs/blogs/recaptioning-and-vlm-instruction-data-2.webp)

The matrix above is the whole argument in one picture, and it is worth internalizing because it dictates the entire recipe. Read it column by column. **Raw alt-text** is strong on exactly one axis that matters enormously: named entities and world knowledge. It knows the dog is a *golden retriever* and the shoe is a *Nike Air Max 90* and the building is *Notre-Dame*. It also has natural, varied human phrasing. But it is sparse (five words is typical), structurally useless (often literal SEO junk or navigation text), and frequently *off-image* entirely — describing the page, the product line, or the photographer rather than the pixels. **VLM captions** are the mirror image: dense, structured, describing spatial relations and attributes and counts — but generic ("a dog" not "a golden retriever"), prone to inventing objects, and templated to the point that ten thousand of them all open with "This image shows." The **fused** column is the punchline. If you can keep the entities and phrasing from alt-text and the density and structure from the VLM, and cross-check the two against each other for faithfulness, you get a caption that is better than either source on every axis at once.

That is not a hypothetical. It is the measured result behind CapsFusion, which we will get to, and it is why every serious VLM data pipeline built after 2023 recaptions.

### The failure modes alt-text bakes in

It helps to name the specific pathologies, because each one shows up as a distinct symptom downstream.

- **Sparsity.** A model trained on five-word captions learns to produce five-word captions and to attend to only the single most salient object. Ask it to describe a complex scene and it names the dog and stops.
- **World-knowledge leakage without grounding.** Alt-text says "Eiffel Tower" for a photo of a keychain shaped like the Eiffel Tower. The model learns a spurious association between the token and a shape, and later confidently mislabels souvenirs as landmarks.
- **Boilerplate contamination.** "Click here", "Advertisement", "Photo by John Smith / Getty Images" — these appear in millions of pairs and the model learns to emit them as if they were descriptions.
- **Language and register drift.** Alt-text is multilingual, abbreviated, and full of hashtags and camelCase. A model trained on it produces captions that read like Instagram tags, not sentences.

The fix for all four is the same move: generate a fresh description from the pixels with a model whose only job is description, then reconcile it with whatever real information the alt-text carried.

## The recaptioning recipe

**The core recipe is three moves: caption with a VLM, fuse with the alt-text, filter what does not survive alignment.** None of the three is optional, and the order matters.

![The recaptioning pipeline](/imgs/blogs/recaptioning-and-vlm-instruction-data-3.webp)

The pipeline above shows the data flow. One web pair enters and immediately splits: the raw alt-text goes down one path (it carries the entities), and the image goes to a captioning VLM down the other (it produces density). Both streams meet at a fusion step — usually a language model prompted to merge them — which produces a single dense caption that preserves the alt-text's real information. Then a filter, typically a CLIP image-text similarity score plus near-duplicate removal, drops the pairs where the caption and image do not actually agree. What ships as a training pair is the fused, filtered caption, not the original alt-text and not the raw VLM output.

<figure class="blog-anim">
<svg viewBox="0 0 760 260" role="img" aria-label="A five-word alt-text is enriched into a dense caption as detail chips fade in one by one" style="width:100%;height:auto;max-width:820px">
<style>
.e1-card{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5;rx:10}
.e1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.e1-alt{font:700 20px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--text-primary,#1f2937);text-anchor:middle}
.e1-chip{fill:var(--accent,#6366f1);opacity:.14;stroke:var(--accent,#6366f1);stroke-width:1.5;rx:9}
.e1-chiptxt{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.e1-arrow{stroke:var(--text-secondary,#6b7280);stroke-width:2;fill:none;marker-end:url(#e1-ah)}
@keyframes e1-in{0%,8%{opacity:0;transform:translateY(8px)}22%,92%{opacity:1;transform:translateY(0)}100%{opacity:0;transform:translateY(8px)}}
.e1-g{animation:e1-in 9s ease-in-out infinite;transform-box:fill-box;transform-origin:center}
.e1-c2{animation-delay:.8s}.e1-c3{animation-delay:1.6s}.e1-c4{animation-delay:2.4s}.e1-c5{animation-delay:3.2s}
@media (prefers-reduced-motion:reduce){.e1-g{animation:none;opacity:1;transform:none}}
</style>
<defs><marker id="e1-ah" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="var(--text-secondary,#6b7280)"/></marker></defs>
<text class="e1-lbl" x="120" y="40">raw alt-text</text>
<rect class="e1-card" x="20" y="55" width="200" height="70" rx="10"/>
<text class="e1-alt" x="120" y="97">"dog.jpg"</text>
<line class="e1-arrow" x1="230" y1="90" x2="300" y2="90"/>
<text class="e1-lbl" x="530" y="40">VLM-enriched caption</text>
<rect class="e1-card" x="310" y="55" width="430" height="180" rx="10"/>
<rect class="e1-chip e1-g" x="330" y="75" width="150" height="34"/>
<text class="e1-chiptxt e1-g" x="405" y="97">a brown dog</text>
<rect class="e1-chip e1-g e1-c2" x="330" y="120" width="120" height="34"/>
<text class="e1-chiptxt e1-g e1-c2" x="390" y="142">running</text>
<rect class="e1-chip e1-g e1-c3" x="465" y="120" width="150" height="34"/>
<text class="e1-chiptxt e1-g e1-c3" x="540" y="142">on green grass</text>
<rect class="e1-chip e1-g e1-c4" x="330" y="165" width="180" height="34"/>
<text class="e1-chiptxt e1-g e1-c4" x="420" y="187">wearing a red collar</text>
<rect class="e1-chip e1-g e1-c5" x="525" y="165" width="140" height="34"/>
<text class="e1-chiptxt e1-g e1-c5" x="595" y="187">mid-stride, sunny</text>
</svg>
<figcaption>Recaptioning turns a five-character alt-text into a dense, grounded description: each attribute chip fades in as the VLM adds detail the web pair never had.</figcaption>
</figure>

The animation shows the enrichment concretely: the model starts from what the web gave it (`dog.jpg`) and adds the color, the action, the setting, the accessory, the lighting — the attributes that were always in the pixels but never in the text. Every chip that fades in is a training signal the original pair did not carry.

### Which captioner to use

The captioner is the workhorse, and the choice is a genuine trade-off between throughput and quality. Here is the landscape as of the models most teams reach for.

| Captioner | Params | Throughput | Caption quality | When to use |
| --- | --- | --- | --- | --- |
| BLIP | 0.4B | Very high | Short, generic | Bulk first-pass on billions of pairs |
| BLIP-2 (OPT-2.7B) | 3.4B | High | Good, fluent | Default web-scale recaptioning |
| InstructBLIP | 4B+ | Medium | Instruction-followable | When you need controllable caption style |
| LLaVA-1.5 / 1.6 | 7B-13B | Medium-low | Dense, detailed | High-value subset, dense descriptions |
| GPT-4V / strong API VLM | - | Low, costly | Best available | Seed sets, distillation targets only |

The economics force a two-tier strategy on anyone working at scale. You cannot run a 13B VLM over ten billion images; the GPU-hours are absurd. So the standard pattern is: caption the *entire* corpus with a cheap model (BLIP or BLIP-2), then caption a high-value *subset* — or a seed set you will distill from — with an expensive one (LLaVA-1.6, or GPT-4V for the seed). ShareGPT4V, which we cover as a case study, is literally this pattern: 100K captions from GPT-4V to seed, then a trained captioner to scale to 1.2M.

### A worked recaptioning: one image, three ways

Abstractions are cheap; let us make this concrete. Take a single stock photo: a golden retriever mid-run across a park lawn, red collar, blue frisbee lying on the grass nearby (not in the dog's mouth), warm late-afternoon light. Here is what each source gives you.

**Raw alt-text** (what the web pair actually contained):

```
Golden Retriever - Dog Breeds | PetFinder.com
```

Notice what is there and what is not. The breed is correct and specific — that is real world knowledge you want to keep. Everything else is a site name and a category label. There is no scene, no action, no setting, no color of the collar, nothing about light. A model trained on this learns "golden retriever" points at this pixel pattern and nothing more.

**BLIP-2 caption** (`Salesforce/blip2-opt-2.7b`, five-beam search):

```
a dog running through the grass in a park
```

Now we have density and structure — an action (running), a setting (park), a surface (grass). But the breed is gone ("a dog"), the collar is gone, the frisbee is gone, and the light is gone. And crucially, if the beam had gone slightly differently, BLIP-2 might have written "a dog catching a frisbee" — because frisbees-near-dogs is a strong prior in its training data — and that would be a hallucination, since the dog is not touching the frisbee.

**Fused caption** (alt-text plus BLIP-2 plus a fusion LLM, cross-checked):

```
A golden retriever running across a grassy park lawn, wearing a red
collar, with a blue frisbee resting on the grass nearby in warm
afternoon light.
```

This is the training signal you actually want. The breed came from the alt-text. The action and setting came from BLIP-2. The collar, the frisbee's *position* (resting nearby, not caught), and the light came from a stronger captioner pass that the fusion step reconciled with the others. The fusion step is also where the hallucination gets caught: because BLIP-2 said "running" and the stronger pass said "frisbee resting nearby," the fuser does not assert "catching a frisbee." The disagreement between sources is a *feature* — it flags the uncertain claim.

The lesson of the three-way comparison is that no single source is sufficient and the errors are not correlated. Alt-text errors are omissions; VLM errors are inventions. Fusing them is not just concatenation — it is using each source to check the other.

### Code: caption a batch and fuse

Here is the captioning half of the pipeline with real libraries. This runs on a single GPU with `transformers` and processes images in batches.

```python
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=dtype
).to(device)
model.eval()

@torch.inference_mode()
def caption_batch(images, max_new_tokens=40):
    inputs = processor(images=images, return_tensors="pt").to(device, dtype)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=5,
        repetition_penalty=1.5,   # BLIP-2 loves to loop; this suppresses it
        length_penalty=1.0,
    )
    return [c.strip() for c in processor.batch_decode(out, skip_special_tokens=True)]

# images: list[PIL.Image] loaded from your webdataset shard
paths = ["dog.jpg", "shoe.jpg", "tower.jpg"]
images = [Image.open(p).convert("RGB") for p in paths]
synthetic = caption_batch(images)
```

The fusion half is a language model prompted to merge the raw alt-text with the synthetic caption. CapsFusion distilled this behavior into a fine-tuned LLaMA (they call it CapsFus-LLaMA) so it could run cheaply at billion-pair scale, but you can prototype with any instruct model. The prompt is the important part.

```python
FUSE_SYSTEM = (
    "You merge a noisy web alt-text with a model-generated image caption "
    "into ONE faithful, dense description. Rules: (1) Keep specific named "
    "entities, brands, and proper nouns ONLY from the alt-text. (2) Keep "
    "spatial layout, attributes, counts, and actions from the caption. "
    "(3) If the two sources disagree about whether an object is present, "
    "DROP that object. (4) Never add anything not present in either source. "
    "(5) One paragraph, no marketing language, no 'this image shows'."
)

FUSE_USER = "ALT-TEXT: {alt}\nCAPTION: {cap}\nFUSED DESCRIPTION:"

def build_fuse_prompt(alt, cap):
    return [
        {"role": "system", "content": FUSE_SYSTEM},
        {"role": "user", "content": FUSE_USER.format(alt=alt, cap=cap)},
    ]

# Any instruct LLM works here; swap in CapsFus-LLaMA, a local vLLM server,
# or an API client. The contract is: messages in, one paragraph out.
alt = "Golden Retriever - Dog Breeds | PetFinder.com"
cap = synthetic[0]
messages = build_fuse_prompt(alt, cap)
# fused = your_llm.chat(messages).content
```

Rule (3) is the load-bearing rule. Instructing the fuser to *drop* objects the two sources disagree about is a cheap, effective hallucination guard — it turns the natural disagreement between an omission-prone source and an invention-prone source into a conservative decision. It is not perfect (both sources can be wrong the same way), but it removes the easy hallucinations for free.

### Second-order: the CLIP filter is not optional

After fusion you must filter, and the reason is subtle. Fusion can produce a beautiful caption that is *still wrong* — if BLIP-2 misidentified the scene and the alt-text was uninformative, the fuser has nothing to correct against and will happily write a fluent, dense, incorrect description. The last line of defense is an image-text alignment score. Compute CLIP similarity between the fused caption and the actual image; drop pairs below a threshold (0.26-0.28 cosine on CLIP ViT-L/14 is a common cut). This catches the pairs where the whole chain went wrong at the source. It also catches the pairs where fusion drifted into generic filler that no longer matches the specific image. Budget for throwing away 10-30% of your recaptioned pairs at this stage; the ones you drop are the ones that would have taught the model to hallucinate.

## CapsFusion: why fusing beats either alone

This is the first case study, and it is the one that established fusion as the default. CapsFusion (Yu et al., CVPR 2024) started from an uncomfortable observation: models trained purely on synthetic captions looked great on small benchmarks and then *got worse* as you scaled the data. Something about pure synthetic data hit a ceiling and then declined — the opposite of what more data is supposed to do.

![CapsFusion: fusing beats either source](/imgs/blogs/recaptioning-and-vlm-instruction-data-4.webp)

The before/after above names the two failure modes CapsFusion diagnosed. On the left, "either source alone": raw alt-text is noisy and sparse, and synthetic-only captions suffer what the authors called **scalability deficiency** and **world-knowledge loss**. The scalability deficiency is that synthetic captions are so templated and low-entropy that adding more of them stops helping and eventually hurts — the model overfits to the caption *style* rather than learning from image *content*. The world-knowledge loss is that a captioner trained to describe pixels literally strips out exactly the named entities (brands, landmarks, species, people) that only appear in human-written alt-text. On the right, CapsFusion: use ChatGPT (distilled into CapsFus-LLaMA for scale) to fuse the raw and synthetic captions, and you retain entities *and* density. The measured result was a large gain — on the order of 18-19 CIDEr points on COCO and NoCaps captioning over the synthetic-only baseline — with roughly 11-16x better sample efficiency and about 15x better compute efficiency to reach the same quality.

The intuition worth carrying away: **synthetic captions have low entropy and world knowledge has high entropy, and you need both.** Density without world knowledge plateaus because there is nothing new to learn once the model has memorized the caption template. World knowledge without density is too sparse to teach perception. Fusion is the only source that scales, because it keeps injecting real-world specificity (from alt-text) as the corpus grows, so the model never runs out of genuinely new signal.

There is a defensive lesson here too, and it connects directly to [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation): pure synthetic data collapses, and the tell is that it looks *better* at small scale before it gets worse. If your recaptioned corpus improves and then plateaus or regresses as you add data, suspect homogeneity, and reach for fusion with a real-world source before you reach for more synthetic data.

## LLaVA and visual instruction data

Captions teach perception, but they do not teach a model to answer questions, follow instructions, or reason about an image in conversation. For that you need instruction data: (image, instruction, response) triples. The problem is that this data barely exists in the wild — nobody posts "here is an image and a three-turn conversation reasoning about it" at web scale. So you synthesize it. LLaVA (Liu et al., NeurIPS 2023) introduced the trick that everyone now uses, and it is clever enough to be worth understanding in detail.

![LLaVA visual-instruction synthesis](/imgs/blogs/recaptioning-and-vlm-instruction-data-5.webp)

The graph above shows the key move, and the surprising part is the node labeled "text-only GPT-4 (no pixels)." **LLaVA generates visual instruction data from a language model that never sees the image.** Here is how: take a COCO image that already has human annotations — five captions and a set of object bounding boxes with class labels. Serialize those annotations into text: the five captions become five sentences, and each bounding box becomes a line like `person: [0.41, 0.22, 0.55, 0.78]`. Feed *that text* to GPT-4 (text-only, in 2023) and ask it to generate three kinds of instruction data. The image is turned into a symbolic representation, and a pure language model does the reasoning. It works because the captions and boxes carry enough of the image's content that a strong LLM can generate plausible, grounded conversations without pixels.

The three output types LLaVA generates, with the original counts, are:

| Type | Count | What it teaches |
| --- | --- | --- |
| Conversation | 58K | Multi-turn Q&A about object presence, count, location, action |
| Detailed description | 23K | Long, comprehensive scene descriptions |
| Complex reasoning | 77K | Step-by-step reasoning that requires inference beyond what is visible |

Total: 158K samples that cost a few thousand dollars of API calls to generate and produced the model that started the open VLM era. That is the leverage of instruction synthesis — a tiny amount of generated data, seeded from existing human annotations, unlocks a capability that no amount of caption data provides.

### The VQA-synthesis prompt template

The prompt is where the quality lives. Here is a template in the LLaVA style, adapted so you can run it against any strong instruct LLM. The critical elements are: give the model the symbolic image (captions plus boxes), give it a few in-context examples of the target format (omitted here for length), and constrain it to only assert what the annotations support.

```python
VQA_SYNTH_SYSTEM = """You are generating a multi-turn conversation between a
person asking about a photo and an assistant that can see it. You are given
the photo's captions and object bounding boxes as text. Generate a natural
conversation as if the assistant is looking at the photo.

HARD RULES:
- Only reference objects, attributes, counts, and spatial relations that are
  supported by the captions or boxes. Never invent objects.
- Ask questions a curious person would ask: what, how many, where, what is
  happening, what might happen next.
- Vary the phrasing across turns. Do not start every answer the same way.
- If a question cannot be answered from the given information, have the
  assistant say so rather than guessing.
- Boxes are [x1, y1, x2, y2] normalized to [0, 1]; use them for 'left/right/
  above/below/foreground' reasoning."""

VQA_SYNTH_USER = """CAPTIONS:
{captions}

OBJECTS (class: box):
{boxes}

Generate a 3-turn conversation."""

def build_vqa_prompt(captions, boxes):
    cap_block = "\n".join(f"- {c}" for c in captions)
    box_block = "\n".join(f"{cls}: {b}" for cls, b in boxes)
    return [
        {"role": "system", "content": VQA_SYNTH_SYSTEM},
        {"role": "user", "content": VQA_SYNTH_USER.format(
            captions=cap_block, boxes=box_block)},
    ]

captions = [
    "A golden retriever running across a grassy park.",
    "A dog on a lawn with a frisbee lying nearby.",
]
boxes = [("dog", [0.30, 0.35, 0.62, 0.88]),
         ("frisbee", [0.70, 0.74, 0.82, 0.83]),
         ("tree", [0.02, 0.05, 0.20, 0.70])]
messages = build_vqa_prompt(captions, boxes)
# conversation = your_llm.chat(messages).content
```

Two design choices are doing the heavy lifting. First, passing boxes *with* captions lets the LLM reason about spatial relations ("the frisbee is to the right of the dog") that captions alone do not encode — this is why LLaVA's conversations can answer "where" questions. Second, the rule "have the assistant say so rather than guessing" is what teaches the eventual VLM to refuse ungrounded questions instead of hallucinating an answer. If you skip that rule, you train a model that always answers, which is exactly the behavior you do not want.

### ShareGPT4V: scaling dense captions

The second case study extends the idea from instructions back to captions, at higher quality. ShareGPT4V (Chen et al., 2023) observed that LLaVA's *captions* were still fairly short, and that GPT-4-Vision could produce far denser, more detailed descriptions — averaging around 900+ characters versus the roughly 50-character captions typical of earlier datasets. They collected 100K such captions directly from GPT-4V, then did the two-tier move: trained a captioner (ShareCaptioner) on those 100K to imitate GPT-4V's style, and used it to caption 1.2M more images cheaply. Swapping in this denser caption data at the pretraining-alignment stage improved LLaVA across the board with no architecture change.

ShareGPT4V is the clearest demonstration of the throughput/quality two-tier strategy in practice: pay for quality on a seed set with the best available model, distill that quality into a cheap captioner, and scale. It is also a cautionary example for the homogeneity problem we discuss in troubleshooting — a captioner distilled from a single teacher inherits that teacher's stylistic tics, and 1.2M captions in one voice is a real diversity risk that fusion with alt-text partially mitigates.

## Interleaved image-text: MMC4 and OBELICS

The fourth data family is structurally different from the other three. Captions, VQA, and grounding are all *tightly coupled* pairs: one image, one piece of text about it. Interleaved data is *loosely coupled*: an entire document with images and text in their original positions, where the relationship between any given image and any given sentence is implicit and sometimes distant.

![Interleaved image-text document (MMC4 / OBELICS)](/imgs/blogs/recaptioning-and-vlm-instruction-data-6.webp)

The stack above is what one interleaved training example looks like: not a pair but a whole web page. An intro paragraph sets a topic, then image 1 (a figure), then a paragraph that says "as shown above" and refers back to image 1, then image 2 (a chart), then a closing paragraph. The text and images are kept in their original reading order, and that order is the entire point. A model trained on this learns to resolve cross-references between modalities — to understand that "the figure above" means the image that preceded this sentence, that a caption describes the image next to it, that a chart supports the argument in the surrounding prose. This is the skill that enables few-shot in-context learning from mixed prompts, the capability Flamingo demonstrated and that every modern VLM with in-context abilities depends on.

Two corpora define this space:

- **MMC4** (Multimodal C4, Zhu et al., 2023) took the text-only C4 corpus and put images *back*. They collected candidate images from the source URLs and used CLIP to assign each image to the most similar sentence in the document, reconstructing an interleaved corpus of roughly 101M documents and 571M images (around 43B text tokens in the full set).
- **OBELICS** (Laurencon et al., 2023, from Hugging Face) built interleaved data directly from Common Crawl, preserving the original DOM structure rather than reconstructing it — extracting the actual text-image ordering from web pages. It contains about 141M documents, 353M images, and 115B text tokens, and it was used to train the IDEFICS models.

The two approaches embody a real design choice. MMC4's CLIP-based placement is *reconstructive* — it guesses where images belong and can place them wrong, but it can enrich any text corpus. OBELICS's structure-preserving extraction is *faithful to the original layout* — the ordering is real — but it is limited to what was actually on the web and inherits all of the web's boilerplate and clutter, which means aggressive filtering (see [text extraction and boilerplate removal](/blog/machine-learning/training-data/text-extraction-and-boilerplate-removal) for the general problem).

The second-order consideration with interleaved data is that it is *lower density per image* than caption data — most sentences in a web document are not about any image at all. So it is not a substitute for caption or instruction data; it is a complement that teaches a different skill. The standard recipe uses interleaved data for its in-context/few-shot ability and dense caption plus instruction data for grounding and conversation. Getting the mix right is the next section.

## Grounding and region data

The third family gets a shorter treatment here because it is more specialized, but it matters for any model that needs to point. Grounding data ties phrases to regions: referring expressions ("the red cup on the left") paired with bounding boxes, or dense region captions where every box in an image gets its own description. RefCOCO/RefCOCO+/RefCOCOg are the classic human-annotated referring-expression sets; GRIT (from Kosmos-2) is a large synthetic grounding corpus built by running a grounding model over image-text pairs to link noun phrases in the caption to boxes in the image.

The synthesis pattern mirrors recaptioning: take an existing caption, run a phrase grounder or open-vocabulary detector to link each noun phrase to a region, and emit (phrase, box) pairs. The same hallucination risk applies and is arguably worse — a grounder can confidently box a region for a phrase describing an object that is not there. The mitigation is symmetric to the CLIP filter: only keep (phrase, box) pairs where the region actually scores high for the phrase under a region-text alignment model. Grounding data is where modern VLMs like [DeepSeek-VL and its dynamic-tiling successor](/blog/machine-learning/computer-vision/deepseek-vl-vl2-dynamic-tiling-moe) get their ability to localize, and where the decoupled-encoder design in [Janus-Pro](/blog/machine-learning/computer-vision/janus-pro-decoupled-visual-encoding) pays off by keeping understanding and generation representations separate.

## Balancing task types

**The single most common mistake in VLM data work after "the captions are bad" is getting the task mixture wrong.** You can have perfect caption data, perfect instruction data, and perfect interleaved data, and still train a mediocre model because the proportions are off. This is the multimodal version of the data-mixing problem covered in [data mixing, domain weighting and curriculum](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum), and the same principles apply.

The failure looks like this. Over-weight captions and the model describes everything but cannot follow instructions or answer questions — it narrates when you ask it to reason. Over-weight instruction data and it becomes chatty and confident but its perception degrades, because it stopped seeing enough raw image-caption signal. Over-weight interleaved data and it does great few-shot in-context tricks but its single-image grounding is loose. Skimp on grounding data and it cannot point or localize.

Here is a reasonable starting mixture for a general-purpose VLM's supervised stage, expressed as rough proportions rather than absolute counts, because the right absolute scale depends on your model size:

| Data family | Rough share | Primary capability | Symptom if too low |
| --- | --- | --- | --- |
| Dense captions (recaptioned/fused) | 35-45% | Perception, attributes | Vague, generic descriptions |
| Instruction / VQA (synthesized) | 25-35% | Conversation, reasoning | Cannot follow instructions |
| Interleaved documents | 15-25% | In-context, few-shot | Poor multi-image reasoning |
| Grounding / region | 8-15% | Localization, pointing | Cannot point or count |

Two rules of thumb make this tractable. First, the mixture changes across training stages: pretraining alignment leans heavily on captions and interleaved data (learning to see); the supervised instruction stage leans on VQA and reasoning data (learning to converse). Second, treat these proportions as a starting hypothesis and *measure* — hold out a balanced eval across all four capabilities and reweight toward whatever is weakest. A model is only as good as its worst capability, and the mixture is your primary lever on the floor.

## Troubleshooting

Every failure below is one I have seen bite real VLM training runs. The pattern is always symptom, then the detection method that confirms the cause, then the fix. The most important one is first.

![Caption hallucination: the fourth claim has no pixels](/imgs/blogs/recaptioning-and-vlm-instruction-data-7.webp)

### Caption hallucination

**Symptom:** the recaptioned corpus contains descriptions that assert objects, attributes, or actions that are not in the image. The figure above shows the canonical case: three claims ("a brown dog", "on green grass", "wearing a red collar") map to real regions, but the fourth ("catching a frisbee") grounds to nothing — the frisbee is present in the scene but the dog is not catching it, so the caption invented an action. Downstream, a model trained on this learns to hallucinate the same way, confidently describing objects that are not present.

**Detection:** measure it with CHAIR (Caption Hallucination Assessment with Image Relevance), which computes the fraction of mentioned objects that are not actually in the image, or POPE (Polling-based Object Probing Evaluation), which asks the model yes/no questions about object presence and measures how often it says "yes" to absent objects. At the data level, run an open-vocabulary detector over each image, extract the noun phrases from the caption, and flag any noun phrase with no corresponding detection above threshold.

**Fix:** four levers, in order of cost. (1) Lower decoding temperature and increase `repetition_penalty` on the captioner — much hallucination is high-temperature confabulation. (2) The fusion rule "drop objects the sources disagree about" from earlier, which removes uncorrelated inventions. (3) The CLIP/region-alignment filter, which drops whole captions that do not match. (4) Object-grounded decoding: feed the captioner the detected object list and instruct it to only describe those objects — the most effective and the most expensive.

### Instruction-format overfit

**Symptom:** the trained VLM answers everything in the exact format of your synthetic instruction data — always the same length, always the same opening, always the same structure — regardless of what the user actually asked. Ask for one word and it writes a paragraph; ask for a table and it writes prose.

**Detection:** evaluate on prompts whose format differs from your training distribution and watch the response-length and structure distributions. If response length is nearly constant regardless of the prompt, you have format overfit. Plot the distribution of first-token and first-sentence patterns; a spike means the model memorized one template.

**Fix:** diversify the synthesis prompts so the generated instructions vary in format, length, and style — explicitly ask the generator for short answers, long answers, lists, single words, and multi-turn exchanges in known proportions. Mix in human-authored VQA (VQAv2, OK-VQA) whose formats were not machine-generated. Add format-augmentation: for a fraction of samples, rewrite the response into a different length or structure while preserving content.

### Synthetic-caption homogeneity

**Symptom:** ten thousand captions and they all sound the same. "This image shows a...", "The image depicts...", "In this photo, we can see...". The model trained on them produces captions in exactly one voice and its lexical diversity collapses. This is the low-entropy trap that CapsFusion diagnosed as scalability deficiency.

**Detection:** compute n-gram diversity (distinct-1, distinct-2), self-BLEU across the corpus (high self-BLEU means the captions are similar to each other), and cluster caption embeddings — if the corpus collapses into a few tight clusters, it is homogeneous. A quick tell is counting how many captions start with the same three words; if it is more than a few percent, you have a problem.

**Fix:** vary the captioner's sampling (temperature, nucleus) instead of greedy/beam decoding, which is a major homogeneity source. Use an ensemble of captioners with different training data rather than one. Strip templated openers in post-processing. And — the highest-leverage fix — fuse with alt-text, which injects human phrasing diversity for free. This is the diversity argument for fusion, distinct from the world-knowledge argument.

### Losing fine-grained detail and entities

**Symptom:** after recaptioning, the specific things vanish. Brands become "a shoe", landmarks become "a building", species become "a bird", and any text visible in the image (signs, labels, packaging) disappears entirely. The corpus is denser but *less* informative about the specifics that matter for many downstream tasks.

**Detection:** run named-entity recognition on the alt-text and on the recaptioned text and compare entity counts and recall — if the recaptioned corpus has far fewer proper nouns than the alt-text it replaced, you are losing world knowledge. For in-image text specifically, sample captions of images you know contain text and check whether the text was transcribed.

**Fix:** this is the world-knowledge argument for fusion again — never *replace* alt-text, always fuse with it, so the entities survive. For in-image text, add an OCR pass and feed the transcribed text into the fusion step (this is the whole subject of the sibling post on [OCR, chart, and document data](/blog/machine-learning/training-data/ocr-chart-and-document-data), which is exactly where fine-grained textual detail lives). And keep the original alt-text as a separate field in your data schema even after fusing, so you can always recover the entities if a later task needs them.

### A worked failure: the plateau that was homogeneity

To tie the troubleshooting together, here is a scenario with numbers. A team recaptions 200M pairs with a single LLaVA-based captioner using beam search, no fusion, no alt-text. At 20M pairs their VLM's captioning CIDEr on a held-out set is 98. At 60M it is 104. At 120M it is 103. At 200M it is 100 — it went *down*. They add more data expecting more quality and get less. The detection: self-BLEU across the corpus is 0.71 (very high), distinct-2 is 0.18 (very low), and 34% of captions begin with "A photo of." The diagnosis is scalability deficiency from homogeneity, exactly CapsFusion's finding. The fix: they re-run the pipeline with (1) nucleus sampling instead of beam search, (2) fusion with the original alt-text, and (3) the CLIP filter. Self-BLEU drops to 0.42, distinct-2 rises to 0.34, and CIDEr at 200M pairs climbs to 112 — above the previous peak, and now *increasing* with scale. Same images, same model, same compute budget for training; the only change was the caption pipeline.

## When to recaption, and when not to

Recaptioning is not free — it costs GPU-hours for the captioner, LLM calls for the fusion, and engineering for the filter. Here is when it earns its keep and when it does not.

**Reach for recaptioning when:**

- Your image-text pairs come from web scrapes where the text is alt-text, filenames, or SEO copy. This is the default case and recaptioning is almost always worth it.
- You are training for dense description, detailed VQA, or any task where the model must attend to more than the single most salient object.
- Your synthetic-only or alt-text-only corpus shows a quality plateau or regression as you scale — that is the homogeneity/scalability tell, and fusion is the fix.
- You need the model to know specific entities (brands, species, landmarks) *and* describe scenes densely — only fusion gives you both.

**Skip or limit recaptioning when:**

- Your pairs already come from high-quality human captions (COCO, human-annotated internal data). Recaptioning these can *lose* information; keep the human captions and only add synthesis for the task types they lack.
- You are training a contrastive model like CLIP where short, entity-rich captions are actually well-matched to the objective — dense captions can hurt contrastive alignment. Measure before you recaption.
- Budget is tight and your alt-text is unusually clean (some curated sources are). Filter hard first; recaption only the subset that filtering flags as low-quality.
- The downstream task is narrow and you have task-specific supervised data — fine-tune on that directly rather than rebuilding the whole caption corpus.

The meta-lesson across all four families is the one the CapsFusion result made undeniable: **for training, the caption is a design decision, not a given.** The web handed you a string; your job is to decide what the model should learn from that image, and then to manufacture the text that teaches exactly that — dense where the web was sparse, grounded where the model would confabulate, specific where a captioner would go generic, and diverse where a single generator would collapse into one voice. Recaptioning, fusion, instruction synthesis, and interleaving are four ways of exercising that decision. Used together, they turn a billion junk strings into the best training signal you have.

## Further reading

- CapsFusion: Rethinking Image-Text Data at Scale (Yu et al., CVPR 2024) — the fusion result and the scalability-deficiency diagnosis.
- Visual Instruction Tuning (Liu et al., NeurIPS 2023) — LLaVA, and the text-only-LLM synthesis trick.
- ShareGPT4V: Improving Large Multi-Modal Models with Better Captions (Chen et al., 2023) — dense captions and the two-tier captioner strategy.
- OBELICS (Laurencon et al., 2023) and Multimodal C4 (Zhu et al., 2023) — the two canonical interleaved corpora.
- Sibling posts: [image-text pairs at scale](/blog/machine-learning/training-data/image-text-pairs-at-scale), [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation), [data mixing, domain weighting and curriculum](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum), and [OCR, chart, and document data](/blog/machine-learning/training-data/ocr-chart-and-document-data).
