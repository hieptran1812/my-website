---
title: "OpenAI to Z on Kaggle: how five winners hunted Amazon sites with lidar, satellites and LLMs"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "OpenAI's 500,000 USD Kaggle hackathon asked 225 teams to find possible archaeological sites under the Amazon canopy with open data and OpenAI models. All five winners, taken apart: what data they used, what they trained, how they checked themselves, and why the three that used the LLM least placed highest."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "hackathon", "geospatial", "lidar", "remote-sensing", "archaeology", "computer-vision", "reproducibility"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 23
---

> **TL;DR.** The OpenAI to Z Challenge (Kaggle, 15 May to 29 June 2025) asked teams to use open data and OpenAI models to find **possibly hidden archaeological sites in the Amazon**, with a focus on Brazil. 225 teams entered; an expert panel picked five winners, and 1st to 3rd took 250k, 100k and 50k USD. **The decisive lever was labelled data, not the language model:** the top three each trained a classical detector on labels borrowed from published site catalogues, a canopy-error trick, or a Maya lidar dataset, and could show a number that checked it. 4th and 5th built polished platforms around GPT and o-series reasoning with no labelled evaluation. **The one lesson:** when domain experts judge, the evidence chain wins; the LLM belongs at the end, explaining what a trained model and an independent signal already agree on. Every site in this post is **the team's own claim, not verified on the ground**.

![The OpenAI to Z Challenge on one card: six and a half weeks, 225 teams, a two-source gate, prize bars to scale, and the lever strip with Data lit](/imgs/blogs/kaggle-openai-to-z-challenge-winners-1.webp)

The card above is the whole contest. Most posts in this series are about a leaderboard with a hidden test set. This one has neither: nobody, not the host, not the judges, not the teams, knows where the undiscovered sites are. That single fact is why the winning entries look less like LLM demos and more like careful remote-sensing papers.

Every candidate site, confidence score and validation number below is **the team's own claim**. No winner reports a field visit, and I found no source saying any candidate has since been verified. Where a team's public artifacts contradict its writeup (TerraReveal's live app) or have vanished (the 3rd-place repo now returns 404), I say so. The figures are illustrative redraws, not copies of any team's imagery.

New to the series? [How Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work) covers judged hackathons; the closest sibling is OpenAI's other 2025 hackathon, the [gpt-oss red-teaming challenge](/blog/machine-learning/learn-from-kaggle/kaggle-gpt-oss-red-teaming-challenge-findings).

## 1. The brief in 60 seconds

As a friend would put it: "OpenAI put up half a million dollars for anyone who could point at a spot in the Amazon rainforest and convince archaeologists that something human-made is hiding under the trees, using only free data and OpenAI's models."

Formally, from the Overview, Prizes and metadata pages:

| Item | Value |
| --- | --- |
| Dates | 2025-05-15 to 2025-06-29 (about six and a half weeks) |
| Task | Find possibly hidden archaeological sites in the Amazon biome, focus on Brazil |
| Data | Competitors find their own: lidar tiles, Sentinel imagery, colonial diaries, papers |
| Model requirement | "Your quest must be supported using OpenAI models to be eligible" (o3, o4-mini, GPT-4.1 named) |
| Checkpoint advice | Host PDF includes "predict and verify the lat/long with two independent methods" |
| Entrants | 225 teams, 309 competitors, 225 submissions |
| Prizes | 1st 250,000 USD, 2nd 100,000 USD, 3rd 50,000 USD; 4th and 5th listed with no cash amount |

One arithmetic note: the metadata states a 500,000 USD pool, but the cash prizes sum to 250k + 100k + 50k = 400k USD. The Prizes page adds OpenAI-credit early-sharing and checkpoint prizes, but no source itemises the remaining 100k, so treat the gap as unexplained.

There is **no notebook budget**, no GPU limit and no hidden test. The constraint was not hardware, it was belief: a panel of archaeologists had to be persuaded.

## 2. The judging rubric

![The rubric to scale: a pass/fail gate on public sources, then five equal 20-point criteria, and the funnel from 225 teams to 5 finalists judged live](/imgs/blogs/kaggle-openai-to-z-challenge-winners-2.webp)

The rubric has two stages, and the figure draws both. First a **pass/fail gate**: at least two verifiable public sources (a lidar tile ID, a DOI, a satellite scene ID), links that open without a paywall, and no plagiarism. Fail it and nothing else is read. Then five criteria at **20 points each**:

$$
S \;=\; \mathbb{1}[\text{gate passed}] \cdot \sum_{k=1}^{5} s_k, \qquad 0 \le s_k \le 20, \qquad S_{\max} = 100
$$

where $s_k$ is the score on criterion $k$ and the indicator zeroes the total when the gate fails. In plain words:

| Criterion | What it rewards | Weak entry |
| --- | --- | --- |
| Evidence depth | Independent signals converging on one point | One image and a story |
| Clarity | Overlays and measurements a reader can follow | A wall of model output |
| Reproducibility | A stranger can rerun it from public IDs | "See the attached report" |
| Novelty | A method or source others did not use | The default NDVI notebook |
| Presentation craft | Includes answering judges live | A slick video, no method |

The **top five were judged in a livestream** by subject-matter experts, OpenAI staff and a "mystery AI leader". The winners post (602618) has the archaeologist panel weighing "potential archaeological impact, investigative ingenuity, depth and creativity of insights, and reproducibility (the ease with which experts can retrace and verify every step of evidence)". Five of 225 is 2.2% of the field.

There are **no per-winner rubric scores** and **no per-winner judge commentary**; the winners post names only the 1st-place team. Everything below about *why* an entry placed where it did is my inference, labelled as such.

Read the rubric with an engineer's eye: three of the five criteria (evidence depth, clarity, reproducibility) are about the **audit trail**, not the discovery. That is the right incentive when nobody can check the answer, and it is the key to the results.

## 3. What made it hard: the canopy, the pixel size, and no ground truth

![Why sites hide, drawn as one illustrative cross-section: optical satellites see only the canopy, a 30 m global DEM smooths the ditch away, and a 1 m lidar terrain model traces it](/imgs/blogs/kaggle-openai-to-z-challenge-winners-3.webp)

Three problems stack on top of each other, and every winner is a different answer to the same three.

**The canopy.** The Amazon's pre-Columbian earthworks (ring ditches, mounds, geometric enclosures called geoglyphs) sit under closed forest. Sentinel-2 sees the top of the trees; its 10 m pixels can catch a vegetation difference over buried soil, not the earthwork. Radar-derived global elevation models such as SRTM or NASADEM read somewhere inside the canopy, so their "terrain" carries a bias of tree height.

**The pixel size.** Global elevation models come at 30 m, and the 3rd-place writeup argues that structures smaller than 30 m vanish in that grid. The figure draws the pixel widths in their true 10 : 30 : 1 ratio; the ditch survives only in the last row. Airborne lidar fixes this: some laser pulses slip through gaps in the leaves, and the last returns are gridded into a **digital terrain model** (DTM) at about 1 m. The catch is coverage: it exists only where someone flew it.

**No ground truth.** In a normal competition a hidden label file decides who is right. Here, the honest label for every unexplored square kilometre is "unknown", and a model's output in new forest can only be checked by another signal or a person walking there. The host's advice, "predict and verify the lat/long with two independent methods", is the workaround: if two methods with different failure modes agree, coincidence is less likely.

So the real contest was **how to manufacture evidence without labels**. Three answers won: borrow labels from published catalogues (1st), turn an instrument's error into signal (2nd), borrow labels from another rainforest (3rd).

## 4. The winners table: all five

Five winners, places from the tracks API, all with a Kaggle Writeup: this post covers 5 of 5.

| Place | Team | Core method | Region | Source |
| --- | --- | --- | --- | --- |
| 1st, 250k USD | Black Bean (Yao, tianmiao11, Yingjie Zhang) | Two ResNet-18s (elevation, optical) on 4,067 known sites, cross-modal pairing, 67 candidates | Whole biome | [writeup](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/amazon-archeological-site-discovery-a-deep-learnin) |
| 2nd, 100k USD | Commencis (Dorukhan Arslan, Ece Ünal, Pınar Ersoy, Ebubekir), GlyphTrack | GEDI-vs-DEM residual plus PU LightGBM, GPT triage | Acre, Brazil | [writeup](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/glyphtrack-detection-of-anomalies-under-canopy) |
| 3rd, 50k USD | mxsx (Maja Somrak) | Segmentation trained on Maya lidar, fine-tuned on Tapajós, run on 1 m lidar | Brazilian lidar tiles | [writeup](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/amazonian-archaeology-discovery) |
| 4th, no cash listed | CloudJas, Archaios | Azure platform: classical lidar processing, four-agent LLM "team", RAG | User-supplied | [writeup](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/archaios-explore-hidden-world-with-open-ai) |
| 5th, no cash listed | GnosisPoet, TerraReveal | GPT-4.1, o3, o4-mini; CNN plus XGBoost claimed; 12 anomalies | Peru | [writeup](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/terrareveal) |

Only 1st place searched the whole biome, and 5th worked in Peru despite the Brazil focus.

## 5. The landscape: a feature matrix across all five winners

![Feature matrix across the five winners: data, method and evidence features, filled where the writeup shows it, with an N of 5 count per row](/imgs/blogs/kaggle-openai-to-z-challenge-winners-4.webp)

Counting only cells the writeups support (TerraReveal's "claimed" cells are drawn but not counted):

- **3 of 5 trained their own detector on their own labelled dataset**, and those three are exactly places 1 to 3.
- **2 of 5 report any quantitative evaluation** (Black Bean's validation accuracy, GlyphTrack's AUC and AP). The 3rd place's numbers sat in the repo that now 404s, so that cell is "?".
- **2 of 5 state a candidate count** (67 and 12), and **0 of 5 report field or independent ground-truth validation.**
- **1 of 5 used historical texts** (Archaios), although the brief explicitly invited colonial diaries.

### Table stakes

Every winner cleared the gate with named public datasets, presented results on a map, and kept the OpenAI model **outside the core detector**: describing, triaging, interpreting or orchestrating. Where an LLM did look at imagery (Archaios), a classical relief pipeline made the images.

### What separated 1st to 3rd from 4th and 5th

My inference: the top three each **trained on real labels** and showed a number that checked the model. The bottom two built **platforms** where GPT and o-series models reason over imagery, with no labelled evaluation. With three of five criteria rewarding the audit trail, that ordering is what you would predict.

### Where the winners contradict each other

| Question | One side | Other side |
| --- | --- | --- |
| Is airborne lidar essential? | mxsx: yes, 30 m DEMs cannot see features under 30 m | GlyphTrack: use GEDI and DEM error instead; Black Bean won with 30 m NASADEM |
| How big a search area? | Black Bean: the whole biome | GlyphTrack: Acre; TerraReveal: one Peruvian box; Archaios: whatever is uploaded |
| How much LLM? | Black Bean: a one-line describer | Archaios: a four-agent expert panel with RAG |

The describer won: that is the headline of this hackathon.

## 6. 1st place: Black Bean, a basin-wide supervised search

![Black Bean as a map: overlapping 2 km windows, elevation and optical hits, known-site exclusion circles, and pairs within 2 km becoming candidates](/imgs/blogs/kaggle-openai-to-z-challenge-winners-5.webp)

### The intuition

If you cannot label the unknown forest, label the known one. Archaeologists have published thousands of Amazonian site locations. Treat each as a positive, sample random forest far from all of them as negatives, train a classifier, and slide it across the basin. Wherever it fires and nothing has been reported is a candidate.

The trap is that such a classifier also learns "looks like where archaeologists happened to survey": near rivers, roads, clearings. Black Bean's answer is a second model that sees different physics, and a rule that keeps only places where both agree.

### The mechanism

From the writeup:

**Labels.** 4,067 positives from six published site datasets, including 1,181 from the Science 2023 "10,000 earthworks" paper and 1,811 from a PeerJ machine-learning paper, so the other four datasets supply the remaining 4,067 - 1,181 - 1,811 = 1,075. Then 4,067 negatives by grid-stratified random sampling inside the biome, each at least 2 km from any known site. The classes are exactly balanced.

**Two inputs, two models.** Elevation comes from NASADEM via Earth Engine (`NASA/NASADEM_HGT/001`) at 30 m pixels, as 64x64 patches (about 1.9 km): the raw height plus four derived terrain metrics, five bands in all. Optical comes from Sentinel-2 surface reflectance, bands B2, B3, B4 and B8, as 206x206 patches at 10 m (about 2.1 km) from the least cloudy 2024 scene. Different pixel grids, so **two separate ResNet-18 models**; separate beat mixed in their trial. Each model reached **more than 70% validation accuracy**. (If ResNets are new to you, [why convolution works](/blog/machine-learning/computer-vision/why-convolution-works) is the background.)

**Inference.** A 2 km window slides with a 1 km stride, so neighbouring windows overlap by half. Windows with probability above 0.8 are kept, clustered with DBSCAN, and anything within 2 km of a known site is dropped. Then the decisive step: **cross-modal pairing**, an elevation hit and an optical hit within 2 km of each other, matched one to one. That left **67 candidates**, each inspected by eye and posted to a public explorer site.

**Where OpenAI sat.** ChatGPT writes "a brief description" per candidate on the explorer; no OpenAI model is in the detection pipeline.

### Why pairing is the whole trick

Above 70% on a balanced set is modest: chance is 50%, and over a whole biome the false positives would swamp any real sites. Pairing attacks exactly this. As an illustrative assumption (not a measured number), if each model fires falsely on 5% of empty windows and their mistakes are independent, an empty spot passes both at 0.05 x 0.05 = 0.0025, twenty times rarer than either alone. Real errors are partly correlated (both can learn "near a river"), so the true gain is smaller, but the direction holds.

The pairing step as an illustrative toy on invented coordinates (**not Black Bean's code**; their repo is in Sources):

```python
# Illustrative toy of cross-modal pairing. NOT Black Bean's code:
# coordinates are invented, in km on a flat local grid.
import math

dem_hits = [(3.0, 4.0), (10.0, 2.0), (6.5, 9.0), (15.0, 15.0)]   # NASADEM model, p > 0.8
s2_hits  = [(3.8, 5.1), (11.2, 2.4), (6.0, 12.5), (1.0, 14.0)]   # Sentinel-2 model, p > 0.8
known    = [(15.5, 14.0)]                                         # published sites

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def far_from_known(p, radius=2.0):
    return all(dist(p, k) >= radius for k in known)

dem = [p for p in dem_hits if far_from_known(p)]
s2 = [p for p in s2_hits if far_from_known(p)]

# all DEM x S2 pairs within 2 km, closest first, each hit used at most once
pairs = sorted((dist(a, b), i, j) for i, a in enumerate(dem)
               for j, b in enumerate(s2) if dist(a, b) <= 2.0)
used_d, used_s, candidates = set(), set(), []
for d, i, j in pairs:
    if i in used_d or j in used_s:
        continue
    used_d.add(i); used_s.add(j)
    mid = ((dem[i][0] + s2[j][0]) / 2, (dem[i][1] + s2[j][1]) / 2)
    candidates.append((mid, round(d, 2)))

print(f"DEM hits kept {len(dem)} of {len(dem_hits)}, S2 hits kept {len(s2)} of {len(s2_hits)}")
for mid, d in candidates:
    print(f"candidate at ({mid[0]:.2f}, {mid[1]:.2f}) km, modalities {d} km apart")
print(f"{len(candidates)} candidates from {len(dem_hits) + len(s2_hits)} raw hits")
```

```console
DEM hits kept 3 of 4, S2 hits kept 4 of 4
candidate at (10.60, 2.20) km, modalities 1.26 km apart
candidate at (3.40, 4.55) km, modalities 1.36 km apart
2 candidates from 8 raw hits
```

Eight raw hits become two candidates: one elevation hit dies near a known site, and the rest find no partner from the other modality within 2 km.

### What the team says it did not do

The writeup is candid. The labels were "not supervised by team members" for lack of archaeological knowledge, so noise from six catalogues went in unchecked. There is no held-out test set, no published baseline, and no field or lidar check of the 67. The team names soil and climate bands as the next step, arguing that the coarse (about 1 km) soil predictors of a prior Random Forest paper cause false negatives.

Why did it win? The panel gave no reasons, so this is inference: the only basin-wide search, a short ranked list an expert can check, a public ID for every input, public code and explorer, and the host's two-methods recipe made literal.

## 7. 2nd place: GlyphTrack, turning instrument error into signal

![An illustrative transect: GEDI ground returns under a canopy-biased DEM, the estimated 20 m bias removed, and residual bumps and hollows beyond 2 m flagged as micro-relief anomalies](/imgs/blogs/kaggle-openai-to-z-challenge-winners-6.webp)

### The intuition

Commencis started from a constraint: airborne lidar "can't be an available solution sometimes, as we experienced". What free, global data reaches the ground at all? **GEDI**, a lidar on the International Space Station that measures ground height at sparse footprints. Compare GEDI's ground with a global DEM and the difference is mostly canopy bias. Model the bias, subtract it, and the residual should be near zero. Where it is not, something on the ground is higher or lower than predicted: a mound or a ditch.

### The mechanism

From the writeup, two models fused:

1. **Micro-relief.** GEDI ground returns are matched to DEM heights. A robust regression predicts the canopy bias, and the residual bumps and hollows are scored as **Pglyph** using slope and ruggedness. No airborne lidar, no CNN.
2. **Landscape.** A **Positive-Unlabelled LightGBM** ensemble ranks 3 km tiles across Acre on elevation, soil phosphorus and distance to the two nearest river orders. Tiles aggregate into 20 km super-tiles, reported at **AUC about 0.95 and AP about 0.84** (team-reported).

For the top tile the two scores are averaged, and GPT appears twice: it triages to pick "the most promising tile" after thresholding, and it interprets the fused table plus an annotated DEM with "an archaeology-focused prompt" (historical and mythological context included). The writeup names neither the GPT version nor the DEM product, and gives no candidate count.

**Positive-Unlabelled learning** fits this problem well: an "unlabelled" tile is not a known negative, it might hold an unreported site, so PU learning treats it as unlabelled. The classic Elkan and Noto result (a general method; the writeup does not say which PU recipe Commencis used) is that if labelled positives are a random sample of all positives,

$$
P(\text{labelled} = 1 \mid x) \;=\; c \cdot P(y = 1 \mid x),
$$

where $y$ is the true class and $c$ is the fraction of true positives that got labelled. So a model predicting "labelled" ranks tiles in the same order as one predicting "is a site", which makes ranking metrics such as AUC and AP the honest ones to report. The [LightGBM deep dive](/blog/machine-learning/traditional-machine-learning/lightgbm-deep-dive) covers the booster itself.

The residual step as a toy, on the illustrative numbers in the figure (**not the team's code or data**):

```python
# Illustrative toy of the GlyphTrack idea. NOT the team's code or data.
# A DEM over forest reads high by the canopy; GEDI gives true ground heights
# at sparse footprints. Fit the bias, keep the residual, flag big ones.
gedi = [200, 201, 203, 202, 204, 209, 203, 198, 204, 205, 206, 205, 207]
dem  = [220, 221.5, 222.5, 222.3, 223.8, 224.5, 223, 221.8, 223.7, 225.2, 225.6, 225.1, 226.8]
x    = [250 * i for i in range(len(gedi))]

diffs = sorted(d - g for d, g in zip(dem, gedi))
bias = diffs[len(diffs) // 2]            # median: a robust one-number "regression"
for xi, d, g in zip(x, dem, gedi):
    r = g - (d - bias)                   # ground minus bias-corrected DEM
    flag = "bump" if r > 2 else "hollow" if r < -2 else ""
    if flag:
        print(f"x={xi:5d} m  residual={r:+.1f} m  {flag}")
print(f"canopy bias estimate: {bias} m")
```

```console
x= 1250 m  residual=+4.5 m  bump
x= 1750 m  residual=-3.8 m  hollow
canopy bias estimate: 20 m
```

The median is the simplest robust estimator: the two anomalies barely move it. That is why a robust regression beats least squares here, where a big mound would drag the bias up and partly hide itself.

**Why it placed (inference).** The only winner needing neither airborne lidar nor a CNN, so the most transferable to forest nobody has flown, with ranking metrics that match its learning setup. Its weakness is the writeup: about 425 words, the reproducible detail living in a linked notebook.

## 8. 3rd place: mxsx, borrowing labels from the Maya

![Borrowing Maya labels: the Chactun tile counts to scale, two equal 240 m tiles at different resolutions, the class mapping from Maya to Amazon, and the Brazilian lidar search area](/imgs/blogs/kaggle-openai-to-z-challenge-winners-7.webp)

### The intuition

Maja Somrak's entry starts where the 30 m approach stops. If small earthworks need 1 m lidar, the detector must be trained on high-resolution lidar, and the best labelled archaeological lidar is Maya, not Amazonian. The **Chactun** dataset from Mexico has more than 12,000 expert-annotated structures in three classes: building, platform and aguada (a water reservoir). Train segmentation there, map Amazonian feature types onto those classes, fine-tune on a few Amazonian examples, and search Brazilian lidar.

### The mechanism

From the writeup:

- **Chactun data.** 2,094 tiles of 480x480 px at 0.5 m, which is 240 m on the ground. 1,343 tiles contain buildings, 1,131 platforms and only 78 aguadas (tiles can hold several classes), so aguada tiles were augmented. Dropping tiles with suspected unlabelled structures (terraces, causeways) left 1,303, a cut of 791 or about 38%; the team trained on both sets.
- **Visualisation layers.** The Relief Visualization Toolbox produces slope, sky-view factor and positive openness with the Maya study's exact settings. Sky-view factor measures how much sky a point "sees", so a ditch floor stands out at any sun angle.
- **Class mapping.** Amazonian mounds map to the platform class, for example; Amazon Dark Earths (anthropogenic soils) are excluded as undetectable by lidar.
- **Few-shot fine-tune** on hand-labelled tiles from known sites in the Tapajós National Forest (an ORNL DAAC lidar set of 1.2 GB).
- **Search area.** NASA's Sustainable Landscapes Brazil lidar: 3,154 DTM files, 125 GB, covering 3,125 km², so each file is about 1 km² and about 40 MB. Tiles are 240x240 px at 1 m (the Chactun footprint at half the resolution), kept only within 10 km of rivers and lakes and away from modern land use (RAISG).

### What is missing

The architecture, hyperparameters and any IoU were deferred to GitHub ("See more on Github"), and that repo returned **HTTP 404** when this series checked it on 2026-10-28. The writeup never mentions an OpenAI model, although the rules required one; how OpenAI models were used is **not stated in the writeup**. There is no candidate count. Today you have the recipe but not the model; the failure modes you would hit are in [debugging segmentation](/blog/machine-learning/debugging-training/debugging-segmentation).

**Why it placed (inference).** The most technically ambitious idea in the field: cross-continental transfer of expert labels at the resolution the earthworks live at, with a domain argument for every exclusion. It loses ground on reproducibility (now) and evidence depth (one modality, no count).

## 9. 4th place: Archaios, a cloud lab for archaeologists

**The user.** Archaios is not a discovery run, it is a **tool**. The writeup calls today's lidar-to-interpretation process "complex, tedious, and heavily dependent on expert manual interpretation", so CloudJas built an Azure platform where an archaeologist uploads LAS, LAZ or E57 point clouds plus field reports and gets back processed layers, analysis and a ranked view.

**The lidar pipeline is classical geoprocessing, not deep learning:** outlier removal, progressive-densification ground classification, gridding at 0.5 to 2.0 m with void filling, a multi-scale **Local Relief Model** (sigma 1 to 5), a white top-hat filter, an 8-direction composite hillshade and slope. More than 20 parameters live in a drag-and-drop workflow builder with an exportable audit trail, the team's reproducibility answer. Spectral layers (NDVI, true and false colour) come from Google Earth Engine; the sensor is not named.

**The LLM is the analyst.** An image analyzer flags areas of interest on the DTM, DSM, hillshade and spectral outputs. Then **Microsoft Semantic Kernel** orchestrates a "virtual archaeological team" of four agents (Archaeological Analyst, Terrain Specialist, Environmental Expert, Team Coordinator) that reaches a consensus on "archaeological authenticity". The agents draw on a RAG layer over **100+ vectorised field reports** (OpenAI embeddings, Cosmos DB DiskANN search). Neither the writeup nor the README names the chat model. (Agent debates as a design pattern are covered in [debate and critique in multi-agent systems](/blog/machine-learning/ai-agent/debate-and-critique-multi-agent); the retrieval half in [building an effective RAG system](/blog/machine-learning/large-language-model/how-to-build-effective-rag-system).)

**Evidence.** A 0 to 100% confidence score rises when a feature appears in both topographic and spectral layers, their version of the two-methods checkpoint. But there is **no candidate count, no coordinates, and no validation against known sites or a baseline**; the findings section is one hillshade image. The repo is public under MIT, with a live app and no video.

**Why it placed (inference).** The most complete reproducible *tool* in the field. Evidence depth is its weakest axis: it never shows the agents' consensus agreeing with anything an archaeologist already knows.

## 10. 5th place: TerraReveal, and an app that disagrees with its writeup

**The writeup is thin**, about 340 words, pointing to an attached report the Writeup API does not expose. Searches found no public repo.

**What it claims.** In the **Peru** Amazon (the brief focused on Brazil), **12 high-confidence anomalies**, two highlighted: Site 5 at (-11.7123, -69.5678), a "large elevated earthwork" at 0.90 confidence, and Site 4 at (-11.3567, -69.2134), a "geometric settlement complex" at 0.85. Models: **GPT-4.1** for context, **o3 and o4-mini** for "spatial and geometric reasoning", a **CNN** for detection and **XGBoost** for scoring. Data: Sentinel-2, SRTM, OpenTopography lidar "for cross-validation". All anomalies "validated within ±50 m spatial tolerance".

Two checks. Four decimals of latitude is about 11 m of precision, from inputs whose finest pixels are 10 to 30 m. And "validated within ±50 m" never says against what: no labelled test, no baseline, no known Peruvian sites. The 0.90 and 0.85 are the team's own scores.

**What the live app shows.** The series' research pass read the app's JavaScript bundle on 2026-10-28, and it **does not match the writeup**:

| Writeup says | App bundle shows |
| --- | --- |
| GPT-4.1, o3 and o4-mini | GPT-4.1 (`GPT-4.1-2025-04-14`), o3 and **GPT-4o-mini**; no o4-mini |
| CNN plus XGBoost ensemble | "Multi-model ensemble for confidence scoring"; the string "XGBoost" does not appear |
| OpenTopography lidar for cross-validation | A routine fetches **SRTM** through OpenTopography's raster endpoint for lat -12 to -11, lon -70 to -69 (which contains both headline sites), falling back to Landsat 8 |
| Open-source platform | A Lovable-generated React app with a Supabase backend; no repo found |

Parts of the dashboard are hardcoded demo content: an activity feed with fixed confidences (87.3, 94.1, 78.9) and an "o3-mini" label, and a "2.4GB processed" string. XGBoost could run server-side, but nothing public shows it. The fair reading: the app is a UI shell, not evidence for the pipeline. Its data records do cite SRTM v3.0 (DOI 10.5066/F7PR7TFT) and the Sentinel-2 Copernicus DOI, which satisfy the gate.

**Why it placed (inference).** It leaned hardest on the sponsor's models and shipped a polished platform, but it is the entry a skeptic can verify least.

## 11. Where the LLM sat, and why that predicts the order

![Where the OpenAI model sat in each winner's pipeline: four stages per row, LLM badges marked, clustering in the lower places](/imgs/blogs/kaggle-openai-to-z-challenge-winners-8.webp)

Mark every stage an OpenAI model performs and the badge counts read 1, 1, 0, 3, 2 from 1st to 5th. The two entries with the LLM in the detect or cross-check stage placed 4th and 5th; the three with it only at the end (or undocumented) placed 1st to 3rd.

The point is not that LLMs are bad at archaeology; it is **what an LLM's output can be checked against**. A 70% validation accuracy and an AUC of about 0.95 are numbers on labels. A four-agent consensus on "archaeological authenticity" is text, and so is "0.90 confidence" from a pipeline nobody can rerun. When judges must "retrace and verify every step", a stage whose output cannot be scored adds presentation but subtracts evidence.

The pattern matches the [LLM judges competition](/blog/machine-learning/learn-from-kaggle/kaggle-llms-you-cant-please-them-all-solutions): an LLM verdict is persuasive and unaudited at once. The winners used the LLM where persuasion is the point and kept it away from where evidence is the point.

## 12. What the host and the forum taught everyone

- **The checkpoints post** (578996) carried the "two independent methods" advice the top two built on, plus early-sharing prizes.
- **The gate** made public IDs mandatory: every winner's data is reachable today, even where its code is not.
- **The winners post** (602618, 144 words) names only Black Bean and says "the remainder of the winners have been selected on Kaggle", suggesting the challenge ran beyond one platform. I found no per-finalist livestream verdicts.

No leaderboard meant no shake-up and no leak; the nearest analogue is the most LLM-forward entries placing lowest.

## 13. What did not work, or was missing

Aggregated across the five writeups:

| Gap | Who | Consequence |
| --- | --- | --- |
| Labels not checked by an archaeologist | Black Bean, self-reported | Catalogue noise trained in unchecked |
| Mixed-modality single model | Black Bean, tried and dropped | Two separate ResNet-18s did better |
| 30 m DEM for small structures | mxsx, argued | Motivated the 1 m lidar route |
| Airborne lidar unavailable | GlyphTrack, self-reported | Motivated the GEDI residual route |
| Method detail outside the writeup | GlyphTrack, mxsx (404), TerraReveal | Readers must chase links; one is dead |
| No labelled evaluation of an LLM stage | Archaios, TerraReveal | Scores with nothing to check against |
| Field or independent verification | All five | Every site is a hypothesis |

## 14. Lessons that transfer

1. **When nobody holds the answer key, manufacture evidence from disagreeing instruments.** Black Bean's elevation-versus-optical pairing and GlyphTrack's micro-versus-landscape average are the host's two-methods checkpoint made literal.
2. **Borrow labels before you invent them.** Published catalogues (1st), PU learning over known positives (2nd) and a Maya dataset (3rd) are three ways to train when the target has no labels.
3. **An instrument's error can be your signal.** GlyphTrack modelled canopy-induced DEM bias and used the residual as the feature; the move works wherever a cheap sensor is biased in a learnable way.
4. **Put the LLM where its output does not need to be scored.** Describing a candidate is a fine LLM job; deciding it is authentic is not, unless you can evaluate that decision.
5. **Hand experts a short, ranked, checkable list.** 67 candidates with an explorer beat an unbounded platform.
6. **Keep the evidence inside the writeup.** A 404 repo and an unexposed attachment both cost their teams auditability.

### If you enter a hackathon like this

- Clear the gate first: list every dataset with its tile ID, scene ID or DOI in one table.
- Build at least two detection signals with different physics, and show where they agree.
- Report one honest number on held-out labels, and say what chance is.
- State your candidate count and give coordinates at the precision your pixels support.
- Use the sponsor's model visibly but at the end, and name the version.
- Put architecture, hyperparameters and metrics in the writeup itself, not only in a repo.
- Say plainly what you did not verify. Black Bean did, and won.

Other judged hackathons in this series: [Gemma impact hackathons](/blog/machine-learning/learn-from-kaggle/kaggle-gemma-impact-hackathons-winners), [Tunix Hack](/blog/machine-learning/learn-from-kaggle/kaggle-tunix-hack-reasoning-winners), [early Gemma and Gemini hackathons](/blog/machine-learning/learn-from-kaggle/kaggle-gemma-gemini-early-hackathons-winners), [BigQuery AI](/blog/machine-learning/learn-from-kaggle/kaggle-bigquery-ai-hackathon-winners) and [Measuring Progress Toward AGI](/blog/machine-learning/learn-from-kaggle/kaggle-measuring-agi-cognitive-abilities-winners).

## 15. Sources

- Competition overview: https://www.kaggle.com/competitions/openai-to-z-challenge
- Winners post: https://www.kaggle.com/competitions/openai-to-z-challenge/discussion/602618
- Thank-you and next steps: https://www.kaggle.com/competitions/openai-to-z-challenge/discussion/587178
- Checkpoints and early-sharing prizes: https://www.kaggle.com/competitions/openai-to-z-challenge/discussion/578996
- 1st, Black Bean writeup: https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/amazon-archeological-site-discovery-a-deep-learnin
  - Code: https://github.com/tianmiao11/amazon-archaeology-explorer/
  - Explorer: https://tianmiao11.github.io/amazon-archaeology-explorer/
  - Video: https://www.youtube.com/watch?v=evNM9eyw0-k
- 2nd, GlyphTrack writeup: https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/glyphtrack-detection-of-anomalies-under-canopy
  - Notebook: https://kaggle.com/code/ceeluna/glyphtrack-detection-of-anomalies-under-canopy
- 3rd, mxsx writeup: https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/amazonian-archaeology-discovery
  - Repo (returned HTTP 404 on 2026-10-28): https://github.com/tuajlajt/amazon_archaeology/
- 4th, Archaios writeup: https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/archaios-explore-hidden-world-with-open-ai
  - Code (MIT): https://github.com/Cloud-Jas/Archaios
  - Live app: https://archaios-fphhghfjdbf8dmbz.centralindia-01.azurewebsites.net/
- 5th, TerraReveal writeup: https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/terrareveal
  - Live app (bundle read 2026-10-28): https://www.terrareveal.com/
  - Video: https://www.youtube.com/watch?v=0V7lpPIOAe0
