# OpenAI to Z Challenge (2025): all 5 winners

- **Slug:** `openai-to-z-challenge`. **URL:** https://www.kaggle.com/competitions/openai-to-z-challenge
- **Dates:** 2025-05-15 to 2025-06-29 (competition metadata). Winners post: https://www.kaggle.com/competitions/openai-to-z-challenge/discussion/602618
- **Entrants:** 225 teams, 309 competitors, 225 submissions (metadata).
- **Prize pool:** $500,000 (metadata). Prizes page: 1st $250k, 2nd $100k, 3rd $50k. The tracks API lists 4th and 5th place with **no cash amount**.
- **Rubric:** pass/fail gate of at least two verifiable public sources (lidar tile ID, DOI, scene ID). Then five criteria at 20 points each: evidence depth, clarity, reproducibility, novelty, presentation craft. The top five were judged in a livestream. The winners post says the expert archaeologist panel weighed "potential archaeological impact ..., investigative ingenuity, depth and creativity of insights, and reproducibility".
- **Bank coverage:** `kaggle-llm-writeup-bank-part4.md`, H2 "## 3. OpenAI to Z Challenge (2025)". It covers 1st to 3rd (Black Bean, GlyphTrack, mxsx) in depth, so their sections below only add what the bank lacks. 4th and 5th were absent from the bank ("not fetched") and get full depth here.
- **Sources used:** the five Kaggle Writeups (`wu/openai-to-z-challenge/`). Also the Archaios GitHub README (https://github.com/Cloud-Jas/Archaios), the TerraReveal live app's JS bundle (https://www.terrareveal.com/, a Lovable/Supabase build), and the TerraReveal YouTube oEmbed (uploader "kris zwart"). The 602618 winners post (144 words) mentions only Black Bean by name. There is no per-winner judge commentary.

## Winners table

Places come from the tracks API awards in `winners_openai-to-z-challenge.txt`. The 602618 post names only 1st place.

| Place | Team | Project | Writeup | Video / demo / code (from writeup links) |
|---|---|---|---|---|
| 1st ($250,000) | Black Bean (Yao, tianmiao11, Yingjie Zhang) | Amazon Archeological Sites Discovery - a Deep Learning Approach | [link](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/amazon-archeological-site-discovery-a-deep-learnin) | [video](https://www.youtube.com/watch?v=evNM9eyw0-k), [explorer](https://tianmiao11.github.io/amazon-archaeology-explorer/), [GitHub](https://github.com/tianmiao11/amazon-archaeology-explorer/) |
| 2nd ($100,000) | Commencis (Dorukhan Arslan, Ece Ünal, Pınar Ersoy, Ebubekir) | GlyphTrack: Detection of Anomalies Under Canopy | [link](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/glyphtrack-detection-of-anomalies-under-canopy) | [Kaggle notebook](https://kaggle.com/code/ceeluna/glyphtrack-detection-of-anomalies-under-canopy); 4 images; no video link |
| 3rd ($50,000) | mxsx (Maja Somrak) | Amazonia Archaeology Discovery | [link](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/amazonian-archaeology-discovery) | [GitHub](https://github.com/tuajlajt/amazon_archaeology/) (returned HTTP 404 on 2026-10-28); no video link |
| 4th (no cash in API) | CloudJas | Archaios - Explore hidden world with Open AI | [link](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/archaios-explore-hidden-world-with-open-ai) | [GitHub](https://github.com/Cloud-Jas/Archaios) (MIT), [live app](https://archaios-fphhghfjdbf8dmbz.centralindia-01.azurewebsites.net/); no video link |
| 5th (no cash in API) | GnosisPoet (author shown as "Gnosis Poet") | TerraReveal | [link](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/terrareveal) | [video](https://www.youtube.com/watch?v=0V7lpPIOAe0), [live app](https://www.terrareveal.com/); no public repo found |

## Per-winner synthesis

### 1st: Black Bean, Amazon Archeological Sites Discovery (bank covers in depth)
See bank §3 for the data (4,067 positives from 6 published datasets plus 4,067 negatives at least 2 km from known sites), the two ResNet-18s, the sliding window, DBSCAN, cross-modal pairing and the **67 candidates**. Facts the bank lacks:
- **Inputs (writeup):** NASADEM is read through Earth Engine as `NASA/NASADEM_HGT/001`, at 30 m pixels and **64x64 patches** (about 2 km). Sentinel-2 is the 206x206 patch, so the two modalities have different pixel grids for the same footprint.
- **Model choice (writeup):** the team says "several different machine learning models/approaches were evaluated and Resnet18 showed an advantage". Each model is >70% on its validation set. There is no held-out test and no comparison with a published baseline. The intro motivates the work against a Random Forest paper whose soil predictors are about 1 km resolution, arguing that predictors that coarse cause false negatives.
- **OpenAI role:** **ChatGPT** writes "a brief description" per candidate on the explorer site. No OpenAI model sits in the detection pipeline.
- **Self-stated limits (writeup):** the training labels were "not supervised by team members" for lack of archaeological knowledge. Soil and climate bands are the next step.
- **No ground-truth validation:** each of the 67 candidates was inspected by eye, with no field or published-lidar check.

### 2nd: Commencis, GlyphTrack (bank covers in depth)
See bank §3 for the GEDI-vs-DEM residual `Pglyph`, the Positive-Unlabelled LightGBM on 3 km tiles, and the 20 km super-tiles (AUC about 0.95, AP about 0.84, writeup). Facts the bank lacks:
- **The writeup is short (about 425 words).** The reproducible detail is in the linked notebook `ceeluna/glyphtrack-detection-of-anomalies-under-canopy` (not pulled here).
- **Where GPT appears:** **twice**. It does the triage that picks "the most promising tile" after thresholding, and it interprets the fused table plus annotated DEM with "an archaeology-focused prompt". The writeup never names the GPT version.
- **Framing (writeup):** airborne lidar "can't be an available solution sometimes, as we experienced", so the team treats canopy-induced DEM error as the signal. It is the only winner that needs no airborne lidar and no CNN.
- **Scope:** the state of Acre (Brazil). The writeup gives no count of candidate points.
- Commencis also published a company blog post on GlyphTrack: https://www.commencis.com/thoughts/commencis-ais-glyphtrack-using-ai-to-rediscover-lost-amazonian-civilizations/ (seen in search results, not read).

### 3rd: mxsx (Maja Somrak), Amazonia Archaeology Discovery (bank covers in depth)
See bank §3 for the Chactun-to-Amazon semantic-segmentation transfer. Facts the bank lacks:
- **Data sizes (writeup):**
  - Chactun: **2,094 tiles** of 480x480 px (240 m at 0.5 m). By class, 1,343 tiles contain buildings, 1,131 platforms and only 78 aguadas, so the aguada tiles were augmented by rotate, mirror and scale. After dropping tiles suspected of unlabelled structures (terraces, causeways), **1,303 tiles** remain, and the team trained on both the full and the filtered set.
  - Amazon pilot: Tapajós National Forest lidar/DTM (ORNL DAAC ds 1514, 1.2 GB).
  - Discovery: NASA Sustainable Landscapes Brazil, **3,154 DTM files, 125 GB, covering 3,125 km²**.
- **Preprocessing:** the Relief Visualization Toolbox produces slope, sky-view factor and positive openness with the Maya study's exact settings. Tiles are 240x240 px at 1 m.
- **Search-area filter:** keep tiles within 10 km of rivers and lakes, then drop tiles intersecting modern activity using RAISG shapefiles.
- **Not in the writeup:** the architecture, the hyperparameters and any IoU are all deferred to GitHub ("See more on Github"), and that repo now returns 404. The writeup also never mentions an OpenAI model. How OpenAI models were used, which the rules require, is **not stated in the writeup**. It gives no candidate count.

### 4th: CloudJas, Archaios: Explore hidden world with Open AI (full depth; new)
**Problem and user.** The user is the working archaeologist with a lidar dataset. The writeup argues that today's process (raw point clouds, then elevation models, then topography, then correlation with history) is "complex, tedious, and heavily dependent on expert manual interpretation" and does not scale to large regions or concurrent projects. Archaios is a **cloud platform**, not a discovery run: users upload LAS/LAZ/E57 point clouds plus field reports, journals and historical records, and the system processes, analyses and ranks them.

**Data.** User-uploaded lidar. The writeup and README name no specific public tile set, OpenTopography included. Spectral layers come from **Google Earth Engine** (NDVI, TrueColor, FalseColor composites) for each lidar dataset's coordinates. The README does not name Sentinel-2. The text corpus is **"100+ vectorized archaeological field reports"** (writeup, README). No labelled training set is built.

**Lidar pipeline (writeup).** The pipeline is classical geoprocessing, not deep learning:
- Statistical outlier removal.
- When points are unclassified, **progressive-densification ground classification** (grid cells, lowest points as seeds, configurable 0.5 to 2.0 m cell size and slope thresholds).
- Adaptive grid interpolation at 0.5 to 2.0 m, distance-transform void filling, and optional 3x3 median and Gaussian smoothing.
- Micro-relief enhancement: a **multi-scale Local Relief Model** (sigma 1 to 5), a **white top-hat** morphological filter and Laplacian sharpening.
- DSM with density-adaptive resolution and E57 artifact handling.
- Hillshade with controllable azimuth and altitude, an **8-direction composite**, and a 2 to 98% percentile stretch.
- Slope clipped to 0 to 60° by default.
- CRS auto-detection with fallbacks.

More than 20 parameters are exposed in a **drag-and-drop workflow builder**, with conditional branching, presets and an exportable audit trail. That last item is their reproducibility answer.

**Architecture.** Azure throughout:
1. Blob uploads (big files chunked) fire HTTP triggers.
2. **Durable Functions** orchestrate everything. They extract metadata, launch **Container App Jobs** running Python lidar/raster/GEE processors, wait for an external completion event, then fan out to a GEE sub-orchestrator.
3. The README lists .NET 8, Python 3.10+, Neo4j (the "graph relationships" layer), Docker, and the MIT license.

**OpenAI role and agents.** The LLM is the analyst, not the detector:
- A **pre-analysis Image Analyzer** looks at the DTM, DSM, hillshade, NDVI and spectral outputs to flag areas of interest.
- **Microsoft Semantic Kernel multi-agent orchestration** then runs a "virtual archaeological team": Archaeological Analyst, Terrain Specialist, Environmental Expert and Team Coordinator. It produces a chronological discussion timeline and a **final consensus on "archaeological authenticity"**.
- Agents pull historical context from a RAG layer: **OpenAI embedding models**, chunked documents, and **Cosmos DB DiskANN** vector search. The same layer powers a researcher chat with citation tracking.
- Neither the writeup nor the README names a specific OpenAI chat model ("OpenAI language models for reasoning and summarization").

**Output and evaluation.** Results combine topographic and spectral evidence into a **0 to 100% confidence score**, which rises when a feature appears in both layers. That is their version of the host's two-independent-methods checkpoint. A gamified leaderboard lets contributors submit discoveries. **No candidate count, no coordinates in text, and no validation against known sites or a baseline.** The "Findings" section is a single hillshade image. **No video link** appears in the writeup links. There are three screenshots and a live Azure app URL.

**Why it placed (inference, not stated by judges):** it was the most complete reproducible *tool* in the field, with an auditable workflow and an open MIT repo. The evidence-depth axis is its weakest.

### 5th: GnosisPoet, TerraReveal (full depth; new; writeup is thin)
**The writeup is thin: about 340 words.** It points to an attached Markdown report ("🌿 TerraReveal_AI-Powered Discovery of Hidden Amazonian Archaeological Sites.md") that the Writeup API does not expose. Its link list has only a YouTube video, two carousel images ("TerraReveal UI", "TerraReveal Discovery PDF") and the app URL. Searches on the web and on GitHub found no repo.

**Claims (writeup):**
- **Region:** the **Peru** Amazon basin. Note that the challenge brief focuses on Brazil.
- **Results:** **12 high-confidence anomalies**, with two highlighted:
  - Site 5 at (-11.7123, -69.5678), "large elevated earthwork", **0.90 confidence**.
  - Site 4 at (-11.3567, -69.2134), "geometric settlement complex", **0.85**.
- **Models:** **GPT-4.1** for archaeological context interpretation, **o3 and o4-mini** for "spatial and geometric reasoning", a **CNN** for pattern detection, and **XGBoost** for ensemble classification and confidence scoring.
- **Data:** Sentinel-2 at 10 m, NASA SRTM at 30 m, and OpenTopography lidar "for cross-validation".
- **Method:** cross-modal analysis (optical, elevation, vegetation), "cryptographic hash logging for reproducibility", and all anomalies "validated within ±50m spatial tolerance".
- **Platform:** CC0 and open-source. The web UI uses React and Mapbox, and exports GeoJSON, CSV and JSON reports. The subtitle claims "detects undocumented pre-Columbian earthworks with 90% confidence".

**What the live app shows (JS bundle read 2026-10-28):** this is new evidence, and partly contradicts the writeup.
- The app is a Lovable-generated React app with a Supabase backend.
- Its methodology export lists `GPT-4.1-2025-04-14 for archaeological analysis`, "OpenAI embedding models for pattern recognition" and "Multi-model ensemble for confidence scoring".
- The UI's model panel lists GPT-4.1, **OpenAI o3** ("multi-step reasoning for complex site validation") and **GPT-4o-mini** ("fast processing for anomaly detection"). It does not list o4-mini.
- The data-source records give SRTM v3.0 (DOI 10.5066/F7PR7TFT) and Sentinel-2 (ESA Copernicus DOI). These satisfy the two-public-sources gate.
- A `loadTwoRealDataSources` routine fetches SRTM through the OpenTopography raster endpoint for the bbox lat -12 to -11, lon -70 to -69, which contains both headline sites, and falls back to Landsat 8.
- **The string "XGBoost" does not appear in the bundle.** It may run server-side, but that cannot be verified.
- Parts of the dashboard are visibly hardcoded demo content, such as an activity feed with fixed confidences (87.3, 94.1, 78.9) and an "o3-mini" label, and a "2.4GB processed" preview string. Treat the app as a UI shell, not as evidence for the pipeline.

**Evaluation.** The writeup's confidences come from the team's own scoring (writeup). There is no labelled test, no baseline and no comparison against known Peruvian sites. The "±50m validation" is undefined: validated against what is not stated.

**Takeaway for the series:** 5th place went to the entry that leaned hardest on the OpenAI models as analysts (GPT-4.1, o3, o4-mini named) and on a polished interactive platform. The top three used LLMs least and classical detectors most.

## Feature matrix (Y / N / ?)

| Row | 1st Black Bean | 2nd GlyphTrack | 3rd mxsx | 4th Archaios | 5th TerraReveal |
|---|---|---|---|---|---|
| OpenAI model named | ChatGPT (version ?) | "GPT" (version ?) | none in writeup | "OpenAI LMs" + embeddings | GPT-4.1, o3, o4-mini (app: + GPT-4o-mini) |
| OpenAI role: analyst (interprets/triages) vs in-pipeline detector | describer only | triage + interpreter | ? | multi-agent analyst + RAG | analyst + "spatial reasoning" |
| Fine-tuned / trained own model | Y (2x ResNet-18) | Y (PU-LightGBM + robust regression) | Y (segmentation, few-shot FT) | N | ? (CNN + XGBoost claimed, untrained details) |
| CV / DL detector | Y (CNN classifier) | N (GBM + residual stats) | Y (semantic segmentation) | N (classical relief filters + LLM vision) | ? (CNN claimed) |
| Data: airborne/ALS lidar | N | N | Y (Sustainable Landscapes 1 m) | Y (user uploads) | claimed "for cross-validation" |
| Data: Sentinel-2 | Y | N | N | ? (GEE composites, sensor unnamed) | Y |
| Data: SRTM / NASADEM / global DEM | Y (NASADEM) | Y (DEM, product ?) | N (DTM from lidar) | N | Y (SRTM) |
| Data: GEDI | N | Y | N | N | N |
| Data: historical texts / reports | N | N (GPT "mythological knowledge") | N | Y (100+ field reports, RAG) | N |
| Own labelled dataset built | Y (4,067+4,067) | Y (PU labels, Acre) | Y (hand-labelled Tapajós tiles) | N | ? |
| Number of candidate sites | 67 | ? (one top tile) | ? | ? | 12 |
| Ground-truth validation (field / independent) | N | N | N | N | N (undefined ±50 m) |
| Two independent modalities fused | Y (DEM x S2 pairing) | Y (micro x landscape) | N (single lidar modality) | Y (topo x spectral score) | Y (claimed) |
| Eval rigour | small quantitative (val acc >70%) | small quantitative (AUC/AP) | ? (in 404 repo) | none | none (self-scored) |
| Video | Y | N | N | N | Y |
| Live demo | Y (explorer) | N (notebook) | N | Y | Y |
| Open code | Y | Y (notebook) | was Y, now 404 | Y (MIT) | N (claims CC0, no repo found) |

**Counted summary:**
- 3 of 5 trained a dedicated detector, and those are exactly places 1 to 3.
- 5 of 5 used OpenAI models outside the detection core, or did not document them at all (mxsx).
- 4 of 5 fused two modalities; mxsx is the exception.
- 0 of 5 report field or independent ground-truth validation.
- 2 of 5 give a candidate count (67, 12).
- 1 of 5 used historical texts (Archaios).
- 1 of 5 used GEDI (GlyphTrack).
- 3 of 5 have a live demo; 2 of 5 have a video.

## Consensus vs divergence
- **Table stakes:**
  - Open, citable data: every winner names public datasets, which is the pass/fail gate.
  - A map- or overlay-centred presentation.
  - OpenAI models used as an interpretation layer, never as the detector.
  - No winner did field verification.
- **What separated 1st to 3rd from 4th to 5th:** the top three each **trained a model on real labels** (published site catalogues, PU labels, Maya expert masks) and could show a quantitative check (validation accuracy, AUC/AP, segmentation transfer). 4th and 5th built **platforms** that use GPT/o-series models to reason over imagery, with no labelled evaluation. Judges valued "potential archaeological impact" and retraceable evidence (602618) over LLM breadth.
- **Divergences:**
  - Lidar is essential ([mxsx](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/amazonian-archaeology-discovery) argues the 30 m DEMs are too coarse for features under 30 m) *vs* lidar is often unavailable, so work around it ([GlyphTrack](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/glyphtrack-detection-of-anomalies-under-canopy) uses GEDI plus DEM error, and [Black Bean](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/amazon-archeological-site-discovery-a-deep-learnin) uses 30 m NASADEM successfully).
  - Basin-wide supervised search (Black Bean) *vs* one-state focus (GlyphTrack, Acre) *vs* Peru (TerraReveal) *vs* bring-your-own-region tooling (Archaios).
  - The LLM as an orchestrated multi-agent "expert panel" ([Archaios](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/archaios-explore-hidden-world-with-open-ai)) *vs* the LLM as a one-line describer ([Black Bean](https://www.kaggle.com/competitions/openai-to-z-challenge/writeups/amazon-archeological-site-discovery-a-deep-learnin)). The describer won.

## UNVERIFIED
- 4th and 5th prize amounts: the API shows none. The livestream verdicts and any judge comments per finalist were not found.
- TerraReveal's attached Markdown report, its CNN architecture, its XGBoost features and its "±50 m validation" target were not retrievable. XGBoost is absent from the client bundle.
- Archaios: which OpenAI chat model is used, and how many sites it flagged.
- mxsx: the segmentation architecture, the hyperparameters, any IoU, and the OpenAI usage (GitHub 404).
- GlyphTrack: the GPT version and the DEM product; the notebook was not pulled.
- Whether any candidate from any winner has since been field-verified: no source found.
