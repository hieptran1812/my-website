---
title: "BigQuery AI Hackathon on Kaggle: what all 11 winners built with LLM functions inside SQL"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "Google's 100,000 USD BigQuery AI hackathon asked 276 teams to solve a business problem with Gemini, embeddings and multimodal tables called straight from SQL. A walk through the weighted rubric, the functions the winners used, a feature matrix across all 11 awarded entries, and why breadth beat measured accuracy."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "hackathon", "bigquery", "vector-search", "rag", "gemini", "multimodal", "sql"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 23
---

> **TL;DR.** BigQuery AI: Building the Future of Data (Kaggle, 12 August to 22 September 2025) paid 100,000 USD for projects that solve a real business problem with large language models called from inside SQL. Three categories (Generative AI, Vector Search, Multimodal) each paid 15k, 9k and 6k USD, plus two 5k Honorable Mentions: 11 awarded entries out of 276 teams. **Nobody fine-tuned anything**: all 11 used hosted Gemini or an embedding model through BigQuery, and the edge was **breadth of the sponsor's toolbox wired to one believable user**, with a dollar number attached. The three 1st places are the broadest builds and report no quality metric; the two most rigorous evaluations finished 2nd. **The one lesson:** in a sponsor hackathon the rubric is the leaderboard, and 30 of its 110 points need no model at all.

![The hackathon on one card: 41 days, 276 teams, three categories with their SQL functions, and a 100,000 USD pool drawn as coin stacks to scale](/imgs/blogs/kaggle-bigquery-ai-hackathon-winners-1.webp)

The card above is the whole contest. There is no hidden test set and no score: a panel read every submission, re-ran every winner, and paid for the project that best showed a data warehouse with Gemini living inside it. The host awarded exactly 11 entries and published no rubric scores, so this post covers **every prize winner** and invents nothing beyond them. Every row count, hour saved, ROI and recall figure below is **the team's own claim** from its writeup unless marked as the host's.

The [gpt-oss red-teaming post](/blog/machine-learning/learn-from-kaggle/kaggle-gpt-oss-red-teaming-challenge-findings) shows the other face of a judged Kaggle event, where the panel rewarded explaining a failure; here it rewarded a working product story. Siblings in this wave cover the [Gemma impact hackathons](/blog/machine-learning/learn-from-kaggle/kaggle-gemma-impact-hackathons-winners), [OpenAI to Z](/blog/machine-learning/learn-from-kaggle/kaggle-openai-to-z-challenge-winners), the [Tunix hack](/blog/machine-learning/learn-from-kaggle/kaggle-tunix-hack-reasoning-winners), the [early Gemma and Gemini hackathons](/blog/machine-learning/learn-from-kaggle/kaggle-gemma-gemini-early-hackathons-winners) and [Measuring AGI](/blog/machine-learning/learn-from-kaggle/kaggle-measuring-agi-cognitive-abilities-winners).

## 1. The brief in 60 seconds

As a friend would put it: "Google wanted proof that you can run a real AI product without leaving the warehouse. Pick a business problem, write SQL that calls Gemini, show it saves money, and publish everything so they can run it."

Formally, from the Overview and Evaluation pages and the host's winners thread:

| Item | Value |
| --- | --- |
| Dates | 2025-08-12 to 2025-09-22, deadline 23:59 UTC (41 days) |
| Hard rule | Use at least one of three approaches: Generative AI, Vector Search, Multimodal |
| Generative AI | `AI.GENERATE`, `AI.GENERATE_BOOL`, `AI.GENERATE_TABLE` and friends, `ML.GENERATE_TEXT` |
| Vector Search | `ML.GENERATE_EMBEDDING`, `VECTOR_SEARCH`, optionally a vector index |
| Multimodal | Object Tables over files in Cloud Storage, `ObjectRef` columns |
| Deliverable | A Kaggle writeup plus public code, and ideally a blog or video, an architecture diagram and a friction-feedback survey |
| Entrants | 276 teams, 276 submissions, 344 competitors |
| Prizes | 3 categories x (15,000 + 9,000 + 6,000 USD) + 2 Honorable Mentions x 5,000 USD |

The prize arithmetic checks out (3 x 30,000 + 2 x 5,000 = 100,000 USD), and 11 awards from 276 teams is a 4.0% hit rate. There is no notebook budget: compute was whatever your cloud project paid for, and the binding constraint was that the judges had to be able to run it.

## 2. The judging rubric, with weights

![The rubric drawn to scale: a 110-point bar split into ten sub-criteria, coloured by whether a line needs a working system, a story with a number, or only an artifact](/imgs/blogs/kaggle-bigquery-ai-hackathon-winners-2.webp)

This rubric publishes weights, with explicit 0, 5, 10 and 15 anchors per line. Notice how much of the bar is not about the model.

| Block | Sub-criterion | Points |
| --- | --- | --- |
| Technical implementation (35) | Clean, working code | 20 |
| | Core use of BigQuery AI | 15 |
| Innovation and creativity (25) | Novelty | 10 |
| | Impact on a metric (hours saved, revenue) | 15 |
| Demo and presentation (20) | Clarity of problem and solution | 10 |
| | Explanation plus architecture diagram | 10 |
| Assets (20) | Public blog or video | 10 |
| | Public GitHub repository | 10 |
| Bonus (10) | Feedback on BigQuery AI friction points | 5 |
| | Completed survey | 5 |

The base is 100 and the bonus takes the ceiling to 110:

$$
\underbrace{35}_{\text{technical}} + \underbrace{25}_{\text{innovation}} + \underbrace{20}_{\text{demo}} + \underbrace{20}_{\text{assets}} + \underbrace{10}_{\text{bonus}} = 110
$$

Group the lines by what earns them and three piles appear. **A working system** earns 35 (clean code 20, core BigQuery AI use 15). **A story with a number** earns 45 (novelty 10, metric impact 15, clarity 10, explanation and diagram 10). **An artifact or a form** earns 30 (blog or video 10, GitHub 10, friction feedback 5, survey 5). That last pile is 30 of 110 points, 27%, and none of it depends on how good your model is. The host said as much in the winners thread: "many of the non-winning submissions were missing multiple artifacts".

### How the judges actually judged

The host's announcement ([612730](https://www.kaggle.com/competitions/bigquery-ai-hackathon/discussion/612730)) describes the process, and it is stricter than most hackathons:

- **Every submission was read by a human**, some several times, and vetted by more than one judge.
- **Submissions missing artifacts were filtered** after checking, as was anything without BigQuery AI and anything not publicly accessible.
- **Every winning submission was replicated by the judges**, "sometimes making fixes or making the necessary DDL". Entries with "too many glaring holes" were passed over.
- Some strong entries were **dropped over responsible-AI concerns**.
- Two unpaid "Commendable Noteworthy Submissions" (QualityNexus, Semantic Song Finder) took no prize and are not covered here.

Replication turns "clean, working code" from a style point into a gate: if a judge cannot rebuild your tables, the other 90 points never get counted.

## 3. BigQuery AI in five SQL shapes

Every winner combines a handful of SQL shapes; learn these five and the feature matrix in section 6 reads itself. The snippets are **illustrative toys**: names are invented, they were **not run against BigQuery** for this post, and the syntax follows Google's documentation at contest time, which keeps moving. They are a map of the shapes, never any team's code.

![Toy rows invented for illustration, modeled on TriLink's follow-up workflow: a calls table where each SQL function appends one new column per row](/imgs/blogs/kaggle-bigquery-ai-hackathon-winners-3.webp)

A table goes in, each function call adds a column, and the output is still a table you can join, filter and chart. Nothing leaves SQL.

**Shape 1, a yes/no question per row.** `AI.GENERATE_BOOL` sends one prompt per row and returns a boolean, which is exactly what a triage `WHERE` clause wants. TriLink used it to flag calls that need a follow-up; CLARIS used it to raise insurance alerts.

```sql
-- Illustrative toy, not run against BigQuery. Table and connection names invented.
SELECT
  call_id,
  AI.GENERATE_BOOL(
    ('Does this customer need a follow-up call? Transcript: ', transcript),
    connection_id => 'us.gemini_conn',
    endpoint => 'gemini-2.5-flash'
  ).result AS needs_followup
FROM support.calls
WHERE call_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY);
```

**Shape 2, structured extraction.** `AI.GENERATE_TABLE` takes a prompt per row and an output schema and returns typed columns: patent knowledge graphs, 50-field financial extractions, wildfire incident tables.

```sql
-- Illustrative toy, not run against BigQuery.
SELECT *
FROM AI.GENERATE_TABLE(
  MODEL `demo_project.support.gemini_flash`,
  (SELECT ticket_id,
          CONCAT('Classify this support ticket: ', description) AS prompt
   FROM support.tickets),
  STRUCT('device STRING, urgency STRING, need_on_site_visit BOOL' AS output_schema)
);
```

**Shape 3, embed then search.** `ML.GENERATE_EMBEDDING` writes a vector column; `VECTOR_SEARCH` joins a query table to a base table by distance. This pair is the core of retrieval-augmented generation done in SQL, the same pattern the [RAG system design post](/blog/machine-learning/large-language-model/how-to-build-effective-rag-system) builds in Python.

```sql
-- Illustrative toy, not run against BigQuery.
CREATE OR REPLACE TABLE kb.solution_emb AS
SELECT *
FROM ML.GENERATE_EMBEDDING(
  MODEL `demo_project.kb.text_embedder`,
  (SELECT solution_id, solution_text AS content FROM kb.solutions),
  STRUCT('RETRIEVAL_DOCUMENT' AS task_type)
);

SELECT query.call_id, base.solution_id, distance
FROM VECTOR_SEARCH(
  TABLE kb.solution_emb, 'ml_generate_embedding_result',
  (SELECT call_id, ml_generate_embedding_result FROM support.call_emb),
  top_k => 3, distance_type => 'COSINE'
);
```

**Shape 4, files as rows.** An Object Table is an external table whose rows are files in Cloud Storage (PDFs, audio, images). Combined with a multimodal Gemini call, a folder of 400 patent PDFs becomes a table you can `SELECT` from.

```sql
-- Illustrative toy, not run against BigQuery.
CREATE EXTERNAL TABLE docs.patent_pdfs
WITH CONNECTION `us.gemini_conn`
OPTIONS (object_metadata = 'SIMPLE', uris = ['gs://demo-bucket/patents/*.pdf']);
```

**Shape 5, the rest of the warehouse.** XGBoost with `ML.EXPLAIN_PREDICT`, K-Means, `AI.FORECAST` (TimesFM), `ML.TRANSCRIBE` for speech, `ML.PROCESS_DOCUMENT` for layout parsing, and BigFrames, a pandas-like Python API over all of it.

## 4. What made it hard

Three things, none of them a modelling problem.

**Replication by strangers.** A private bucket or an undocumented setup step is a project the panel cannot score. TriLink shipped all its datasets in the repository "so judges need not re-simulate", plus three setup notebooks: a design decision driven entirely by the judging.

**No data.** Enterprise data is private. Winners built synthetic data (TriLink's telco), collected public documents (wildfire reports, 10-K filings, ESG PDFs), or used a BigQuery public dataset (Google Patents, Open Targets, NOAA storms, Labeled Patents).

**Proving impact without users.** A six-week project has no production traffic. Most winners answered the 15-point metric line with a back-of-envelope ROI model; the few who measured quality were, as section 6 shows, not rewarded as much as you might expect.

## 5. The winners table: all 11

The host announced places within each category. The order below follows the host thread.

| Category, place (USD) | Team | Project | What it is, in one line | Source |
| --- | --- | --- | --- | --- |
| Generative AI 1st (15k) | David Babu | TriLink | A synthetic telco with five SQL-native workflows and an agent on top | [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/trilink-architecting-the-ai-native-digital-service) |
| Generative AI 2nd (9k) | Veysel Serifoglu | The AI Patent Analyst | Patent PDFs to a component knowledge graph to semantic search | [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/the-ai-patent-analyst) |
| Generative AI 3rd (6k) | intellitrend | ESG and Financial Reports Insights Agent | Annual and ESG PDFs to a 50-field table and 5-year forecasts | [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/new-writeup-1757758945521) |
| Vector Search 1st (15k) | SpeakAura.AI | SpeakAura AI | Speech therapy for stammering: transcribe, score, find similar sessions | [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/ai-powered-speech-therapy-for-stammering-speakaura) |
| Vector Search 2nd (9k) | ReDrugAI | ReDrugAI | Drug repurposing by mechanism-of-action similarity on Open Targets | [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/redrugai0421) |
| Vector Search 3rd (6k) | laoyu84 | Causal RAG | 10-K "why" questions answered by generated code over a causal graph | [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/causal-rag) |
| Multimodal 1st (15k) | MASHUP | OncOmix AI | Multi-omics for 500 breast-cancer patients joined to PubMed retrieval | [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/oncomix-ai) |
| Multimodal 2nd (9k) | seyyedreza madani | Grid Incident RAG | 153 wildfire incident reports to RAG plus extracted incident tables | [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/grid-incident-rag) |
| Multimodal 3rd (6k) | Illya Moskvin, Jacob Bridges | Auto-Updating Documentation | Slack, Zoom and change requests folded into living docs | [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/auto-updating-documentation) |
| Honorable Mention (5k) | sl02 | Patent Intelligence | Semantic search over 2.9M patents with explainable matches | [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/patent-intelligence-semantic-search-for-innovation) |
| Honorable Mention (5k) | Tiyab K. | CLARIS | County-level storm risk narratives and alerts for insurers | [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/county-level-augmented-risk-intelligence-system-cl) |

One curiosity: Auto-Updating Documentation opens its writeup by targeting the Generative AI track, and the judges placed it 3rd in Multimodal. The panel assigned categories itself.

## 6. The landscape: a feature matrix across all 11

![Feature matrix: the 11 winners as columns grouped by category, 13 features as rows, filled, empty or unknown cells, and an N of 11 count per row](/imgs/blogs/kaggle-bigquery-ai-hackathon-winners-4.webp)

Cells come from each team's writeup (or its linked notebook where the writeup was thin); a question mark means it does not say. Recounted from the cells: **0 of 11 fine-tuned a model**; every winner called hosted Gemini or an embedding model through BigQuery, except Patent Intelligence, which embedded offline with a small open model. **9 of 11** made a generation call (ReDrugAI unclear) and **9 of 11** used `VECTOR_SEARCH`, including 4 of the 6 Generative AI and Multimodal placings. **5 of 11** used `AI.GENERATE_TABLE`, **6 of 11** ingested files through Object Tables or `ObjectRef` (PDFs in four), and only **2 of 11** each used BigFrames, `AI.FORECAST` or a tool-using agent. All 11 published code, 7 a video, 4 a live demo. Just **4 of 11** report any quantitative quality measure, and two compare against a baseline.

### Table stakes

Nearly every winner shared four things: a public repo plus an architecture diagram (the asset and explanation lines); a quantified business impact, which nearly every judges' note mentions; a pipeline kept **inside BigQuery SQL** rather than calling Gemini from Python; and three or more functions crossing category lines, so a "Vector Search" winner usually also generates text.

### What separated the 1st places

![Evaluation evidence against placement: each winner as a dot, rows for 1st, 2nd, 3rd and Honorable Mention, columns from nothing measured to a quantitative comparison against a baseline](/imgs/blogs/kaggle-bigquery-ai-hackathon-winners-5.webp)

The three category winners are the **broadest builds**: TriLink wires five workflows, six or more functions, a BigQuery ML model and an agent; SpeakAura runs six functions behind a deployed app; OncOmix claims all three approaches. And none of the three reports a quality metric. The two most rigorous evaluations, ReDrugAI's leave-one-out recall and Grid Incident RAG's 358-question judged test, both finished 2nd. Causal RAG, which compared against two baselines, finished 3rd.

That is what the rubric says: "impact on a metric" means hours saved, not recall, and breadth scores on core use and novelty at once. The honorable mentions are the narrowest users of BigQuery AI.

### Where the winners contradict each other

- **Native versus external embeddings.** Most used `ML.GENERATE_EMBEDDING`; Patent Intelligence found it too slow at 2.9M rows (section 9). Everyone else embedded thousands of rows, not millions.
- **Free prompts versus schemas.** TriLink deliberately used plain `AI.GENERATE` over `AI.GENERATE_TABLE` for coaching "for more control on the prompt design"; the Patent Analyst and ESG built their core on `AI.GENERATE_TABLE` schemas.
- **Model size.** Causal RAG argues Gemini 2.5 Flash with code generation matches 2.5 Pro and is more stable; ESG chose 2.5 Pro for extraction.

## 7. The 1st places in depth

### TriLink (Generative AI 1st, David Babu)

![TriLink as five lanes off one warehouse: follow-up, coaching, churn, ticket triage and an ADK agent, each a short chain of BigQuery AI functions](/imgs/blogs/kaggle-bigquery-ai-hackathon-winners-6.webp)

**Problem.** Call-centre agents, managers and field technicians whose useful data (transcripts, tickets, device photos) is unstructured; the warehouse as the "central nervous system" of an AI-native company.

**Data.** Entirely synthetic: a fictional Ontario telco with 100,000 customers across mobile (about 75k), internet (about 75k) and home security (about 42k) tables. Gemini 2.5 Flash generated thousands of call transcripts from customer profiles and Flash-Lite rated them. Two knowledge bases hold 175 business problems and 100 technical issues with their fixes.

**Five actions, all SQL in one notebook**, as the lanes in the figure show:

1. *Proactive follow-up:* `AI.GENERATE_BOOL` triages yesterday's transcripts, embeddings plus `VECTOR_SEARCH` find similar resolved calls and knowledge-base fixes, and `AI.GENERATE` writes a personalised email. The team reports 370 of about 960 weekly calls flagged (39%) and about 0.0055 USD per email.
2. *Agent coaching:* `VECTOR_SEARCH` finds the three most similar calls handled by high performers (rating above 6), and prompts extract coaching notes.
3. *Churn retention:* a BigQuery ML XGBoost model's `ML.EXPLAIN_PREDICT` attributions become a plain-language problem statement, matched to the solutions base and turned into a retention email.
4. *Ticket triage:* an `ObjectRef` over the customer's device photo plus the text description goes into `AI.GENERATE_TABLE` with a schema (device, hardware or software, urgency, `need_on_site_visit`, technician report, do-it-yourself steps). On 100 sample tickets, the team reports 62% needed no visit.
5. *Agent:* a Google Agent Development Kit root agent routes to a call-centre assistant and a technician assistant, each with function tools wrapping the workflows above.

**Evaluation.** None on output quality; impact is an assumption-driven ROI per action (6.4M USD a year in avoided truck rolls, 420k USD in retention, 2,600 to 5,200 manager hours). The judges cited "a very clear impact for each one of the processes". The lesson is that each function is attached to a named person doing a named job, so breadth reads as a product rather than a feature tour.

### SpeakAura AI (Vector Search 1st, shafi5110 and biddahari)

**Problem.** People who stammer (the writeup cites about 70M worldwide, 11 to 12M in India) and their therapists, for whom therapy is costly and progress is hard to measure.

**Pipeline.** Audio goes to Cloud Storage and an Object Table, then `ML.TRANSCRIBE` with the Chirp model produces word-level transcripts. Disfluency markers (pauses, repetitions, prolongations, fillers, severity) are computed in SQL and Python, and `VECTOR_SEARCH` over transcript embeddings returns similar past sessions with their therapy plans. `AI.GENERATE` with an output schema writes therapy plans and read-aloud scripts. Therapy PDFs go through `ML.PROCESS_DOCUMENT` to ground a chat assistant, and `AI.FORECAST` projects a fluency score (100 minus 100 times severity) for a progress dashboard on Cloud Run.

**Evaluation.** None: no detection accuracy, no retrieval metric, no user study; the headline is "reduces therapist prep time by 40%+". SpeakAura is the clearest case of a sympathetic user plus six functions plus a live URL carrying a category.

### OncOmix AI (Multimodal 1st, team MASHUP)

**Problem.** Oncology clinicians and researchers who must connect a patient's multi-omic profile to a fast-moving literature, and who distrust ungrounded AI.

**Data.** The TCGA breast-cancer cohort of 500 patients (clinical, RNA expression, mutation, copy number, protein, methylation, treatment) plus "thousands" of PubMed paper summaries, cleaned and joined with BigFrames. "Multimodal" here means structured omics plus unstructured text, not images or audio.

**Pipeline.** Abstracts are embedded with BigFrames' `TextEmbeddingGenerator`. Each patient's profile becomes a query such as "breast cancer, ER positive, HER2 negative, PIK3CA mutation" and runs through `VECTOR_SEARCH`; hits are tagged strong at 75% similarity or more and moderate from 60 to 74%. Gemini, via `GeminiTextGenerator`, writes a structured summary with verbatim cited evidence. A Literature Validation Score crosses cohort biomarker frequency with literature footprint, separating "ready now" markers (ERBB2, ESR1, BRCA1/2) from a watch list (TP53, PTEN) and under-explored ones (PD-L1, LAG3, CTLA4). Model versions are not named.

**Evaluation.** Qualitative: three illustrative patient summaries. The authors call it a research prototype and caveat that the immune-checkpoint "gap" may be an artefact of their limited literature slice, honesty that makes a medical demo credible.

## 8. The 2nd places: where the measurement lives

### The AI Patent Analyst (Generative AI 2nd, Veysel Serifoglu)

**Problem.** IP teams reading patent PDFs by hand to find functionally similar components.

**Data.** 403 PDFs (197 English, the other 206 French or German) from Google's public Labeled Patents bucket, joined to the matching BigQuery public table.

**Pipeline, all in BigQuery:** an Object Table over the PDFs; multimodal `ML.GENERATE_TEXT` with Gemini 2.5 Flash extracts title and summary and describes the drawings; `AI.GENERATE_TABLE` with an "expert analyst" prompt builds a nested component, function and interconnection graph; then `gemini-embedding-001` produces **two vectors per component**, one for what the component does and one for the patent it sits in, blended by a SQL function with 0.7 weight on function and 0.3 on context, then searched with `VECTOR_SEARCH`.

The blend is the one genuinely clever retrieval idea in the field, and it is easy to see why with a toy:

```python
# Illustrative toy of the Patent Analyst's dual-embedding blend (0.7 function,
# 0.3 context) followed by a cosine nearest-neighbour search. Invented 3-d vectors;
# the real system used gemini-embedding-001 inside BigQuery. NOT the team's code.
import numpy as np

def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)

def blend(function_vec, context_vec, w_function=0.7):
    return unit(w_function * unit(function_vec) + (1 - w_function) * unit(context_vec))

components = {   # (what the part does, what the whole patent is about)
    "hinge in a folding phone": blend([0.9, 0.1, 0.0], [0.1, 0.9, 0.1]),
    "hinge in a car door":      blend([0.9, 0.0, 0.1], [0.0, 0.1, 0.9]),
    "battery in a phone":       blend([0.0, 0.2, 0.9], [0.1, 0.9, 0.1]),
}
query = unit([1.0, 0.1, 0.05])   # "a part that lets two rigid panels rotate"
for name, vec in sorted(components.items(), key=lambda kv: -float(kv[1] @ query)):
    print(f"{float(vec @ query):.3f}  {name}")
```

```console
0.931  hinge in a folding phone
0.902  hinge in a car door
0.133  battery in a phone
```

The car-door hinge lands next to the phone hinge because 70% of each vector describes what the part does, and the battery, which shares the phone hinge's patent context, falls far behind. That is the behaviour an engineer hunting for prior art across industries wants. (The math of why a weighted sum of unit vectors behaves this way is in the [geometry of embeddings post](/blog/machine-learning/mathematics/vectors-dot-products-and-the-geometry-of-embeddings).)

**Evaluation.** No retrieval metric. The claim is 605 expert-hours and over 90,000 USD manual against 15 minutes and under 20 USD, "over 4,500x ROI" (90,000 / 20 = 4,500, so it holds at the stated bounds). Recomputed throughput: 403 in 15 minutes is 1,612 an hour; the manual 403 over 605 hours is 0.67, where the writeup says about 0.6.

### ReDrugAI (Vector Search 2nd, Ting-Wei Chu and Hung Yu Hsiao)

**Problem.** Find approved drugs with known safety that might treat another disease, especially rare diseases with few treatments.

**Data.** The Open Targets Platform in its BigQuery public dataset form: drugs, targets, diseases, mechanisms of action (MoA) and known drug associations.

**Method.** Disease names, synonyms and descriptions are embedded with `text-embedding-005` (768 dimensions) behind a vector index, and `VECTOR_SEARCH` keeps diseases within cosine distance 0.3 of the target. MoA descriptions are embedded too, with pairwise similarities precomputed. A candidate's base score is its best MoA similarity to known treatments, multiplied by a tier: x1 for MoA only, x2 for a shared binding target, x5 for an exact MoA and target match. The final score adds half the mean score from similar diseases:

$$
\text{final}(d) = s_{\text{primary}}(d) + 0.5 \cdot \frac{1}{K}\sum_{k=1}^{K} s_{\text{similar},k}(d)
$$

A toy with invented numbers shows how the tiers and the similar-disease term interact:

```python
# Illustrative toy of ReDrugAI's published scoring rule. NOT the team's code:
# the drugs, similarities and targets below are invented to show the arithmetic.
TIER = {"moa_only": 1.0, "shared_target": 2.0, "exact_moa_and_target": 5.0}

def candidate_score(max_moa_sim, tier):
    return max_moa_sim * TIER[tier]          # base in [0, 1] times tier multiplier

primary = {"drug_A": candidate_score(0.62, "moa_only"),
           "drug_B": candidate_score(0.41, "shared_target"),
           "drug_C": candidate_score(0.18, "exact_moa_and_target")}
similar = {"drug_A": [candidate_score(0.30, "moa_only"), candidate_score(0.25, "moa_only")],
           "drug_B": [candidate_score(0.10, "moa_only"), candidate_score(0.05, "moa_only")],
           "drug_C": [candidate_score(0.90, "shared_target"), candidate_score(0.70, "shared_target")]}

final = {d: primary[d] + 0.5 * sum(similar[d]) / len(similar[d]) for d in primary}
for d, s in sorted(final.items(), key=lambda kv: -kv[1]):
    print(f"{d}: primary={primary[d]:.3f}  similar_mean={sum(similar[d])/2:.3f}  final={s:.3f}")
```

```console
drug_C: primary=0.900  similar_mean=1.600  final=1.700
drug_B: primary=0.820  similar_mean=0.075  final=0.857
drug_A: primary=0.620  similar_mean=0.275  final=0.758
```

Drug C has the weakest raw similarity (0.18) and ends first: the x5 tier lifts it to 0.9 and neighbouring diseases add 0.8. The multipliers are a hand-set prior, defensible for pharmacology but not ablated in the writeup.

**Evaluation, the strongest of the 11.** A leave-one-out test on 100 randomly sampled diseases: hide the known drugs, recommend, and check exact drug-ID overlap. Recall is 21.55% for the overall top 10, 23.90% for the similar-disease top 5, and 45.44% combined; the two parts sum to 45.45, so the combined figure is their sum within rounding. The team argues the other 54.56% are candidates rather than failures, which is an interpretation, not a validated claim. Whether any generative step exists is unclear from the writeup.

### Grid Incident RAG (Multimodal 2nd, Reza Madani)

**Problem.** Cross-incident patterns hidden in 153 California Public Utilities Commission wildfire reports of 20 to 200 pages each.

**Pipeline, "everything in SQL":** an Object Table over the PDFs; Document AI layout parsing (tables, multi-column text, reading order), which succeeded on 130 of 153 PDFs (85%) and produced 3,334 chunks (the cost table counts 3,229 embedded, and the writeup gives both); `textembedding-gecko` vectors; `VECTOR_SEARCH` with metadata pre-filters on utility, year and fire-threat tier; RAG answers through `ML.GENERATE_TEXT` with citations; and `AI.GENERATE_TABLE` with a dynamic schema that extracted 126 incident records with 40-plus fields into six tables. A BigQuery ML K-Means model yields four risk profiles.

**Evaluation.** An LLM judge over 358 questions auto-generated from the extracted tables: correctness 3.76 out of 5, faithfulness 100%, citation recall 58.9%. There is no baseline, and questions generated from the system's own extraction test consistency more than truth, but nothing else in the field was evaluated at that scale. (The [LLM-as-a-judge failure modes post](/blog/paper-reading/ai-interpretability/one-token-to-fool-llm-as-a-judge) is a good companion on how far to trust such a judge.)

![Cost and time at scale, team-reported: projected in-warehouse embedding time against an offline GPU for 2.9M patents, and Grid Incident RAG's whole-project bill broken into its items](/imgs/blogs/kaggle-bigquery-ai-hackathon-winners-7.webp)

**Cost accounting.** The whole project cost 39.57 USD, of which Document AI was 32.50 for 2,600 pages, evaluation 1.34 and embeddings 0.10; the itemised parts sum to 33.94, so 5.63 USD sits in items our source does not break out. That is about 0.004 USD per query. The right panel of the figure makes the point: parsing, not the LLM, is the bill.

## 9. The 3rd places and the honorable mentions

**ESG and Financial Reports Insights Agent (Generative AI 3rd, intellitrend).** Annual and sustainability PDFs go through an Object Table into `ML.GENERATE_TEXT` with Gemini 2.5 Pro at temperature 0, then `AI.GENERATE` answers a block of 50-plus predefined questions, and `AI.GENERATE_TABLE` extracts a 50-plus field schema (revenue, net income, EPS, Scope 1 to 3 emissions, diversity). `AI.FORECAST` with TimesFM 2.0 gives five-year revenue forecasts; BigFrames and Looker Studio handle views. The writeup contradicts itself on its corpus (52 pharma files in one section, 32 bank reports in another), measures no extraction accuracy, and its "Agent" uses no tools.

**Causal RAG (Vector Search 3rd, laoyu84).** The target is 10-K "why" questions plain RAG fails, such as what drove Salesforce's FY2025 net-margin rise. The author hand-built, offline, a causal graph among accounting variables plus a CSV of FY2018 to FY2025 statement figures. At query time Gemini 2.5 Flash, through `ML.GENERATE_TEXT`, extracts company and year, then writes Python that prints facts only ("DO NOT make up any data"), which runs locally; Flash then acts as a "consultant" turning facts into insights, and finally writes a `VECTOR_SEARCH` query pre-filtered by company and year to fetch evidence chunks. There is no ReAct loop: the whole program is regenerated on failure. The evaluation compares against plain BigQuery RAG and Gemini 2.5 Pro with thinking on a small hand-built set, shown only as an image, so set size and scores are unverified. It pairs well with the [advanced RAG for financial documents post](/blog/machine-learning/ai-agent/advanced-rag-financial-documents) and the [graph RAG guide](/blog/machine-learning/large-language-model/graph-rag-complete-guide-with-case-studies).

**Auto-Updating Documentation (Multimodal 3rd, Illya Moskvin and Jacob Bridges).** The problem is documentation divergence: decisions made in chat never reach the docs. The design takes the last verified document and its timestamp, collects "ephemera" newer than it (Slack JSON, Zoom transcripts, change requests, PR summaries) from an Object Table, and asks `AI.GENERATE` on `gemini-2.0-flash-lite`, with a fixed seed of 12345 for run-to-run consistency, to write the next version for human review. Prompts are themselves objects joined by type, one per (document type, ephemera type) pair, so a chat can change a spec but only a PR summary changes the docs. One function and a toy dataset partly generated with Claude Sonnet 4: it won on a crisp problem statement, not breadth.

**Patent Intelligence (Honorable Mention, sl02).** The largest data in the field: 49,184,081 Google Patents rows profiled, filtered to 2.9M English patents from January to June 2024 (14.4 GB). The author tried `ML.GENERATE_EMBEDDING`, measured 48,000 rows in 6 hours (0.45 s a row), and projected about 15 days for 2.9M; the left panel of the last figure draws that against 6.5 hours for `all-MiniLM-L6-v2` on a Kaggle GPU. Recomputed: 2.9M x 0.45 s is 362.5 hours, 15.1 days, and 362.5 / 6.5 is 55.8, matching the claimed "56x faster". BigQuery then does `VECTOR_SEARCH` only. On 10 queries, semantic search returned 300 results and keyword 180 with 5 in common, and the author says plainly that without ground truth this measures difference, not quality.

**CLARIS (Honorable Mention, Tiyab K.).** County-level storm risk for insurers from two NOAA datasets in BigQuery. `ML.GENERATE_TEXT` on gemini-2.5-flash writes risk narratives; `AI.GENERATE_BOOL` raises three alerts, with a prompt trick of counting digits to judge damage magnitude (10 digits means billions) and explicit FALSE conditions to avoid alert fatigue; `AI.GENERATE_TABLE` classifies risk; embeddings and `VECTOR_SEARCH` find analogous counties. The notebook runs generation on a 100-row sample. The writeup reports no results.

## 10. What the host and the forum taught everyone

The host thread (section 2) is the most useful document in the competition; a separate thread announced a judging delay ([610964](https://www.kaggle.com/competitions/bigquery-ai-hackathon/discussion/610964)), unsurprising when every winner needs its own cloud setup to replicate. The friction-feedback bonus made teams write down what hurt, and those notes are the most reusable engineering content of the event.

## 11. What did not work

Aggregated across the 11 writeups, with who reported it:

- **In-warehouse embedding at millions of rows** (Patent Intelligence): 0.45 s a row, about 15 days projected for 2.9M rows. Offline embedding on a GPU took 6.5 hours.
- **Vector index on Standard edition** (Patent Intelligence): the index was built but query-time use needs Enterprise Plus, so search fell back to brute force.
- **`LIMIT`/`OFFSET` paging without `ORDER BY`** (Patent Intelligence): duplicated rows; fixed with `ROW_NUMBER()`.
- **Mixing embedding models of the same dimension** (Grid Incident RAG): "nonsensical results". It also reports 20% chunk overlap plus dropping low-quality chunks raised citation recall by 33%.
- **Document parsing coverage** (Grid Incident RAG): layout parsing failed on 23 of 153 PDFs.
- **Long context as a substitute for retrieval** (Causal RAG): Gemini 2.5 Pro returned internal errors when fed eight annual reports at once, which motivated the code-generation design.
- **`AI.GENERATE_TABLE` for free-form aggregation** (TriLink): plain prompts gave more control over coaching summaries.

## 12. Lessons that transfer

1. **The rubric is the leaderboard; read it literally.** 30 of 110 points are artifacts and forms. The host named missing artifacts as the most common reason good entries lost ([612730](https://www.kaggle.com/competitions/bigquery-ai-hackathon/discussion/612730)).
2. **Design for replication.** Every winner was re-run. Ship the data, the DDL and setup notebooks, as TriLink did, so a stranger can rebuild your tables.
3. **Breadth, attached to people, beats depth on one function.** The three 1st places were the broadest builds in their categories, and each tied its functions to a named user and job.
4. **Put a number on impact, and label it as an estimate.** Nearly every judges' note mentions ROI, time or cost; the 15-point metric line pays for it. The best writeups say which numbers are assumptions.
5. **Measure quality anyway.** ReDrugAI's recall and Grid Incident RAG's judged test did not win their categories, but they are the claims a reader can check.
6. **Know the product's scaling edges.** In-warehouse embedding speed and index editions decided one architecture outright.

### If you enter a hackathon like this

- Map every rubric line to an artifact before writing code.
- Pick one user with a job and a cost; add functions only where they serve that user.
- Keep the pipeline in the sponsor's product; move work out only with numbers showing it does not scale.
- Make setup one command, data included; assume the judge has a clean project.
- Report one quality metric with a baseline, and say what it does not measure.
- Write the friction feedback as you go.

The next post in this track, on the Gemini vibe-coding sprints and course capstones, shrinks the build window to 48 hours. The [vector database post](/blog/machine-learning/ai-agent/vector-database) covers the indexes these teams reached for.

## 13. Sources

- Competition: [BigQuery AI: Building the Future of Data](https://www.kaggle.com/competitions/bigquery-ai-hackathon), [Evaluation](https://www.kaggle.com/competitions/bigquery-ai-hackathon/overview/evaluation)
- Host threads: [winners and judging process (612730)](https://www.kaggle.com/competitions/bigquery-ai-hackathon/discussion/612730), [judging delay (610964)](https://www.kaggle.com/competitions/bigquery-ai-hackathon/discussion/610964)
- TriLink: [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/trilink-architecting-the-ai-native-digital-service), [GitHub](https://github.com/hazardscarn/TriLink/tree/main), [video](https://youtu.be/xmQJxaG4Lyc)
- The AI Patent Analyst: [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/the-ai-patent-analyst), [GitHub](https://github.com/veyselserifoglu/bq-ai-patent-analyst), [pipeline notebook](https://www.kaggle.com/code/fissalalsharef/bigquery-ai-the-patent-analyst-project), [live app](https://patent-search-analytics.streamlit.app/), [video](https://youtu.be/6WVi1cdd5fw)
- ESG and Financial Reports Insights Agent: [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/new-writeup-1757758945521), [notebook](https://github.com/intellitrend-global/google_hackathon_bq_ai/blob/main/process_reports.ipynb), [blog](https://intellitrend.co.uk/transforming-corporate-reporting-with-google-cloud-ai/)
- SpeakAura AI: [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/ai-powered-speech-therapy-for-stammering-speakaura), [GitHub](https://github.com/hariprasath0622/speak-aura-ai), [Medium](https://medium.com/@hariprasath0622/speakaura-ai-revolutionizing-speech-therapy-with-ai-8cd2484a5b02)
- ReDrugAI: [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/redrugai0421), [GitHub](https://github.com/catmei/ReDrugAI), [Medium](https://medium.com/@quantcat666/redrugai-repurposing-existing-drugs-as-new-treatments-with-google-bigquery-ai-ae0e5c256516)
- Causal RAG: [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/causal-rag), [GitHub](https://github.com/Laoyu84/bq_causal_rag), [video](https://youtu.be/C3GfcaQYhoo)
- OncOmix AI: [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/oncomix-ai), [GitHub](https://github.com/sharath6umich/OncOmix-AI-BigQuery.git), [video](https://youtu.be/VzOfrgzVV80)
- Grid Incident RAG: [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/grid-incident-rag), [GitHub](https://github.com/srmadani/Grid-Incident-Rag), [Medium](https://medium.com/@seyyedreza.madani/turning-153-wildfire-pdfs-into-answers-a-sql-only-rag-system-88ca9982a5fe)
- Auto-Updating Documentation: [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/auto-updating-documentation), [GitHub](https://github.com/jacobbridges/kaggle-bigquery-ai-2025-public), [video](https://www.youtube.com/watch?v=qNa2ULGu_PE)
- Patent Intelligence: [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/patent-intelligence-semantic-search-for-innovation), [GitHub](https://github.com/sl2902/patent_art), [notebook](https://www.kaggle.com/code/laxmsun/patent-intelligence-semantic-search-for-innovatio?scriptVersionId=262939955)
- CLARIS: [writeup](https://www.kaggle.com/competitions/bigquery-ai-hackathon/writeups/county-level-augmented-risk-intelligence-system-cl), [notebook](https://www.kaggle.com/code/tiyabk/county-level-augmented-risk-intelligence-system)
