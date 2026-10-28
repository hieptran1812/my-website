# Google: AI Assistants for Data Tasks with Gemma (2024): every prize winner

- **Slug / URL:** `data-assistants-with-gemma`, https://www.kaggle.com/competitions/data-assistants-with-gemma
- **Dates:** 2024-02-21 to 2024-04-14 (final deadline 2024-04-14 23:59 UTC). Winners: https://www.kaggle.com/competitions/data-assistants-with-gemma/discussion/499090 (posted ~2024-04-30). Mid-point prizes: https://www.kaggle.com/competitions/data-assistants-with-gemma/discussion/487380 (2024-03-28, notebooks shared in the first five weeks).
- **Entrants:** 6,971 users joined (bank); no team/submission counts in the API metadata (notebook-judged; submission via a form).
- **Prize pool:** $50,000 = 5 × $10k, one per category; plus mid-point recognition for 5 early public notebooks (reward not stated in 487380) and 25 swag prizes for well-documented Gemma variants on Kaggle Models.
- **Task:** a tutorial notebook showing **Gemma (1st gen, 2B/7B, plus 1.1 and CodeGemma as they released)** doing one of five assistant tasks: teach basic DS concepts; answer Python questions; summarise Kaggle solution writeups; explain concepts from writeups; answer Kaggle-platform questions.
- **Rubric:** yes/no gates (Compliant, Topical, Open) then Technical 10 (few-shot, RAG, fine-tuning), Descriptive 10, Useful 10, Robust 10.
- **Bank coverage:** `kaggle-llm-writeup-bank-part4.md` H2 "## 14." covers three category winners in one line each and none of the honorable mentions. Every winner, honorable mention and mid-point notebook gets a section here.
- **Sources:** no Writeup objects exist; each entry's "writeup" is its notebook, pulled with `pullnb.py` (markdown + first 800 chars of each code cell). 14 of 15 host-listed notebooks pulled; **abireltaief/google-gemma-s-magic-on-kaggle-write-ups returned HTTP 403** (private or deleted). Lucamassaron's mid-point notebook also pulled.

## Winners table (host order in 499090)

| Category | Award | Author | Notebook | Kaggle Model / extras |
|---|---|---|---|---|
| Explain/teach basic DS concepts | Winner ($10k) | sitaberete | [Build AI Agents With Google's LLM Gemma](https://www.kaggle.com/code/sitaberete/build-ai-agents-with-google-s-llm-gemma) | model [sitaberete/dsi-coder](https://www.kaggle.com/models/sitaberete/dsi-coder/); dataset DSI-Coder |
| | Honorable mention | woosungyoon | [Teacher Gemma: heuristic algorithm](https://www.kaggle.com/code/woosungyoon/teacher-gemma-heuristic-algorithm) | |
| | Honorable mention | hserdaraltan | [Data Science with Gemma, LlamaIndex and Pinecone](https://www.kaggle.com/code/hserdaraltan/data-science-with-gemma-llamaindex-and-pinecone) | |
| Python Q&A | Winner ($10k) | davidtroxellucla | [PyGEM: A Chatbot for Python Questions using Gemma](https://www.kaggle.com/code/davidtroxellucla/pygem-a-chatbot-for-python-questions-using-gemma) | model [pygem-using-googles-gemma-for-a-python-chatbot](https://www.kaggle.com/models/davidtroxellucla/pygem-using-googles-gemma-for-a-python-chatbot/) |
| | Honorable mention | kyushikmin | [Building Self-Critic Python Agent with Gemma](https://www.kaggle.com/code/kyushikmin/building-self-critic-python-agent-with-gemma) | |
| | Honorable mention (+ mid-point, filed under "Other") | inoueu1 | [Make A Smart Python Assistant with Gemma](https://www.kaggle.com/code/inoueu1/make-a-smart-python-assistant-with-gemma) | |
| Summarise Kaggle writeups | Winner ($10k) (+ mid-point) | toshik | [Gemma meets LangChain: summarize kaggle writeups](https://www.kaggle.com/code/toshik/gemma-meets-langchain-summarize-kaggle-writeups) | model [toshik/mindgemmap](https://www.kaggle.com/models/toshik/mindgemmap/Keras/v1/1) |
| | Honorable mention (+ mid-point) | jacoporepossi | [Text Summarization with Gemma](https://www.kaggle.com/code/jacoporepossi/text-summarization-with-gemma) | |
| | Honorable mention | astura | [Gemma: extracting structured data from write-ups](https://www.kaggle.com/code/astura/gemma-extracting-structured-data-from-write-ups) | |
| Explain concepts from writeups | Winner ($10k) (+ mid-point) | nghihuynh | [Unleashing Gemma's Power by Prompt Engineering](https://www.kaggle.com/code/nghihuynh/unleashing-gemma-s-power-by-prompt-engineering) | Streamlit app "GemmaBot" |
| | Honorable mention | ianakoto | [Gemma LLM Kaggle solution writeup agent](https://www.kaggle.com/code/ianakoto/gemma-llm-kaggle-solution-writeup-agent) | |
| | Honorable mention | abireltaief | [Google Gemma's magic on Kaggle write-ups](https://www.kaggle.com/code/abireltaief/google-gemma-s-magic-on-kaggle-write-ups) | NOT PULLED (403) |
| Kaggle platform Q&A | Winner ($10k) | vbookshelf | [KaggleBot: Gemma-7b-it RAG w few-shot prompting](https://www.kaggle.com/code/vbookshelf/kagglebot-gemma-7b-it-rag-w-few-shot-prompting) | dataset gemma-comp-data |
| | Honorable mention | penpentled | [KaggleQA: ReAct with Kaggle API](https://www.kaggle.com/code/penpentled/kaggleqa-react-with-kaggle-api) | model gemma-kaggleqa-react |
| | Honorable mention | tarique7 | [KaggleGPT powered by Gemma](https://www.kaggle.com/code/tarique7/kagglegpt-powered-by-gemma) | |
| (mid-point only) DS concepts | Mid-point | lucamassaron | [Data Science AI Assistant with Gemma 2b-it](https://www.kaggle.com/code/lucamassaron/data-science-ai-assistant-with-gemma-2b-it) | |

## Per-winner synthesis: category winners

### DS concepts winner: "Build AI Agents With Google's LLM Gemma" (sitaberete)
**What it is.** An 11,000-word tutorial building the "DSI Crew" (Data Science Instructor Crew): specialised Gemma models orchestrated by a Gemma agent, with interactive quizzes after each chapter (the host singled out the quizzes and explanations).

**Models.** Compares gemma-2b, gemma-2b-it and gemma-7b-it side by side on three teaching prompts; explains why the pretrained 2B repeats itself; loads in **fp16 (T4/P100 lack bf16)** and **4-bit NF4 via bitsandbytes**, with a memory table (7B ≈ 3.98 GB int4 vs 31.91 GB fp32). Uses 2b-it as the "coder" and **7b-it as the agent**.

**Data.** The author built the **DSI-Coder dataset (181 entries)** because public code datasets were too task-specific and lacked explanations: ~10 seed examples from Gemini, the rest generated with GitHub Copilot autocomplete in VS Code, each reviewed by hand; examples kept under 1,024 tokens so none is truncated.

**Fine-tune.** **QLoRA** on gemma-2b-it with TRL `SFTTrainer`: **r = 8, alpha = 8, dropout 0.1, batch 1, lr 1e-4, 1 epoch** ("we did only one epoch on a few examples" because good prompting already works). Published as Kaggle Model `sitaberete/dsi-coder` (transformers/2b).

**Retrieval.** Chapter 4: Chroma + `BAAI/bge-small-en` over a Markdown version of the Python Data Science Handbook with a parent-document retriever (child chunks indexed, parents stored); chapter 5: FAISS + `BAAI/bge-small-en-v1.5` over a Wikipedia Data Science Articles summary dataset, with LangChain's Wikipedia retriever as a fallback for freshness.

**Agent, and the key negative result.** First a LangChain **ReAct agent** (`create_react_agent`) with three tools (handbook retriever, Wikipedia, the fine-tuned coder). It chose tools correctly but returned incomplete answers, assuming the user would read the tool output, and prompt changes didn't fix it. So the author **simplified the agent into a router**: Gemma 7B-it only decides whether a question goes to the programming tool (DSI-Coder) or to a full RAG generator. Also warns readers not to ship the "ignore irrelevant context and answer anyway" prompt line used for the demo.

**Eval:** qualitative only (side-by-side outputs), no metric. Host quote: "Gemma-7b acts as a routing agent to determine whether a given prompt should be answered using knowledge from Wikipedia or knowledge from a fine-tuned Gemma variant ... an excellent learning resource" (499090).

### Python Q&A winner: "PyGEM" (davidtroxellucla)
**Data.** Four sources: two **novel datasets** the author built (`python_code_questions.jsonl` basic code tasks; `Python_Conceptual_Questions.jsonl` no-code concept Q&A; both human-reviewed + AI-generated) plus `flytech/python-codes-25k` and MBPP; ~2,000 prompts used for fine-tuning.

**Model and fine-tune.** **KerasNLP `gemma_instruct_2b_en`, LoRA rank 16**, sequence length **375** (memory-limited), lr 8.5e-4 with an adaptive scheduler (ReduceLROnPlateau) and early stopping in the experimental code; "ten epochs ... on a single P100 GPU took less than 3 hours" (notebook). Published as a Kaggle Model (Keras `PyGEM.keras` + LoRA weights).

**RAG.** Chroma + `avsolatorio/GIST-small-Embedding-v0` over a Python knowledge base, chunks inserted into a prompt template.

**Evaluation (why the host picked it).** **ROUGE-L and ROUGE-2 (recall, precision, F) on unseen data at three stages**: base, base + RAG, fine-tuned + RAG. Reported: RAG adds +0.02 to 0.03 ROUGE-L F; fine-tuning "almost doubled" it to ~0.4; ROUGE-2 follows the same trajectory (notebook). Plus an ipywidgets chat app. Limitations stated: 375-token context, the code-vs-concept trade-off in the training mix. Host: "quality of the responses were assessed via ROUGE score both before and after both fine-tuning and RAG" (499090).

### Writeup summarisation winner: "Gemma meets LangChain" + MindGemmap (toshik)
**Pipeline.** Writeup collection follows Darek's `kaggle-winning-solutions-methods` approach: scrape farid.one/kaggle-solutions for top-solution links, merge with the writeups shipped in competition data, and join Meta Kaggle `Teams.csv` for private/public scores. Summaries with **KerasNLP `gemma_instruct_2b_en` wrapped as a LangChain LLM**, `load_summarize_chain(chain_type="map_reduce")` over `RecursiveCharacterTextSplitter(chunk_size=1000, overlap=50)`. Output per competition: an overall summary, a **comparison matrix** across top solutions, and individual summaries.

**Fine-tuned variant.** **MindGemmap**, a Gemma 2B fine-tune (Keras weights `MindGemmap.weights.h5`) that converts plain text into a **Graphviz DOT file** rendered as a mind map, "impossible for original Gemma 2B" (notebook). The model card describes the use case but not the training data or hyperparameters. Published to Kaggle Models, which also targeted the 25 swag prizes for documented variants. Failure path handled ("Failed to generate the mind map for this solution").

**Eval:** none quantitative. Host: "a tool that can help us to better extract learnings from the treasure trove of knowledge ... buried away in Kaggle Competition solution writeups" (499090).

### Explain-concepts-from-writeups winner: "Unleashing Gemma's Power by Prompt Engineering" (nghihuynh)
**Pipeline.** Writeups from Meta Kaggle (Competitions, CompetitionTags, Tags, Teams, ForumTopics, ForumMessages). Covers zero-shot, few-shot, chaining and chain-of-thought prompting, quantised vs full Gemma-2b-it (Keras and transformers implementations; a GPU-memory comparison figure).

**Evaluation (small but explicit).** Extraction of key DS concepts from a writeup compared against a **human-checked Gemini-generated ground truth**, scored with **BLEU, ROUGE (rouge1/rougeL), Levenshtein, and "G-eval" (Gemma self-evaluation)**. Result on 4-bit Gemma-2b-it: **one-shot BLEU 0.1244 vs zero-shot 0.0641; G-eval 0.7500 vs 0.7203** (notebook). Conclusion: one example in the prompt roughly doubles BLEU; self-evaluation improves cohesiveness.

**Product.** A **Streamlit app (GemmaBot)** run from Kaggle via localtunnel, using 4-bit Gemma-2b-it, that organises writeups and supports follow-up questions to explain unclear concepts. Host: "a useful self-evaluation approach that compliments more traditional statistical measures such as ROUGE ... the provided Streamlit app was a fantastic way to quickly gather and organize Kaggle solution writeups" (499090). No fine-tuning.

### Kaggle-platform Q&A winner: "KaggleBot: Gemma-7b-it RAG w few-shot prompting" (vbookshelf)
**Data.** 14 curated text files on the Kaggle platform (competitions, notebooks, datasets, etc.) in the `gemma-comp-data` dataset; the author stresses the chunking strategy as "vital".

**Retrieval.** `all-MiniLM-L6-v2` embeddings + **FAISS exhaustive search (IndexFlatL2)**, then a **cross-encoder reranking** step; worked example: the correct chunk for "who is D. Sculley" was 6th after vector search and 1st after reranking ("the reranking step greatly improves the quality of the final result").

**Generation.** **gemma-7b-it (transformers, 4-bit) sharded across 2 × T4** (29 GB vs P100 16 GB, needed so evaluation doesn't crash). **Few-shot prompting with three Q/A examples** from a hand-corrected `df_corrected_data.csv` so answers don't mention "the context", plus **two heuristics (rules)** for output style. Shows base-model hallucinations (e.g. "Kaggle datasets are curated by...") as motivation.

**Evaluation.** **25 representative questions** run end to end (~3 min), outputs reviewed manually and shown in full for judges. Host: "We were really impressed at how accurately it responded to our questions and how well it knew when to decline to respond" (499090). No fine-tuning.

## Per-entry synthesis: honorable mentions and mid-point notebooks (short)

- **woosungyoon, "Teacher Gemma heuristic algorithm" (DS concepts HM).** gemma-2b-it with hybrid retrieval (FAISS over a wiki-STEM index + BM25), and a "heuristic algorithm": an **ensemble of three baseline answers computed as a batch** and a **refinement chain** that passes each step's answer forward as a few-shot example. Compares base, base + RAG, ensemble and chain answers qualitatively; finding: RAG over-relies on documents, the refinement chain improves thematic consistency.
- **hserdaraltan, "Data Science with Gemma, LlamaIndex and Pinecone" (DS concepts HM).** Three-way comparison: untuned Gemma, **KerasNLP LoRA fine-tune (rank 8, lr 5e-5, 3 epochs, batch 1, input length 128)** on the "1000 Data Science Concepts" dataset, and fine-tune + **LlamaIndex RAG over a Pinecone index** (`BAAI/bge-base-en-v1.5`). Qualitative comparison; fine-tune gives concise accurate answers.
- **kyushikmin, "SCAPPY: Self-Critic Agent for Programming in Python" (Python HM).** gemma-7b QLoRA (TRL, 2 epochs, batch 8, max length 4096) as a QA model, wrapped in a **self-critic loop**; **MBPP pass@1 evaluation**: self-critique adds **+3.4% code runs without error and +2.2% test-case pass rate** (notebook), without training on MBPP train.
- **inoueu1, "Make A Smart Python Assistant with Gemma" (Python HM + mid-point, which filed it under "Other").** Tests Gemma on hand-picked problems against **GPT-3.5/GPT-4-Turbo**, executes generated code and times it; KerasNLP **LoRA rank 4 on TPU** with `coseal/Magicoder-Evol-Instruct-110K-sft`; evaluates on CODAL-Bench samples. Finding: none of the author's fine-tunes beat **CodeGemma 7B instruct**, which became the final model.
- **jacoporepossi, "Text Summarization with Gemma" (summarisation HM + mid-point).** transformers gemma-2b-it, LangChain **Stuff / MapReduce / Refine** strategies, `HTMLHeaderTextSplitter` for long writeups, few-shot discussion; a small **LoRA adapter (r = 6, lr 2e-4, 12 steps, length 512)** on CNN/DailyMail summarisation. Tutorial-style, no metric reported.
- **astura, "Extracting structured data from write-ups" (summarisation HM).** `gemma_instruct_2b_en` with iterative prompt refinement to pull a fixed schema from Meta Kaggle writeups. Most useful as a *failure catalogue*: Gemma imposed familiar framings on unfamiliar domains (gravitational-wave data called "computer vision", ASL fingerspelling miscategorised, "NLP" for non-NLP comps); question-style prompts beat giving a schema.
- **ianakoto, "Gemma LLM Kaggle solution writeup agent" (explain-concepts HM).** **Gemma 7B instruct on TPU (8 cores, model-parallel layout map)**; synthetic QA data generated with Gemma Instruct and aligned with real human QA from Meta Kaggle forums, then **LoRA rank 8, lr 5e-5, 22 epochs, sequence length 1024**. Flags noisy question/answer tagging in forum data and hallucination caveats.
- **abireltaief, "Google Gemma's magic on Kaggle write-ups" (explain-concepts HM).** Notebook not retrievable (HTTP 403 on 2026-10-28); no content verified.
- **penpentled, "KaggleQA: ReAct with Kaggle API" (platform HM).** **ReAct agent** (thought/action/observation) whose tools are the **live Kaggle API** (datasets, kernels, competitions) and Kaggle docs. Base gemma-2b-it "has no idea what it's doing" while Gemini handled it, so the author **distilled Gemini ReAct traces into Gemma** (`kaggleqa_react_gemini.csv`; LoRA r = 8, lr 8e-5, 600 steps), published as a Kaggle Model. Shows visible reasoning and sources.
- **tarique7, "KaggleGPT powered by Gemma" (platform HM).** KerasNLP Gemma 2B fine-tuned (LoRA, AdamW with weight decay) on **~2,000 instruction-response pairs**, plus LangChain FAISS RAG; qualitative before/after; notes RAG limits generation to the retrieved context.
- **lucamassaron, "Data Science AI Assistant with Gemma 2b-it" (mid-point, DS concepts).** RAG from scratch: a DS dataset built programmatically from the **Wikipedia API**, `thenlper/gte-large` embeddings, **Google ScaNN** search, `generate_summary_and_answer()` at temperature 0.4 with an "expert" role; then the same RAG on **gemma.cpp (2b-it-sfp compressed weights) for CPU-only devices**. Host praised the from-scratch RAG explanation and the GemmaCPP path (487380).

## Feature matrix

Columns: SB = sitaberete (DS winner), PG = PyGEM (Python winner), TK = toshik (summ. winner), NH = nghihuynh (explain winner), VB = vbookshelf (platform winner), WY = woosungyoon, HA = hserdaraltan, KM = kyushikmin, IN = inoueu1, JR = jacoporepossi, AS = astura, IA = ianakoto, PP = penpentled, TQ = tarique7, LM = lucamassaron (mid-point). abireltaief omitted (not retrievable).

| Row | SB | PG | TK | NH | VB | WY | HA | KM | IN | JR | AS | IA | PP | TQ | LM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Main model | 2b-it + 7b-it | 2b-it (Keras) | 2b-it (Keras) + MindGemmap | 2b-it (4-bit) | 7b-it (4-bit) | 2b-it | 2B (Keras) | 7b | CodeGemma 7b-it | 2b-it | 2b-it | 7b-it | 2b-it | 2B (Keras) | 2b-it (+ gemma.cpp) |
| Framework | transformers + TRL | KerasNLP | KerasNLP + LangChain | KerasNLP + transformers | transformers | transformers | KerasNLP + LlamaIndex | transformers + TRL | KerasNLP (TPU) | transformers + LangChain | KerasNLP | KerasNLP (TPU) | transformers + PEFT | KerasNLP + LangChain | transformers + gemma.cpp |
| Fine-tuned (LoRA/QLoRA) | Y (r8) | Y (r16) | Y | N | N | N | Y (r8) | Y | Y (r4, not final) | Y (r6) | N | Y (r8) | Y (r8, distilled) | Y | N |
| Published model variant | Y | Y | Y | N | N | N | ? | ? | ? | N | N | ? | Y | ? | N |
| Own dataset built | Y (181) | Y (2 sets) | N (scraped writeups) | Y (Gemini ground truth) | Y (14 docs + few-shot set) | N | N | N | N | N | N | Y (synthetic QA) | Y (Gemini traces) | ? | Y (Wikipedia API) |
| RAG | Y (Chroma, FAISS) | Y (Chroma) | N (map-reduce) | N | Y (FAISS + rerank) | Y (FAISS + BM25) | Y (Pinecone) | N | N | N | N | N | Y (docs + API) | Y (FAISS) | Y (ScaNN) |
| Agent / routing / tools | Y (ReAct → router) | N | N | N | N | partial (ensemble + chain) | N | Y (self-critic) | N | N | N | N | Y (ReAct + Kaggle API) | N | N |
| Few-shot prompting | ? | N | N | Y | Y | Y | N | N | N | Y | N | N | Y | N | N |
| Synthetic / LLM-generated training data | Y (Gemini + Copilot) | Y (partly) | ? | N | N | N | N | N | Y (Magicoder) | N | N | Y | Y (Gemini) | ? | N |
| Eval rigour | qualitative | small quant + ablation (ROUGE, 3 stages) | none | small quant (BLEU/ROUGE/G-eval, 0 vs 1 shot) | qualitative (25 Qs, manual) | qualitative comparison | qualitative comparison | benchmark + baseline (MBPP pass@1) | small quant vs GPT-3.5/4 | none | qualitative (failure catalogue) | none | qualitative | qualitative | qualitative |
| Interactive app | quizzes | ipywidgets | N | Streamlit | N | N | N | N | N | N | N | N | N | N | N |
| CPU / edge runtime | N | N | N | N | N | N | N | N | N | N | N | N | N | N | Y (gemma.cpp) |

**Counts (15 retrievable entries; 5 category winners):** fine-tuned 10/15 (winners 3/5: SB, PG, TK); RAG 9/15 (winners 3/5); agent/router/self-critic 4/15 (winners 1/5: SB); published a model variant to Kaggle Models: 4 confirmed (SB, PG, TK, PP; 3 of them winners); built own dataset 8/15 (winners 5/5 in some form); any quantitative metric 4/15 (PG ROUGE, NH BLEU/ROUGE/G-eval, KM MBPP, IN vs GPT), of which 2 are winners; 7B-class models 5/15; edge/CPU runtime 1/15.

## Consensus vs divergence

- **Table stakes:** tutorial-quality explanation (the rubric's Descriptive and Useful criteria), Gemma 2B-it as the workhorse, LangChain or KerasNLP, and at least one of RAG / few-shot / LoRA.
- **What separated winners from honorable mentions:** each winner has one *visible, checkable* differentiator the host names in its citation: a router agent plus a published fine-tune (SB), a before/after ROUGE ablation (PG), a published mind-map model (TK), a self-evaluation metric plus a Streamlit app (NH), and a 25-question manual eval with good refusals (VB). Honorable mentions often had comparable or stronger techniques (KM's MBPP pass@1 is the most rigorous eval in the field; PP's ReAct over the live Kaggle API is the most agentic) but less polish or less fit to the category.
- **Contradictions:**
  - *ReAct agents with Gemma 1:* SB found a ReAct agent picked tools correctly but gave incomplete answers and downgraded it to a router (https://www.kaggle.com/code/sitaberete/build-ai-agents-with-google-s-llm-gemma); PP found base Gemma 2B couldn't do ReAct at all and fixed it by distilling Gemini traces (https://www.kaggle.com/code/penpentled/kaggleqa-react-with-kaggle-api). Both agree raw Gemma 1 is too weak for free-form agents.
  - *Fine-tune vs stronger base:* PG reports fine-tuning nearly doubled ROUGE-L; IN found none of its fine-tunes beat CodeGemma 7B and shipped the stock specialist model (https://www.kaggle.com/code/inoueu1/make-a-smart-python-assistant-with-gemma).
  - *Schema prompting:* AS found giving Gemma a schema worked worse than question-style prompts; NH found adding one example to the prompt doubled BLEU.
- **Through-line to later Gemma hackathons:** routing between retrieval and a fine-tuned model, publishing the variant, and measuring each technique's increment reappear in the Gemma language-tuning, Gemma 3n and Gemma 4 winners.

## UNVERIFIED
- abireltaief's notebook content (403).
- MindGemmap training data and hyperparameters (not in notebook or model card).
- PyGEM exact ROUGE numbers per stage (only "+0.02 to 0.03" and "almost doubled to ~0.4" stated in text; exact values are in plots/screenshots); whether 10 epochs or 3 (code shows `epochs=3` in a commented cell, text says ten).
- Code cells were truncated to 800 characters by `pullnb.py`, so some hyperparameters (e.g. tarique7's LoRA rank and LR, sitaberete's exact SFT steps) may be incomplete.
- Mid-point prize amounts (487380 does not state them); total entrant/submission counts.
