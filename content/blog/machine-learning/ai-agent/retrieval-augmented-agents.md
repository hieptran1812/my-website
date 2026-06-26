---
title: "Retrieval-Augmented Agents: Wiring RAG into the Reasoning Loop"
date: "2026-06-27"
description: "How to integrate retrieval into the agent reasoning loop — query generation, retrieval scheduling, re-ranking, and the failure modes that plain RAG never sees."
tags: ["ai-agents", "rag", "retrieval", "memory", "llm", "machine-learning", "nlp", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 42
---

There is a version of RAG that works beautifully in a demo: user asks a question, system retrieves three passages, LLM answers. Every blog post about RAG is secretly describing that version. It is not the version you build when retrieval needs to happen inside an agent that reasons across ten steps, uses five tools, and may need to retrieve different facts at different moments in the same turn.

The agent version is harder. The chatbot version retrieves once, at the start, with the user's literal question as the query. The agent version has to answer questions like: *Which step in this reasoning chain actually needs external knowledge? What query should I issue — the user's words or a reframed hypothesis? How do I inject retrieved context into a prompt that is already full of tool schemas, memory, and instructions? What do I do when retrieval comes back empty?* None of that appears in a "RAG tutorial."

This post is about that harder version. We will walk through every layer: when to retrieve, how to generate good queries from agent state, how to schedule retrieval across a multi-step turn, how to re-rank results, where to inject them in the prompt, and how to handle the full taxonomy of retrieval failures — empty results, low-relevance results, and conflicting results. Along the way, we will look at case studies from real systems where the standard pipeline broke in interesting ways.

![RAG wired into the ReAct reasoning loop](/imgs/blogs/retrieval-augmented-agents-1.webp)

The diagram above is the mental model. Each agent turn runs a retrieve-inject cycle between the "Think" and "Act" steps. That cycle is not a single call to a vector database — it is a small pipeline of its own: query generation, retrieval, re-ranking, and injection. Get any one of those wrong and the LLM acts on bad context, or worse, acts without context at all and hallucinates with confidence.

---

## 1. RAG in agents vs RAG in chatbots: the key differences

The standard chatbot RAG pipeline has a clean interface: one user turn, one retrieval call, one LLM call. The retrieval query is the user's message. The retrieved context is prepended to the prompt. The LLM answers. Done.

An agent breaks every one of those assumptions.

**Multiple retrieval moments per turn.** A ReAct agent running a research task might take 8–12 reasoning steps in a single user turn. Some of those steps are pure reasoning ("given what I know, the next thing to check is X"), some are tool calls (calculator, code executor, web scraper), and some are knowledge lookups that need retrieval. Applying chatbot RAG here means either retrieving once at the start (which means the context is stale by step 6) or retrieving at every step (which is expensive and retrieves irrelevant stuff on the pure-reasoning steps).

**The query is not the user's message.** In a chatbot, the user asks "What was Apple's revenue in Q3 2024?" and you retrieve on that query. In an agent, the user might ask the same question, but by step 4, the agent has decomposed it into a sub-task: "What is AAPL's most recent 10-Q? What quarter is that?" The retrieval query should reflect the *current reasoning state*, not the original user message. Retrieving on the original message at step 4 returns passages about Apple's revenue history — not the specific filing the agent needs right now.

**Prompt real estate is already contested.** Chatbot RAG inserts retrieved context into a prompt that contains the system message and the user turn. Agent prompts additionally carry: tool schemas (often 1,000–3,000 tokens of JSON), a scratchpad of prior reasoning steps, an episodic memory buffer, and the current task. Adding 800 tokens of retrieved context to an already-full prompt is a real tradeoff, not a free operation.

**Failure modes compound.** If retrieval fails in a chatbot, the LLM either says "I don't know" or makes something up — one failure, caught at one point. If retrieval fails in an agent, the consequences cascade: the agent may hallucinate a fact in step 3, base the tool call in step 5 on that hallucination, and produce a confidently wrong final answer in step 10. Retrieval failures in agents need explicit handling, not just a "sorry, couldn't find that" fallback.

**Memory competes with retrieval.** Agents typically have episodic memory (prior turns in this session), semantic memory (a long-term knowledge store), and working memory (the current scratchpad). Retrieval from a vector store is one memory system among several. You need a routing layer that decides: does this sub-task need external retrieval, or can it be answered from episodic memory? That routing decision does not exist in chatbot RAG.

Here is a concise comparison:

| Dimension | Chatbot RAG | Agent RAG |
|---|---|---|
| Retrieval calls per turn | 1 | 0–N (variable) |
| Query source | User's literal message | Current reasoning state |
| Failure blast radius | Single answer | Multi-step cascade |
| Context competition | Low | High (tools + memory + task) |
| Injection timing | Always at start | Conditional, mid-turn |
| Staleness risk | Low | High (topic shifts mid-turn) |

---

## 2. When to retrieve: query-triggered vs proactive vs scheduled retrieval

Before we discuss *how* to retrieve, we need to settle *when*. There are three fundamentally different retrieval timing strategies, and choosing the wrong one for your workload is the most common source of silent failure in agentic RAG systems.

![Retrieval timing strategies compared](/imgs/blogs/retrieval-augmented-agents-2.webp)

### Query-triggered retrieval

The agent retrieves *on demand* — when its reasoning process determines that it needs external knowledge. This is the cleanest model intellectually: retrieval happens exactly when it is needed, which means no wasted calls on pure-reasoning steps and no stale context from prefetching.

The implementation is also the most natural when retrieval is modeled as a tool (see Section 7). The LLM decides to call `retrieve(query)` and blocks until results come back before continuing its reasoning.

The downside is latency. A retrieval call to a remote vector store typically takes 50–200 ms. In an 8-step reasoning chain with 3 retrieval calls, that is 150–600 ms of additional wall-clock time before the first token of the final answer is generated. For interactive applications where users expect sub-second responses, this is often not acceptable.

Query-triggered retrieval also requires that the LLM is genuinely capable of deciding when to retrieve. Weaker models over-retrieve (calling retrieve on every step, including the pure-math ones) or under-retrieve (reasoning through knowledge gaps instead of looking them up). You need to calibrate retrieval triggering through prompt engineering and fine-tuning before relying on it.

### Proactive retrieval

The system prefetches relevant documents *before* the agent starts reasoning, or *speculatively* prefetches at the end of each step in anticipation of the next step's needs.

The user-perceived latency advantage is significant. If retrieval happens in parallel with the previous step's computation, the results are waiting when needed. In a pipeline where each LLM call takes ~1 second, prefetching during that second means the retrieval latency is fully hidden.

The cost is over-retrieval. You are fetching context for reasoning steps that may not need it, which inflates token count and can crowd out more relevant context. The silent staleness risk is subtler: if the agent's reasoning takes an unexpected turn (which agents frequently do), the prefetched context is no longer relevant, but it is already in the prompt. The LLM may anchor on it anyway.

Proactive retrieval works well for predictable pipelines — research agents with a known sequence of sub-tasks, document-processing agents where the document structure is known up front, or conversational agents where user intent can be classified early and prefetch queries can be constructed from that classification.

### Scheduled / background retrieval

A separate process maintains a freshness-gated cache of frequently-accessed knowledge. The agent queries this cache synchronously (zero latency on a hit) while the cache refresh happens asynchronously.

This is the right model for monitoring agents, report-generation agents, and any use case where the knowledge domain is predictable but the data changes frequently (market data, news feeds, product catalogs). The cache TTL is the primary design knob: set it too short and you are paying for frequent background refreshes; set it too long and agents act on stale data.

The cache-miss fallback path is critical. When the cache misses (new query pattern, expired entry), you fall back to query-triggered retrieval with its latency cost. If your cache hit rate is low, you are paying the operational cost of maintaining the cache while still paying the latency cost of on-demand retrieval most of the time.

---

## 3. Query generation: turning agent state into retrieval queries

The query you send to the retrieval system is the single highest-leverage decision in the retrieval pipeline. A bad query returns bad results regardless of how good your vector index, re-ranker, or injection strategy is. Most retrieval bugs are secretly query bugs.

![Query generation variants: HyDE, step-back, multi-query](/imgs/blogs/retrieval-augmented-agents-3.webp)

### The vocabulary mismatch problem

The core failure mode is vocabulary mismatch: the agent's current reasoning uses different words than the documents it needs to find. The agent might be reasoning about "the time complexity of the algorithm described in the paper" while the relevant document talks about "computational complexity analysis" and "asymptotic behavior." Bi-encoder embeddings close some of this gap (semantics over syntax), but not all of it — especially for technical domains where specialized jargon is semantically distinct from lay terms.

### HyDE: Hypothetical Document Embeddings

HyDE addresses vocabulary mismatch by having the LLM generate a *hypothetical document* — a synthetic passage that looks like what the ideal retrieved document would contain — and then embedding that synthetic document for retrieval instead of the original query.

The intuition: if you need to find a document about "asymptotic complexity of quicksort," generating the sentence "Quicksort has an average-case time complexity of O(n log n) and a worst-case of O(n²), depending on pivot selection" and embedding that will find more relevant passages than embedding "time complexity of the algorithm."

```python
def hyde_query(agent_state: str, llm: LLM, k: int = 5) -> list[Document]:
    """Generate a hypothetical document and retrieve using its embedding."""
    # Step 1: generate the hypothetical document
    prompt = f"""Given this question or task:
{agent_state}

Write a 2-3 sentence passage that would perfectly answer this question.
Write as if you are the author of a technical document."""
    
    hypothetical_doc = llm.generate(prompt, max_tokens=150)
    
    # Step 2: embed the hypothetical doc, not the original query
    embedding = embed_model.encode(hypothetical_doc)
    
    # Step 3: retrieve using the hypothetical doc's embedding
    results = vector_store.search(embedding, k=k)
    return results
```

HyDE works well when the LLM has enough knowledge to generate a plausible hypothetical document. It fails when the LLM does not know the answer (it cannot write a plausible passage) or when the domain is so specialized that the LLM's prior knowledge is incorrect and it generates a confident-but-wrong hypothetical. In those cases, HyDE retrieves documents that support the LLM's misconception.

### Step-back prompting

Step-back prompting addresses the specificity mismatch problem: the agent is reasoning about a specific instance, but the relevant documents describe the general case. An agent reasoning about "why did the AAPL stock drop on March 15, 2024" may need documents about "how earnings misses affect tech stock prices in general" before it can reason about the specific case.

The technique: before issuing the retrieval query, have the LLM reformulate it as a more abstract question. "Why did AAPL drop on March 15" becomes "What general factors cause large-cap tech stocks to drop significantly in a single trading session?" The retrieved general documents provide the framework that the agent applies to the specific case.

```python
def stepback_query(specific_query: str, llm: LLM) -> str:
    """Reformulate a specific query as a more abstract concept question."""
    prompt = f"""Given this specific question:
{specific_query}

Reformulate it as a more abstract, general question that captures
the underlying concept or principle. This abstract question should
help find documents that explain the general case.

Return only the reformulated question, nothing else."""
    return llm.generate(prompt, max_tokens=100)
```

### Multi-query generation

Multi-query generation hedges against any single query being poorly formulated by issuing N parallel queries — N paraphrases or framings of the same information need — and merging the results.

The standard recipe is to generate 3–5 paraphrases that vary the vocabulary, framing, and specificity:

```python
def multi_query(question: str, llm: LLM, n: int = 4) -> list[str]:
    """Generate N paraphrases of the original question for diverse retrieval."""
    prompt = f"""Generate {n} different ways to phrase this question for document retrieval.
Vary the vocabulary, specificity, and framing. Do not repeat the original.

Original question: {question}

Return each paraphrase on a new line, numbered 1–{n}."""
    
    response = llm.generate(prompt, max_tokens=200)
    paraphrases = [line.split('. ', 1)[1] for line in response.strip().split('\n') 
                   if line and line[0].isdigit()]
    return [question] + paraphrases[:n-1]  # include original

def multi_query_retrieve(question: str, llm: LLM, vector_store, k_per_query: int = 10) -> list[Document]:
    """Issue N queries and merge results with Reciprocal Rank Fusion."""
    queries = multi_query(question, llm)
    all_results = [vector_store.search(q, k=k_per_query) for q in queries]
    return reciprocal_rank_fusion(all_results, k=60)
```

Merging uses Reciprocal Rank Fusion (RRF): for each document, compute $\text{score} = \sum_i \frac{1}{k + \text{rank}_i(d)}$ where the sum runs over all query result lists and $k = 60$ is a smoothing constant. RRF is more robust than score-averaging because it handles the case where queries produce incomparable raw scores.

The tradeoff: N parallel retrieval calls, N × embedding computations. For agents where latency is critical, multi-query with N=4 adds roughly 3× the retrieval cost. Use it for research agents and batch pipelines, not for interactive query answering.

### When to use which

| Failure mode | Best technique | When to skip it |
|---|---|---|
| Vocabulary mismatch | HyDE | LLM lacks domain knowledge |
| Generality gap | Step-back | Already at the right specificity level |
| Single-query coverage | Multi-query | Latency-critical interactive use |
| All three | Multi-query + HyDE | High-stakes research, offline batch |

---

## 4. Retrieval scheduling: once per turn vs per-step vs on-demand tool call

We have established *when* to retrieve in a general sense. Now let's be precise about the interaction between retrieval timing and the agent's reasoning loop structure.

### Once-per-turn retrieval

The simplest approach: retrieve once at the start of each user turn, using the user's message as the query (possibly enriched via HyDE or step-back). Inject the results into the system prompt or the first user message. All subsequent reasoning steps have access to the same retrieved context.

This works when the retrieval query maps cleanly to the entire turn — a factual question, a document summarization task, a single-topic research request. It fails when the agent's reasoning shifts topic mid-turn, when the relevant knowledge is distributed across multiple retrieval queries, or when early reasoning steps establish facts that determine what to retrieve later.

### Per-step retrieval

Retrieve at every reasoning step, using the current reasoning state (the last thought or partial action) as the query. This maximizes context relevance but is the most expensive option: N reasoning steps × retrieval cost.

The practical mitigation: cache retrieval results at the embedding level. If two consecutive steps produce similar embeddings (cosine similarity > 0.9), skip the second retrieval call and use the cached results. This cuts retrieval cost by 30–60% for typical agent traces with related sub-tasks.

```python
class CachedRetriever:
    def __init__(self, vector_store, cache_threshold: float = 0.90):
        self.vector_store = vector_store
        self.threshold = cache_threshold
        self._cache: list[tuple[np.ndarray, list[Document]]] = []
    
    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        query_emb = embed_model.encode(query)
        
        # Check cache for similar recent query
        for cached_emb, cached_docs in self._cache[-3:]:  # check last 3
            similarity = cosine_similarity(query_emb, cached_emb)
            if similarity >= self.threshold:
                return cached_docs  # cache hit: skip retrieval
        
        # Cache miss: real retrieval
        docs = self.vector_store.search(query_emb, k=k)
        self._cache.append((query_emb, docs))
        if len(self._cache) > 10:
            self._cache.pop(0)
        return docs
```

### On-demand retrieval as a tool call

The agent explicitly invokes retrieval by calling a `retrieve` tool, just as it might call a `calculator` or `web_search` tool. The LLM decides when retrieval is needed based on its confidence in its own knowledge.

This is the most flexible approach and the one most consistent with the agentic RAG philosophy. It requires that the LLM is calibrated to know when it does not know something — a property that is not uniformly present across model families, and that degrades with weaker models.

Concretely, you expose a tool:

```python
retrieve_tool = {
    "type": "function",
    "function": {
        "name": "retrieve_context",
        "description": "Search the knowledge base for documents relevant to a query. "
                        "Call this when you need factual information you are not confident about.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — phrase it as a question or "
                                   "as key terms from the fact you need to find."
                },
                "k": {
                    "type": "integer",
                    "description": "Number of documents to retrieve (default 5).",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}
```

The LLM invokes this tool as needed during its reasoning trace. The agent framework intercepts the tool call, runs the retrieval pipeline, and injects the results back into the conversation as a tool response.

The key design decision: should the tool return raw document text, or structured metadata? For most use cases, returning the passage text plus minimal metadata (source URL, date, relevance score) is the right tradeoff. The LLM can use the source metadata to express confidence and cite its sources; the passage text is what it actually reasons over.

---

## 5. Re-ranking: cross-encoder re-ranking, LLM-as-reranker, diversity re-ranking

Raw bi-encoder retrieval is fast but imprecise. Bi-encoders compute embeddings for query and documents independently, so they cannot model the *interaction* between query and document — the fine-grained lexical matching, the semantic overlap of specific phrases, the relevance of a particular passage structure to a particular question type.

Re-ranking is the precision pass that runs after the recall pass.

![Re-ranking pipeline: bi-encoder → cross-encoder → diversity](/imgs/blogs/retrieval-augmented-agents-4.webp)

### Cross-encoder re-ranking

A cross-encoder takes a (query, document) pair concatenated as a single sequence and produces a relevance score. Because the model sees both at once, it can model token-level interactions that bi-encoder embeddings miss. Cross-encoders typically improve NDCG@10 by 15–30% over bi-encoder recall alone on standard IR benchmarks.

The cost: cross-encoder inference is O(N) in the number of candidates, and each inference is a full forward pass over a sequence of length (query + document). For top-100 candidates at 512 tokens each, this is 100 forward passes — practical for offline indexing, expensive for online agents.

The standard recipe: retrieve top-100 with bi-encoder, re-rank top-100 to top-20 with cross-encoder, inject top-5. The cross-encoder runs over 100 candidates, not the full corpus.

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query: str, candidates: list[Document], top_k: int = 20) -> list[Document]:
    """Re-rank candidates using a cross-encoder, return top_k."""
    pairs = [(query, doc.text) for doc in candidates]
    scores = cross_encoder.predict(pairs)
    
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
```

For latency-sensitive applications, quantized cross-encoders (INT8 or FP16) reduce inference time by 2–4× with minimal quality loss. The MiniLM family of cross-encoders is a reasonable default: MSMARCO-MiniLM-L-6 runs in ~20 ms per 100 candidates on a CPU.

### LLM-as-reranker

For high-stakes retrieval where you need top-1 or top-3 precision to be very high, you can use the LLM itself as a reranker. This exploits the LLM's language understanding at the expense of latency and cost.

The standard approach is the "relevance score" prompt:

```python
def llm_rerank(query: str, candidates: list[Document], llm: LLM, top_k: int = 10) -> list[Document]:
    """Use the LLM to score each candidate's relevance to the query."""
    scored = []
    for doc in candidates:
        prompt = f"""On a scale of 0–10, how relevant is this passage to the query?

Query: {query}
Passage: {doc.text[:500]}

Reply with only a number from 0–10. No explanation."""
        score_str = llm.generate(prompt, max_tokens=5)
        try:
            score = float(score_str.strip())
        except ValueError:
            score = 0.0
        scored.append((score, doc))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
```

A faster variant is listwise re-ranking: send the LLM all N candidates at once and ask it to return a ranked list of indices. This uses a single LLM call instead of N calls:

```python
def llm_rerank_listwise(query: str, candidates: list[Document], llm: LLM) -> list[Document]:
    passages = "\n\n".join(f"[{i+1}] {doc.text[:400]}" for i, doc in enumerate(candidates))
    prompt = f"""Query: {query}

Below are {len(candidates)} passages. Rank them by relevance to the query.
Return only the passage numbers in order from most to least relevant, e.g.: 3, 7, 1, 5, ...

Passages:
{passages}"""
    
    ranking_str = llm.generate(prompt, max_tokens=100)
    indices = [int(x.strip()) - 1 for x in ranking_str.split(',') if x.strip().isdigit()]
    return [candidates[i] for i in indices if i < len(candidates)]
```

Listwise re-ranking is sensitive to the "lost-in-the-middle" problem: LLMs tend to pay more attention to passages at the beginning and end of the list than the middle. Mitigations: randomize the order before re-ranking, or run two rounds with reversed order and average the ranks.

### Diversity re-ranking (MMR)

Cross-encoder and LLM re-rankers optimize for relevance. They do not penalize redundancy. If the top-5 passages after re-ranking all say slightly different versions of the same fact, you have wasted 4/5 of your context budget.

Maximal Marginal Relevance (MMR) balects relevance with diversity:

$$\text{MMR}(d_i) = \lambda \cdot \text{relevance}(d_i, q) - (1 - \lambda) \cdot \max_{d_j \in S} \text{sim}(d_i, d_j)$$

where $S$ is the set of already-selected documents and $\lambda \in [0, 1]$ is a tradeoff parameter ($\lambda = 0.7$ is a common default). MMR greedily selects documents that are both relevant and dissimilar from what has already been selected.

```python
def mmr(query_emb: np.ndarray, candidates: list[Document], k: int = 5, lam: float = 0.7) -> list[Document]:
    """Select k documents maximizing relevance to query and diversity from each other."""
    doc_embs = [embed_model.encode(doc.text) for doc in candidates]
    
    selected_indices = []
    remaining = list(range(len(candidates)))
    
    for _ in range(k):
        if not remaining:
            break
        
        mmr_scores = []
        for i in remaining:
            rel = cosine_similarity(query_emb, doc_embs[i])
            if selected_indices:
                max_sim = max(cosine_similarity(doc_embs[i], doc_embs[j]) 
                              for j in selected_indices)
            else:
                max_sim = 0.0
            mmr_scores.append(lam * rel - (1 - lam) * max_sim)
        
        best = remaining[np.argmax(mmr_scores)]
        selected_indices.append(best)
        remaining.remove(best)
    
    return [candidates[i] for i in selected_indices]
```

In practice, the full re-ranking pipeline is: bi-encoder (top-100) → cross-encoder (top-20) → MMR (top-5). For most agentic workloads, this pipeline adds 50–100 ms and improves answer quality measurably.

---

## 6. Injection strategies: where retrieved context goes in the prompt

Having good retrieved documents is necessary but not sufficient. *Where* you put them in the prompt matters because LLMs pay non-uniform attention across the context window — early and late positions are attended to more strongly than the middle, a phenomenon documented in the "Lost in the Middle" paper and consistently reproduced across model families.

![Prompt anatomy with injected retrieval context](/imgs/blogs/retrieval-augmented-agents-5.webp)

### The five injection positions

**Position 1: Before the system prompt.** Retrieved context becomes the very first thing the model sees. This maximizes attention on the retrieved material but competes with the system instructions for framing. Not recommended — it disrupts the "persona before information" ordering that models are trained to expect.

**Position 2: After the system prompt, before memory.** Good for contexts where the retrieved material is foundational to the entire task. If the agent is a domain expert system and the retrieved material is domain reference content (a medical protocol, a legal document, a product specification), placing it here positions it as permanent context the agent should reason within.

**Position 3: Middle injection (after memory, before task).** The standard RAG placement. Retrieved documents sit between the historical context and the current question, modeling the "research materials" the agent has gathered. The risk is the lost-in-the-middle effect — if the task is near the end of a long context window and retrieved documents are in the middle, the LLM may effectively ignore them.

**Position 4: Immediately before the current task.** The most attention-maximizing placement for the retrieved documents. By placing them directly before the question, you exploit primacy-recency effects simultaneously — the question itself benefits from recency, and the documents immediately precede it so they are still within the high-attention "pre-task" zone.

**Position 5: Interleaved with reasoning steps.** In a multi-step agent, inject retrieved context as a tool response at the exact step that requested it, rather than at the start. This is the natural placement when retrieval is a tool call. The model sees: `[prior reasoning] → [tool call: retrieve] → [tool response: passages] → [continue reasoning]`. This is typically the best placement for per-step or on-demand retrieval.

### Injection format

How you format the retrieved passages matters as much as where you put them. The minimal format is:

```
<retrieved_context>
[1] (source: docs.company.com/api-reference, score: 0.94)
The rate limit for the standard tier is 100 requests per minute...

[2] (source: docs.company.com/pricing, score: 0.87)
Enterprise customers have a dedicated rate limit of 10,000 RPM...
</retrieved_context>
```

The `<retrieved_context>` XML tag helps models with XML-structured prompting (Claude, Gemini). The numbered format makes it easy for the LLM to cite specific passages ("According to [1]..."). Including the relevance score gives the LLM an explicit signal to discount low-confidence results.

### Handling long retrieved content

When retrieved passages are long (full documents, multi-paragraph excerpts), injection can exhaust the context window. Two strategies:

**Chunk and select:** before injection, run a sentence-level relevance pass over each retrieved document and extract the 2–4 most relevant sentences. Inject those sentences plus the document metadata, not the full text. This reduces injection overhead by 60–80%.

**Compression:** have a small LLM (7B or 13B parameter model) run a retrieval-focused summary pass over each document, extracting the sentence(s) that are most relevant to the current query. Inject the compressed summaries. This is more expensive than sentence selection but handles cases where the key information spans multiple sentences.

---

## 7. Retrieval as a tool: the agent decides when to retrieve

The highest-leverage architectural change you can make to a naive agentic RAG system is to convert retrieval from a background process (always runs at the start) to an explicit tool (the agent decides to call it).

![Always-retrieve vs retrieval-as-tool](/imgs/blogs/retrieval-augmented-agents-6.webp)

The before/after above is not hypothetical — in production systems we have measured, converting from always-retrieve to retrieval-as-tool reduces retrieval calls from 100% of reasoning steps to ~20–35% of reasoning steps, cutting total token cost by 40–60% with negligible impact on answer quality. The quality holds because the steps that do not retrieve are genuinely computation-bound (math, code, reasoning from facts already in context), and forcing retrieval on those steps just injects irrelevant documents that the LLM correctly ignores anyway.

### Tool design choices

The most important design decision is the tool's description. LLMs are extremely sensitive to the description field — it is what they use to decide whether to call the tool. A poorly worded description leads to systematic over- or under-retrieval.

**Overly general (over-retrieves):**
```
"Search the knowledge base for information."
```
The model calls this on almost every step because "information" is always at least marginally relevant.

**Overly specific (under-retrieves):**
```
"Search for product documentation when the user asks about a specific feature."
```
The model only calls this when the prompt explicitly matches the use case; it misses retrieval opportunities on paraphrases.

**Calibrated:**
```
"Search the company knowledge base for relevant context. Call this tool when:
- You need factual information about company products, policies, or procedures
- You need to verify a claim that you are uncertain about
- The user asks about a specific case or example that requires sourced data
Do NOT call this for: general reasoning, math calculations, or information you are already confident about."
```

The negative examples in the description are crucial. Without them, weaker models over-retrieve on the "negative" cases.

### Multi-tier retrieval tools

For agents with access to multiple knowledge stores (a public web index, a private document corpus, a structured database), expose separate retrieval tools per store and let the model route:

```python
tools = [
    {
        "name": "search_knowledge_base",
        "description": "Search the company's internal documentation, wikis, and technical specifications."
    },
    {
        "name": "search_web",
        "description": "Search the public web for recent events, external research, and general knowledge. Use when the question requires information published after your training cutoff or from external sources."
    },
    {
        "name": "query_database",
        "description": "Run a structured query against the analytics database. Use for specific metrics, counts, or time-series data about users, products, or transactions."
    }
]
```

The model learns to route based on the information type. This is more precise than routing at the retrieval layer (where you'd need a classifier to decide which store to query) and takes advantage of the LLM's language understanding to interpret the intent behind the query.

### Confidence-gated retrieval

One refinement: before calling the retrieval tool, have the LLM emit a confidence score for its current answer. If confidence is above a threshold (say 0.85), skip retrieval; if below, retrieve. This requires either a separate calibration prompt or a model that produces well-calibrated logit-based confidence.

```python
def confidence_gated_step(question: str, context: str, llm: LLM) -> dict:
    """Attempt to answer, retrieve only if confidence is low."""
    # First pass: attempt answer without retrieval
    response = llm.generate(f"""Context: {context}

Question: {question}

Answer the question if you are confident. If you are not confident, 
respond with "RETRIEVE: <query>" where <query> is the retrieval query to use.

Response:""")
    
    if response.startswith("RETRIEVE:"):
        query = response[9:].strip()
        retrieved = retriever.retrieve(query)
        # Second pass: answer with retrieved context
        return {"retrieved": True, "docs": retrieved}
    else:
        return {"retrieved": False, "answer": response}
```

---

## 8. Handling retrieval failures: empty results, irrelevant results, conflicting results

Production retrieval systems fail. A well-designed agentic RAG system has explicit handling for each failure mode. Without it, retrieval failures propagate silently into agent reasoning and produce confident-sounding wrong answers.

![Retrieval failure handling decision tree](/imgs/blogs/retrieval-augmented-agents-8.webp)

### Empty results

Empty results occur when no document in the corpus has sufficient similarity to the query. Common causes: the query is outside the corpus domain, the document was indexed with different terminology, the query is too specific, or the retrieval threshold is too high.

**Fallback strategy 1: Query widening.** Remove filters, reduce specificity, and broaden the query before retrying. Programmatic widening: strip rare terms, replace specific entities with broader categories, or remove temporal constraints.

```python
def widen_query(original_query: str, llm: LLM) -> str:
    """Broaden a query that returned no results."""
    prompt = f"""This query returned no results: "{original_query}"

Rewrite it as a broader, more general query that might find relevant documents.
Keep the core information need but remove overly specific constraints."""
    return llm.generate(prompt, max_tokens=100)

def retrieve_with_fallback(query: str, retriever, max_retries: int = 2) -> list[Document]:
    results = retriever.retrieve(query, k=5)
    
    for attempt in range(max_retries):
        if results:
            return results
        widened = widen_query(query, llm)
        results = retriever.retrieve(widened, k=5)
    
    # Final fallback: return empty and let the agent reason from memory
    return []
```

**Fallback strategy 2: Parametric fallback.** If retrieval returns nothing after widening attempts, tell the agent explicitly and let it reason from its parametric knowledge. The key is making this explicit in the prompt:

```
Retrieved context: [No relevant documents found. Reasoning from training knowledge only — confidence may be lower.]
```

This prevents the agent from hallucinating with confidence. The explicit "(confidence may be lower)" signal prompts the agent to hedge appropriately.

### Irrelevant results

Irrelevant results are more dangerous than empty results because they look like real context. The agent processes them as if they were relevant, potentially anchoring on the irrelevant content.

Detection: relevance scoring at retrieval time. Every retrieval result should carry a similarity score. If all top-k results have scores below a threshold (commonly 0.6 for cosine similarity with normalized embeddings), treat this as a soft failure and do not inject the results.

```python
RELEVANCE_THRESHOLD = 0.60

def filtered_retrieve(query: str, retriever, k: int = 5) -> tuple[list[Document], bool]:
    """Retrieve with relevance filtering. Returns (docs, all_filtered_out)."""
    results = retriever.retrieve(query, k=k, return_scores=True)
    
    filtered = [(doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD]
    
    if not filtered:
        return [], True  # all results below threshold
    
    docs = [doc for doc, _ in filtered]
    return docs, False
```

When all results are below threshold, escalate to cross-encoder re-ranking. Cross-encoders have higher precision than bi-encoders on relevance judgment; a passage that scores 0.55 on bi-encoder cosine similarity might score 8.5/10 on cross-encoder relevance. Use the cross-encoder as a second opinion before discarding.

Log every retrieval miss — the corpus of failed queries is your highest-signal dataset for indexing improvements, new document ingestion, and query rewriting fine-tuning.

### Conflicting results

Conflicting results occur when the retrieved passages make contradictory claims. This is particularly common for time-sensitive information (policy documents with multiple versions, product specs after a feature change) and for contested factual domains (clinical studies with contradictory findings, legal rulings across jurisdictions).

The safest handling strategy is source recency: when documents conflict, prefer the more recently published one and note the conflict to the agent.

```python
def detect_conflicts(docs: list[Document]) -> list[tuple[Document, Document]]:
    """Identify pairs of documents with potentially conflicting claims."""
    conflicts = []
    for i, d1 in enumerate(docs):
        for d2 in docs[i+1:]:
            # Simple heuristic: high similarity but opposite sentiment/assertion
            similarity = cosine_similarity(embed(d1.text), embed(d2.text))
            if similarity > 0.7:  # similar topic...
                if contradicts(d1.text, d2.text):  # ...but conflicting claims
                    conflicts.append((d1, d2))
    return conflicts

def resolve_conflict_prompt(d1: Document, d2: Document, query: str) -> str:
    return f"""Two sources disagree on this topic: "{query}"

Source 1 ({d1.date}): {d1.text[:300]}

Source 2 ({d2.date}): {d2.text[:300]}

Use the more recent source ({max(d1.date, d2.date)}) as the primary reference, 
but note the disagreement in your response."""
```

For high-stakes domains (medical, legal, financial), conflicting results should trigger an escalation rather than an autonomous resolution. Log the conflict, surface it to the user, and do not let the agent make decisions on the basis of self-resolved contradictions.

---

## 9. Multi-hop retrieval: iterative retrieval where each result informs the next query

Multi-hop retrieval is the pattern you need when answering a question requires bridging multiple pieces of evidence that are stored separately. "What was the market cap of the company founded by the engineer who led GPT-3's training at the time of their Series B?" requires: who led GPT-3 training → what company did they found → when was its Series B → what was the market cap at that date. No single document contains all four pieces.

![Multi-hop retrieval: iterative query trace](/imgs/blogs/retrieval-augmented-agents-7.webp)

### The iterative retrieval loop

The structure is simple: retrieve, update state, generate next query, repeat.

```python
def multi_hop_retrieve(question: str, llm: LLM, retriever, max_hops: int = 4) -> dict:
    """Iteratively retrieve until the question can be answered."""
    context_so_far = ""
    all_retrieved = []
    
    current_query = question  # start with original question
    
    for hop in range(max_hops):
        # Retrieve on current query
        docs = retriever.retrieve(current_query, k=3)
        all_retrieved.extend(docs)
        
        # Update context
        context_so_far += f"\n\n[Hop {hop+1} results]\n"
        context_so_far += "\n".join(doc.text for doc in docs)
        
        # Ask LLM: can I answer now, or do I need another hop?
        check_prompt = f"""Question: {question}

Context gathered so far:
{context_so_far}

Can you answer the question with this context? 
- If yes, answer it.
- If no, say "NEED: <what specific information is still missing>"

Response:"""
        
        response = llm.generate(check_prompt, max_tokens=200)
        
        if not response.startswith("NEED:"):
            # Answer found
            return {"answer": response, "hops": hop+1, "docs": all_retrieved}
        
        # Generate next query from the gap
        missing = response[5:].strip()
        next_query_prompt = f"""I need to find: {missing}

Based on the context I have:
{context_so_far[-500:]}

Write a retrieval query to find this missing information."""
        
        current_query = llm.generate(next_query_prompt, max_tokens=100)
    
    # Max hops reached without answer
    return {"answer": "Could not find sufficient evidence.", "hops": max_hops, "docs": all_retrieved}
```

### When multi-hop retrieval goes wrong

**Compounding errors.** If hop 1 retrieves a slightly wrong fact (wrong entity name, wrong date), hop 2 retrieves based on that wrong fact, and the error compounds. By hop 3, you are far from the right answer. Mitigation: include the confidence score of each retrieved document in the state, and have the LLM flag low-confidence chains.

**Infinite loops.** The agent keeps retrieving because each new document raises new questions. Set a hard hop limit (3–5 is typical for most QA tasks), and have a "best effort answer" path that fires when the limit is hit.

**Off-track retrieval.** After hop 2, the agent generates a query for "Series B valuation" and retrieves a document about Series B mechanics in general, not the specific company's Series B. This happens when the query generation step loses track of the specific entity context. Mitigation: maintain an entity state dict and enforce that generated queries reference the tracked entities.

### Multi-hop vs regular RAG: when to use which

Use multi-hop retrieval when:
- The question involves two or more entities connected by a relationship not stored in any single document
- The agent's trace shows repeated retrieval calls on related sub-topics
- Your eval set shows low recall on "bridge questions" (questions that require linking two facts)

Do not use multi-hop retrieval when:
- A single well-formed query retrieves the answer (use multi-query instead)
- Latency is critical (multi-hop is inherently sequential and slow)
- The knowledge base is small enough to exhaustively search (just retrieve top-20 and let the LLM reason over the full set)

---

## 10. Cost model: retrieval frequency × result size × context injection overhead

Retrieval is not free. Every call costs latency, compute, and tokens — and in an agentic system where retrieval may happen multiple times per turn, those costs multiply.

![Retrieval cost model: frequency × result size](/imgs/blogs/retrieval-augmented-agents-9.webp)

### The three cost levers

**Retrieval frequency** is the number of retrieval calls per user turn. For an always-retrieve agent doing 8 steps per turn, this is 8. For a retrieval-as-tool agent where 25% of steps retrieve, this is ~2. The difference is 4×.

**Result size** is the number of tokens in the retrieved passages injected into the prompt. Each retrieved passage typically runs 200–500 tokens; top-k of 5 passages is 1,000–2,500 tokens per call. At 8 retrieval calls per turn, that is 8,000–20,000 tokens of retrieval overhead per user turn — significant relative to the total context window.

**Injection overhead** is the LLM inference cost of processing the retrieved context. This is not a free operation: larger prompts take longer to process (KV-cache misses, longer attention computation), and at scale, the difference between a 2,000-token prompt and a 10,000-token prompt is material. For Claude and GPT-4 class models at standard API rates, each 1,000 additional tokens per call adds ~$0.01–0.03 to the per-turn cost.

### Back-of-envelope for a production agent

```python
# Cost model for a production customer support agent
RETRIEVAL_CALLS_PER_TURN = 2.5      # average for retrieval-as-tool
PASSAGES_PER_CALL = 4               # top-4 after re-ranking
TOKENS_PER_PASSAGE = 350            # average passage length
INPUT_TOKEN_PRICE = 0.000003        # $/token for GPT-4o input
DAILY_TURNS = 10_000                # daily active turns

retrieval_tokens_per_turn = RETRIEVAL_CALLS_PER_TURN * PASSAGES_PER_CALL * TOKENS_PER_PASSAGE
# = 2.5 × 4 × 350 = 3,500 tokens/turn

daily_retrieval_token_cost = DAILY_TURNS * retrieval_tokens_per_turn * INPUT_TOKEN_PRICE
# = 10,000 × 3,500 × 0.000003 = $105/day = ~$3,150/month

# Compare: always-retrieve (8 calls/turn)
always_retrieve_cost = (8 / 2.5) * daily_retrieval_token_cost
# ≈ $10,080/month — 3.2× more expensive
```

The cost difference between always-retrieve and retrieval-as-tool is not 3.2× in retrieval API calls (those are cheap) — it is 3.2× in LLM input tokens, which drives most of the cost at scale.

### Optimization strategies

**Passage compression.** Before injection, compress each retrieved passage to its most relevant sentences using a small model. A 400-token passage compressed to 80 tokens saves 320 tokens per passage × 4 passages × 2.5 calls = ~3,200 tokens/turn — nearly as much as switching from always-retrieve to retrieval-as-tool.

**Caching at the prompt level.** Modern LLM APIs (Claude's prompt caching, GPT-4's context caching) allow the system prompt and frequently-retrieved documents to be cached at the KV layer. If your agent frequently retrieves the same documents (a product FAQ, a technical specification), put those documents in the system prompt prefix and cache them — paying only the cache storage cost on repeated calls.

**Result size tuning.** Most teams default to top-5 or top-10 retrieved passages. In practice, the marginal value of passage 6+ is often low — the answer is usually in the top-3. Run an ablation over your eval set: what is NDCG@k for k=1,2,3,4,5? If NDCG@3 is within 5% of NDCG@5, use 3 and save 40% of injection overhead.

---

## 11. Case studies

### Case study 1: The enterprise knowledge bot that retrieved against the wrong tenant

**Context:** A SaaS company built an internal knowledge bot for customer support. The RAG system retrieved from a shared vector index across all customer tenants. For the first few months, it worked well. Then a support agent noticed the bot occasionally cited documentation from a competitor's product.

**What happened:** The index was multi-tenant but the retrieval system lacked tenant-scoped filtering. When a customer support agent asked "how do I configure the SMTP integration," the query was semantically close to passages from another customer's SMTP integration documentation that happened to be in the same index. The bi-encoder did not distinguish between tenants.

**Root cause:** The architecture treated retrieval as a purely semantic problem (find the most similar passages) and ignored the access control dimension. Retrieval should have been `retrieve(query, tenant_id=current_tenant)` with hard filtering, not `retrieve(query)` followed by a hope that the semantic similarity would align with tenant boundaries.

**Fix:** Added mandatory metadata filtering at the retrieval layer. The vector store index was rebuilt with tenant tags on all documents, and every retrieval call was decorated with a `filter={"tenant_id": current_tenant_id}` predicate evaluated before semantic similarity scoring. The fix reduced cross-tenant leaks to zero and had no measurable impact on retrieval quality for correct-tenant queries.

**Lesson:** Multi-tenant retrieval is a security boundary, not a semantic preference. Hard filtering must happen at the index level, not the re-ranking level.

---

### Case study 2: The research agent that halted on the first empty retrieval

**Context:** A pharmaceutical research agent was designed to synthesize literature reviews. It used multi-hop retrieval to chain from a drug name to its mechanism of action to clinical trial outcomes. The team used query-triggered retrieval with a relevance threshold of 0.65.

**What happened:** For novel compounds with limited published literature, the first retrieval hop frequently returned nothing above threshold. The agent's `retrieve_with_fallback` function fell through to "no documents found" and the agent's next reasoning step said "insufficient evidence to continue" — halting mid-synthesis for 35% of compound queries.

**Root cause:** The fallback logic treated all empty-result cases identically. In reality, three distinct cases existed: (1) the compound is well-studied but the query was poorly formed, (2) the compound is novel with truly limited literature, and (3) the query used terminology inconsistent with the corpus vocabulary. Only case 1 benefited from query widening; case 2 needed an alternative knowledge source; case 3 needed terminology normalization.

**Fix:** The team built a three-path fallback: first, semantic similarity score inspection (if top-1 result is 0.55–0.64, run cross-encoder re-rank before discarding); second, query normalization using a drug name standardization API (INN names, IUPAC identifiers, trade names); third, web search as an alternative source when the proprietary corpus is sparse.

After the fix, the completion rate for novel compound queries improved from 65% to 91%.

---

### Case study 3: The multi-hop agent that looped on ambiguous entities

**Context:** A financial intelligence agent was designed to answer questions like "what did the CFO say about margin pressure in the Q3 call?" using a corpus of earnings call transcripts. The multi-hop architecture: identify the company → find the Q3 transcript → find the CFO's statements → extract margin-related remarks.

**What happened:** For companies with common names (e.g., "Apollo" could be Apollo Global Management, Apollo Medical, or Apollo Tyres), the first hop retrieved transcripts from the wrong company 23% of the time. The agent happily proceeded with the wrong entity, producing confident-sounding summaries of the wrong company's CFO remarks.

**Root cause:** Entity disambiguation happened entirely through semantic similarity. "Apollo Q3 earnings" was close to multiple companies' transcripts. The agent had no mechanism to detect entity ambiguity and ask for clarification before proceeding.

**Fix:** The team added an entity disambiguation step before the first retrieval hop. For company names with multiple candidates in the index, the agent was instructed to either use additional context (ticker symbol, industry, jurisdiction) to disambiguate, or to surface the ambiguity to the user before proceeding. The disambiguation logic used metadata filtering: the agent first ran a structured lookup against a company name → ticker mapping, then used the ticker as a hard filter on retrieval.

After the fix, the wrong-entity rate dropped to 2.3%.

---

### Case study 4: The chatbot that ignored its retrieved context

**Context:** A customer service chatbot used standard RAG: retrieve top-5 passages, inject at the beginning of the user turn, have the LLM answer. The retrieved passages were accurate, but A/B testing showed the bot answered incorrectly 28% of the time on questions where the answer was in the retrieved context.

**What happened:** The team ran attention analysis and found the model was attending strongly to passages 1 and 5 but very weakly to passages 2, 3, and 4. The correct answer was in passage 3 in 40% of the failing cases — right in the lost-in-the-middle zone.

**Root cause:** The classic lost-in-the-middle failure. The injection position (all passages prepended, in order of bi-encoder score) systematically placed relevant passages in low-attention positions.

**Fix:** Three changes: (1) re-ordered passages so the highest-relevance passage was first, second-highest was last, and middle passages were in between (exploiting primacy and recency simultaneously); (2) moved retrieved context injection from before the user message to immediately before the assistant's response slot; (3) reduced top-k from 5 to 3 (higher quality, less noise, fewer middle-zone positions).

Answer accuracy on the failing questions improved from 72% to 89%.

---

### Case study 5: The agent that over-retrieved on structured calculations

**Context:** A financial modeling agent combined retrieval with a Python code execution tool. It could retrieve data from a market data corpus and run Python calculations on that data. The agent was built with always-retrieve (retrieval on every step).

**What happened:** The agent was retrieving on pure computation steps — steps where it was writing Python code to calculate EBITDA from numbers already in its context. These retrieval calls returned irrelevant documents about financial modeling methodology, which the agent then sometimes incorporated into its reasoning, leading to calculation errors.

**Root cause:** The always-retrieve architecture made no distinction between knowledge-acquisition steps (where retrieval is valuable) and computational steps (where retrieval is actively harmful — it introduces noise into an otherwise deterministic calculation).

**Fix:** Switched to retrieval-as-tool with a calibrated tool description that explicitly excluded retrieval for mathematical calculations and code execution. Added a classifier at the retrieval call site that checked whether the current agent step was "reasoning/knowledge" or "computation/tool-call" — the latter blocked retrieval entirely.

After the fix, the calculation error rate dropped from 11% to 1.8%. The reduction in unnecessary retrieval also cut average turn latency by 340 ms.

---

### Case study 6: Stale retrieval in a scheduled news summary agent

**Context:** A hedge fund ran a news summary agent that generated morning briefings. The agent used a shared Redis cache with a 4-hour TTL. Documents were ingested from news feeds every 30 minutes.

**What happened:** On the morning of a major surprise central bank announcement, the cache contained documents from before the announcement (cached 2 hours earlier). The agent's morning briefing described "expected rate decision" when the actual decision had already been announced.

**Root cause:** The TTL-based cache worked fine in normal conditions but had no mechanism to invalidate on high-volatility events. The news feed ingestion was real-time, but the cache was not invalidated when new documents arrived — it simply waited for TTL expiration.

**Fix:** Added a pub/sub channel on the news ingestion pipeline. When a new document matched a "high-priority entity" list (central banks, major index movers, scheduled economic releases), an invalidation signal was sent to the cache layer. The cache for affected queries was cleared and rebuilt on next access.

The fix also added a "cache age" field to the retrieved context injection format, so the LLM could see that it was working with context from a specific timestamp and hedge appropriately on time-sensitive claims.

---

### Case study 7: Multi-query overload in a real-time customer support system

**Context:** A customer support platform deployed multi-query retrieval (4 paraphrases per query) to improve retrieval recall. Initial offline evals showed a 12% improvement in answer accuracy.

**What happened:** In production, the median response latency jumped from 1.2 seconds to 4.8 seconds. Support agents started routing around the bot. The 12% accuracy improvement was real but irrelevant at 4× latency.

**Root cause:** Multi-query retrieval with 4 paraphrases meant 4 bi-encoder inference calls, 4 vector store searches, and 400 cross-encoder inference calls (100 candidates × 4 queries) per user turn, all in the request-response path.

**Fix:** The team moved to async multi-query: issue the first retrieval call synchronously, stream the initial response, and issue the remaining 3 retrieval calls asynchronously in background. If the background calls return before the LLM finishes generating, incorporate them into a "refinement" pass. This reduced median latency to 1.6 seconds while preserving most of the accuracy gain.

For truly latency-critical queries (the support system had a "live escalation" mode where a human would take over if the bot did not answer in 2 seconds), the system fell back to single-query retrieval entirely.

---

### Case study 8: Conflicting context in a legal document analysis agent

**Context:** A legal tech company built an agent to analyze contracts and answer questions about specific clauses. The corpus contained multiple versions of standard contracts, amendment documents, and legal memos. Multi-hop retrieval was used to trace clause evolution.

**What happened:** The agent frequently produced answers that combined language from different versions of the same contract — for example, citing the liability cap from the 2021 version and the governing law from the 2023 amendment, neither of which was the current operative version.

**Root cause:** The retrieval system had no version tracking. From the agent's perspective, all documents in the corpus were equally valid sources. The conflict detection logic (cosine similarity > 0.7 between conflicting pairs) was not triggered because the documents described the same topic but did not technically contradict each other at the semantic level — they were successive versions, not opposing claims.

**Fix:** Added document lineage metadata to the corpus: each document was tagged with a `superseded_by` field pointing to any later version. Retrieved documents were post-filtered: if a document had a `superseded_by` field pointing to another document in the corpus, the older document was dropped from the result set.

The agents also received a modified injection format that included the document version and effective date, and were instructed to cite the "most recent applicable version" explicitly in their answers. Clause-attribution accuracy improved from 61% to 94%.

---

## 12. When to use agent-native retrieval vs plain RAG

![Recommended retrieval strategy by agent type](/imgs/blogs/retrieval-augmented-agents-10.webp)

Not every agent needs the full stack described in this post. The grid above gives a quick reference, but here is the reasoning behind the recommendations.

### Use plain RAG (once-per-turn, no tool, no multi-hop)

- The agent always needs context, the context always comes from the same query, and the user's question is a good enough query. Customer support FAQ bots fit this profile.
- You are building an MVP and latency/cost/complexity matter more than peak accuracy.
- Your agent's turns are short (2–3 steps) and the relevant context does not change within a turn.

### Use retrieval-as-tool (on-demand)

- The agent has diverse step types (reasoning, calculation, knowledge lookup, code execution) and not all steps need retrieval.
- You have measured or expect >30% of steps are retrieval-irrelevant (math, pure reasoning, tool execution).
- Your LLM (GPT-4, Claude Sonnet or higher) is well-calibrated on retrieval decisions.

### Use multi-hop retrieval

- Questions frequently require chaining across multiple documents.
- Your eval set shows low recall on "bridge questions."
- Latency is tolerable (multi-hop is inherently sequential: 3 hops × 150 ms = 450 ms minimum).

### Use proactive/scheduled retrieval

- The agent's knowledge needs are highly predictable (domain expert, scheduled report generation).
- You have latency constraints that rule out on-demand retrieval.
- The knowledge domain is well-defined and the cache TTL can be set with confidence.

### Do not use retrieval

- All relevant information is already in the context (summarization agents, code agents where the code is provided).
- The agent's accuracy is limited by reasoning, not knowledge (mathematical proof agents, planning agents with fully-specified state).
- The document corpus is small enough to fit in the context window entirely.

### The decision matrix

| Agent type | Step diversity | Latency budget | Knowledge volatility | Recommendation |
|---|---|---|---|---|
| QA / factual lookup | Low | Medium | Low | Retrieval-as-tool |
| Research / synthesis | High | High | Medium | Multi-hop + multi-query |
| Customer support | Medium | Low | Low | Proactive + scheduled |
| Code generation | High | Low | Low | Retrieval-as-tool (docs only) |
| Monitoring / alerts | Low | Very low | High | Scheduled (TTL-gated) |
| Planning agent | High | High | Low | None or once-per-turn |
| Legal / medical analysis | Medium | High | Low | Multi-hop + conflict detection |

---

## Putting it together

Agentic RAG is a spectrum, not a binary. At one end: always-retrieve with no re-ranking and no failure handling — it works for demos, fails in production. At the other end: query-triggered multi-hop retrieval with HyDE query generation, cross-encoder re-ranking with MMR diversity, adversarial conflict detection, and per-step injection — it works in production for high-stakes applications and is worth the complexity.

Most production systems live in the middle. The right default: retrieval-as-tool (the agent decides when), single retrieval call per decision, top-3 passages after cross-encoder re-ranking, injected immediately before the current task. That setup captures 80% of the quality benefit at 20% of the complexity cost.

Add multi-hop when your eval set shows it is needed. Add multi-query when you have latency budget and retrieval coverage is the bottleneck. Add conflict detection when the corpus has overlapping or versioned documents and wrong answers carry real cost.

The most important thing is to treat retrieval failures as first-class events. Empty results, irrelevant results, and conflicting results each need explicit code paths, not silent pass-through. An agent that hallucinates because retrieval failed without a fallback will do so confidently, repeatedly, and in exactly the cases where retrieval was most needed.

For foundations on the vector storage layer, see [vector database internals](/blog/machine-learning/ai-agent/vector-database). For the broader picture of how retrieval relates to other memory systems in agents, see [RAG vs agent memory](/blog/machine-learning/ai-agent/rag-vs-agent-memory) and [multi-signal memory retrieval](/blog/machine-learning/ai-agent/multi-signal-memory-retrieval). For the baseline retrieval architecture that the agent patterns extend, see [basic RAG](/blog/machine-learning/ai-agent/basic-rag).
