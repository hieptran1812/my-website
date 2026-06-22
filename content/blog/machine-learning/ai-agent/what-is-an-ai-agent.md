---
title: "What Is an AI Agent? The Perceive-Reason-Act-Remember Loop"
date: "2026-06-22"
description: "A precise definition of AI agents through the four-pillar loop, with an agency spectrum that shows exactly where a chatbot ends and an agent begins."
tags: ["ai-agents", "llm", "agent-architecture", "machine-learning", "deep-learning", "nlp", "system-design", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 50
---

Every engineer who has shipped a product in the last two years has heard the pitch: "We should add an AI agent." Sometimes the feature that ships is genuinely agentic — it calls tools, loops until done, adapts to what it finds, and accomplishes tasks that would have required a human operator. More often it is a chatbot with extra steps: a fixed if/else tree wearing an LLM-powered coat, dressed up in agent marketing because that is what investors and product managers want to hear.

The damage from this confusion runs in both directions. Teams over-engineer chatbots into full agent frameworks — paying 10× the latency and 20× the token cost for a problem that a single well-crafted prompt would have solved. Teams also under-engineer genuinely agentic tasks — wrapping a multi-step problem inside a single LLM call, then debugging the resulting mess when the model hallucinates step three because it ran out of context window before it got to the relevant information.

Getting the definition right is a load-bearing engineering decision. This post builds that definition from first principles through the four-pillar loop, then puts it to work on the agency spectrum so you can call the right shot for your next system. By the end you will have a precise vocabulary for the difference between a chatbot, a RAG application, a reactive agent, and an autonomous agent — and a practical decision framework for knowing which one to build.

![The Perceive-Reason-Act-Remember Loop](/imgs/blogs/what-is-an-ai-agent-1.webp)

## The Chatbot Illusion

Here is what a chatbot actually does: it accepts a message, runs one LLM inference, and returns a response. The context window accumulates turn-by-turn history, which creates the feeling of continuity, but nothing persists outside the conversation. No tool is called unless a human manually pastes in the result. No plan is formed beyond "predict the next token." When the conversation window fills or the session ends, the slate is wiped.

Now here is what a large fraction of "AI agents" shipped in 2023–2024 actually do: they accept a message, run one LLM inference, call a single tool, and return a response. The tool call is real, but there is still no loop, no plan, no persistent state. The user's intent is either satisfied in one shot or it fails. That is not an agent. That is a chatbot with a function call bolted on.

The distinction matters the moment your task requires more than one tool call or more than one reasoning step. Consider a concrete example: "Research this company, write a one-page briefing, and schedule a follow-up meeting if the revenue trend is negative." That task requires querying a financial API, reading web search results, deciding whether a follow-up trigger fires based on the data, writing a structured document, and conditionally calling a calendar API — in a sequence whose exact shape is not known at the start. You cannot know whether you need to call the calendar API until you have interpreted the financial data. No single LLM call handles this. No fixed if/else tree generalizes to the full range of companies a user might name or the full range of financial situations they might be in.

The chatbot illusion persists because LLMs are extraordinarily good at simulating agency for easy cases. Ask a chatbot to book a flight and it will cheerfully give you step-by-step instructions for booking the flight yourself. It looks like it is helping. It is not acting. Ask a chatbot for a research summary and it will synthesize whatever is in its training data into something that reads like research. It looks like it is researching. It is not retrieving. Ask a chatbot to analyze your CSV file and it will describe what kind of analysis would be appropriate. It looks like it is analyzing. It is not executing.

The confusion is not the user's fault. The language models themselves speak in the first person about "doing" and "finding" and "checking" things, when they are in fact predicting sequences of tokens about doing and finding and checking. Users anthropomorphize the output; the product teams lean into that anthropomorphization; the marketing teams run with it; and by the time the engineering team is asked why the "agent" cannot actually do the task, everyone has lost track of what "agent" was supposed to mean.

There is a precise vocabulary for the difference, and it starts with four pillars.

## The Four Pillars: Perceive, Reason, Act, Remember

Every system that deserves to be called an AI agent implements, at minimum, four capabilities operating in a closed loop. Strip out any one of them and you are no longer looking at an agent — you are looking at a simpler system that may or may not be the right tool for your task.

**Perceive** is the ability to receive and interpret information from the environment. For an LLM-based agent this is not limited to user text — it includes tool outputs, database query results, file contents, API responses, browser page HTML, shell stdout and stderr, timestamps, environment variables, system state, and in multimodal systems, images and audio. The richer the perception, the broader the range of tasks the agent can attempt. An agent that can only perceive text is severely limited in what it can act on. An agent that can perceive a screenshot, read the error message in a terminal, and parse the output of a debugging command has the raw material to solve problems that text-only agents cannot even formulate.

**Reason** is the ability to map from a perceived state to a decision about what to do next. In an LLM-based agent, reasoning is the inference call: the model takes the assembled context — system prompt, prior turns, retrieved facts, tool results — and predicts what the best next action is. This is not magical planning by a silicon brain. It is autoregressive prediction over a rich context, where the training data included vast amounts of human reasoning, planning, debugging, and problem-solving across nearly every domain. When the prompt is well-structured and the context contains the right information, this prediction reliably produces multi-step plans, self-corrections, and conditional branches. When the context is poorly assembled or the task exceeds the model's reliable reasoning horizon, prediction degrades — and the degradation is not always visible in the output, which is what makes hallucination and context loss so dangerous.

**Act** is the ability to change state in the world. Actions are not limited to returning text to a user. A well-designed agent can call REST APIs, execute shell commands, write to files, query and update databases, send emails and messages, spawn child agents, manipulate browser state, trigger workflows, and interact with external services. The full action taxonomy — read, write, compute, communicate, orchestrate — runs from non-destructive read-only lookups all the way to irreversible side effects. The key property that separates an agent from a chatbot is that its actions have consequences outside the conversation window. The agent changes something. Those changes persist after the conversation ends.

**Remember** is the ability to maintain state across the steps of a task and, in a full agent, across sessions. Memory in a production agent is stratified into multiple types with different persistence, access patterns, and cost profiles: in-context memory (what fits in the active window), episodic memory (a log of prior observations retrieved via semantic search), semantic memory (a knowledge base of domain facts and tool documentation), and procedural memory (learned patterns about which tool sequences work for which task classes). The loop between Act and Remember is what allows an agent to observe the result of an action, update its world model, and use that updated model to inform the next Perceive step. Without Remember, each step starts from scratch. With Remember, the agent builds up an understanding of the world — and of the task — that compounds across steps.

These four pillars form a cycle, not a line. Each pillar feeds into the next: perception produces a context that reasoning operates on; reasoning produces a decision that action executes; action produces an observation that memory stores; memory produces grounding that shapes the next perception. An agent halts when it determines the task is complete or when an external condition — time limit, error budget, human override, iteration cap — forces termination. The loop is the defining structural property. A system with all four pillars but no loop is not an agent; it is a pipeline.

## The Agency Spectrum

The four-pillar definition is binary — a system either has all four pillars operating in a loop or it does not — but "having" a pillar is not a boolean. Each pillar exists on a quality gradient. The perception system may be simple (text-only, single document) or rich (multimodal, multi-source with real-time retrieval). The reasoning system may be shallow (single LLM call, no planning) or deep (multi-step chain of thought, explicit sub-task decomposition). The action space may be narrow (one or two read-only tools) or broad (web, file system, APIs, orchestration). The memory system may be in-context only (vanishes at session end) or persistent across sessions with rich episodic recall.

That gradient is what the agency spectrum captures.

![The Agency Spectrum: Five Tiers of Autonomy](/imgs/blogs/what-is-an-ai-agent-3.webp)

**Tier 1 — Hard-coded Script.** No LLM, no perception beyond predefined inputs, no reasoning, no memory. A Python function that calls an API and formats the response. Completely deterministic. The right choice when the task is well-defined, the input space is narrow, and you need every run to be auditable and reproducible. Do not underestimate Tier 1. A well-engineered script is faster, cheaper, and more reliable than any agent for the tasks it can handle.

**Tier 2 — Chatbot.** An LLM that maintains a conversation history within a context window. Perception is limited to the conversation turns. No external tools. No persistent memory outside the session. Reasoning is real but reactive — the model responds to what the user writes, rather than planning ahead. The right choice for interactive Q&A, text generation, editing, summarization, and any task that can be completed in one turn with the information available in the user's message or the base model's weights.

**Tier 3 — RAG Application.** A chatbot augmented with retrieval. At query time, the system fetches relevant documents from a vector store or search index and injects them into the prompt before the LLM call. The LLM can now reason over information that was not in its training data. Still predominantly single-turn or short-turn. Still no persistent state across sessions. Still no loop. The right choice when the task requires knowledge that cannot fit in the base model's weights — a company's internal documentation, a product catalog, a regulatory database, a domain-specific knowledge base that changes faster than model training cycles.

**Tier 4 — Reactive Agent.** The system loops. It calls tools, observes the results, feeds those results back into the next LLM call, and continues until a stop condition is met. It maintains state within a session. It can recover from partial failures by retrying a different approach. It can route dynamically based on what it finds. This is the tier where most production "AI agents" should land, and where the engineering challenges start to bite in earnest: context management, tool error handling, loop termination, observability, cost control.

**Tier 5 — Autonomous Agent.** The system sets and pursues its own sub-goals, manages memory across sessions, self-corrects based on feedback, and can operate without human input for extended periods — hours or days. Current examples include systems like Devin (software engineering over multi-hour sessions), deep research agents (multi-hour web research and synthesis), and certain robotic control loops operating in physical environments. The engineering surface area at this tier is enormous: reliability, safety, cost, alignment, and human oversight all become first-class concerns that require dedicated engineering, not just prompt engineering.

Most tasks that "feel agentic" are Tier 3 or Tier 4. The engineering discipline here is resisting the gravitational pull toward Tier 5 when Tier 4 would suffice. It is also resisting the impulse to build Tier 4 infrastructure when a Tier 3 RAG pipeline would handle 95% of user queries.

## Chatbot vs. Workflow vs. Agent

The agency spectrum compares systems by autonomy tier, but practitioners also need a horizontal comparison: for a given task, should I build a chatbot, a deterministic workflow, or an agent? These three patterns differ along six dimensions that determine which is correct.

![Chatbot vs. Workflow vs. Agent: Six Dimensions](/imgs/blogs/what-is-an-ai-agent-4.webp)

A **chatbot** has no tools, no persistent state, no multi-step reasoning, no dynamic routing, and no self-correction. Its cost is low. Use it when the task fits in one turn and the information needed is either in the base model's weights or in the user's message. Do not add complexity to a chatbot because complexity feels more professional. The best chatbot is the simplest one that accomplishes the task.

A **workflow** adds tool calls and branching, but in a pre-scripted form. The engineer defines the directed acyclic graph at build time: step A calls tool X; if the result matches condition Y, go to step B; otherwise go to step C. The LLM may generate text at individual nodes — writing an email body, summarizing a document, classifying an input — but it does not decide the routing between nodes. Workflows are deterministic, debuggable, and cheap. Use them when the task structure is stable and known in advance: the same tools, the same sequence, the same branching logic run every time.

An **agent** adds dynamic routing, persistent state, and self-correction. The LLM decides at each step which tool to call and what to do with the result. The set of tools required, the number of steps, and the routing logic are all determined at runtime by the model's interpretation of the current context. This flexibility is exactly what makes agents useful — and also what makes them expensive, unpredictable, and hard to debug. Use them when the task structure is genuinely unknown at build time.

The critical insight that many teams miss: agents are not strictly better than workflows. A well-engineered workflow beats an agent on cost, latency, reliability, and debuggability for any task whose structure is fixed. The moment you find yourself hardcoding agent tool sequences in the system prompt — "First, call the CRM tool. Then, call the policy tool. Then, decide whether to escalate." — you have a workflow masquerading as an agent. Refactor it. Give it proper branching logic, remove the LLM from the routing layer, and pay the chatbot rate for the text generation nodes instead of the agent rate for the whole execution.

The right question is not "should we use an agent?" but "what is the minimum tier that accomplishes the task?" Answer that question honestly and you will make better architecture decisions than 80% of the teams currently building with LLMs.

## What Perceive Really Means

Perception is the most underspecified pillar in most agent designs. Practitioners treat it as "the user's message arrives and gets concatenated to the prompt." Production agents need something fundamentally richer than this.

![Full Taxonomy of Agent Perception Inputs](/imgs/blogs/what-is-an-ai-agent-5.webp)

The full taxonomy of what an agent can perceive spans five categories, each requiring different engineering investment:

**Text inputs** are the obvious ones: user messages, documents, PDF extracts, code files, email threads, commit messages, log lines. These arrive as strings and are relatively easy to handle — chunk, embed, retrieve, inject into the prompt. The failure mode is not perception itself but context management: injecting too much text at once degrades the model's ability to attend to any particular part of it. A 200-page policy document injected raw produces worse results than a 500-token chunk that was semantically retrieved based on the current query.

**Structured data** is where most teams underinvest. An agent reasoning over a customer support ticket needs the ticket's status history, the customer's order records, the product's return policy as a typed JSON object — not a prose summary of these things. When structured data is serialized to prose before the LLM sees it ("The customer placed order 8821 on March 12, which was delivered on March 19"), precision is lost and hallucination risk increases. Passing the raw JSON with field names intact ("order": {"id": 8821, "placed": "2024-03-12", "delivered": "2024-03-19"}) keeps the model honest about values. A model that sees the raw data will not confuse order IDs with transaction amounts; a model that sees a prose summary might.

**Tool outputs** are a special perception category because they are the product of the agent's own prior actions. After a shell command runs, stdout and stderr are new perceptions. The agent has to interpret them to decide whether the command succeeded, what to do about a partial failure, and whether the output contains the information it was looking for. After a web search, the result snippets are perceptions that need to be ranked and filtered before they are useful. After an API call, the response body needs to be validated against an expected schema. Each of these steps is an active perception problem — it requires the agent to interpret structured and semi-structured data in the context of its current goal, not just pass the data forward.

**Environment state** is what distinguishes system-automation agents from pure text agents. An agent managing a CI/CD pipeline needs to know which jobs are running, which have failed, and what the current deployment status is before deciding what to do next. An agent managing a calendar scheduling task needs the current time, the user's timezone, and the availability windows of all attendees. An agent managing a code editing task needs to know the current state of the file on disk — not the state when the editing session started, but the current actual state including any changes made by other processes or by the agent's own prior edits.

**Multimodal inputs** are the frontier. An agent that can perceive images can understand screenshots, UI layouts, charts, architectural diagrams, and rendered code output. An agent that can perceive audio can transcribe and reason over spoken instructions, meeting recordings, and customer service calls. These modalities dramatically expand the action space — an agent with vision can interact with GUIs that expose no API, scrape data from rendered charts that are not available in structured form, and validate visual outputs against specifications. They also add new failure modes: vision models hallucinate chart values, audio transcription introduces errors that compound in downstream reasoning, and visual reasoning over complex diagrams is significantly less reliable than text reasoning.

The practical design lesson: every input type your agent receives should have an explicit representation and parsing strategy in your context assembly logic. Do not assume the LLM will correctly interpret a blob of JSON without schema context, or that a 400-line shell output is safe to inject raw. Perception engineering — deciding what to observe, how to parse it, how to filter it, and what level of detail to include in context — is as important as reasoning engineering. Often more so.

A worked example of perception engineering: an agent tasked with diagnosing a production incident receives a 5,000-line application log as its perception input. Injecting the full log raw takes 5,000 tokens, pushes other context out of the window, and makes it harder for the model to identify the relevant log lines. A better perception strategy: pre-filter to extract only the error and warning lines (reducing to 300 lines), group them by timestamp into incident windows (reducing to 5 windows), and inject only the window containing the peak error rate. The model receives 200 tokens of highly relevant log context instead of 5,000 tokens of noise. The same reasoning capability — the same model, the same prompt structure — produces dramatically better diagnosis because the perception layer did its job. Perception engineering is not glamorous. It is also not optional.

Perception also has an important temporal dimension that is easily overlooked. An agent perceiving a file at the start of a long task may be working from a stale snapshot by step 10 if other processes write to that file during the session. An agent perceiving a stock price at the start of a financial analysis may be working from a price that has moved significantly by the time the agent acts on its recommendation. Production agents need to reason about the freshness of their perceptions and re-query when there is a risk that the observed state has changed. This is particularly critical for write actions: the agent should re-read the resource it is about to modify immediately before modifying it, to confirm that its in-context representation matches the current disk state.

## What Reason Really Means

Reasoning is where the LLM earns its keep — and where the most dangerous misconceptions live.

The misconception: LLMs reason by "thinking." They have a model of the world, they run forward simulations, they evaluate options, and they select the best one. The reality is simpler and simultaneously more remarkable: LLMs reason by prediction. When a model appears to plan — "First I will query the API, then I will check the response, then I will decide whether to retry" — it is predicting that this sequence of tokens follows from the context it has been given. The training data included vast amounts of human reasoning in exactly this form: debugging sessions, planning documents, decision logs, engineering postmortems, code reviews. The model has learned to predict the tokens that constitute good reasoning, which — in practice — produces output that functions as good reasoning.

This matters for agent design because it implies two constraints that purely planning-based approaches do not face. First, the quality of reasoning is context-sensitive: give the model a poorly assembled context and the reasoning output degrades, not because the model "got worse" but because the context did not contain the information needed to ground the predictions. Second, reasoning degrades over chain length. Each step of the reasoning chain introduces prediction error. That error accumulates. Empirically, most frontier models maintain reliable coherence up to 5–7 reasoning steps; beyond that, error rates climb significantly and the quality-per-step drops.

![How Reasoning Bridges Perception to Action](/imgs/blogs/what-is-an-ai-agent-6.webp)

The reasoning pipeline for a production agent has four stages, each with its own failure mode and optimization surface:

**Context assembly** determines what goes into the prompt window. This is the highest-leverage design decision in the agent. The composition of system instructions, memory retrievals, tool schemas, conversation history, and the current observation determines the full input to the model. A poorly assembled context — too long, poorly ordered, containing irrelevant noise, missing key constraints — is the single most common cause of bad agent behavior that practitioners misattribute to "the model being bad." The model is often fine; the context is the bug. Investing in context assembly quality — what to include, in what order, at what level of detail — pays larger reliability dividends than switching to a more expensive model.

**LLM inference** is the prediction call itself. The model produces a completion given the assembled context. At this stage, three things can go wrong. Hallucination: the model generates a factual assertion not grounded in the context window. Context loss: the model "drops" an earlier constraint because it was diluted by the mass of later tokens. Format failure: the model does not produce the structured output format required for the downstream parsing step. All three are addressable — hallucination via grounding and retrieval, context loss via context management and pinned memories, format failure via function-calling APIs and structured output modes — but only if you have the observability to detect them.

**Action selection** converts the inference output into an executable operation. If the model produces free-form JSON for a tool call, parse failures are common. If the model uses the function-calling or tool use APIs provided by frontier model providers, parse failure rates drop by an order of magnitude because the API enforces schema compliance. The action selection step should always validate the parsed action against the tool's input schema before dispatching — an invalid tool call that is caught before execution is far cheaper than discovering the invalidity after the tool has run or, worse, after an irreversible side effect has occurred.

**Multi-step planning** is the capability that differentiates Tier 4 and Tier 5 agents from simpler systems. A model that can decompose a complex task into sub-tasks, assign them to specific tools, establish ordering constraints, and track completion state is doing something qualitatively different from a model that just produces one action at a time. The architectures that structure this planning — ReAct (interleaved reasoning and action), Plan-and-Execute (upfront plan generation followed by execution), Tree of Thoughts (parallel exploration of multiple plan branches) — each make different tradeoffs between upfront planning cost, flexibility to adapt mid-execution, and reliability under uncertainty. We cover these in depth in the Track B posts.

## What Act Really Means

The action space of an agent is the set of all operations it can perform on the world. Designing this space correctly — what to include, how to structure the schemas, how to handle errors — is one of the highest-leverage decisions in agent architecture. The design of the action space determines the blast radius of failures, the cost structure of the agent loop, and the range of tasks the agent can accomplish.

A practical taxonomy for agent actions runs across five categories:

**Read actions** are non-destructive information retrieval: `search_web()`, `query_db()`, `read_file()`, `get_calendar_events()`, `list_directory()`. These are always safe to retry, produce no side effects, and should be the default action type for any information-gathering step. Every agent should have more read actions than write actions. The more your agent can observe before acting, the fewer costly mistakes it makes. The failure mode for read actions is staleness — a cached search result that does not reflect current state, a file that was modified since the retrieval — not irreversibility.

**Write actions** create or modify state: `write_file()`, `update_db_record()`, `send_email()`, `create_calendar_event()`, `post_message()`, `deploy_code()`. Write actions introduce the blast radius problem. A malformed database update can corrupt records. A misdirected email cannot be un-sent. An incorrect deployment can take a production service down. Every write action in an agent should either (a) have a reversibility mechanism — versioning, soft delete, draft/commit pattern — or (b) require an explicit confirmation step before execution. Never let an agent write to production systems without one of these guardrails.

**Compute actions** transform data: `run_python_code()`, `call_api()`, `process_image()`, `run_test_suite()`. These are often idempotent but can be expensive in time and cost. A code execution that takes 90 seconds is a 90-second toll on every loop iteration that requires it. The failure mode is incorrect computation that consumes resources without producing the intended output: a buggy script that loops forever, an API call that hits a rate limit and blocks, an image processing job that crashes on a malformed input. Compute actions require timeout management, resource limits, and sandbox isolation.

**Communicate actions** send information to humans or external systems: `notify_user()`, `post_to_slack()`, `escalate_ticket()`, `send_webhook()`. These are often irreversible at the human layer even if they are technically reversible at the system layer — you can delete a Slack message, but you cannot un-read it. Communicate actions also introduce human latency into the loop. Design agents to batch communication rather than fire a message for every intermediate result. An agent that sends a Slack notification for each of its 15 reasoning steps will train users to ignore all of them.

**Orchestrate actions** are the highest-level: `spawn_subagent()`, `call_workflow()`, `delegate_to_specialist_agent()`, `fork_parallel_task()`. These are the actions that make multi-agent systems possible. They also introduce a new failure mode: the spawned agent or workflow can fail, loop, or produce output that the parent agent cannot interpret. Orchestration actions require the most careful error handling — what does the parent agent do when a sub-agent returns an error? How does it handle a sub-agent that does not respond within the expected time window? What is the fallback when the specialist is unavailable?

The before/after comparison below makes concrete what happens when a scripted bot — which has no dynamic Act capability — meets a task that requires dynamic action selection.

![Scripted Bot vs. Agent: Same Customer Query](/imgs/blogs/what-is-an-ai-agent-7.webp)

The scripted bot hits a decision point not encoded in its script (a sizing error on a fulfilled order, which triggers an exception clause in the return policy that the script author never anticipated) and returns a canned failure response. The agent retrieves order history, reads the full policy via a tool call, identifies the exception clause dynamically, and selects the correct action. The API calls are identical in both cases — both systems have access to the same CRM and policy systems. What differs is the routing: the agent composes the tools dynamically based on what it finds; the script executes a fixed sequence.

A critical design principle that most agent tutorials omit: the action space should be the minimum set of actions required to complete the task, not the maximum set that could theoretically be useful. Every action you add is a surface for the model to use incorrectly, a new error mode to handle, and an additional security boundary to protect. Start with read-only actions. Add write actions only when the task demonstrably requires them. Add compute and orchestrate actions last, when you have evidence that the simpler action types are insufficient. The principle of minimum viable action space is not conservatism; it is how you build agents that remain controllable as their autonomy increases.

## What Remember Really Means

Memory is the pillar that most sharply distinguishes a stateless chatbot from a genuine agent. Without memory, every turn starts from the same blank context. The agent cannot build on prior observations, cannot track which sub-tasks are complete, cannot learn from errors within a session, and — in a full agent — cannot maintain consistent behavior across sessions. The multi-step task is impossible. The agent would need to re-discover its progress from scratch at each step.

Production agent memory is stratified into four types, each with different persistence, access patterns, cost profiles, and engineering implications:

**In-context memory** is everything in the active context window right now. This is the fastest and cheapest memory — retrieval is zero latency, no external system required — but it is also the most expensive to expand (every additional token costs money, and models slow down as context length grows) and the most volatile (it disappears when the session ends or when the context window overflows). In-context memory holds the current task goal, the system prompt, the immediate tool outputs, the most recent reasoning chain, and any other information that needs to be immediately available for the next reasoning step. Every agent has in-context memory. The question is what it has in addition.

**Episodic memory** is a persistent log of prior observations, tool results, and reasoning steps, stored in an external system (typically a vector database) and retrieved via semantic similarity search. When an agent starts a new turn or a new session, it queries episodic memory for prior experiences relevant to the current task. This is what enables the cross-session behaviors that make agents feel genuinely intelligent rather than amnesiac: "I ran into a rate limit on the financial API last time — let me use the backup endpoint." "The user's preferred format for reports is bullet-point summaries, not prose." "This customer has contacted us three times about the same issue." Episodic memory is read/write: the agent writes new observations at the end of each turn and reads from prior observations at the start.

**Semantic memory** is the agent's knowledge base: facts about the world, domain knowledge, tool documentation, product specifications, policy documents, API schemas. This is typically a read-only vector store populated offline — it does not update during agent operation. Retrieval augmentation — the RAG layer — is semantic memory in action. The agent does not learn new facts into semantic memory during a task; it retrieves facts that were indexed in advance and uses them to ground its reasoning. Semantic memory is the solution to the base model's knowledge cutoff problem and to the problem of keeping agents current with rapidly changing domain knowledge.

**Procedural memory** is the agent's repository of learned patterns: which sequence of tool calls reliably solves a class of problem, how to handle a specific API's error codes, which prompt formulations work best for a given tool's output format. In current production systems, procedural memory is usually encoded directly in the system prompt — the agent engineer writes down the patterns that work and the model follows them. More sophisticated systems use few-shot examples dynamically retrieved from a success log based on the current task type. This is a nascent area: systems that can identify successful patterns from their own execution logs and incorporate them into future runs are beginning to close the gap between Tier 4 and Tier 5.

The interaction between memory types is where the richest agent behaviors emerge — and also where some of the hardest engineering problems live. Memory retrieval is not free: a vector similarity search over episodic memory takes 50–200ms and charges embedding API costs. A semantic memory lookup over a large knowledge base may retrieve 5 documents, each of which needs to be re-ranked for relevance before injection. Procedural memory encoded in the system prompt must fit within the fixed prompt budget — too long and it crowds out the dynamic context; too short and the agent misses important tool-use patterns. Every memory type has a cost function, and production agents need to manage those costs explicitly.

The interaction between memory types is also where the richest behaviors emerge. An agent handling a complex customer support ticket might: retrieve the customer's contact history from episodic memory (has this customer opened tickets before?); pull the relevant policy clauses from semantic memory (what are the rules for this product category?); use its in-context window to reason about the current ticket text and the retrieved context; and follow procedural memory (encoded in the system prompt) to know that monetary figures must be confirmed against the transaction record before they are stated.

The most common production memory failure is context overflow: the agent's in-context window fills before the task is complete. Tool outputs from earlier steps get truncated. The agent loses track of constraints established in step 2, repeats actions it already performed in step 5, or contradicts decisions made in step 3 when writing its final output. Preventing this requires active memory management: summarizing completed observations into compact state objects, evicting raw tool outputs that have already been processed, and promoting key facts — the current task goal, critical constraints, irreversible decisions made — to pinned positions in the context that cannot be evicted. We cover this in depth in the Track C posts on context window management and the memory architecture posts.

## The One Complete Agent Turn

Before examining failure modes, it is worth making the per-turn execution structure concrete. Every agent iteration — the quantum of agent execution — follows the same seven-phase structure, regardless of the agent's tier or the specific framework used to implement it.

![One Complete Agent Turn: User Message to Response](/imgs/blogs/what-is-an-ai-agent-2.webp)

**Phase 1 — User Message.** A new input arrives. This might be the initial user query at the start of a session, a follow-up instruction from a human in the loop, or — in a multi-agent system — a message from an orchestrating parent agent. The input is parsed, type-checked, and routed to the perception stage. At this phase, the agent validates that the input is within its operational scope and sets up the turn context.

**Phase 2 — Perceive.** The agent's perception layer processes the input. This includes chunking long documents, embedding and querying the episodic memory store for relevant prior observations, fetching any structured data needed to ground the current task (order records, policy documents, user preferences), and assembling the raw ingredients for context construction. Perception latency is dominated by retrieval: a vector similarity search over a large episodic memory can take 50–200ms, and multiple parallel retrieval calls can push this to 500ms or more.

**Phase 3 — Reason.** The LLM inference call runs with the assembled context. This is typically the highest-latency operation in the turn — 500ms to 5+ seconds depending on model size, prompt length, output length, and whether the model needs to produce a scratchpad reasoning chain before its final output. The output is a structured completion that specifies the next action (a tool call with parameters) or, if the task is complete, the final response.

**Phase 4 — Plan.** If the task requires multiple steps, the reasoning output may include a planning phase: the model decomposes the goal into sub-tasks, assigns them to tools, and establishes ordering constraints. This phase is often implicit — the model reasons about the plan in its chain-of-thought scratchpad rather than producing a separate plan document — but in Plan-and-Execute architectures, this phase is a distinct system call that produces an explicit plan object that governs the subsequent execution loop.

**Phase 5 — Act.** The specified action is dispatched. Tool schemas are validated, the tool's access permissions are checked, rate-limit controls run, and the tool executes. Action latency varies by three orders of magnitude: a cache lookup takes microseconds, a read from a local file system takes milliseconds, a web search takes 500ms–2s, a database query takes 10ms–500ms depending on complexity, a code execution can take seconds to minutes. The agent tracks action cost (latency, API charges) and enforces budget limits.

**Phase 6 — Observe.** The tool result is returned to the agent. The observation is validated against an expected schema, parsed into a structured representation, and written to episodic memory. The agent checks whether the task is complete (did the last action satisfy the goal?) and whether an error requires recovery (did the tool return an error that requires a different approach, a different tool, or a human escalation?).

**Phase 7 — Respond / Loop.** If the task is complete, the agent assembles the final response, writes a session summary to episodic memory, and terminates. If more steps are required, the observation is injected into the context (after any necessary summarization or eviction to manage context length), the iteration counter increments, and the loop returns to Phase 2. The loop continues until the task is complete, a terminal error is encountered, the iteration cap is reached, or an external condition forces termination.

This seven-phase structure is the same whether you are using LangChain, LlamaIndex, AutoGen, Claude's tool use API, or a bespoke agent framework. The frameworks differ in how they implement each phase and what abstractions they expose, but the underlying execution structure is universal.

## Where Agents Fail: The Three Primary Failure Modes

Every production agent will eventually fail. The diagnostic value of the four-pillar framework is that it maps the three most common failure modes to the specific pillar they corrupt. The pillar name tells you where to look for the fix. This makes the framework not just descriptive but operationally useful.

![Three Agent Failure Modes and the Pillars They Break](/imgs/blogs/what-is-an-ai-agent-8.webp)

### Failure Mode 1: Hallucination (corrupts Reason)

Hallucination occurs when the LLM's output contains factual assertions not grounded in the context window. The model fabricates an API response it did not actually see. It invents a clause in the policy document that does not exist. It reports a numerical value that differs from the one in the retrieved record. It asserts that a step succeeded when it was never executed.

In a single-turn chatbot, hallucination is annoying but usually caught by the human reader, who can ask a follow-up question or check the source. In an agent loop, a hallucinated observation from step 2 becomes the input context for step 3. The agent's reasoning at step 3 is grounded in a false world-state. The action at step 3 is based on that false reasoning. The observation at step 3 goes back into the context. By step 5, the agent may be taking confident, energetic, well-reasoned action based on a model of the world that it constructed from a hallucination three steps back — and the well-reasoned quality of the steps after the hallucination makes it even harder to spot in the trace.

**The signal:** the agent's actions do not match the actual state of the external system. The tool was called correctly, the parameters were syntactically valid, but the values used were derived from a hallucinated prior observation rather than the actual retrieved data. In production observability, this shows up as a divergence between the agent's stated world-state and the actual world-state, which requires cross-referencing the agent's claims against the raw tool outputs in the trace.

**The structural fix:** retrieval grounding with citation enforcement. Every factual claim the agent makes should be traceable to a specific token span in a specific retrieved document in the current context window. If the agent says "the order was placed on March 15," that date should appear in the retrieved order record, not be inferred from a prose description that the model summarized two steps ago. RAG with per-claim citation tracking is the implementation pattern; structured prompts that require the model to cite its source for each factual assertion are the prompting technique; automated fact-checking against the raw tool outputs is the observability layer. We cover this in depth in the Track C posts on retrieval-augmented agents.

### Failure Mode 2: Infinite Loop (corrupts Act)

An infinite loop occurs when the agent's exit condition is never satisfied and the iteration cap is not set. This can happen for several reasons. The task description is underspecified: "improve this code until it is good" has no well-defined completion state. The tool is consistently returning an error the agent cannot resolve: "access denied" on every attempt at an action the agent believes is necessary. The agent's self-correction logic creates an oscillation: "the code failed, try a different approach" leads to approach B, which also fails, which leads the agent to try approach A again, which also fails, in a cycle that repeats indefinitely. The model is producing syntactically valid tool calls whose parameters are subtly wrong in a way that the tool's error messages do not make clear.

**The signal:** token count and cost are climbing steeply without task progress. The agent is repeating variations of the same actions. The iteration count exceeds any reasonable estimate of what the task requires. In production observability, this shows up as an anomalous session length in the latency distribution and an anomalous token consumption in the cost distribution — both far to the right of the normal range for that task type.

**The structural fix** has two components. First, a hard iteration cap: no agent should ever take more than N steps without a human review. For most production tasks, N is between 20 and 50. Setting N = 100 because "the task might be complex" is not a principled decision; it is an invitation to a $50 API bill from a single runaway session. Second, an explicit multi-condition exit prompt: "Stop when the task is complete, OR after 30 iterations, OR if the same error appears three times in a row, OR if no progress has been made in the last five steps." All four conditions, stated explicitly, with the response for each: complete the task, declare failure, escalate, or pause for human review.

### Failure Mode 3: Context Loss (corrupts Remember)

Context loss occurs when the agent's context window fills up and earlier observations are truncated or evicted without being preserved in external memory. The agent loses track of constraints established early in the task, repeats actions it already performed, contradicts decisions it made in earlier steps, or produces a final output that ignores information from step 2 because that information has been pushed out of the context window by the accumulation of later steps.

Context loss is not the same as the context window filling up — that is an infrastructure fact, not a failure mode. The failure mode is the absence of active memory management that would prevent important early information from being lost. A well-designed agent, when its context begins to fill, summarizes completed observations into compact state representations and promotes key constraints to pinned positions in the context that are never evicted. Context loss happens when this management is absent: the agent just concatenates all outputs sequentially and lets the LLM provider handle truncation, which typically means the oldest content is cut first.

**The signal:** the agent contradicts earlier decisions or repeats earlier observations. "I will now search for X" when it searched for X three steps ago and already processed the results. "The policy allows refunds within 30 days" when the retrieved policy says 60 days and the agent correctly stated 60 days two steps ago. In the trace, this shows up as actions that are inconsistent with the agent's earlier successful retrievals.

**The structural fix:** an explicit session state object that is continuously maintained and pinned in the context. At the start of each step, the agent writes a structured summary of the current task state: what has been completed, what was found, what constraints are in effect, what the current sub-goal is. This state object replaces the raw concatenation of prior observations in the context. The raw observations are written to episodic memory and are available for retrieval, but they do not take up primary context window space. The state object is typically 200–500 tokens and remains readable regardless of how long the session has been running.

The failure mode and recovery matrix below maps each failure mode to its detection, mitigation, and prevention strategies in a single view.

![Failure Modes × Recovery Strategies](/imgs/blogs/what-is-an-ai-agent-9.webp)

A key observation from this matrix: each failure mode has a distinct detection signal that is observable without deep analysis of the model's reasoning. Step-counter-exceeds-N is a simple counter check. Monetary-figure-differs-from-tool-output is a schema comparison. Agent-contradicts-prior-state is a semantic similarity check between the current claim and the prior session state. These signals do not require you to understand why the model produced the wrong output. They require only that you instrument the agent to emit the right events and that you set the right thresholds. Observability is not optional in production agents; it is the mechanism by which you detect and recover from the failure modes that are inevitable in any sufficiently complex agent.

## Case Studies: Real Systems on the Agency Spectrum

### Case Study 1: Siri, 2011–2023 — The Perpetual Tier 2

Siri launched as a genuine innovation: a voice interface that could understand natural language and call device APIs. But its architecture was fundamentally Tier 2. Each utterance was a fresh inference. Siri had no episodic memory across conversations, no multi-step reasoning loop, no self-correction. If you asked "What's the weather in Paris?" and followed up with "What about tomorrow?", Siri treated the second question as a standalone query with no connection to the first.

The diagnosis from the four-pillar lens: Siri had Perceive (voice-to-text, intent classification) and Act (call device APIs, play music, set alarms), but its Reason layer was a shallow intent classifier rather than an LLM-based planner, and it had essentially no Remember at the cross-conversation level. Every conversation started at zero context.

The lesson for engineers: significant commercial success is achievable at Tier 2. For a large fraction of voice assistant queries — "set a timer for 10 minutes," "call Mom," "what's 15% of 84?" — stateless one-shot responses are exactly what users need. The failure mode was scope creep. As users expected more complex multi-turn interactions driven by the conversational metaphor the UI established, Siri's architecture could not accommodate them without a complete rebuild of the reasoning layer.

### Case Study 2: AutoGPT, 2023 — The Tier 5 Prototype

AutoGPT was released in March 2023 and briefly became one of the fastest-growing GitHub repositories in history. Its promise was bold: give it a goal, and it will autonomously pursue it with web browsing, code execution, and file system access. No human in the loop required.

AutoGPT was a genuine Tier 5 prototype. It had all four pillars: real-time web perception via a browsing tool, GPT-4 reasoning, a rich action space including web search, code execution, and file system access, and a vector store for persistent memory. In controlled demonstrations, it completed multi-step research and development tasks that would have required hours of human work.

In practice, it suffered from all three failure modes simultaneously and severely. Hallucination: the model would assert it had completed a sub-task when it had not, based on a hallucinated completion signal. Infinite loops: without a robust exit condition, AutoGPT would spin on tasks that were genuinely infeasible, consuming tokens until the budget was exhausted. Context loss: the memory architecture was rudimentary, causing the agent to lose track of earlier sub-task results as the context grew across a multi-hour session.

The diagnosis: the architecture was directionally correct but the reliability engineering layer was absent. AutoGPT proved the concept was viable and served as the empirical foundation for the wave of more carefully engineered agent frameworks that followed. The lesson for the field: autonomous operation at Tier 5 requires not just the four pillars but a rigorous failure mode engineering stack layered on top of them. The difference between a demo and a production system is almost entirely that stack.

### Case Study 3: GitHub Copilot, 2021–2023 — Correctly Positioned at Tier 3

GitHub Copilot is a study in correctly scoped agency. In its original form, it was a RAG application: it retrieved code context from the current file and nearby files, injected that context into a code completion prompt, and returned single-turn suggestions. No loop. No persistent state. No multi-step planning.

This was exactly the right design for the task. Code autocomplete — complete the current line or block — has a clear, single-turn structure. The input is fully observable in the current editor context. The output is immediately evaluated by the developer, who accepts or rejects each suggestion. There is no reason to build a reasoning loop into this interaction; the developer is the loop.

Copilot graduated toward Tier 4 with the introduction of Copilot Chat (multi-turn code Q&A with document retrieval) and then Tier 4+ with Copilot Workspace (multi-file editing with an explicit plan-and-execute loop for implementing GitHub issues). Each tier upgrade was driven by the task changing, not by a desire to add complexity for its own sake.

The lesson: identify the minimum tier required to accomplish the task. Then build that tier. GitHub did not architect an autonomous software engineering agent and then progressively constrain it; they built the simplest system that achieved target user behavior and upgraded the tier as the task scope expanded and as user demand validated the investment.

### Case Study 4: A Customer Service Agent — Reactive Done Right

A fintech company deployed a customer support agent for handling account queries, transaction disputes, and product questions. The agent was Tier 4: it called tools (CRM query, transaction history lookup, policy document retrieval, ticket creation, escalation routing), maintained state within a session, and looped until the issue was resolved or escalated to a human agent.

The initial deployment suffered from all three failure modes in production. Hallucination: the agent invented transaction amounts when the CRM query returned empty results for a transaction ID. It would state "Your transaction of $47.23 on March 12 was processed successfully" when no such transaction existed in the system — the transaction ID in the query was malformed. Infinite loops: disputes on transactions from more than 180 days ago triggered a retry loop because the system returned a specific error code that the agent interpreted as a transient retrieval failure and retried indefinitely. Context loss: long sessions with multiple follow-up questions caused the agent to lose track of which transaction was being disputed, and it would respond to a follow-up question about transaction X with information it had retrieved about transaction Y three turns earlier.

The fixes applied: a grounding rule enforced at the prompt level ("any monetary amount must be sourced from the transaction record — if the record is empty or the transaction ID is invalid, report that fact explicitly rather than generating a value"); an explicit error-code mapping that treated the 180-day policy limitation as a terminal condition rather than a transient error; a session state object that pinned the disputed transaction ID as a non-evictable element in the context. Iteration cap set at 15 with automatic escalation.

Results: hallucination rate on monetary figures dropped from 3.2% to 0.1%. Infinite loops ceased in production within the first week. Session coherence on long sessions (defined as the percentage of final responses that correctly referenced the originally disputed transaction) improved from 78% to 97%.

The lesson: the failure mode taxonomy is directly actionable. The diagnostic — which pillar is corrupted — tells you which layer to fix. The fixes are independent; you can address hallucination without re-architecting the loop or the memory system.

### Case Study 5: Devin — The First Commercial Tier 5

Devin, released by Cognition AI in early 2024, was the first commercially deployed software engineering agent that credibly operated at Tier 5 for a real task class. It could accept a GitHub issue, spin up a development environment, plan the implementation, write code, run tests, debug failures, iterate on the implementation, and open a pull request — over a session that might last hours — without human intervention for the core execution loop.

Devin's architecture had all four pillars at production quality: perception of code repositories, terminal output, test results, and error messages; reasoning via an LLM with an explicit planning step that decomposed the issue into implementation steps; a rich action space including bash execution, code editing, browser, and communication tools; and persistent memory that preserved context across the hours-long development sessions required for non-trivial tasks.

Its failure modes were instructive for the field. For tasks requiring long-range planning — implement a feature that touches ten files and requires coordinated architectural changes across services — context loss remained the dominant failure mode even with sophisticated memory management. For tasks requiring deep knowledge of an unfamiliar third-party library — implement a feature using an API that Devin had not seen in its training data — hallucination of API signatures and behavior was the primary failure mode.

The lesson: Tier 5 reliability is not a solved problem. It requires continuous engineering on the failure mode stack as the task distribution shifts. A fix that works for 80% of the task distribution may not generalize to the tail 20% where sessions are longer, the tool landscape is less familiar, or the user's specification is more ambiguous.

### Case Study 6: LLM-Powered Data Analysis — The Wrong Tier for the Task

A data engineering team built an "AI analyst agent" to answer questions about their data warehouse. Users asked questions in natural language; the agent generated SQL, executed it against the warehouse, interpreted the results, and returned a formatted answer. The team built it as a full Tier 4 reactive agent with a loop, session memory, and multiple tool types.

After three months in production, they audited the execution logs. Their finding: 94% of queries were answered in exactly one tool call — one SQL generation, one execution, one interpretation. The loop never fired more than once in 94% of sessions. The session memory was never accessed in 89% of sessions, because users asked standalone questions rather than multi-turn analytical sequences.

They refactored to a Tier 3 RAG application: no loop, no session memory, just SQL generation from natural language with schema retrieval and a single execution. The results were stark: median query latency dropped from 4.2 seconds to 1.1 seconds. Cost per query dropped 65%. SQL correctness (measured by execution success rate) was unchanged. Hallucination rate on query results dropped slightly — a simpler prompt with a cleaner structure was easier for the model to follow precisely.

The lesson: always measure whether the loop fires before committing to loop infrastructure. If the loop fires in fewer than 30% of sessions, your task likely does not require agent architecture. Build the simpler system.

### Case Study 7: A Research Agent for Competitive Intelligence — Tier 4 Required

A strategy consulting firm deployed a research agent to gather competitive intelligence on companies. The task: given a company name, produce a structured briefing covering financials, products, leadership, and recent news. Multiple sources had to be queried — web search, financial data APIs, news aggregators, company website. The number of sources and the specific queries depended on what the agent found: a public company needed filings; a private company needed web-sourced revenue estimates; a startup needed product launch announcements and funding news.

This task genuinely required Tier 4. The routing logic could not be specified in advance because it depended on what was discovered. The number of tool calls ranged from 3 (a well-documented large public company) to 18 (an obscure private company with minimal coverage). Self-correction was essential: if the first search query returned stale results, the agent needed to retry with a date-restricted query or a different search engine.

The agent was initially deployed without iteration limits. On one early run, a company with an unusual name (a common English word used as a brand) caused every search query to return irrelevant results about the word itself. The agent retried with variation after variation for 40 minutes before hitting a session timeout, consuming $12.40 of API calls for a briefing that should have cost under $1.

The structural fixes: a per-source retry cap of 3 attempts, a confidence threshold per section (if the financial section cannot be populated within 3 tries, mark it as "data unavailable" and proceed to the next section), and a hard total step cap of 30 with automatic failover to a human researcher above that cap. The agent now completes 98% of briefings within 14 steps and $0.95.

### Case Study 8: Perplexity AI — Citation as the Hallucination Fix

Perplexity AI built a web-native answer engine that retrieves live search results and synthesizes them into a cited answer. In the four-pillar framework, it is primarily Tier 3 with elements of Tier 4 — it sometimes follows up with clarifying searches when the first results are insufficient to fully answer the query.

What makes Perplexity instructive is the decision to make citation a first-class structural requirement rather than a nice-to-have. Every claim in the answer is linked to the specific source it came from. This is not cosmetic design — it is a structural fix for the hallucination failure mode. The model is constrained to produce claims grounded in the retrieved sources, and the citations make every claim auditable by the user. A user who sees a citation they do not trust can check the source in one click. A model that knows it must cite cannot easily hallucinate information that has no source.

The agency tier is deliberately conservative. Perplexity does not maintain session state across queries (Tier 3, not Tier 4). It does not take write actions. It does not orchestrate sub-agents. It does one thing — answer questions with live web retrieval and mandatory citations — and it does that one thing with high reliability.

The lesson: reliability often comes from constraint rather than capability. Adding the citation requirement constrained the model's output space in a way that improved answer quality. The conservative agency tier made the system easier to reason about, test, and trust. Tier 3 with excellent reliability and UX beats Tier 5 with high complexity and variable reliability for most commercial use cases.

## When to Reach for an Agent — and When Not To

The framework above implies a decision procedure. Here is the practical version that you can apply on your next project.

**Reach for an agent (Tier 4+) when:**

The task structure is genuinely unknown at build time. If you cannot write the complete decision tree for your task — because the right sequence of steps depends on what the agent discovers during execution — you need an agent. If you can write the complete decision tree, even a complex one, you have a workflow.

The task requires more than one tool call and the sequence of calls depends on intermediate results. "Search for X, then based on what you find, either do Y or Z" is a reactive agent task. "Always search for X then always do Y then always do Z" is a workflow.

The task requires self-correction from tool failures that are not predictable at build time. If a tool can fail in ways that require dynamic recovery — trying a different query, a different tool, a different format — that is a reactive agent task. A workflow can handle pre-defined retries; an agent handles dynamic recovery.

The task maintains state across a long session where the input to step N depends on the results of step N-6 in a way that cannot be captured by a compact intermediate representation. Context management at this level of complexity is what agents are built for.

The task requires genuine multi-step planning where the plan cannot be fully specified before execution begins. If the right plan depends on what the agent finds in its first few steps, explicit planning architectures are the right approach.

**Do not reach for an agent when:**

The task can be solved in one LLM call with the right prompt and context. Approximately 70% of tasks that engineers initially believe require agents are solved by better prompting, richer context injection, or a simpler retrieval pipeline. Do the prompt engineering work before committing to agent infrastructure.

The task structure is fixed and all branches are known. Build a deterministic workflow. It will be faster, cheaper, more reliable, and easier to debug and maintain. When the user asks for new branches, add them to the workflow. Upgrade to an agent only when the workflow's branching logic becomes too complex to maintain — typically when you have more than 20 distinct branches or when the routing logic requires semantic reasoning that a rules engine cannot express.

You need deterministic, auditable outputs on every run. Agents introduce non-determinism at every loop iteration. If your task is regulated (financial advice, medical information, legal document generation), high-stakes, or requires byte-for-byte reproducibility for auditing, agent architecture adds risk without proportional benefit.

Your latency budget is tight. Every loop iteration adds at minimum one LLM inference call (500ms to 5s). A five-step agent loop has a minimum latency of 2.5 seconds at the inference layer alone, before tool execution time. If your SLA is under 2 seconds and your task genuinely requires 5 steps, you have a hard engineering problem that agent choice does not solve.

You are optimizing for minimal cost before you have validated demand. Agents cost 3× to 10× more per interaction than chatbots or simple workflows. If you have not yet validated that users will use the feature, that they will pay for it at the cost it requires, and that the task genuinely cannot be solved at a lower tier, start with the simpler architecture. Upgrade when you have evidence.

The agency spectrum is a design tool, not a status symbol. Sitting at Tier 3 because Tier 3 solves your problem is not a failure of ambition — it is correct engineering. The discipline is knowing which tier your task actually requires, building at that tier with appropriate engineering investment in the failure mode stack, and resisting the pull toward unnecessary complexity in either direction.

For a detailed treatment of agentic design patterns — ReAct, Plan-and-Execute, and their production variants — see [Agentic Design Patterns and Case Studies](/blog/machine-learning/ai-agent/agentic-design-patterns-and-case-studies). For a hands-on guide to building your first production agent, including tool schemas, loop termination, and observability instrumentation, see [Building Effective Agents: Hands-On Guide](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide).

The next post in this series — [Agent Loop Anatomy](/blog/machine-learning/ai-agent/agent-loop-anatomy) — dissects a single agent turn at the microsecond level: token budgets, latency budgets, and the six places errors inject. That post is where the framework built in this post gets turned into production engineering decisions.

![Eight Real-World Systems on the Agency Spectrum](/imgs/blogs/what-is-an-ai-agent-10.webp)

The eight systems above span the full spectrum from single-turn voice commands to multi-hour autonomous development loops. The pattern holds consistently: agency tier correlates with task complexity and the degree to which task structure is unknown at build time. Systems positioned correctly at their tier deliver reliable, cost-efficient experiences. Systems over-tiered for their task deliver complexity without proportional benefit. Systems under-tiered for their task fail at the edge cases their architecture was never designed to handle.

The framework developed in this post — the four-pillar loop, the agency spectrum, the comparison of chatbot versus workflow versus agent, the three failure modes and their pillar mapping — is a vocabulary for making these architectural decisions explicitly. Most teams currently make them implicitly, driven by what framework demos look like and what the product roadmap says. Explicit decisions can be revisited when evidence changes. Implicit decisions calcify into technical debt.

One more observation before the engineering gets started: the cost of getting the tier wrong scales with the complexity of the task. A chatbot deployed as an agent for a simple Q&A use case costs 3–10× what it should and runs slower, but users can live with it. An agent architecture deployed for a task that genuinely required a Tier 5 autonomous system — where the reactive Tier 4 agent cannot complete multi-hour work autonomously — fails visibly and repeatedly in production. The mismatch between task requirements and architecture tier is the root cause of the majority of high-profile "AI agent" failures in production from 2023 to 2025. The framework in this post is the diagnostic tool. The engineering judgment is knowing the difference — and making it explicit — before you start building.
