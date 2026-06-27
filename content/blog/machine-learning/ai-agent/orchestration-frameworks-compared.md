---
title: "Orchestration Frameworks Compared: LangGraph, AutoGen, CrewAI, and Building Your Own"
date: "2026-06-27"
description: "A practical comparison of LangGraph, AutoGen, CrewAI, and raw Python for multi-agent orchestration — abstractions, tradeoffs, escape hatches, and when each framework pays its overhead."
tags: ["ai-agents", "multi-agent", "langgraph", "autogen", "crewai", "orchestration", "llm", "machine-learning", "production-ml"]
category: "machine-learning"
subcategory: "AI Agent"
author: "Hiep Tran"
featured: true
readTime: 39
---

Here is what actually happens when you pick the wrong orchestration framework: you spend three days wiring together a beautiful CrewAI crew for a document-processing pipeline. The demo works. Then someone asks "can we add a retry loop if the extraction confidence is below 0.7?" You stare at the sequential process definition. There is no `if` branch. You add a second crew that calls the first crew's output. Six weeks later your codebase has four nested crews, each calling the next via a string-output hack, and a new engineer spends an entire sprint just understanding the control flow.

This is not a hypothetical. I have seen it happen three times across different companies. The framework abstraction that made the demo fast made the production system slow to evolve.

The purpose of this guide is not to tell you "use LangGraph, it is the best." LangGraph is the best — for a specific category of problems. For others, AutoGen, CrewAI, or a handful of raw Python functions will serve you better. The hard part is knowing which category your problem sits in before you have sunk two months into the wrong framework.

![Abstraction layers in agent orchestration: raw LLM API at the bottom, thin tools layer, orchestration framework, application at the top](/imgs/blogs/orchestration-frameworks-compared-1.webp)

The diagram above is the mental model. Every orchestration framework lives between your raw LLM API calls and your application logic. The framework is not free — it adds a layer with real overhead in cognitive complexity, debugging friction, version instability, and in some cases, runtime performance. Whether that overhead is worth paying depends entirely on what problem you are solving.

## 1. Why Frameworks Exist and What They Cost You

The case for a framework is compelling on the surface. Multi-agent systems have real coordination problems: how do you pass state between agents? How do you retry a failed agent without re-running the ones that succeeded? How do you handle a tool call that returns an error? How do you stop an agent that has gone into an infinite reasoning loop? Solving these problems from scratch every time is genuinely tedious, and a good framework eliminates the boilerplate.

But frameworks are also opinionated about what multi-agent coordination looks like. That opinion is baked into the abstraction. LangGraph's opinion is that agent workflows are state machines with typed shared state. AutoGen's opinion is that agent coordination is a conversation between participants. CrewAI's opinion is that tasks belong to role-named agents and execute in sequence. Each opinion is correct for a large class of problems and actively wrong for others.

The cost you pay for a framework has four components:

**Abstraction overhead.** Every framework adds at least one level of indirection between your code and the LLM call. That indirection costs you in understanding — when something goes wrong, you need to know both your code and the framework internals. LangGraph's graph execution engine, AutoGen's ConversableAgent hierarchy, CrewAI's task delegation model: all of these are things you will need to learn deeply before you can debug production issues.

**Version instability.** Agent frameworks are extremely young and change fast. LangGraph 0.0.x to 0.1.x to 0.2.x introduced multiple breaking changes. AutoGen 0.x has broken the ConversableAgent API several times. CrewAI has deprecated and replaced its task delegation model more than once. Every time the framework version breaks, you own the migration. If you are locked in, the migration is expensive. If you have kept thin, well-defined boundaries between your code and the framework, migration is manageable.

**Escape hatch friction.** The moment your use case sits outside the framework's happy path, you hit friction. For LangGraph, the happy path is a well-structured directed graph with no runtime-dynamic topology changes. For AutoGen, the happy path is a conversational loop with a fixed set of agents. For CrewAI, the happy path is a sequential or hierarchical execution of well-defined tasks. Any deviation forces you into escape hatches — subclassing, monkey-patching, or dropping down to the layer below.

**Lock-in risk.** Frameworks make it easy to write code that is tightly coupled to their APIs. After six months of building, it is common to find that your business logic is deeply entangled with framework primitives. Migrating away becomes a full rewrite rather than a refactor.

None of these costs are dealbreakers. They are tradeoffs. A framework that costs you three weeks of escape hatch wrestling is still a net win if it saved you four months of infrastructure work. The question is whether the specific tradeoffs of a given framework match the specific requirements of your project.

## 2. LangGraph: Graph-Based State Machines

LangGraph was released by LangChain in early 2024, initially as a thin wrapper around LangChain's agent infrastructure, and has since grown into an independent framework with its own compilation model, checkpointing system, and deployment infrastructure. The core abstraction is a directed graph where nodes are Python callables (usually agents or tool executors) and edges define the routing logic.

![LangGraph state machine: nodes as agents, edges as conditional routing, shared typed state](/imgs/blogs/orchestration-frameworks-compared-2.webp)

### The state machine model

The key insight LangGraph offers is that agent workflows are state machines. Every LangGraph graph operates on a typed shared state object — typically a TypedDict — that every node can read from and write to. The node does not return a value; it returns an update to the state. The edge (or conditional edge) then reads the updated state to decide which node runs next.

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator

class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]  # append-only
    intent: str
    context: str
    code_result: str
    verdict: str
    retry_count: int

def router_node(state: AgentState) -> AgentState:
    """Classify intent from the latest message."""
    intent = classify_intent(state["messages"][-1])
    return {"intent": intent}

def search_agent(state: AgentState) -> AgentState:
    """Run a web search and update context."""
    results = web_search(state["messages"][-1])
    return {"context": results}

def critic_node(state: AgentState) -> AgentState:
    """Quality check the current state."""
    score = evaluate_output(state)
    verdict = "pass" if score >= 0.8 else "retry"
    return {"verdict": verdict, "retry_count": state["retry_count"] + 1}

def route_from_router(state: AgentState) -> str:
    """Conditional edge: returns the name of the next node."""
    if state["intent"] == "research":
        return "search_agent"
    return "code_agent"

def route_from_critic(state: AgentState) -> str:
    if state["verdict"] == "pass" or state["retry_count"] >= 3:
        return END
    return "router"

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("search_agent", search_agent)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_from_router)
workflow.add_edge("search_agent", "critic")
workflow.add_conditional_edges("critic", route_from_critic)

app = workflow.compile(checkpointer=MemorySaver())

# Run with a thread ID for resumability
config = {"configurable": {"thread_id": "thread-001"}}
result = app.invoke({"messages": ["Find info on LLM scaling laws"], 
                     "intent": "", "context": "", "code_result": "",
                     "verdict": "", "retry_count": 0}, config)
```

The type system is not just for readability. When you declare `messages: Annotated[List[str], operator.add]`, you are telling LangGraph how to merge concurrent state updates. If two nodes run in parallel and both append to `messages`, LangGraph uses the `operator.add` reducer to merge the results. This is how LangGraph handles distributed execution — the reducer function defines the merge semantics.

### Checkpointing and resumability

The most underappreciated LangGraph feature is the checkpointing system. Every node execution is saved to a persistence store (SQLite, PostgreSQL, Redis). If an agent crashes midway through a 20-step workflow — due to a rate limit error, a network timeout, or an application restart — you can resume from the last successful checkpoint rather than re-running from the start. For long-running workflows that cost $5–50 in LLM API calls, this is not optional; it is a business requirement.

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver

# Development: SQLite
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)

# Production: PostgreSQL
with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
    app = workflow.compile(checkpointer=checkpointer)
    
# Resume an interrupted run
config = {"configurable": {"thread_id": "thread-001"}}
state = app.get_state(config)  # inspect current state
app.invoke(None, config)  # resume from checkpoint
```

### Human-in-the-loop

LangGraph's `interrupt_before` and `interrupt_after` mechanisms let you pause execution at a specific node and wait for human approval. This is the cleanest implementation of human-in-the-loop patterns I have seen in any framework. The checkpoint system stores the full state, so the human can review what the agent was about to do, approve or modify it, and the execution resumes with the updated state.

```python
# Interrupt before the "send_email" node for human approval
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["send_email"]
)

# Run until interruption
result = app.invoke(inputs, config)
# Returns when interrupted

# Human reviews the state
state = app.get_state(config)
print(state.values["draft_email"])

# Human approves (or modifies state)
app.update_state(config, {"draft_email": modified_draft})

# Resume
app.invoke(None, config)
```

### LangGraph strengths

LangGraph excels when your workflow has complex conditional routing, retries, or loops; when you need checkpointing for long-running expensive workflows; when you need human-in-the-loop at specific decision points; when your state needs to be inspectable and strongly typed; and when you are building a production system that needs to be maintained by a team. The graph-as-code model also makes it easy to visualize the workflow — `app.get_graph().draw_mermaid_png()` gives you a diagram that matches the code.

### LangGraph weaknesses

The learning curve is genuinely steep. The abstraction mismatch between "I want the agent to do X" and "I need to model that as a graph with typed state transitions" is not obvious for developers new to state machine thinking. The most common mistake is writing a graph that is logically a simple pipeline but expressed as six nodes with five edges — the framework overhead is real cost with minimal benefit over raw Python. The second common mistake is writing cycles without a proper termination condition, producing infinite loops that burn through API budget.

Observability requires an additional layer — LangSmith if you are on the LangChain ecosystem, or custom OTEL instrumentation if you are not. The LangSmith integration is excellent but costs money and creates another vendor dependency. Without it, debugging a multi-hop LangGraph workflow means reading raw log output or adding print statements inside nodes, which is painful.

| Dimension | LangGraph Score | Why |
|---|---|---|
| Abstraction level | Medium | State machine is powerful but has a learning curve |
| Debuggability | Good | TypedDict state is inspectable; checkpoints help |
| Flexibility | High | Any graph topology, any Python in nodes |
| Production readiness | Strong | Checkpointing, retry, persistence built-in |
| Community | Large | LangChain ecosystem, active development |
| Learning curve | Steep | Graph model + state management is non-trivial |
| Lock-in risk | Medium | Graph topology easy to port; state schema is yours |
| Observability | Good (with LangSmith) | Full trace visibility with LangSmith integration |

## 3. AutoGen: Conversational Multi-Agent

AutoGen is Microsoft Research's framework for multi-agent conversation. The mental model is different from LangGraph: instead of a directed graph with typed state, AutoGen treats multi-agent coordination as a conversation between participants. Agents send and receive messages. The GroupChatManager is a special agent that reads the conversation history and decides who speaks next.

![AutoGen GroupChat conversational loop: UserProxy, GroupChatManager, AssistantAgent, ToolCallerAgent, CriticAgent all connected through a shared message thread](/imgs/blogs/orchestration-frameworks-compared-3.webp)

### The conversational model

The core abstraction in AutoGen is `ConversableAgent`. Every agent — whether it is a human proxy, an AI assistant, a tool caller, or a custom function — is a `ConversableAgent`. Agents communicate by calling `receive()` on each other, and the conversation history is a list of `{"role": "user/assistant", "content": "..."}` dicts that grows with every exchange.

![Framework failure modes: each framework's distinct failure signature across infinite loops, opaque state, branching, escape hatches, and version breaks](/imgs/blogs/orchestration-frameworks-compared-8.webp)

```python
import autogen

config = {
    "config_list": [{"model": "gpt-4o", "api_key": "..."}],
    "timeout": 60,
    "temperature": 0,
}

# Define agents
assistant = autogen.AssistantAgent(
    name="planner",
    llm_config=config,
    system_message="""You are an expert software engineer. 
    Plan and write code to solve problems. 
    Always output a single Python code block.""",
)

user_proxy = autogen.UserProxyAgent(
    name="user",
    human_input_mode="NEVER",  # fully automated
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding",
        "use_docker": True,  # sandbox execution
    },
)

critic = autogen.AssistantAgent(
    name="critic",
    llm_config=config,
    system_message="""You are a code reviewer. 
    Review code for correctness and style. 
    End your review with APPROVED or REVISE.""",
)

# GroupChat: all agents in one room
groupchat = autogen.GroupChat(
    agents=[user_proxy, assistant, critic],
    messages=[],
    max_round=15,
    speaker_selection_method="auto",  # LLM selects next speaker
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=config)

# Kick off the conversation
user_proxy.initiate_chat(
    manager,
    message="Write a Python function that finds all prime numbers up to N using a sieve.",
)
```

The `speaker_selection_method="auto"` is the magic and the source of most AutoGen problems. The GroupChatManager sends the entire conversation history to the LLM and asks it to pick the next speaker. This means the speaker selection costs an LLM call every round, and the decision is opaque — you cannot inspect why the manager chose agent B over agent A in round 7. For reproducible workflows, this is a significant problem.

### AutoGen strengths

AutoGen genuinely shines for creative or research-style tasks where the optimal sequence of agent turns is not knowable in advance. If you are building a "red team vs. blue team" adversarial evaluation system, a debate-style fact-checker with multiple expert personas, or a research assistant that needs to dynamically decide whether to search, code, or critique, AutoGen's conversational model is more natural than a hardcoded graph. The `human_input_mode="ALWAYS"` option also makes it easy to build interactive agents that can pause and ask a human for input mid-conversation.

The two-agent case (one `UserProxy` + one `AssistantAgent`) is AutoGen's simplest and most reliable pattern. It maps cleanly to "user asks, assistant responds, user executes code, loop until done." For code generation workflows where the user proxy runs the generated code and feeds back the stdout, this pattern is extremely effective.

### AutoGen weaknesses

The conversational state (the growing list of messages) becomes the framework's biggest liability at scale. After 20 rounds, you have a 20,000-token context window full of conversation history, and every LLM call in the loop consumes that entire history. Token costs compound. More critically, the conversation history is opaque — it is a list of strings, not a typed state object. Debugging why an agent made a particular decision in round 15 requires reading through the full conversation trace, which is rarely a pleasant experience.

The `max_round` termination condition is a blunt instrument. You set it to 20 and hope the task finishes before round 20. If it finishes at round 3, you waste nothing. If it needs 25 rounds, it silently terminates with an incomplete result. More sophisticated termination conditions (based on the content of the latest message) require custom `is_termination_msg` functions that are fragile to prompt format changes.

Breaking changes in AutoGen 0.x are severe. The API changed enough between minor versions that upgrading from 0.1 to 0.2 required rewriting initialization logic for every agent. As of 2026, the framework is more stable, but the memory of those breaking changes makes engineering teams cautious about adopting it for production systems.

| Dimension | AutoGen Score | Why |
|---|---|---|
| Abstraction level | High | Conversational model hides routing logic completely |
| Debuggability | Hard | 20-turn conversation thread is hard to parse |
| Flexibility | Medium | Custom speaker_selection partially helps; still constrained |
| Production readiness | Medium | No checkpointing; token costs compound |
| Community | Large | Microsoft backing; active research community |
| Learning curve | Very steep | ConversableAgent hierarchy is complex |
| Lock-in risk | High | Business logic entangled in system prompts |
| Observability | Weak | No built-in tracing; raw message history only |

## 4. CrewAI: Role-Based Crews

CrewAI takes the most human-relatable abstraction of the three: you have a crew of workers, each with a role and a set of skills, and you give them tasks to execute in a defined order. The mental model is familiar to anyone who has managed a team. That familiarity is both the strength and the limitation.

![CrewAI sequential process: Crew kickoff triggers ResearchAgent, then AnalystAgent, then WriterAgent, each receiving the previous output as context](/imgs/blogs/orchestration-frameworks-compared-4.webp)

### The crew model

A crew consists of agents and tasks. An agent has a `role`, a `goal`, and a `backstory` — these become the system prompt that shapes its behavior. A task has a `description`, an `expected_output`, and an `agent` assignment. The crew's `process` can be `sequential` (each task runs in order, passing its output to the next) or `hierarchical` (a manager agent delegates tasks to worker agents).

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileWriterTool

search_tool = SerperDevTool()
file_writer = FileWriterTool()

# Define agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive, accurate information on {topic}",
    backstory="""You are an expert researcher at a top consulting firm.
    You excel at finding relevant information from multiple sources 
    and synthesizing it into clear insights. You always cite sources.""",
    tools=[search_tool],
    verbose=True,
    memory=True,
    llm="claude-opus-4-5",
)

analyst = Agent(
    role="Business Analyst",
    goal="Extract actionable insights from research data",
    backstory="""You are a senior analyst who transforms raw research 
    into structured business intelligence. You identify patterns, 
    risks, and opportunities that others miss.""",
    verbose=True,
    memory=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, compelling reports from analyzed data",
    backstory="""You produce executive-quality reports that balance 
    technical depth with clarity. Your reports are always actionable.""",
    tools=[file_writer],
    verbose=True,
)

# Define tasks
research_task = Task(
    description="""Research {topic} thoroughly. 
    Find: market size, key players, recent trends, challenges.
    Search at least 5 sources. Include URLs.""",
    expected_output="A detailed research summary with at least 500 words and source citations",
    agent=researcher,
)

analysis_task = Task(
    description="""Analyze the research data. 
    Identify the top 3 opportunities and top 3 risks.
    Rank them by potential business impact.""",
    expected_output="A structured analysis with ranked opportunities and risks",
    agent=analyst,
    context=[research_task],  # depends on research output
)

writing_task = Task(
    description="""Write a 2-page executive report combining the research 
    and analysis. Include an executive summary, key findings, and recommendations.""",
    expected_output="A formatted executive report in markdown",
    agent=writer,
    context=[research_task, analysis_task],
    output_file="report.md",
)

# Define and run the crew
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,
    verbose=True,
    memory=True,  # shared memory across agents
)

result = crew.kickoff(inputs={"topic": "AI agent frameworks in 2026"})
print(result.raw)
```

### Hierarchical process

CrewAI's hierarchical process is more powerful than sequential, though it comes with its own complexity. In hierarchical mode, a manager agent (usually a powerful LLM like GPT-4 or Claude Opus) reads all tasks and delegates them to worker agents it deems appropriate. This can handle some branching cases that sequential cannot, but the delegation logic is in the manager's prompt, making it as opaque as AutoGen's speaker selection.

```python
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.hierarchical,
    manager_llm="claude-opus-4-5",  # explicit manager LLM
    verbose=True,
)
```

### CrewAI strengths

CrewAI is genuinely the fastest path from "I have an idea for a multi-agent workflow" to a working demo. The role/goal/backstory model for defining agent behavior is intuitive and maps cleanly to how non-engineers think about delegation. For content creation workflows — research, draft, review — it is nearly perfect. The learning curve is shallow enough that a product manager can read and understand a CrewAI workflow without knowing Python deeply.

For teams building internal tools or exploring agent capabilities, CrewAI reduces the time-to-demo from days to hours. That speed advantage is real and should not be dismissed.

### CrewAI weaknesses

Sequential process has no conditional branching. If your research task finds "insufficient data," there is no way to branch to a "gather more data" task and then retry the analysis — at least not without deeply hacky workarounds. The output of one task is passed as a string to the next; there is no structured state. If the writer task needs to know whether the analyst flagged high risk, it has to parse that from a natural language string, which is fragile.

The `backstory` field in agent definitions is the root cause of a subtle and pervasive problem: your business logic is encoded in a natural language prompt field. Changing agent behavior requires editing prose, not code. This is manageable at small scale and becomes a maintenance nightmare at production scale when five engineers are modifying overlapping backstory fields to fix subtle output quality issues.

Version instability is worse in CrewAI than in LangGraph. The deprecation cycle is fast, and features that worked in CrewAI 0.28 may have been removed or renamed in 0.35. Always pin your CrewAI version in production.

| Dimension | CrewAI Score | Why |
|---|---|---|
| Abstraction level | High | role/goal/backstory hides all routing in prompts |
| Debuggability | Medium | sequential is inspectable; hierarchical is opaque |
| Flexibility | Low | sequential process has no if/else; hierarchical is fragile |
| Production readiness | Weak | no checkpointing; no retry; fast deprecation |
| Community | Growing | active user base; good docs |
| Learning curve | Easy | familiar mental model; excellent tutorials |
| Lock-in risk | High | business logic in backstory strings |
| Observability | Weak | verbose logs only; no structured tracing |

## 5. Raw Python / No Framework: The Underrated Option

Here is the option that most blog posts skip because it is not exciting: don't use a framework. Write a Python function that calls an LLM. Write another function that calls a different LLM. Connect them with an `if` statement. Ship it.

This is not laziness. For a substantial fraction of multi-agent use cases, a few hundred lines of well-structured Python with explicit control flow is strictly better than any framework — more debuggable, more portable, more performant, and easier to test.

```python
import anthropic
import asyncio
from typing import Optional
from dataclasses import dataclass

client = anthropic.Anthropic()

@dataclass
class AgentResult:
    output: str
    confidence: float
    needs_retry: bool

def call_agent(system_prompt: str, user_message: str, 
               model: str = "claude-opus-4-5", max_tokens: int = 4096) -> str:
    """Single LLM call with retry on rate limits."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_message}],
        system=system_prompt,
    )
    return response.content[0].text

def research_agent(query: str) -> str:
    return call_agent(
        system_prompt="You are a research analyst. Find key facts and cite sources.",
        user_message=query,
    )

def analysis_agent(research_output: str, query: str) -> AgentResult:
    response = call_agent(
        system_prompt="""Analyze research output. 
        Rate confidence 0-1. 
        Output JSON: {"analysis": "...", "confidence": 0.85}""",
        user_message=f"Query: {query}\n\nResearch: {research_output}",
    )
    import json
    data = json.loads(response)
    return AgentResult(
        output=data["analysis"],
        confidence=data["confidence"],
        needs_retry=data["confidence"] < 0.7,
    )

def writer_agent(analysis: str, query: str) -> str:
    return call_agent(
        system_prompt="You are a technical writer. Write clear, concise reports.",
        user_message=f"Query: {query}\n\nAnalysis: {analysis}",
    )

def run_pipeline(query: str, max_retries: int = 3) -> str:
    """Explicit, debuggable, testable pipeline."""
    for attempt in range(max_retries):
        research = research_agent(query)
        result = analysis_agent(research, query)
        
        if not result.needs_retry:
            return writer_agent(result.output, query)
        
        print(f"Attempt {attempt + 1}: confidence {result.confidence:.2f}, retrying...")
        query = f"{query} (Note: previous attempt had low confidence, be more thorough)"
    
    # Best effort on final attempt
    return writer_agent(result.output, query)

# Parallel execution with asyncio
async def parallel_research(queries: list[str]) -> list[str]:
    """Run multiple research tasks concurrently."""
    async def async_research(q: str) -> str:
        return await asyncio.to_thread(research_agent, q)
    
    return await asyncio.gather(*[async_research(q) for q in queries])
```

![Framework boilerplate vs raw Python: CrewAI's 90 lines of configuration vs the same pipeline in 28 lines of raw Python](/imgs/blogs/orchestration-frameworks-compared-9.webp)

This code does not look impressive. It fits on a screen. You can read it top to bottom and understand every decision. You can add a `print` statement anywhere to see what is happening. You can write a unit test for every function. You can swap in a different LLM by changing one string. You can add a retry decorator that logs every retry attempt to your existing monitoring system.

What this code does not give you: automatic state management, built-in checkpointing, a pretty graph visualization, or the ability to demo it in a conference talk. For internal tools, ETL pipelines, and well-understood workflows with fewer than five agents, those absences are not losses.

| Dimension | Raw Python Score | Why |
|---|---|---|
| Abstraction level | None | Direct LLM calls; no indirection |
| Debuggability | Best | Print statements work; straightforward stack traces |
| Flexibility | Maximum | Any Python is valid |
| Production readiness | You own it | Implement what you need; nothing forced |
| Community | N/A | Standard Python; no framework-specific knowledge |
| Learning curve | Varies by developer | Familiar to any Python developer |
| Lock-in risk | Zero | No framework dependencies |
| Observability | You own it | Integrate with any observability stack |

## 6. Comparison Across 8 Dimensions

The table below crystallizes the tradeoffs across four options. Read each row as a question: "For this dimension, which option is my bottleneck?"

![Framework comparison across 8 dimensions: abstraction, debug, flexibility, production readiness, community, learning curve, lock-in, observability](/imgs/blogs/orchestration-frameworks-compared-5.webp)

### Abstraction level

**LangGraph** sits at medium abstraction — you write Python nodes and the graph compiler handles scheduling and state merging. You can read the compiled graph as code. **AutoGen** is high abstraction — the GroupChatManager makes routing decisions opaque to you. **CrewAI** is high abstraction — the backstory-based prompts and sequential execution hide control flow behind role definitions. **Raw Python** has zero abstraction overhead — what you see is what executes.

High abstraction is only beneficial when you gain something proportional to what you lose. AutoGen gains emergent conversation dynamics at the cost of reproducibility. CrewAI gains role-based clarity at the cost of control flow expressiveness.

### Debuggability

**Raw Python** wins this unconditionally. Stack traces are clean, `print` works, debuggers work, unit tests work. **LangGraph** is second — the TypedDict state is inspectable at any point, and checkpoints let you replay from any node. The LangSmith integration provides full execution traces with input/output per node. **CrewAI** sits in the middle — sequential execution is easy to follow, but hierarchical delegation and memory state are murky. **AutoGen** is the hardest to debug — the growing conversation history is the state, and reading 15 rounds of agent dialogue to understand why something went wrong is genuinely painful.

### Flexibility

**Raw Python** is unbounded — any Python is valid. **LangGraph** is highly flexible within the graph model — you can add nodes, change edge logic, add parallelism, and modify state schema at any point. **AutoGen** is moderately flexible — you can customize speaker selection with a function, but the conversational architecture resists fundamental changes. **CrewAI** is the least flexible — sequential process has no branching, and working around it requires nesting crews or using the hierarchical process as a proxy.

### Production readiness

**LangGraph** leads on this dimension: checkpointing, state persistence, human-in-the-loop, and LangGraph Cloud for deployment are all production-grade features. **Raw Python** comes second — you own the implementation, which means you implement exactly what you need and nothing you don't. This is a net positive if your team has the engineering bandwidth. **AutoGen** is moderate — the lack of checkpointing and the token cost of growing conversation histories are genuine production risks. **CrewAI** is weak — no checkpointing means a 3-hour research crew run lost to a network error cannot resume.

## 7. Framework Selection Decision Tree

Six questions, answered in order, will get you to a recommendation in most cases.

![Framework selection decision tree: six questions leading to four framework recommendations](/imgs/blogs/orchestration-frameworks-compared-6.webp)

**Question 1: Is this a prototype/demo, or is it going to production?**

If prototype: use CrewAI for linear tasks (research → write → review), or AutoGen for conversational/debate-style tasks. The speed advantage is real, and the weaknesses do not matter for a demo. If production: proceed to question 2.

**Question 2: Does your workflow have complex conditional branching or retry loops?**

If yes: LangGraph. The conditional edge model handles arbitrary branching cleanly, and cycles are first-class citizens. If no: proceed to question 3.

**Question 3: Does your workflow need checkpointing (long-running, expensive, or resumable)?**

If yes: LangGraph. It is the only framework with production-grade checkpointing built in. If no: proceed to question 4.

**Question 4: Is your workflow primarily conversational — agents debating, critiquing, or building on each other's messages?**

If yes: AutoGen. The conversational model is genuinely the best fit for research teams, red team/blue team, or multi-persona debate. If no: proceed to question 5.

**Question 5: Is your workflow a simple linear sequence with well-defined handoffs?**

If yes: evaluate whether the simplicity justifies a framework at all. A three-agent linear pipeline is 30 lines of raw Python. CrewAI adds maybe 20 lines for a minimal gain. If the team is non-technical and needs to read/modify the pipeline, CrewAI's role-based model is worth the overhead. Otherwise, raw Python wins. If the pipeline has more than 5 agents or complex context passing: CrewAI sequential or LangGraph basic pipeline.

**Question 6: Does your team have the bandwidth to learn and maintain a framework?**

This is the question that overrides all previous answers. If your team is two engineers building a side project, the learning curve of LangGraph — state machines, graph compilation, reducer functions, checkpointing — may cost more than it saves. Raw Python with a thin retry/logging wrapper is a legitimate production choice for small teams.

## 8. Escape Hatches: Migrating Off a Framework

Every framework has an escape hatch — a way to drop down to a lower abstraction level when the framework's model does not fit your problem. Knowing the escape hatch before you need it is the difference between a three-hour migration and a three-week rewrite.

![Framework migration path: raw Python PoC to LangGraph state machine to observability layer to escape hatches and raw fallback](/imgs/blogs/orchestration-frameworks-compared-7.webp)

### LangGraph escape hatches

The cleanest escape hatch in LangGraph is to make a node a raw Python callable that ignores the state machine model entirely. The node receives the state, calls whatever code it wants — including direct LLM API calls, external API calls, or even a sub-pipeline built in raw Python — and returns a state update. LangGraph does not care what happens inside a node.

```python
def escape_hatch_node(state: AgentState) -> AgentState:
    """This node bypasses the graph model; calls LLM directly."""
    # Direct Anthropic call, no LangChain wrappers
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=8096,
        messages=[{"role": "user", "content": state["messages"][-1]}]
    )
    return {"messages": [response.content[0].text]}

# This node participates in the graph like any other
workflow.add_node("escape_hatch", escape_hatch_node)
```

If you are migrating away from LangGraph entirely, the incremental approach is: identify the nodes that carry the most value from the graph model (checkpointing, conditional routing), keep those as LangGraph nodes, and replace the rest with raw Python functions. Gradually expand the raw Python surface until the LangGraph overhead is minimal, then remove the graph wrapper.

### AutoGen escape hatches

AutoGen's `ConversableAgent` is the core class. Everything else inherits from it. The cleanest escape hatch is to override `generate_reply()` directly, which bypasses the LLM call and lets you return whatever you want:

```python
class CustomAgent(autogen.ConversableAgent):
    def generate_reply(self, messages=None, sender=None, config=None):
        """Bypass AutoGen's LLM call entirely."""
        last_message = messages[-1]["content"]
        # Your custom logic here
        result = my_custom_pipeline(last_message)
        return True, result
```

For migrating off AutoGen, the key insight is that the value AutoGen provides is the `GroupChatManager`'s speaker selection. If you can replace that with explicit routing logic in LangGraph or raw Python, you have eliminated the core dependency. Extract your agent logic into plain Python functions, then build the routing layer in your replacement framework.

### CrewAI escape hatches

The cleanest CrewAI escape hatch is to implement a custom tool that is actually a mini-pipeline. CrewAI tasks can call tools, and tools are arbitrary Python callables. A tool that internally calls three LLMs in sequence with conditional branching is completely valid and invisible to CrewAI's sequential model.

```python
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class ComplexAnalysisInput(BaseModel):
    research_text: str = Field(description="Research to analyze")

class ComplexAnalysisTool(BaseTool):
    name: str = "complex_analysis"
    description: str = "Performs multi-step analysis with branching logic"
    args_schema: Type[BaseModel] = ComplexAnalysisInput
    
    def _run(self, research_text: str) -> str:
        # Fully custom logic: if/else, loops, multiple LLM calls
        # CrewAI sees this as a single tool call
        if len(research_text) < 500:
            return simple_analysis(research_text)
        else:
            step1 = detailed_analysis(research_text)
            if step1.needs_clarification:
                step2 = clarify(research_text, step1)
                return combine(step1, step2)
            return step1.output
```

Migrating off CrewAI is typically straightforward because the business logic is mostly in the system prompts. Extract the `role`, `goal`, and `backstory` fields, turn them into a single `system_message` string, and call the LLM directly. The task descriptions become user messages. The sequential execution becomes explicit function calls. A full CrewAI migration to raw Python can often be done in a day.

## 9. The Framework Evolution Problem

Agent frameworks in 2026 are in approximately the same state as web frameworks in 2004. Django would not be released until 2005. Rails was brand new. Everyone was choosing between raw PHP, various PHP frameworks, Java servlets, and early Ruby experiments. The winning frameworks were not obvious, and several popular choices in 2004 are now completely forgotten.

This matters because framework choice is a long-term commitment. The code you write today will still exist in 2028, by which time the agent framework landscape will look completely different. Three dynamics are worth watching:

**Framework consolidation.** Today there are dozens of agent frameworks. By 2028, there will likely be three to five. The survivors will be the ones with strong backing (Microsoft/AutoGen, LangChain/LangGraph), large developer communities, or unique capabilities that nothing else provides. Smaller frameworks with no differentiated value proposition will be abandoned by their maintainers.

**LLM provider integration.** Anthropic, OpenAI, and Google are all building first-party agent primitives — computer use, tool calling with parallel execution, multi-agent handoffs. As these capabilities mature, the value proposition of third-party orchestration frameworks shrinks. The question "why would I use LangGraph when Anthropic's native agent APIs handle routing?" will have a sharper answer in two years.

**Standardization.** The Model Context Protocol (MCP) and similar standards are pushing toward a world where agent components are interchangeable. An orchestration framework that builds on standards rather than proprietary APIs has a longer half-life than one that locks you into its own ecosystem.

The framework-agnostic hedge is to keep your business logic — your system prompts, your tool implementations, your state schemas — in plain Python classes that do not import from any framework. Your framework integration should be a thin adapter layer that can be replaced. This is not easy to maintain under deadline pressure, but it is the only reliable defense against framework evolution risk.

## 10. Mixing Frameworks: When It Makes Sense

Using multiple frameworks in one system is generally a bad idea, but there are specific cases where it is the right architectural decision.

**LangGraph outer loop + CrewAI inner crew.** LangGraph manages the high-level state machine (route to research phase vs. writing phase vs. review phase, handle retries, checkpoint between phases). CrewAI runs the inner crew within each phase. This works because CrewAI's `crew.kickoff()` is a Python callable that can be wrapped in a LangGraph node. The outer loop gets LangGraph's checkpointing and retry model; the inner crew gets CrewAI's role-based task model.

```python
from crewai import Crew, Agent, Task, Process

def research_crew_node(state: AgentState) -> AgentState:
    """LangGraph node that internally runs a CrewAI crew."""
    crew = Crew(
        agents=[researcher_agent, fact_checker_agent],
        tasks=[research_task, fact_check_task],
        process=Process.sequential,
    )
    result = crew.kickoff(inputs={"query": state["current_query"]})
    return {"research_output": result.raw}

# This node is registered in a LangGraph graph like any other
workflow.add_node("research_phase", research_crew_node)
```

The key requirement: the LangGraph state object must be the source of truth. The inner CrewAI crew is stateless from LangGraph's perspective — it runs, returns a string, and the result is merged into the LangGraph state. No CrewAI-specific state escapes into the LangGraph layer.

**AutoGen inner loop for adversarial evaluation.** If you have a production system built in LangGraph and need to add an adversarial evaluation step where two agents debate the quality of an output, AutoGen's GroupChat is a natural choice for the inner loop. The same principle applies: wrap the AutoGen conversation in a LangGraph node, pass in the relevant context from the state, and return the conversation's final verdict to the state.

**When not to mix.** If both frameworks are trying to manage the same state — if AutoGen's message history and LangGraph's TypedDict are both tracking the same information — you have a synchronization problem that will produce subtle bugs. The rule is: one framework owns the state, the other provides a capability (usually a specialist workflow pattern).

## 11. Case Studies

### Case Study 1: The Legal Document Analyzer (LangGraph, worked)

A legal technology startup needed to build a contract review system. The workflow: extract clauses → classify each clause by type → flag risky clauses → generate redline suggestions → human lawyer reviews flagged items → export to Word. The human-in-the-loop requirement and the retry loop ("if confidence < 0.6, send to a different LLM") made this a perfect LangGraph use case.

The implementation used a TypedDict state with a `clauses` list, a `flags` dict keyed by clause ID, and a `human_approvals` set. The graph had seven nodes and three conditional edges. The `interrupt_before=["export_to_word"]` checkpoint let the lawyer review every flagged clause before the export ran. The SQLite checkpointing survived the five server restarts they had during a week of load testing.

The outcome: a system that cost $0.12 per contract in LLM API calls, completed 95% of reviews without human intervention, and processed 3,000 contracts in the first month of production. The LangGraph overhead was real — the initial setup took two weeks for one engineer — but the production reliability requirements justified every hour of it.

### Case Study 2: The Marketing Copy Generator (CrewAI → LangGraph migration, backfired then fixed)

A marketing SaaS company started with CrewAI: a researcher agent, a copywriter agent, and a critic agent in sequential process. The demo was impressive. Three weeks after launch, the product team asked for a feature: if the critic scores copy below 7/10, regenerate with the specific weaknesses addressed.

The developer's first attempt was a second crew that called the first crew. Then a third crew to handle the "addressed weaknesses" pass. After six weeks, they had three nested crews with a combined 600 lines of CrewAI configuration that no one on the team fully understood. Every change to one crew's backstory potentially affected the outputs of the downstream crews in unpredictable ways.

The migration to LangGraph took five days. The critic's score became a field in the TypedDict state. The retry loop became a conditional edge from `critic_node` back to `copywriter_node` with a `max_retries` counter in state. The result: 180 lines of LangGraph code that replaced 600 lines of nested CrewAI configuration, with explicit state at every step and a clear audit trail. The migration would have been faster if they had started with LangGraph.

### Case Study 3: The Research Team Simulation (AutoGen, worked)

A hedge fund's research team wanted to automate their pre-investment analysis. The process was inherently conversational: a "bull" analyst argued for the investment, a "bear" analyst argued against it, a "quant" ran numbers when claims needed validation, and a "generalist" synthesized the debate into a recommendation. The sequence of turns was not predetermined — the generalist would interrupt the debate at any point to ask the quant to run specific calculations.

AutoGen's GroupChat was the right call. The `speaker_selection_method="auto"` let the GroupChatManager decide dynamically when the quant needed to intervene. The conversation history was the audit trail that compliance required. The system ran 120 investment analyses in the first month, each producing a 500-1000 word debate transcript that the research team could review.

The limitation they hit: at round 15 of a complex debate, the context window costs became significant. A single analysis could cost $8–12 in API calls, mostly from the growing conversation history. They mitigated it by adding message summarization every five rounds, which reduced costs by 40% at the cost of some conversation coherence.

### Case Study 4: The ETL Pipeline (Raw Python, worked)

A data team needed to enrich a database of 50,000 companies with LLM-generated summaries, industry classifications, and risk scores. The workflow was simple: for each company, fetch its website text, call an LLM to extract structured data, validate the output, and write to the database. No conditional branching beyond "retry on rate limit." No shared state between records. No human-in-the-loop.

They spent one afternoon evaluating LangGraph and CrewAI and decided against both. The overhead — defining a TypedDict state for a workflow that had exactly two "states" (pending and done), defining nodes for three sequential LLM calls — was pure ceremony with no benefit. The final implementation was 120 lines of Python using asyncio.gather to process 100 companies concurrently, with a simple retry decorator and structured logging to their existing monitoring system.

Processing time: 8 hours for 50,000 records at $0.004 per record. No framework involved, no version migration risk, no new dependencies. The code was readable by every member of the data team on day one.

### Case Study 5: The Customer Support Bot (LangGraph, wrong choice initially)

A customer support platform wanted to add an AI triage bot. Route the ticket to billing, technical, or general support. For billing and technical tickets, extract structured data (account ID, error codes). For general tickets, generate a first-response draft. Simple conditional routing to three handlers.

The first engineer built it in LangGraph because they knew it and were comfortable with it. Four nodes (router, billing_extractor, tech_extractor, general_drafter), three conditional edges. The implementation took three days and worked correctly.

Six months later, the team wanted to add a fifth category (account cancellation, requiring legal-approved messaging). The LangGraph modification took two hours, including tests. But looking at the code afterward, two other engineers pointed out that the same system could have been built in 50 lines of raw Python with a `match` statement. The LangGraph overhead was unnecessary for this level of complexity, and the team had carried that overhead for six months.

The lesson is not that LangGraph was wrong — it worked fine. The lesson is that the team had not asked "does this workflow need what LangGraph provides?" before reaching for it. For routing complexity that fits in a `match` statement, raw Python is the right answer.

### Case Study 6: The Document QA System (AutoGen failure)

An enterprise software company needed a document Q&A system. Users ask questions; the system searches a vector database, retrieves relevant chunks, and generates an answer with citations. Classic RAG, except they wanted two agents to improve quality: one to retrieve and one to synthesize.

They used AutoGen's two-agent pattern: a `UserProxy` that executed the vector search tool, and an `AssistantAgent` that synthesized the results. This worked well in testing. In production, they hit two problems.

First, long documents filled the context window faster than expected. By the 10th turn of a complex Q&A session, the growing message history was consuming 12,000 tokens per request, and response times had degraded from 1.5 seconds to 6 seconds.

Second, when the system was wrong, debugging was nearly impossible. The `UserProxy`'s code execution logs, the `AssistantAgent`'s synthesis reasoning, and the tool call results were all interleaved in the growing message history. Finding where the retrieval went wrong required reading through thousands of lines of conversation transcript.

They rewrote the system in LangGraph in two weeks. The state object held the retrieved chunks and the generated answer separately. The retrieval node and synthesis node were independent, each with their own logging. Response times dropped back to 1.5 seconds, and debugging time per incident dropped from hours to minutes.

### Case Study 7: The Code Review Agent (LangGraph + raw Python hybrid)

A developer tools company built an automated code review system. The high-level flow was: analyze the PR diff → classify the types of issues → for each issue type, run a specialized checker → aggregate results → generate summary. This was clearly a LangGraph graph (conditional dispatch to specialized checkers based on classification output).

But one of the specialized checkers — the security vulnerability scanner — needed to run 12 pattern-matching checks in parallel, each with different tools and prompts. Running these as 12 LangGraph nodes with a fan-out edge pattern was theoretically correct but practically unreadable. The graph visualization became a tangle of edges.

The solution: the security checker was a raw Python function using `asyncio.gather` that ran 12 concurrent LLM calls. From LangGraph's perspective, it was a single node that took state and returned state. From the security checker's perspective, it was 12 concurrent API calls with no framework overhead. The hybrid worked cleanly because the boundary was clear: LangGraph owns the top-level routing and state management; raw Python owns the parallelism within a single node.

### Case Study 8: The Regulatory Compliance Checker (CrewAI to raw Python)

A fintech company built a regulatory compliance checker using CrewAI. Five agents: a policy reader, a transaction analyzer, a risk scorer, a flag generator, and a report writer. The sequential process was correct — each agent really did depend on the previous. CrewAI was a reasonable choice.

Eight months later, they needed to add a new regulatory framework (EU AI Act compliance, not just US banking regulations). The challenge: the policy reader needed to read different policy documents based on the transaction's geography. CrewAI's sequential process could not express this. They tried adding a sixth agent before the policy reader to pre-select documents, but the output of a "pre-selector" agent could not be used to dynamically configure the policy reader's tools.

The rewrite to raw Python took three days. The geography check became a two-line `if` statement. The dynamic tool configuration became a function argument. The five agents became five Python functions with explicit inputs and outputs. The entire system was 200 lines of Python. The CrewAI version had been 400 lines of framework configuration wrapping 200 lines of actual logic. The migration saved 400 lines of code while adding the geographic routing feature that had been impossible to add before.

## 12. When to Adopt, When to Build Thin Wrappers, When to Go Raw

The framework-vs-raw-Python decision is not binary. There is a middle path — a thin wrapper — that gives you some structure without the full overhead of a framework.

![Project profiles mapped to recommended frameworks](/imgs/blogs/orchestration-frameworks-compared-10.webp)

**Adopt a framework when:**

Your workflow has complex state management requirements that would require significant custom code to implement correctly (typed state with merge semantics, checkpointing, replay). You need human-in-the-loop at specific decision points. Your workflow has conditional routing logic that would require more than ~50 lines of custom router code. Your team is large enough that the framework's conventions provide value as a shared language. You need built-in observability and your team does not have the bandwidth to build custom tracing.

The clearest signals for LangGraph: retry loops, conditional routing, human approval gates, expensive workflows that need checkpointing. The clearest signals for AutoGen: adversarial/debate-style agents, dynamic speaker selection, research team simulation. The clearest signals for CrewAI: rapid prototyping, linear workflows with clear role assignments, non-technical team members who need to read and modify the pipeline.

**Build a thin wrapper when:**

Your workflow is a simple sequence of LLM calls, but you want consistent retry logic, rate limit handling, and logging across all calls. You want structured outputs enforced consistently. You want a common interface for swapping LLM providers. A thin wrapper might be 50-100 lines of code that gives you:

```python
class Agent:
    def __init__(self, system_prompt: str, model: str = "claude-opus-4-5"):
        self.system_prompt = system_prompt
        self.model = model
        self.client = anthropic.Anthropic()
    
    def run(self, message: str, context: dict = None) -> str:
        """Structured call with retry and logging."""
        if context:
            message = f"Context: {json.dumps(context)}\n\n{message}"
        
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": message}],
                    system=self.system_prompt,
                )
                return response.content[0].text
            except anthropic.RateLimitError:
                time.sleep(2 ** attempt)
        raise RuntimeError("Max retries exceeded")

class Pipeline:
    def __init__(self, agents: list[tuple[str, Agent]]):
        self.agents = agents  # [(name, agent), ...]
    
    def run(self, initial_input: str) -> dict[str, str]:
        """Sequential pipeline with full output capture."""
        outputs = {}
        current_input = initial_input
        for name, agent in self.agents:
            output = agent.run(current_input, context=outputs)
            outputs[name] = output
            current_input = output
        return outputs
```

This is 45 lines. It gives you retry logic, logging hooks (you can add them), and a clean interface for building sequential pipelines. It is not LangGraph. It is not CrewAI. It is a thin wrapper over the Anthropic SDK that you wrote yourself and fully understand.

**Go raw when:**

Your workflow is fewer than five agents, the routing is a simple `if` statement or `match`, you need maximum observability into every API call, performance is a concern (framework overhead is real and measurable), your team is small and the framework learning curve does not amortize, or you are building a one-time automation rather than a product.

The honest answer for a substantial fraction of "should I use an agent framework?" questions is: start raw. Build the minimal thing that works. Identify the specific pain points — "I need checkpointing" or "I need to express this conditional routing cleanly." Then add exactly the framework that addresses those specific pain points, and no more.

The most expensive engineering decisions are the ones made at the start of a project, when you know the least about what the project actually needs. A framework adopted on day one for a use case that turns out to be simple sequential LLM calls is a dependency you carry for the lifetime of the project. A framework adopted on week four, after you have learned what your actual requirements are, is a decision grounded in evidence.

## The Decision Framework for Your Next Project

Before you pick a framework, answer five questions in writing:

1. What is the conditional routing logic in my workflow? Draw it on a whiteboard. Count the branches. If the branch count is zero or one, you do not need a state machine.

2. Does my workflow have retry loops? How deep can they go? If a loop can exceed 10 iterations, you need an iteration limit and a way to inspect state mid-loop.

3. How expensive is the workflow end-to-end? If a full run costs more than $1, you need checkpointing. LangGraph is currently the only major framework that provides production-grade checkpointing.

4. Will non-engineers need to read or modify this pipeline? If yes, CrewAI's role-based model has genuine value. If no, the value is minimal.

5. What is the failure mode I am most afraid of? If the answer is "the agent runs forever burning API budget," you need explicit iteration limits and LangGraph's `max_recursion` configuration. If the answer is "I cannot debug why it got the wrong answer," you need raw Python or LangGraph with LangSmith. If the answer is "the framework breaks and I cannot upgrade," you need a thin wrapper or raw Python.

The frameworks are good tools. Use them when they are the right tool. Do not use them when they are not.

---

For further reading on the patterns that underlie these frameworks:

- [Multi-agent topologies](/blog/machine-learning/ai-agent/multi-agent-topologies) — the architectural patterns (sequential, parallel, hierarchical, swarm) that all frameworks implement
- [Agent-as-tool pattern](/blog/machine-learning/ai-agent/agent-as-tool-pattern) — the composition primitive that enables LangGraph + CrewAI hybrids and raw Python pipelines that call sub-agents
- [Shared state and coordination](/blog/machine-learning/ai-agent/shared-state-and-coordination) — the state management problem in depth, including the tradeoffs between typed state (LangGraph), conversation history (AutoGen), and explicit function arguments (raw Python)
