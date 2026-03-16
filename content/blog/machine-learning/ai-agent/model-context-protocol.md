---
title: "Model Context Protocol (MCP): The USB-C Port for AI Applications"
publishDate: "2026-03-16"
category: "machine-learning"
subcategory: "AI Agent"
tags: ["mcp", "ai-agent", "protocol", "anthropic", "tool-calling", "llm", "integration"]
date: "2026-03-16"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Model Context Protocol (MCP) is an open standard that lets AI applications connect to any external system through a unified interface — like USB-C for AI. This guide covers the architecture, primitives, transport layers, and real-world use cases with code examples."
---

## Introduction

Every AI application eventually hits the same wall: **it needs to talk to the outside world.**

Your AI coding assistant needs to read files, search documentation, and run commands. Your AI customer support bot needs to query databases, look up orders, and send emails. Your AI research assistant needs to search the web, read PDFs, and manage citations.

Before MCP, every integration was **custom-built**. If you wanted Claude to talk to Slack, you'd build a Slack integration for Claude. If you wanted GPT to talk to Slack, you'd build a completely separate Slack integration for GPT. And if you wanted either of them to also talk to GitHub, Jira, and your internal database? That's 2 AI apps x 4 external systems = **8 custom integrations** to build and maintain.

```
Before MCP: N x M integration problem

AI Apps          External Systems
┌─────────┐     ┌──────────┐
│ Claude   │────→│ Slack    │  Custom integration #1
│          │────→│ GitHub   │  Custom integration #2
│          │────→│ Jira     │  Custom integration #3
│          │────→│ Database │  Custom integration #4
└─────────┘     └──────────┘
┌─────────┐     ┌──────────┐
│ GPT     │────→│ Slack    │  Custom integration #5
│          │────→│ GitHub   │  Custom integration #6
│          │────→│ Jira     │  Custom integration #7
│          │────→│ Database │  Custom integration #8
└─────────┘     └──────────┘

Total: 8 custom integrations (and growing!)
```

**Model Context Protocol (MCP)** solves this with a standardized interface. Build one MCP server for Slack, and **every** MCP-compatible AI application can use it. Build one MCP client in your AI app, and it can connect to **every** MCP server.

```
After MCP: N + M solution

AI Apps (MCP Clients)     MCP Servers
┌─────────┐              ┌──────────┐
│ Claude   │──┐      ┌──→│ Slack    │  1 server
│          │  │      │   └──────────┘
└─────────┘  │  ┌───┤   ┌──────────┐
┌─────────┐  ├──│MCP│──→│ GitHub   │  1 server
│ GPT     │──┤  └───┘   └──────────┘
│          │  │      │   ┌──────────┐
└─────────┘  │      ├──→│ Jira     │  1 server
┌─────────┐  │      │   └──────────┘
│ VS Code  │──┘      └──→│ Database │  1 server
└─────────┘              └──────────┘

Total: 3 clients + 4 servers = 7 components (not 12 integrations)
And adding a new AI app or new service costs +1, not +N
```

This is the same pattern as USB-C for hardware: one standardized port that works with everything.

## Who Uses MCP Today?

MCP is already supported by major AI applications:

- **Claude Desktop** and **Claude Code** (Anthropic)
- **ChatGPT** (OpenAI)
- **Visual Studio Code** (GitHub Copilot)
- **Cursor** (AI code editor)
- **Windsurf** (AI code editor)
- **Zed** (code editor)

And hundreds of MCP servers exist for services like GitHub, Slack, Google Drive, PostgreSQL, Puppeteer, Sentry, Notion, Linear, and many more.

## Architecture: The Three Players

MCP follows a **client-server architecture** with three distinct roles:

```
┌─────────────────────────────────────────────────────────┐
│                    MCP HOST                               │
│            (e.g., Claude Desktop, VS Code)                │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  MCP Client   │  │  MCP Client   │  │  MCP Client   │  │
│  │  (1:1 conn)   │  │  (1:1 conn)   │  │  (1:1 conn)   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                  │                  │            │
└─────────┼──────────────────┼──────────────────┼───────────┘
          │                  │                  │
          ▼                  ▼                  ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  MCP Server   │  │  MCP Server   │  │  MCP Server   │
   │  (Filesystem) │  │  (GitHub)     │  │  (Database)   │
   └──────────────┘  └──────────────┘  └──────────────┘
```

### MCP Host

The **host** is the AI application the user interacts with — Claude Desktop, VS Code, Cursor, etc. It:

- Creates and manages multiple MCP clients
- Routes messages between the AI model and MCP clients
- Enforces security policies and user consent
- Controls which servers the AI can access

### MCP Client

Each **client** is a component inside the host that maintains a **dedicated 1:1 connection** to a single MCP server. The client:

- Handles the protocol lifecycle (initialization, capability negotiation, shutdown)
- Translates between the host's internal format and MCP messages
- Manages subscriptions and notifications from its connected server

### MCP Server

A **server** is a program that exposes capabilities to AI applications through the MCP protocol. It:

- Declares what it can do (tools, resources, prompts) during initialization
- Responds to requests from clients
- Can be a local process (communicating via stdio) or a remote service (communicating via HTTP)

**Example**: A GitHub MCP server might expose tools like `create_issue`, `list_pull_requests`, `merge_pr`, and resources like repository file contents.

### How It All Connects: A Complete Message Flow

Let's trace a complete interaction from user input to final response to see how all three players work together:

```
User types: "Create a GitHub issue for the login bug"

┌──────────────────────────────────────────────────────────────────────┐
│ Step 1: Host receives user message                                    │
│   Claude Desktop sends user message + available tools to Claude API   │
│   Available tools include all tools from all connected MCP servers    │
└──────────────────────┬───────────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 2: Model decides to use a tool                                   │
│   Claude API returns: "I want to call create_github_issue             │
│   with arguments {repo: 'myorg/app', title: 'Fix login bug'}"        │
└──────────────────────┬───────────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 3: Host routes to correct client                                 │
│   Claude Desktop identifies: create_github_issue belongs to the       │
│   GitHub MCP client → forwards the tool call                          │
└──────────────────────┬───────────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 4: Client sends JSON-RPC request to server                       │
│   MCP Client → GitHub MCP Server:                                     │
│   {"method": "tools/call", "params": {"name": "create_github_issue",  │
│    "arguments": {"repo": "myorg/app", "title": "Fix login bug"}}}     │
└──────────────────────┬───────────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 5: Server executes and responds                                  │
│   GitHub MCP Server calls GitHub API → creates issue #142             │
│   Returns: {"result": {"content": [{"type": "text",                   │
│   "text": "Created issue #142"}]}}                                    │
└──────────────────────┬───────────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 6: Result flows back to model                                    │
│   Server → Client → Host → Claude API (with tool result)              │
│   Claude generates final response: "I've created GitHub issue #142    │
│   for the login bug in myorg/app."                                    │
└──────────────────────┬───────────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Step 7: Host displays response to user                                │
│   Claude Desktop shows: "I've created GitHub issue #142..."           │
└──────────────────────────────────────────────────────────────────────┘
```

Understanding this flow is essential: the MCP server never talks to the LLM directly. It only talks to the MCP client, which talks to the host, which talks to the model. This separation is a deliberate design choice for security and modularity.

## The Protocol: JSON-RPC 2.0 Under the Hood

MCP messages are **JSON-RPC 2.0** messages, UTF-8 encoded. This means every message is a JSON object with a standard structure:

```json
// Request (client → server or server → client)
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": { "location": "Tokyo" }
  }
}

// Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{ "type": "text", "text": "Tokyo: 22°C, sunny" }],
    "isError": false
  }
}

// Notification (no response expected)
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": { "uri": "file:///project/src/main.rs" }
}
```

Three message types:
- **Requests**: Have an `id`, expect a response
- **Responses**: Match a request's `id`, contain `result` or `error`
- **Notifications**: No `id`, fire-and-forget

### Connection Lifecycle

Every MCP connection goes through a precise initialization handshake:

```
Client                              Server
  │                                    │
  │  ──── initialize request ────→     │  Step 1: Client sends protocol version
  │                                    │          and its capabilities
  │  ←── initialize response ────      │  Step 2: Server responds with its
  │                                    │          protocol version and capabilities
  │  ── initialized notification ──→   │  Step 3: Client confirms ready
  │                                    │
  │  ← ─ ─ normal operation ─ ─ →      │  Step 4: Exchange messages
  │                                    │
  │  ──── shutdown / disconnect ──→    │  Step 5: Clean termination
  │                                    │
```

Here's what the initialization looks like in practice:

```json
// Step 1: Client sends capabilities
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "sampling": {},        // "I can handle LLM sampling requests"
      "elicitation": {}      // "I can prompt the user for input"
    },
    "clientInfo": {
      "name": "claude-desktop",
      "version": "1.2.0"
    }
  }
}

// Step 2: Server responds with its capabilities
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "tools": { "listChanged": true },     // "I have tools, and they can change"
      "resources": { "subscribe": true },    // "I have resources you can subscribe to"
      "prompts": { "listChanged": true }     // "I have prompt templates"
    },
    "serverInfo": {
      "name": "github-mcp-server",
      "version": "2.1.0"
    }
  }
}

// Step 3: Client confirms initialization complete
{
  "jsonrpc": "2.0",
  "method": "notifications/initialized"
}
```

This handshake is critical: **both sides discover what the other supports.** If a server doesn't declare `tools` in its capabilities, the client won't try to call tools. If the client doesn't declare `sampling`, the server won't try to request LLM completions. This capability negotiation makes the protocol extensible without breaking backward compatibility.

## The Three Core Primitives

MCP exposes three types of capabilities, each controlled by a different party:

```
┌─────────────────────────────────────────────────┐
│                 MCP Primitives                    │
├─────────────┬─────────────────┬─────────────────┤
│   TOOLS     │   RESOURCES     │   PROMPTS       │
│             │                 │                  │
│  Controlled │  Controlled by  │  Controlled by   │
│  by MODEL   │  APPLICATION    │  USER            │
│             │                 │                  │
│  "Do this"  │  "Read this"    │  "Use this       │
│  (actions)  │  (data)         │   template"      │
├─────────────┼─────────────────┼─────────────────┤
│  Functions  │  File contents  │  Slash commands  │
│  API calls  │  DB records     │  Reusable prompts│
│  Commands   │  API responses  │  Workflow starters│
└─────────────┴─────────────────┴─────────────────┘
```

### 1. Tools: Actions the Model Can Take

Tools are **executable functions** that the LLM can invoke. They're the most powerful primitive — the model decides when and how to use them based on the conversation context.

**Discovery** — the client asks what tools are available:

```json
// Client → Server
{ "method": "tools/list" }

// Server → Client
{
  "tools": [
    {
      "name": "create_github_issue",
      "title": "Create GitHub Issue",
      "description": "Creates a new issue in a GitHub repository",
      "inputSchema": {
        "type": "object",
        "properties": {
          "repo": {
            "type": "string",
            "description": "Repository in owner/name format"
          },
          "title": {
            "type": "string",
            "description": "Issue title"
          },
          "body": {
            "type": "string",
            "description": "Issue body in Markdown"
          },
          "labels": {
            "type": "array",
            "items": { "type": "string" },
            "description": "Labels to apply"
          }
        },
        "required": ["repo", "title"]
      }
    },
    {
      "name": "search_issues",
      "title": "Search Issues",
      "description": "Search for issues across repositories",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": { "type": "string", "description": "Search query" }
        },
        "required": ["query"]
      }
    }
  ]
}
```

**Invocation** — the model calls a tool:

```json
// Client → Server
{
  "method": "tools/call",
  "params": {
    "name": "create_github_issue",
    "arguments": {
      "repo": "myorg/myapp",
      "title": "Fix login page timeout",
      "body": "Users report 504 errors when logging in during peak hours.",
      "labels": ["bug", "high-priority"]
    }
  }
}

// Server → Client
{
  "result": {
    "content": [{
      "type": "text",
      "text": "Created issue #142: 'Fix login page timeout' in myorg/myapp\nURL: https://github.com/myorg/myapp/issues/142"
    }],
    "isError": false
  }
}
```

Tools can return multiple content types — text, images, audio, or resource links:

```json
// A screenshot tool might return:
{
  "content": [
    { "type": "text", "text": "Screenshot of the login page:" },
    { "type": "image", "data": "iVBORw0KGgo...", "mimeType": "image/png" }
  ]
}
```

**Error handling** distinguishes between protocol errors and tool execution errors:

```json
// Protocol error: tool doesn't exist (JSON-RPC error)
{
  "error": { "code": -32602, "message": "Unknown tool: delete_database" }
}

// Tool execution error: the tool ran but failed (still a success response)
{
  "result": {
    "content": [{ "type": "text", "text": "Error: Repository 'myorg/myapp' not found or access denied" }],
    "isError": true
  }
}
```

This distinction matters: protocol errors mean something is misconfigured, while tool errors are normal operational failures the model can reason about and retry.

#### Tool Annotations: Metadata for Smarter UX

Tools can include **annotations** — metadata hints that help the client make better UI and safety decisions without inspecting the tool's actual behavior:

```json
{
  "name": "delete_user",
  "description": "Permanently delete a user account and all associated data",
  "inputSchema": { ... },
  "annotations": {
    "title": "Delete User Account",
    "readOnlyHint": false,        // This tool modifies state
    "destructiveHint": true,      // This tool destroys data (show warning!)
    "idempotentHint": true,       // Safe to retry (deleting already-deleted = no-op)
    "openWorldHint": false        // Only affects this system, not external ones
  }
}
```

These hints let clients implement smarter behavior:
- `destructiveHint: true` → show a red confirmation dialog
- `readOnlyHint: true` → auto-approve without user confirmation
- `openWorldHint: true` → warn that this action affects external systems (e.g., sending an email)
- `idempotentHint: true` → safe to automatically retry on timeout

#### Structured Output: Machine-Readable Tool Results

Tools can define an `outputSchema` so results are both human-readable (in `content`) and machine-parseable (in `structuredContent`):

```json
// Tool definition with output schema
{
  "name": "get_stock_price",
  "description": "Get current stock price for a ticker symbol",
  "inputSchema": {
    "type": "object",
    "properties": { "ticker": { "type": "string" } },
    "required": ["ticker"]
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "ticker": { "type": "string" },
      "price": { "type": "number" },
      "currency": { "type": "string" },
      "change_percent": { "type": "number" }
    }
  }
}

// Tool result includes both human and machine formats
{
  "result": {
    "content": [{
      "type": "text",
      "text": "AAPL: $198.50 (+1.2%)"
    }],
    "structuredContent": {
      "ticker": "AAPL",
      "price": 198.50,
      "currency": "USD",
      "change_percent": 1.2
    }
  }
}
```

The `content` field is for the LLM to reason about (human-readable). The `structuredContent` is for programmatic downstream processing — a client could feed this directly into a charting library, a spreadsheet, or another tool.

#### Pagination: Handling Large Tool Lists

When a server exposes many tools, the `tools/list` response supports **cursor-based pagination**:

```json
// First page request
{ "method": "tools/list", "params": {} }

// Response with 50 tools and a cursor for more
{
  "tools": [ /* first 50 tools */ ],
  "nextCursor": "eyJwYWdlIjoyLCJsaW1pdCI6NTB9"
}

// Next page request
{ "method": "tools/list", "params": { "cursor": "eyJwYWdlIjoyLCJsaW1pdCI6NTB9" } }

// Response with next batch
{
  "tools": [ /* next 50 tools */ ],
  "nextCursor": null  // no more pages
}
```

This same pagination pattern applies to `resources/list` and `prompts/list` as well.

#### Progress Tracking: Long-Running Operations

For tools that take a long time (e.g., running a build, processing a large dataset), MCP supports **progress notifications**:

```json
// Client sends tool call with a progress token
{
  "method": "tools/call",
  "params": {
    "name": "run_test_suite",
    "arguments": { "project": "myapp" },
    "_meta": { "progressToken": "test-run-123" }
  }
}

// Server sends progress updates during execution
{ "method": "notifications/progress", "params": {
    "progressToken": "test-run-123",
    "progress": 15,
    "total": 100,
    "message": "Running unit tests (15/100)..."
}}

{ "method": "notifications/progress", "params": {
    "progressToken": "test-run-123",
    "progress": 78,
    "total": 100,
    "message": "Running integration tests (78/100)..."
}}

// Finally, the tool returns its result
{
  "result": {
    "content": [{ "type": "text", "text": "All 100 tests passed in 45.2s" }]
  }
}
```

The client can display a progress bar to the user while the tool executes. Without progress tracking, long-running tools would appear to hang.

#### Dynamic Tool Lists: Tools That Change

MCP servers can add or remove tools at runtime. When a server's tool list changes, it sends a notification:

```json
// Server notifies client that available tools have changed
{ "method": "notifications/tools/list_changed" }

// Client re-fetches the tool list
{ "method": "tools/list" }
```

This is useful for:
- Servers that connect to external systems on-demand (new tools appear as connections are established)
- Servers that enable/disable tools based on user permissions
- Plugin systems where tools are loaded dynamically

### 2. Resources: Data for Context

Resources are **read-only data** that provide context to the LLM. Unlike tools (which are model-controlled), resources are **application-controlled** — the host application decides when to fetch and include them.

Every resource has a **URI** that uniquely identifies it:

```
file:///project/src/main.rs          → a local file
postgres://localhost/mydb/users      → a database table
https://api.example.com/v1/config    → an API endpoint
git://repo/branch/path               → a git-tracked file
```

**Listing resources:**

```json
// Server exposes available resources
{
  "resources": [
    {
      "uri": "file:///project/src/main.rs",
      "name": "main.rs",
      "title": "Application Entry Point",
      "description": "The main Rust source file",
      "mimeType": "text/x-rust",
      "size": 2048
    },
    {
      "uri": "postgres://localhost/mydb/users",
      "name": "Users Table",
      "description": "Active user records",
      "mimeType": "application/json"
    }
  ]
}
```

**Reading a resource:**

```json
// Client → Server
{ "method": "resources/read", "params": { "uri": "file:///project/src/main.rs" } }

// Server → Client (text content)
{
  "contents": [{
    "uri": "file:///project/src/main.rs",
    "mimeType": "text/x-rust",
    "text": "fn main() {\n    println!(\"Hello, world!\");\n}"
  }]
}

// Server → Client (binary content, e.g., an image)
{
  "contents": [{
    "uri": "file:///project/logo.png",
    "mimeType": "image/png",
    "blob": "iVBORw0KGgoAAAANSUhEUg..."
  }]
}
```

**Resource templates** let servers expose parameterized resources using URI templates (RFC 6570):

```json
{
  "uriTemplate": "postgres://localhost/mydb/{table}",
  "name": "Database Tables",
  "description": "Read any table from the database"
}

// Client can request: postgres://localhost/mydb/users
// Client can request: postgres://localhost/mydb/orders
// Client can request: postgres://localhost/mydb/products
```

**Subscriptions** let clients get notified when resources change:

```json
// Client subscribes to file changes
{ "method": "resources/subscribe", "params": { "uri": "file:///project/src/main.rs" } }

// Later, server sends notification when file changes
{ "method": "notifications/resources/updated", "params": { "uri": "file:///project/src/main.rs" } }

// Client can then re-read the resource to get updated content
```

#### Resource Annotations: Priority and Audience

Resources can carry **annotations** that help the client decide how and whether to use them:

```json
{
  "uri": "file:///project/README.md",
  "name": "README",
  "annotations": {
    "audience": ["user", "assistant"],  // Show to both human and AI
    "priority": 0.9,                    // High priority — include early in context
    "lastModified": "2026-03-15T10:30:00Z"
  }
}

{
  "uri": "file:///project/.env.example",
  "name": "Environment Template",
  "annotations": {
    "audience": ["user"],    // Show only to human, NOT to AI (may contain secrets pattern)
    "priority": 0.3          // Low priority
  }
}
```

The `audience` field is particularly interesting: it lets the server indicate whether a resource is intended for the human user, the AI model, or both. A debug log might be `["user"]` only (the human can read it but there's no benefit in feeding it to the AI).

#### Resources vs. Tools: When to Use Which

This is a common design question. The key distinction:

```
Use RESOURCES when:
  - The data is read-only
  - The application should control when it's fetched
  - The data might be subscribed to for updates
  - It represents static or slowly-changing content
  Examples: file contents, database schemas, configuration, documentation

Use TOOLS when:
  - The operation has side effects (creates, updates, deletes)
  - The model should decide when to invoke it
  - The result depends on dynamic arguments
  - It represents an action, not data
  Examples: creating issues, running queries, sending messages, executing code

Gray area (could be either):
  - Searching a database → Tool (dynamic arguments, model-driven)
  - Getting a specific record by ID → Resource template (postgres://db/{table}/{id})
  - Fetching current weather → Tool (model decides when to check)
  - Reading a fixed config file → Resource (static data, app-driven)
```

### 3. Prompts: Reusable Templates

Prompts are **pre-built interaction templates** that servers expose, typically triggered by users via slash commands or UI menus. They're the simplest primitive but very useful for standardizing common workflows.

```json
// Server exposes prompt templates
{
  "prompts": [
    {
      "name": "code_review",
      "title": "Code Review",
      "description": "Review code for bugs, style, and improvements",
      "arguments": [
        { "name": "code", "description": "The code to review", "required": true },
        { "name": "language", "description": "Programming language", "required": false }
      ]
    },
    {
      "name": "explain_error",
      "title": "Explain Error",
      "description": "Explain an error message and suggest fixes",
      "arguments": [
        { "name": "error", "description": "The error message", "required": true }
      ]
    }
  ]
}
```

When a user triggers a prompt (e.g., types `/code_review` in the chat), the client calls `prompts/get`:

```json
// Client → Server
{
  "method": "prompts/get",
  "params": {
    "name": "code_review",
    "arguments": { "code": "def add(a, b): return a + b", "language": "python" }
  }
}

// Server → Client (returns structured messages to inject into conversation)
{
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "Please review the following Python code for bugs, style issues, and potential improvements:\n\n```python\ndef add(a, b): return a + b\n```\n\nProvide specific, actionable feedback."
      }
    }
  ]
}
```

The key insight: prompts let server developers craft **optimized instructions** for the LLM. Instead of users writing "review my code and look for bugs and style issues and...", the server provides a well-engineered prompt template.

#### Multi-Turn Prompts with Embedded Resources

Prompts can return **multiple messages** (creating a multi-turn conversation template) and can **embed resources** directly:

```json
// A "debug error" prompt that includes relevant log files
{
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "I'm seeing this error in production:"
      }
    },
    {
      "role": "user",
      "content": {
        "type": "resource",
        "resource": {
          "uri": "sentry://issues/latest",
          "mimeType": "text/plain",
          "text": "TypeError: Cannot read property 'email' of undefined\n  at UserService.getProfile (user.ts:42)\n  at Router.handleRequest (router.ts:118)"
        }
      }
    },
    {
      "role": "assistant",
      "content": {
        "type": "text",
        "text": "I can see the error. Let me analyze the stack trace and relevant source code to identify the root cause."
      }
    },
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "Here's the relevant source file:"
      }
    },
    {
      "role": "user",
      "content": {
        "type": "resource",
        "resource": {
          "uri": "file:///project/src/user.ts",
          "mimeType": "text/typescript",
          "text": "class UserService {\n  async getProfile(userId: string) {\n    const user = await this.db.findUser(userId);\n    return { email: user.email, name: user.name }; // line 42\n  }\n}"
        }
      }
    }
  ]
}
```

This creates a pre-structured conversation where the AI already has all the context it needs. The user triggers this with a single slash command, but the prompt template does the heavy lifting of gathering relevant data and structuring the conversation.

#### Argument Auto-Completion

MCP supports **auto-completion** for prompt arguments. When a user starts typing a prompt argument, the client can request completions from the server:

```json
// User is typing a repo name argument for a prompt
{ "method": "completion/complete", "params": {
    "ref": { "type": "ref/prompt", "name": "analyze_repo" },
    "argument": { "name": "repo", "value": "my" }
}}

// Server returns matching completions
{
  "completion": {
    "values": ["myorg/myapp", "myorg/my-library", "myorg/my-docs"],
    "hasMore": false
  }
}
```

This works for resource URI arguments too — enabling tab-completion-like experiences for selecting files, database tables, or any parameterized resource.

## Advanced Features

### Sampling: Letting Servers Use the LLM

This is one of MCP's most powerful features. **Sampling** allows a server to request LLM completions from the client — enabling agentic, multi-step workflows without the server needing its own API key.

```
Normal flow:  User → Client → Model → Client → Server (tool call) → Client → Model

With sampling: Server needs to "think" mid-operation:
              Server → Client → Model → Client → Server (continues work)
```

**Example**: A "smart database migration" server that generates and validates SQL:

```json
// Server calls sampling to get the LLM to generate SQL
// Server → Client
{
  "method": "sampling/createMessage",
  "params": {
    "messages": [
      {
        "role": "user",
        "content": {
          "type": "text",
          "text": "Generate a PostgreSQL migration to add an 'email_verified' boolean column to the users table, defaulting to false. Return only the SQL."
        }
      }
    ],
    "modelPreferences": {
      "hints": [{ "name": "claude-sonnet" }],
      "intelligencePriority": 0.8,
      "speedPriority": 0.5
    },
    "systemPrompt": "You are a PostgreSQL expert. Generate safe, production-ready migrations.",
    "maxTokens": 500
  }
}

// Client routes to LLM and returns result
// Client → Server
{
  "result": {
    "role": "assistant",
    "content": {
      "type": "text",
      "text": "ALTER TABLE users ADD COLUMN email_verified BOOLEAN NOT NULL DEFAULT FALSE;"
    },
    "model": "claude-sonnet-4-20250514",
    "stopReason": "endTurn"
  }
}

// Server can now validate this SQL, run it in a transaction, etc.
```

The **model preferences** system is provider-agnostic — instead of demanding a specific model, the server specifies priorities:

```json
{
  "modelPreferences": {
    "hints": [{ "name": "claude-sonnet" }, { "name": "gpt-4" }],
    "costPriority": 0.3,          // Low priority on cost
    "speedPriority": 0.5,         // Medium priority on speed
    "intelligencePriority": 0.9   // High priority on intelligence
  }
}
```

The client maps these preferences to whatever model it has available. A Claude Desktop client would use Claude; a GPT-based client would use GPT.

**Security**: Sampling includes a human-in-the-loop design. The client can show the server's sampling request to the user for approval before forwarding it to the model, and can show the model's response before returning it to the server.

### Elicitation: Asking Users for Input

Servers can request additional information from users through the client:

```json
// Server needs user's database password during setup
// Server → Client
{
  "method": "elicitation/request",
  "params": {
    "message": "Please enter your database connection password:",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "password": { "type": "string" }
      },
      "required": ["password"]
    }
  }
}

// Client shows a dialog to the user, returns their input
// Client → Server
{
  "result": {
    "action": "accept",
    "content": { "password": "my-secret-password" }
  }
}
```

### Logging: Structured Server Logs

MCP servers can send structured log messages to clients, visible in the host's debugging UI:

```json
// Server sends a log message
{
  "method": "notifications/message",
  "params": {
    "level": "warning",
    "logger": "github-server",
    "data": "Rate limit approaching: 4,850/5,000 requests used this hour"
  }
}
```

Log levels follow syslog conventions: `debug`, `info`, `notice`, `warning`, `error`, `critical`, `alert`, `emergency`. Clients can filter which levels they want to receive by setting the log level during initialization.

### Roots: Telling Servers What to Focus On

**Roots** are a client-side feature that tells the server which files or URIs the client is interested in. This helps the server scope its operations:

```json
// Client declares roots during initialization or via notification
{
  "method": "notifications/roots/list_changed"
}

// Server calls roots/list to discover what the client is working on
{
  "method": "roots/list"
}

// Client responds
{
  "result": {
    "roots": [
      {
        "uri": "file:///Users/me/projects/myapp",
        "name": "My Application"
      },
      {
        "uri": "file:///Users/me/projects/shared-lib",
        "name": "Shared Library"
      }
    ]
  }
}
```

A filesystem MCP server can use roots to know which directories it should index. A Git MCP server can use roots to know which repositories to operate on. Without roots, servers would have to guess or ask the user.

## Transport Layers: How Messages Travel

MCP supports two transport mechanisms, each suited for different deployment scenarios.

### Stdio Transport (Local Servers)

The client launches the server as a **subprocess** and communicates via standard I/O:

```
┌────────────┐     stdin      ┌────────────┐
│            │ ──────────────→ │            │
│   Client   │                 │   Server   │
│ (Host app) │ ←────────────── │ (subprocess)│
│            │     stdout      │            │
└────────────┘                 └────────────┘
                    stderr → (logging only)
```

**Rules:**
- Messages are **newline-delimited JSON** (each message is one line)
- Server MUST NOT write anything non-MCP to stdout (would corrupt the protocol)
- Server MAY write to stderr for human-readable logging
- The client manages the server's process lifecycle (start, stop, restart)
- Zero network overhead — pure inter-process communication

**When to use stdio:**
- Local tools: filesystem access, running commands, local databases
- Development and testing
- Single-user desktop applications
- When you need zero-configuration networking

**Configuration example** (Claude Desktop `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/projects"],
      "env": {}
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxx"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgres://user:pass@localhost:5432/mydb"
      }
    }
  }
}
```

**Claude Code configuration** (`~/.claude/claude_code_config.json`):

```json
{
  "mcpServers": {
    "sentry": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sentry"],
      "env": { "SENTRY_AUTH_TOKEN": "sntrys_xxxx" }
    }
  }
}
```

### Streamable HTTP Transport (Remote Servers)

For remote servers, MCP uses HTTP with optional Server-Sent Events (SSE) for streaming. This is the transport you'd use for cloud-deployed MCP servers that serve multiple clients.

```
┌────────────┐    HTTP POST     ┌────────────┐
│            │ ───────────────→ │            │
│   Client   │                  │   Server   │
│            │ ←─ JSON or SSE ─ │  (remote)  │
│            │                  │            │
│            │    HTTP GET      │            │
│            │ ───────────────→ │            │
│            │ ←──── SSE ────── │            │
└────────────┘                  └────────────┘
```

#### How It Works in Detail

**Client → Server (POST):**
Every JSON-RPC message from the client is sent as an HTTP POST to a single endpoint:

```http
POST /mcp HTTP/1.1
Host: tools.example.com
Content-Type: application/json
Accept: application/json, text/event-stream
Mcp-Session-Id: abc123-session-id
MCP-Protocol-Version: 2025-06-18

{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_weather","arguments":{"city":"Tokyo"}}}
```

The server can respond in two ways:

```http
# Option A: Simple JSON response (for quick results)
HTTP/1.1 200 OK
Content-Type: application/json

{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"Tokyo: 22°C"}]}}
```

```http
# Option B: SSE stream (for long-running operations with progress)
HTTP/1.1 200 OK
Content-Type: text/event-stream

event: message
data: {"jsonrpc":"2.0","method":"notifications/progress","params":{"progressToken":"t1","progress":50,"total":100}}

event: message
data: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"Analysis complete"}]}}
```

**Server → Client (GET for server-initiated messages):**
The client can open a long-lived SSE stream for receiving server-initiated requests (like sampling) and notifications:

```http
GET /mcp HTTP/1.1
Host: tools.example.com
Accept: text/event-stream
Mcp-Session-Id: abc123-session-id
```

The server streams events as they occur. This is how the server can proactively notify the client about resource changes or request sampling.

#### Session Management

```
1. Client sends initialize POST (no session ID yet)
2. Server responds with Mcp-Session-Id header
3. Client includes this header on ALL subsequent requests
4. Server uses it to maintain per-client state

If the server receives a request with an invalid/expired session ID,
it returns HTTP 404 → client must re-initialize.

Client can explicitly end a session:
  DELETE /mcp HTTP/1.1
  Mcp-Session-Id: abc123-session-id
```

#### Resumability

If the SSE stream disconnects (network issues), the client can resume:

```http
GET /mcp HTTP/1.1
Accept: text/event-stream
Mcp-Session-Id: abc123-session-id
Last-Event-ID: evt-42          ← "Resume from where I left off"
```

The server replays any events the client missed since `evt-42`. This requires the server to assign unique IDs to SSE events and buffer recent events.

#### When to Use Streamable HTTP

- **Multi-tenant SaaS tools**: One server instance serving many AI clients
- **Cloud-deployed services**: Running on AWS/GCP/Vercel
- **Authenticated external APIs**: OAuth-protected services
- **Shared team servers**: A company-wide MCP server for internal tools
- **High-availability**: Load-balanced, redundant server deployments

#### Security for Remote Servers

Remote servers face additional attack surfaces:

```
1. DNS Rebinding Prevention:
   Server MUST validate Origin header on every request
   Reject requests where Origin doesn't match expected domains

2. Authentication:
   Use OAuth 2.0 for user-scoped access
   Or API keys for service-to-service authentication
   NEVER rely solely on Mcp-Session-Id for auth (it's for state, not security)

3. Localhost Binding:
   Local development servers MUST bind to 127.0.0.1, NOT 0.0.0.0
   Prevents other devices on the network from connecting

4. TLS:
   Remote servers MUST use HTTPS in production
   Stdio transport doesn't need TLS (local IPC)

5. Session ID Security:
   Must be cryptographically random (e.g., UUIDv4)
   Must be unguessable and unique per session
```

### Choosing the Right Transport

| Factor | Stdio | Streamable HTTP |
|--------|-------|-----------------|
| **Deployment** | Local (same machine) | Remote (any network) |
| **Latency** | Minimal (IPC) | Network-dependent |
| **Concurrency** | 1 client per process | Many clients per server |
| **Auth** | Process-level (inherited) | OAuth, API keys, etc. |
| **State** | Per-process | Session-managed |
| **Networking** | None needed | Full HTTP stack |
| **Best for** | Dev tools, local files | Cloud services, shared tools |

## Real-World Use Cases

### Use Case 1: AI-Powered Development Workflow

**Scenario**: A developer uses Claude Code with multiple MCP servers to manage a full development workflow.

```
MCP Servers connected:
  1. Filesystem server     → Read/write project files
  2. GitHub server         → Manage issues, PRs, code review
  3. PostgreSQL server     → Query and modify database
  4. Sentry server         → View error reports
  5. Puppeteer server      → Take screenshots, test UI
```

**Conversation flow:**

```
Developer: "There's a Sentry error about null pointer in the checkout
            flow. Can you investigate and fix it?"

Claude's actions (via MCP tools):
  1. [Sentry tool] Get latest error details → Stack trace points to
     checkout.ts:142, null user.address
  2. [Filesystem tool] Read checkout.ts → Finds the bug: no null check
     before accessing user.address
  3. [PostgreSQL tool] Query users table → Confirms 3% of users have
     null addresses
  4. [Filesystem tool] Edit checkout.ts → Adds null check with fallback
  5. [Puppeteer tool] Navigate to checkout page → Takes screenshot
     to verify fix doesn't break UI
  6. [GitHub tool] Create branch, commit, open PR → PR #287 with
     description linking to Sentry issue
```

Without MCP, each of these integrations would require custom code. With MCP, Claude Code connects to all five servers through the same protocol.

### Use Case 2: Customer Support AI Agent

**Scenario**: An AI support agent handles customer inquiries using multiple backend systems.

```
MCP Servers:
  1. CRM server (Salesforce)  → Customer profiles, ticket history
  2. Order management server  → Order status, shipping, returns
  3. Knowledge base server    → Product docs, FAQ, policies
  4. Email server             → Send replies, escalation emails
```

**Example interaction:**

```
Customer: "I ordered a laptop last week but got the wrong model. Order #12345."

AI Agent's MCP tool calls:
  1. [Order server] get_order(id="12345")
     → Order: MacBook Air M3, shipped Dec 10, delivered Dec 12
     → Customer actually received: MacBook Pro M3 (warehouse error)

  2. [CRM server] get_customer(order="12345")
     → Customer: Jane Smith, Premium tier, 3-year customer

  3. [Knowledge base] search("wrong item received premium customer policy")
     → Policy: Premium customers get express replacement + return label

  4. [Order server] create_return(order="12345", reason="wrong_item")
     → Return label generated: RMA-789

  5. [Order server] create_replacement(order="12345", item="MacBook Air M3",
                                       shipping="express")
     → Replacement order #12346 created, arriving in 2 days

  6. [Email server] send_email(to=jane@example.com, template="wrong_item_resolution",
                               vars={return_label: "RMA-789", replacement: "#12346"})
     → Confirmation email sent

AI response to customer:
  "I'm sorry about the mix-up, Jane. I've set up a return for the
   incorrect MacBook Pro (return label RMA-789 sent to your email)
   and ordered your correct MacBook Air M3 with express shipping.
   It should arrive by Thursday. Is there anything else I can help with?"
```

The entire resolution happened through standardized MCP tool calls — the AI agent didn't need custom code for each backend system.

### Use Case 3: Data Analysis Pipeline

**Scenario**: A data scientist uses an AI assistant to explore and analyze data across multiple sources.

```
MCP Servers:
  1. PostgreSQL server    → Production database
  2. BigQuery server      → Analytics warehouse
  3. Filesystem server    → Read/write CSV, notebooks
  4. Matplotlib server    → Generate charts and visualizations
```

**Workflow:**

```
Data Scientist: "Compare our user signup rate this month vs last month,
                 broken down by acquisition channel. Make a chart."

AI's MCP actions:
  1. [BigQuery tool] Run query:
     SELECT channel, DATE_TRUNC(signup_date, MONTH) as month,
            COUNT(*) as signups
     FROM analytics.user_signups
     WHERE signup_date >= '2026-02-01'
     GROUP BY channel, month
     ORDER BY channel, month

  2. [BigQuery tool] Results:
     | channel  | month   | signups |
     |----------|---------|---------|
     | organic  | 2026-02 | 12,340  |
     | organic  | 2026-03 | 14,210  |
     | paid     | 2026-02 | 8,500   |
     | paid     | 2026-03 | 7,200   |
     | referral | 2026-02 | 3,100   |
     | referral | 2026-03 | 4,800   |

  3. [Matplotlib tool] Generate grouped bar chart with the data
     → Returns chart as image

  4. [Filesystem tool] Save chart to /reports/signups-march-2026.png

AI: "Here's the comparison. Key findings:
     - Organic signups up 15% (12.3K → 14.2K) — strongest growth
     - Paid signups down 15% (8.5K → 7.2K) — worth investigating ad spend
     - Referral signups up 55% (3.1K → 4.8K) — referral program working well
     Chart saved to /reports/signups-march-2026.png"
```

### Use Case 4: Smart Home / IoT Control

**Scenario**: An AI assistant controls smart home devices through MCP servers.

```
MCP Servers:
  1. Hue server          → Smart lights
  2. Thermostat server   → HVAC control
  3. Calendar server     → Schedule awareness
  4. Weather server      → External conditions
```

```
User: "I'm heading to bed. Set things up for sleep."

AI's MCP actions:
  1. [Calendar tool] Check tomorrow's schedule
     → First meeting at 9:00 AM

  2. [Weather tool] Get tomorrow morning forecast
     → 45°F at 7 AM, clear skies

  3. [Hue tool] Set bedroom lights to 5% warm white, turn off all other lights

  4. [Thermostat tool] Set sleep schedule:
     - Now: cool to 67°F for sleeping
     - 6:30 AM: warm to 72°F (30 min before suggested wake time)

AI: "All set for sleep. Lights dimmed, temperature cooling to 67°F.
     I'll warm the house to 72°F at 6:30 AM — you have your first
     meeting at 9:00 and it'll be 45°F outside. Goodnight!"
```

## Building an MCP Server: Step by Step

Let's build a real MCP server — a **bookmark manager** that stores and retrieves bookmarks.

### TypeScript Implementation

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// In-memory bookmark store (use a database in production)
const bookmarks: Map<string, { url: string; title: string; tags: string[] }> = new Map();

// Create MCP server
const server = new McpServer({
  name: "bookmark-manager",
  version: "1.0.0",
});

// ═══ TOOL: Add a bookmark ═══
server.tool(
  "add_bookmark",
  "Save a new bookmark with a title and optional tags",
  {
    url: z.string().url().describe("The URL to bookmark"),
    title: z.string().describe("A descriptive title"),
    tags: z.array(z.string()).optional().describe("Tags for categorization"),
  },
  async ({ url, title, tags }) => {
    const id = crypto.randomUUID();
    bookmarks.set(id, { url, title, tags: tags ?? [] });
    return {
      content: [
        {
          type: "text",
          text: `Bookmark saved!\nID: ${id}\nTitle: ${title}\nURL: ${url}\nTags: ${(tags ?? []).join(", ") || "none"}`,
        },
      ],
    };
  }
);

// ═══ TOOL: Search bookmarks ═══
server.tool(
  "search_bookmarks",
  "Search bookmarks by title, URL, or tag",
  {
    query: z.string().describe("Search term to match against title, URL, or tags"),
  },
  async ({ query }) => {
    const results = [...bookmarks.entries()]
      .filter(([_, b]) =>
        b.title.toLowerCase().includes(query.toLowerCase()) ||
        b.url.toLowerCase().includes(query.toLowerCase()) ||
        b.tags.some((t) => t.toLowerCase().includes(query.toLowerCase()))
      )
      .map(([id, b]) => `[${b.title}](${b.url}) — tags: ${b.tags.join(", ") || "none"} (id: ${id})`);

    return {
      content: [
        {
          type: "text",
          text: results.length > 0
            ? `Found ${results.length} bookmark(s):\n${results.join("\n")}`
            : "No bookmarks found matching your query.",
        },
      ],
    };
  }
);

// ═══ TOOL: Delete a bookmark ═══
server.tool(
  "delete_bookmark",
  "Delete a bookmark by its ID",
  {
    id: z.string().describe("The bookmark ID to delete"),
  },
  async ({ id }) => {
    if (!bookmarks.has(id)) {
      return {
        content: [{ type: "text", text: `Bookmark ${id} not found.` }],
        isError: true,
      };
    }
    const bookmark = bookmarks.get(id)!;
    bookmarks.delete(id);
    return {
      content: [{ type: "text", text: `Deleted bookmark: ${bookmark.title} (${bookmark.url})` }],
    };
  }
);

// ═══ RESOURCE: All bookmarks as JSON ═══
server.resource(
  "bookmarks://all",
  "bookmarks://all",
  async (uri) => ({
    contents: [
      {
        uri: uri.href,
        mimeType: "application/json",
        text: JSON.stringify([...bookmarks.entries()].map(([id, b]) => ({ id, ...b })), null, 2),
      },
    ],
  })
);

// ═══ PROMPT: Organize bookmarks ═══
server.prompt(
  "organize_bookmarks",
  "Suggest a better organization scheme for all bookmarks",
  async () => {
    const allBookmarks = [...bookmarks.values()]
      .map((b) => `- ${b.title} (${b.url}) [${b.tags.join(", ")}]`)
      .join("\n");

    return {
      messages: [
        {
          role: "user",
          content: {
            type: "text",
            text: `Here are my current bookmarks:\n\n${allBookmarks}\n\nPlease suggest a better organization scheme: recommend tag changes, identify duplicates, and group related bookmarks into categories.`,
          },
        },
      ],
    };
  }
);

// Start the server
const transport = new StdioServerTransport();
await server.connect(transport);
```

### Python Implementation

```python
from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP("bookmark-manager")

# In-memory store
bookmarks: dict[str, dict] = {}

@mcp.tool()
def add_bookmark(url: str, title: str, tags: list[str] | None = None) -> str:
    """Save a new bookmark with a title and optional tags."""
    import uuid
    bookmark_id = str(uuid.uuid4())
    bookmarks[bookmark_id] = {"url": url, "title": title, "tags": tags or []}
    return f"Bookmark saved! ID: {bookmark_id}, Title: {title}"

@mcp.tool()
def search_bookmarks(query: str) -> str:
    """Search bookmarks by title, URL, or tag."""
    query_lower = query.lower()
    results = []
    for bid, b in bookmarks.items():
        if (query_lower in b["title"].lower() or
            query_lower in b["url"].lower() or
            any(query_lower in t.lower() for t in b["tags"])):
            results.append(f"- [{b['title']}]({b['url']}) (tags: {', '.join(b['tags']) or 'none'})")

    if results:
        return f"Found {len(results)} bookmark(s):\n" + "\n".join(results)
    return "No bookmarks found matching your query."

@mcp.tool()
def delete_bookmark(bookmark_id: str) -> str:
    """Delete a bookmark by its ID."""
    if bookmark_id not in bookmarks:
        return f"Error: Bookmark {bookmark_id} not found."
    title = bookmarks[bookmark_id]["title"]
    del bookmarks[bookmark_id]
    return f"Deleted bookmark: {title}"

@mcp.resource("bookmarks://all")
def get_all_bookmarks() -> str:
    """All saved bookmarks as JSON."""
    return json.dumps(
        [{"id": bid, **b} for bid, b in bookmarks.items()],
        indent=2
    )

@mcp.prompt()
def organize_bookmarks() -> str:
    """Suggest a better organization scheme for all bookmarks."""
    lines = [f"- {b['title']} ({b['url']}) [{', '.join(b['tags'])}]"
             for b in bookmarks.values()]
    return (
        f"Here are my current bookmarks:\n\n"
        + "\n".join(lines)
        + "\n\nPlease suggest a better organization scheme."
    )

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### Connecting to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "bookmarks": {
      "command": "node",
      "args": ["path/to/bookmark-server/index.js"]
    }
  }
}
```

Or for Python:

```json
{
  "mcpServers": {
    "bookmarks": {
      "command": "python",
      "args": ["path/to/bookmark_server.py"]
    }
  }
}
```

Now in Claude Desktop, you can say: "Save this article as a bookmark about MCP with tags 'ai' and 'protocol'" — and Claude will call the `add_bookmark` tool through MCP.

## Security Considerations

MCP's design includes several security layers, but as a protocol that gives AI models access to external systems, security is critical.

### Principle of Least Privilege

```
BAD:  Give the AI full database admin access
      → One hallucinated DROP TABLE destroys everything

GOOD: Give the AI read-only access to specific tables
      → MCP server only exposes safe, scoped tools

// Server only exposes what's needed:
tools: [
  "query_users"        // read-only, parameterized queries only
  "get_order_status"   // specific lookup, not arbitrary SQL
]
// NOT: "execute_sql"  // too powerful, too dangerous
```

### User Consent and Approval

MCP hosts should **always require user approval** before executing tools that modify data or interact with external services. This is part of the protocol design:

```
User says: "Delete all my old emails"

Claude Desktop flow:
  1. Claude decides to call email server's delete_emails tool
  2. Claude Desktop shows user: "Claude wants to delete 1,247 emails
     older than 1 year. Allow?"
  3. User reviews and approves/denies
  4. Only then does the tool execute
```

### Input Validation

MCP servers must **never trust tool inputs blindly**:

```python
@mcp.tool()
def read_file(path: str) -> str:
    """Read a file from the allowed project directory."""
    # Validate: prevent path traversal attacks
    allowed_root = "/home/user/project"
    resolved = os.path.realpath(path)
    if not resolved.startswith(allowed_root):
        return "Error: Access denied — path outside allowed directory"

    with open(resolved) as f:
        return f.read()
```

### Key Security Principles

1. **Validate all inputs** — MCP servers receive inputs from AI models, which can be unpredictable
2. **Scope access narrowly** — Expose the minimum set of tools needed, with the minimum permissions
3. **Require confirmation for mutations** — Read operations are usually safe; write/delete operations should require human approval
4. **Rate limit tool calls** — Prevent runaway agents from making thousands of API calls
5. **Audit everything** — Log all tool invocations for debugging and compliance
6. **Treat tool descriptions as untrusted** — Malicious MCP servers could craft descriptions to manipulate the LLM (prompt injection via tool metadata)

## MCP vs. Alternatives

### MCP vs. Direct Function Calling (OpenAI Tools)

```
OpenAI Function Calling:
  - Tools defined in the API request JSON
  - Execution handled by YOUR application code
  - No standardized server interface
  - Tight coupling between app and tool implementation

MCP:
  - Tools defined by external MCP servers
  - Execution handled by the MCP server process
  - Standardized protocol — any client works with any server
  - Loose coupling — swap servers without changing app code
```

MCP doesn't replace function calling — it **standardizes the server side**. The AI model still uses function calling to decide which tools to invoke; MCP standardizes how those tools are discovered, described, and executed.

### MCP vs. LangChain Tools

```
LangChain Tools:
  - Python library with tool implementations
  - Tightly coupled to LangChain framework
  - Tools run in the same process as your app
  - Tool ecosystem locked to LangChain users

MCP:
  - Language-agnostic protocol (any language can implement)
  - Framework-independent (works with any AI app)
  - Tools run in separate processes (isolation, security)
  - Tool ecosystem shared across all MCP clients
```

### MCP vs. REST APIs

```
REST APIs:
  - Designed for application-to-application communication
  - No built-in concept of "what tools does this API offer?"
  - No standard for AI-friendly descriptions, schemas
  - No capability negotiation or subscriptions

MCP:
  - Designed specifically for AI-to-tool communication
  - Built-in discovery: "list all available tools with schemas"
  - AI-optimized descriptions and input schemas
  - Capability negotiation, change notifications, subscriptions
```

MCP servers often **wrap** existing REST APIs, adding the discovery and schema layer that AI models need.

## Testing and Debugging MCP Servers

### The MCP Inspector

The official **MCP Inspector** is a browser-based tool for testing MCP servers interactively:

```bash
# Launch inspector against your local server
npx @modelcontextprotocol/inspector node path/to/your/server.js

# Or for Python servers
npx @modelcontextprotocol/inspector python path/to/your/server.py
```

The inspector provides:
- A visual UI to see all available tools, resources, and prompts
- Forms to call tools with custom arguments and see results
- Real-time logging of all JSON-RPC messages
- Ability to test notifications and subscriptions

### Manual Testing with JSON-RPC

Since MCP uses stdio, you can test servers by piping JSON directly:

```bash
# Start the server and send an initialize request
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | node server.js

# Or use a tool like jq to format the output
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | node server.js | jq .
```

### Automated Testing

For production MCP servers, write automated tests that exercise the protocol:

```python
# Python example: testing an MCP server programmatically
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_bookmark_server():
    server_params = StdioServerParameters(
        command="python",
        args=["bookmark_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()

            # Test: list tools
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            assert "add_bookmark" in tool_names
            assert "search_bookmarks" in tool_names

            # Test: add a bookmark
            result = await session.call_tool("add_bookmark", {
                "url": "https://example.com",
                "title": "Test Bookmark",
                "tags": ["test"]
            })
            assert not result.isError
            assert "Bookmark saved" in result.content[0].text

            # Test: search for it
            result = await session.call_tool("search_bookmarks", {
                "query": "Test"
            })
            assert "Test Bookmark" in result.content[0].text

            # Test: list resources
            resources = await session.list_resources()
            assert any(r.uri == "bookmarks://all" for r in resources.resources)

            print("All tests passed!")

asyncio.run(test_bookmark_server())
```

### Common Debugging Patterns

```
Problem: Server not appearing in Claude Desktop
  → Check claude_desktop_config.json syntax (trailing commas break JSON)
  → Check that the command path is absolute or in PATH
  → Check server stderr output:
    tail -f ~/Library/Logs/Claude/mcp-server-*.log  (macOS)

Problem: Tool calls failing silently
  → Enable debug logging in your server
  → Check that tool inputSchema matches what the model sends
  → Verify JSON-RPC message format (common mistake: missing "jsonrpc":"2.0")

Problem: Server crashes on startup
  → Test the server command manually in a terminal first
  → Check for missing environment variables (env block in config)
  → Check Node.js/Python version compatibility

Problem: Tools not discovered by the model
  → Tool descriptions are too vague — improve them
  → Too many tools (model gets overwhelmed) — reduce to essentials
  → Tool name doesn't match its purpose — rename for clarity
```

## Production Best Practices

### 1. Design Tools for AI Consumption

The most important factor for MCP server quality is **how well the tools are designed for AI understanding**. The model decides which tools to use based solely on their names, descriptions, and input schemas.

```json
// BAD: Vague, unclear what it does
{
  "name": "process",
  "description": "Process the data",
  "inputSchema": {
    "type": "object",
    "properties": {
      "input": { "type": "string" }
    }
  }
}

// GOOD: Clear name, detailed description, typed parameters with descriptions
{
  "name": "analyze_sentiment",
  "description": "Analyze the sentiment of customer feedback text. Returns a sentiment score from -1.0 (very negative) to 1.0 (very positive), along with key phrases that influenced the score. Use this when the user asks about customer satisfaction or feedback tone.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "text": {
        "type": "string",
        "description": "The customer feedback text to analyze (max 5000 characters)"
      },
      "language": {
        "type": "string",
        "enum": ["en", "es", "fr", "de", "ja"],
        "description": "Language of the text. Defaults to 'en' if not specified."
      }
    },
    "required": ["text"]
  }
}
```

Key principles:
- **Names**: Use verb_noun format (`create_issue`, `search_users`, not `issue` or `users`)
- **Descriptions**: Include when to use the tool, what it returns, and edge cases
- **Parameters**: Every parameter should have a description explaining valid values
- **Enums**: Use `enum` for parameters with fixed valid values (not free text)

### 2. Keep Tool Count Manageable

Research shows that LLM tool-calling accuracy degrades as the number of available tools increases. Guidelines:

```
1-10 tools:   Excellent accuracy. Model rarely picks wrong tool.
10-30 tools:  Good accuracy. Descriptions must be distinct and clear.
30-50 tools:  Moderate accuracy. Consider grouping or splitting into
              multiple servers.
50+ tools:    Poor accuracy. Tools must be extremely well-described,
              or use dynamic tool registration based on context.
```

If you have many tools, consider:
- **Multiple specialized servers** instead of one monolithic server
- **Dynamic tools**: Start with a small set and add tools contextually via `notifications/tools/list_changed`
- **Prompt engineering**: Include guidance in the system prompt about when to use which tools

### 3. Handle Errors Gracefully

Return errors as **tool results with `isError: true`**, not as protocol-level errors. This gives the model a chance to reason about the failure and try alternatives:

```python
@mcp.tool()
def query_database(sql: str) -> str:
    """Execute a read-only SQL query against the analytics database."""
    try:
        # Validate: only allow SELECT queries
        if not sql.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed. " \
                   "Please reformulate your query as a SELECT statement."

        result = db.execute(sql)
        return format_results(result)

    except db.TimeoutError:
        return "Error: Query timed out after 30 seconds. " \
               "Try adding a LIMIT clause or simplifying the query."

    except db.SyntaxError as e:
        return f"Error: SQL syntax error: {e}. " \
               "Please check the query syntax and try again."

    except Exception as e:
        # Log the full error server-side for debugging
        logger.error(f"Unexpected error in query_database: {e}", exc_info=True)
        # Return a helpful message to the model
        return "Error: An unexpected error occurred. " \
               "Please try a simpler query or contact support."
```

The error messages are designed for AI consumption: they explain what went wrong AND suggest what to do differently.

### 4. Implement Timeouts and Rate Limits

MCP servers that call external APIs should protect themselves:

```python
import asyncio
from functools import wraps

# Timeout decorator for tools
def with_timeout(seconds=30):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                return f"Error: Operation timed out after {seconds} seconds"
        return wrapper
    return decorator

# Rate limiting
from collections import defaultdict
import time

call_counts: dict[str, list[float]] = defaultdict(list)

def rate_limit(max_calls=60, window_seconds=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            key = func.__name__
            # Remove expired timestamps
            call_counts[key] = [t for t in call_counts[key] if now - t < window_seconds]
            if len(call_counts[key]) >= max_calls:
                return f"Error: Rate limit exceeded ({max_calls} calls per {window_seconds}s). Please wait."
            call_counts[key].append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@mcp.tool()
@rate_limit(max_calls=30, window_seconds=60)
def search_github(query: str) -> str:
    """Search GitHub repositories. Rate limited to 30 calls/minute."""
    ...
```

### 5. Version Your Server

Use semantic versioning and communicate breaking changes through the protocol:

```json
{
  "serverInfo": {
    "name": "my-analytics-server",
    "version": "2.1.0"          // Visible to clients during initialization
  }
}
```

When tools change (renamed, removed, or schema-modified), send `notifications/tools/list_changed` so clients re-discover the new tool set. Don't silently change tool behavior — models may have cached assumptions from previous interactions.

### 6. Log Everything (For Debugging and Audit)

```python
import logging
import json
from datetime import datetime

logger = logging.getLogger("mcp-server")

@mcp.tool()
def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
    """Transfer funds between accounts."""
    # Log for audit trail
    logger.info(json.dumps({
        "event": "tool_call",
        "tool": "transfer_funds",
        "params": {"from": from_account, "to": to_account, "amount": amount},
        "timestamp": datetime.utcnow().isoformat(),
        # Don't log who called it — the MCP server doesn't know the end user
        # The HOST is responsible for user-level audit logging
    }))

    result = banking_api.transfer(from_account, to_account, amount)
    logger.info(json.dumps({
        "event": "tool_result",
        "tool": "transfer_funds",
        "success": result.success,
        "transaction_id": result.id,
    }))
    return f"Transfer complete. Transaction ID: {result.id}"
```

## The MCP Ecosystem

### Popular MCP Servers

The MCP ecosystem has grown rapidly. Here are some of the most useful servers:

**Development:**
| Server | What It Does |
|--------|-------------|
| `@modelcontextprotocol/server-filesystem` | Read/write/search files |
| `@modelcontextprotocol/server-github` | Issues, PRs, code search, repos |
| `@modelcontextprotocol/server-postgres` | Query PostgreSQL databases |
| `@modelcontextprotocol/server-sqlite` | Query SQLite databases |
| `@modelcontextprotocol/server-puppeteer` | Browser automation, screenshots |
| `@modelcontextprotocol/server-git` | Git operations (log, diff, blame) |

**Productivity:**
| Server | What It Does |
|--------|-------------|
| `@modelcontextprotocol/server-slack` | Read/send messages, search channels |
| `@modelcontextprotocol/server-google-drive` | Read/search Google Drive files |
| `@modelcontextprotocol/server-notion` | Query Notion databases and pages |
| `@anthropic/server-linear` | Issue tracking with Linear |

**Data & Analytics:**
| Server | What It Does |
|--------|-------------|
| `@modelcontextprotocol/server-bigquery` | Query Google BigQuery |
| `@modelcontextprotocol/server-snowflake` | Query Snowflake warehouses |
| `@modelcontextprotocol/server-redis` | Read/write Redis data |

**Monitoring:**
| Server | What It Does |
|--------|-------------|
| `@modelcontextprotocol/server-sentry` | Error tracking and debugging |
| `@modelcontextprotocol/server-datadog` | Metrics and monitoring |

### Building Your Own vs. Using Existing

Decision framework:

```
Use an existing MCP server when:
  ✓ An official or well-maintained server exists for your service
  ✓ The existing server covers your use case
  ✓ You don't need custom business logic

Build a custom MCP server when:
  ✓ Integrating with an internal/proprietary system
  ✓ You need custom business logic (e.g., combining multiple APIs)
  ✓ You need specific access controls not available in generic servers
  ✓ The existing server doesn't cover your use case

Wrap an existing REST API as MCP when:
  ✓ The API is well-documented with OpenAPI/Swagger spec
  ✓ You want AI apps to use it without custom integration
  ✓ You can auto-generate tool definitions from the API spec
```

### The Future: MCP Server Registries

As the ecosystem grows, discovery becomes important. Several community initiatives are creating **MCP server registries** — searchable directories of MCP servers similar to npm for Node.js packages. This would allow AI applications to dynamically discover and connect to servers based on the user's needs.

## Design Insights

### Why Three Primitives?

The separation of Tools, Resources, and Prompts isn't arbitrary — it reflects three fundamentally different **control flows**:

```
TOOLS:     Model decides → "I need to call create_issue"
           The AI autonomously chooses when to use tools

RESOURCES: App decides  → "Include this file as context"
           The application controls what data the AI sees

PROMPTS:   User decides → "/code_review" (slash command)
           The user explicitly triggers a workflow
```

Collapsing these into a single concept (just "tools" or just "APIs") would lose this distinction. Each has different security implications, different UX patterns, and different optimization strategies.

### Why JSON-RPC?

MCP chose JSON-RPC 2.0 over alternatives like REST, gRPC, or GraphQL because:

1. **Bidirectional**: Both client and server can initiate requests (critical for sampling and notifications)
2. **Simple**: Just JSON objects with `method`, `params`, `result` — easy to implement in any language
3. **Transport-agnostic**: Works over stdio, HTTP, WebSocket — same message format regardless
4. **Stateful**: Supports sessions, capability negotiation, and subscriptions (REST is stateless by design)

### The Capability Negotiation Pattern

MCP's initialization handshake is designed for **forward compatibility**. When the protocol adds new features:

1. Old clients connecting to new servers → new capabilities are simply not negotiated
2. New clients connecting to old servers → new capabilities are simply not offered
3. No version conflicts, no breaking changes

This is the same pattern used by TLS, HTTP/2, and other successful protocols. It's why MCP can evolve without fragmenting the ecosystem.

### The "Protocol, Not Framework" Philosophy

MCP deliberately avoids being a framework. It doesn't dictate:
- What language you write servers in
- How you implement tool logic
- What database or storage you use
- How you deploy or scale

It only specifies the **wire protocol** — the JSON-RPC messages that flow between client and server. This is why MCP servers exist in Python, TypeScript, Go, Rust, Java, C#, and more. And it's why MCP works equally well for a simple bash wrapper and a complex cloud-deployed microservice.

This is the same approach that made HTTP successful: specify the protocol, let implementations vary.

### The Bidirectionality Insight

Most integration protocols are **unidirectional**: clients send requests, servers send responses. MCP is fundamentally **bidirectional**: servers can also send requests to clients (sampling, elicitation).

This bidirectionality is what enables the advanced agentic patterns. A server that can request LLM completions mid-operation can implement complex multi-step reasoning without the client needing to orchestrate each step. The server becomes a semi-autonomous agent within the broader system.

```
Traditional: Client → Server → Response (one direction)
MCP:         Client ⇄ Server (both directions)

This enables:
  Server: "I found 3 possible fixes. Let me ask the LLM which one is best."
          → sampling/createMessage to Client
          → Client routes to LLM
          → LLM picks option 2
          → Server applies option 2
          → Server returns final result to Client
```

### Why Resources and Tools Are Separate (Not Just "Read" Tools)

A natural question: why not just make resources into read-only tools? The answer lies in **control and context management**:

```
If resources were tools:
  → The MODEL would decide when to read files
  → You'd have no control over what context the AI sees
  → Every file read would be a tool call (expensive, slow)
  → No way to proactively push context updates

Because resources are separate:
  → The APPLICATION decides what context to include
  → You can pre-load relevant files before the model runs
  → Resources can be subscribed to and update in real-time
  → The application can implement smart context selection
    (e.g., only include files relevant to the current task)
```

This matters enormously for production systems. The application might know that a specific config file is always relevant, or that the user's current git branch context should always be included. Making this application-controlled rather than model-controlled gives builders much more flexibility.

## Summary

| Concept | What It Is | Who Controls It |
|---------|-----------|----------------|
| **MCP** | Open standard protocol for AI ↔ external systems | Anthropic (open governance) |
| **Host** | AI application (Claude, VS Code, Cursor) | End user |
| **Client** | Protocol handler inside the host | Host application |
| **Server** | Program exposing tools/resources/prompts | Server developer |
| **Tools** | Executable functions (actions) | Model decides when to use |
| **Resources** | Read-only data (context) | Application decides when to include |
| **Prompts** | Reusable templates (workflows) | User triggers explicitly |
| **Sampling** | Server requests LLM completions | Server initiates, user approves |
| **Elicitation** | Server requests user input | Server initiates, user responds |
| **Roots** | Client tells server what to focus on | Client declares |
| **Annotations** | Metadata hints for tools/resources | Server declares |
| **Stdio** | Local transport (subprocess) | For local servers |
| **Streamable HTTP** | Remote transport (HTTP + SSE) | For remote/cloud servers |

**Key takeaways for practitioners:**

1. **Start simple**: Build a stdio server with 3-5 well-described tools. You can add complexity later.
2. **Tool descriptions are everything**: The model chooses tools based on descriptions alone. Invest time here.
3. **Use the SDK**: Don't hand-roll JSON-RPC. The official TypeScript and Python SDKs handle protocol details.
4. **Test with the Inspector**: Use `npx @modelcontextprotocol/inspector` before connecting to a real AI host.
5. **Security by default**: Validate all inputs, scope access narrowly, require confirmation for mutations.
6. **Resources vs. Tools**: If it's read-only data, make it a resource. If it has side effects, make it a tool.
7. **Keep tool count low**: 10-20 well-designed tools beats 100 mediocre ones.

MCP's core value proposition is turning the **N x M integration problem** into an **N + M ecosystem problem**. Instead of every AI app building custom integrations for every service, build one MCP server per service and one MCP client per AI app. The protocol handles the rest.

As AI applications move from simple chatbots to autonomous agents that interact with dozens of external systems, having a standardized protocol for these interactions becomes not just convenient but essential. MCP is that protocol.

## References

1. [Model Context Protocol — Official Documentation](https://modelcontextprotocol.io/)
2. [MCP Specification — GitHub](https://github.com/modelcontextprotocol/specification)
3. [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
4. [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
5. [MCP Servers Repository](https://github.com/modelcontextprotocol/servers) — Official and community MCP servers
6. [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
7. [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
