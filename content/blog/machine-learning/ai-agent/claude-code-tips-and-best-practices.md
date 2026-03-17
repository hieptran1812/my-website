---
title: "Claude Code Tips and Best Practices from Its Creator"
publishDate: "2026-03-18"
category: "machine-learning"
subcategory: "AI Agent"
tags: ["ai-agent", "claude", "claude-code", "anthropic", "developer-tools"]
date: "2026-03-18"
author: "Hiep Tran"
featured: false
aiGenerated: true
image: "/imgs/blogs/claude-agent-skills.png"
excerpt: "A comprehensive collection of tips and best practices for using Claude Code effectively, sourced from Boris Cherny — the creator of Claude Code — and the Claude Code team at Anthropic."
---

## Introduction

Boris Cherny, the creator of [Claude Code](https://docs.anthropic.com/en/docs/claude-code), has shared extensive insights on how he and his team use the tool. His setup is "surprisingly vanilla" — Claude Code works great out of the box without heavy customization. There is no single correct way to use it: the tool is intentionally built so you can use it, customize it, and hack it however you like.

This article compiles the most valuable tips from Boris's threads and the Claude Code team's collective experience.

## 1. Run Multiple Sessions in Parallel

The single biggest productivity unlock is **parallel execution**. Boris runs 5 Claude instances simultaneously in his terminal, numbering his tabs 1-5 and using system notifications to know when a Claude needs input.

On top of that, he runs 5-10 sessions on `claude.ai/code` alongside local terminal sessions, using:

- The `&` operator to hand off between platforms
- `--teleport` to move sessions around
- The iOS app for launching sessions throughout the day

**Pro tip**: Prefer **git worktrees** over separate checkouts for parallel sessions. Create shell aliases (`za`, `zb`, `zc`) to hop between worktrees instantly.

```bash
# Launch Claude in an isolated worktree
claude --worktree my_worktree

# Optional: launch in a separate Tmux session
claude --worktree my_worktree --tmux
```

Use `/color` to change the prompt input color per session — this gives you instant visual identification when juggling 3-5 parallel sessions.

## 2. Use Plan Mode Before Executing

Most sessions should start in **Plan Mode** (press `Shift+Tab` twice). If the goal is to write a Pull Request:

1. Enter Plan mode
2. Go back and forth with Claude until the plan looks right
3. Switch into **auto-accept edits** mode
4. Claude can usually one-shot the implementation

This prevents downstream issues and enables much higher quality single-shot results.

## 3. CLAUDE.md as Institutional Memory

The Claude Code team shares a single `CLAUDE.md` file checked into their repository. The whole team contributes to it multiple times a week.

> "Anytime we see Claude do something incorrectly we add it to the CLAUDE.md, so Claude knows not to do it next time."

This is the practice of **Compounding Engineering** — building institutional knowledge over time. Tips for maintaining it:

- End every correction with: *"Update your CLAUDE.md so you don't make that mistake again"*
- Ruthlessly edit CLAUDE.md over time to keep it focused
- Maintain a `notes/` directory for project-specific context, linked from CLAUDE.md
- Tag `@claude` in PR code reviews to update CLAUDE.md via the GitHub Action

## 4. Give Claude a Way to Verify Its Work

> "Probably the most important thing to get great results out of Claude Code — give Claude a way to verify its work. If Claude has that feedback loop, it will 2-3x the quality of the final result."

Verification methods vary by domain:

- **Test suites** — run existing tests after changes
- **Browser testing** — Claude uses the Chrome extension to open browsers, test functionality and UX, and iterate until satisfied
- **Simulators** — for hardware or embedded work
- **Bash commands** — for quick sanity checks
- **Background agents** — for very long-running tasks, prompt Claude to verify its work with a background agent when done

Invest significant effort in setting up verification loops — the payoff is enormous.

## 5. Create Slash Commands for Repeated Workflows

Store reusable commands in `.claude/commands/` and check them into git so the whole team benefits.

Example: A `/commit-push-pr` command that uses inline bash to precompute information, reducing back-and-forth interactions.

Other useful custom commands from the team:

- `/techdebt` — find duplicated code after each session
- `/boris` — a skill containing all of Boris's tips (available as a community skill)
- Analytics-engineer agents that write dbt models and test in dev

## 6. Use Subagents for Parallel Work

Deploy specialized subagents for recurring PR workflows:

- **code-simplifier** — reviews code for reuse, quality, efficiency
- **verify-app** — end-to-end verification
- **build-validator** — CI checks

Append "use subagents" to your requests when you want Claude to parallelize compute. This keeps the main agent's context clean by offloading tasks.

Custom agents live in `.claude/agents/` with custom names, colors, and tool sets. Subagents can run in isolated worktrees with `isolation: worktree` in the agent frontmatter.

## 7. Set Up Hooks for Automation

### PostToolUse Hook

Implement formatting hooks that handle the final 10% of code formatting automatically. This prevents CI failures while leveraging Claude's generally strong output quality.

### Agent Stop Hook

For very long-running tasks, use an agent Stop hook for deterministic verification when Claude finishes — more reliable than relying on prompts alone.

### PostCompact Hook

Fires after context compression. Use it to re-inject critical instructions that might be lost during compaction, or to log compaction events.

## 8. Manage Permissions Wisely

Use `/permissions` to pre-allow safe bash commands rather than `--dangerously-skip-permissions`. This reduces unnecessary prompts while maintaining security.

Supports wildcard syntax:

```json
{
  "Bash(bun run *)": "allow",
  "Edit(/docs/**)": "allow"
}
```

Check permission settings into your team's `settings.json` for consistency. For sandboxed environments, use `--permission-mode=dontAsk` or `--dangerously-skip-permissions` to allow uninterrupted progress.

Run `/sandbox` to opt into the open-source sandbox runtime with file and network isolation.

## 9. Integrate External Tools via MCP

Claude can leverage MCP servers and CLI tools for:

- **Slack integration** — paste bug threads directly and just say "fix"
- **BigQuery** — run queries via `bq` CLI (Boris: *"I haven't written SQL in 6+ months"*)
- **Sentry** — retrieve error logs
- **Any database** with CLI, MCP, or API access

All configurations are checked into `.mcp.json` for team sharing.

## 10. Model and Effort Selection

Boris uses **Opus with thinking mode** exclusively:

> "Less steering + better tool use = faster overall"

Despite being larger and slower than Sonnet, Opus requires less correction and produces higher quality results.

Use `/model` to switch between effort levels:

- **Low** — fast, simple tasks
- **Medium** — balanced
- **High** — intelligent (Boris's default)
- **Max** (`/effort max`) — Claude reasons longer using as many tokens as needed

## 11. Prompting Strategies

Effective prompting patterns from the team:

- **Challenge Claude**: *"Grill me on these changes and don't make a PR until I pass"*
- **Request proofs**: *"Diff behavior between main and your feature branch"*
- **Push for quality**: *"Scrap this and implement the elegant solution"*
- **Bug fixing**: *"Go fix the failing CI tests"* — without micromanaging
- **Use voice dictation** (press `fn` twice on macOS) for 3x faster, more detailed prompts

## 12. New Built-in Skills: /simplify and /batch

Two powerful skills announced by Boris for shipping to production faster:

### /simplify

Append `/simplify` to any prompt. It runs parallel agents reviewing your code for reuse opportunities, quality issues, and efficiency improvements.

```
Make this code change then run /simplify
```

### /batch

Interactively plan migrations, then execute them in parallel. Fans work out to dozens of agents in isolated worktrees, where each agent tests end-to-end and creates an independent PR.

```
Migrate src/ from Solid to React
```

Combined, these automate much of the work it takes to (1) shepherd a pull request to production and (2) perform straightforward, parallelizable code migrations.

## 13. Additional Productivity Tips

- **Session naming**: `claude --name "auth-refactor"` for human-readable identification when juggling multiple worktrees
- **Status line**: Customize via `/statusline` to show model, directory, context %, and cost
- **Keybindings**: Every key binding is customizable via `/keybindings`
- **Output styles**: Use "Explanatory" or "Learning" modes when onboarding to a new codebase
- **/loop**: Schedule prompts to run on interval for up to 3 days — great for PR babysitting, deploy monitoring, or Slack summaries
- **/btw**: Ask single-turn questions without interrupting Claude's current work
- **Remote control**: Run `claude remote-control` to spawn and manage sessions from your mobile app

## Key Takeaways

1. **Parallel execution** is the single biggest productivity unlock
2. **Verification feedback loops** 2-3x quality — invest in domain-specific testing
3. **CLAUDE.md as institutional memory** compounds engineering knowledge over time
4. **Plan mode** prevents downstream issues and enables one-shot implementations
5. **Customization checked into git** extends benefits across the entire team
6. There is **no one right way** — experiment and find what works for you

## References

- [Boris Cherny's original thread on X](https://x.com/bcherny/status/2007179832300581177)
- [Boris Cherny's /simplify and /batch announcement](https://x.com/bcherny/status/2027534984534544489)
- [How Boris Uses Claude Code](https://howborisusesclaudecode.com)
- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
