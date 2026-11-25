---
title: "dmux Open-Sourced For Agent Swarms"
publishDate: "2026-02-19"
date: "2026-02-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "tools", "harness-engineering"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2024507517359788224-1.jpg"
excerpt: "jpschroeder open-sources dmux, a tool for running Codex and Claude Code swarms with tmux and worktrees."
---

> Source: [@jpschroeder on X](https://x.com/jpschroeder/status/2024507517359788224) · posted 2026-02-19

![dmux Open-Sourced For Agent Swarms](/imgs/blogs/x-bookmarks/2024507517359788224-1.jpg)

## Summary

The author's team is open sourcing dmux, an internal tool. It uses tmux and git worktrees with claude, codex and opencode, has hooks for worktree automation, A/B comparisons between claude and codex, worktree management and multi-project sessions.

## Background

A git worktree is a separate checkout of the same repo, so several agents can edit in parallel without clashing. tmux keeps each session running.

## What I took away

- Worktrees are the standard way to run parallel agents.
- A/B testing agents on the same task is possible with this setup.

## Reference

- Original post: [https://x.com/jpschroeder/status/2024507517359788224](https://x.com/jpschroeder/status/2024507517359788224)
- Author: [@jpschroeder](https://x.com/jpschroeder) on X, posted 2026-02-19
