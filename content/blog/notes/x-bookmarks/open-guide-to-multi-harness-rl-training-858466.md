---
title: "Open Guide to Multi-Harness RL Training"
publishDate: "2026-10-01"
date: "2026-10-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "harness-engineering", "ai-agents"]
author: "Hiep Tran"
featured: false
excerpt: "A guide to training models with RL inside real agent harnesses like Claude Code, Codex and OpenCode."
---

> Source: [@maximelabonne on X](https://x.com/maximelabonne/status/2105796335584858466) · posted 2026-10-01

## Summary

Maxime Labonne praises the collaboration and the open recipe, suggesting training LFM2.5-2.6B on all harnesses. The quoted post releases a guide to multi-harness RL: the same model behaves differently in each agent harness, so they built an open way to train any model with RL on any task set inside harnesses like Claude Code, Codex and OpenCode.

## Background

A harness is the tool and prompt wrapper an agent runs in. A model trained in one harness may struggle in another because tools, formats and prompts differ.

## What I took away

- Train and evaluate agents in the harnesses they will actually run in.
- Harness mismatch can explain surprising drops in agent quality.

## Reference

- Original post: [https://x.com/maximelabonne/status/2105796335584858466](https://x.com/maximelabonne/status/2105796335584858466)
- Author: [@maximelabonne](https://x.com/maximelabonne) on X, posted 2026-10-01
