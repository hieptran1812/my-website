---
title: "TRL Adds Fine-Tuning On Agent Traces"
publishDate: "2026-06-04"
date: "2026-06-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "ai-agents", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2062553572433080797-1.jpg"
excerpt: "TRL can now fine-tune models on agent traces from Claude Code, Codex, OpenClaw, Pi and others."
---

> Source: [@adithya_s_k on X](https://x.com/adithya_s_k/status/2062553572433080797) · posted 2026-06-04

![TRL Adds Fine-Tuning On Agent Traces](/imgs/blogs/x-bookmarks/2062553572433080797-1.jpg)

## Summary

The author says you can now fine-tune models on agent traces directly with TRL, including Claude Code, Codex, OpenClaw and Pi traces. They feel this is a step toward a standard stack for fine-tuning agentic models. The quoted post announces official trl support, with datasets v5.

## Background

An agent trace is the recorded steps an agent took: messages, tool calls and results. Supervised fine-tuning (SFT) on such traces teaches a model to behave like the agent that produced them. TRL is Hugging Face's training library.

## What I took away

- Agent traces are becoming training data.
- One command can train on local, synthetic or community traces.

## Reference

- Original post: [https://x.com/adithya_s_k/status/2062553572433080797](https://x.com/adithya_s_k/status/2062553572433080797)
- Author: [@adithya_s_k](https://x.com/adithya_s_k) on X, posted 2026-06-04
