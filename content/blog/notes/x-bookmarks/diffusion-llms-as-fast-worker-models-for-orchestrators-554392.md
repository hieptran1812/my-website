---
title: "Diffusion LLMs as Fast Worker Models for Orchestrators"
publishDate: "2026-10-19"
date: "2026-10-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "ai-agents", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "Tri Dao suggests pairing a smart orchestrator model with a fast diffusion LLM worker, reacting to the Mercury 3 release."
---

> Source: [@tri_dao on X](https://x.com/tri_dao/status/2112263206085554392) · posted 2026-10-19

## Summary

@tri_dao says he has often wanted to pair a smart orchestrator or architect model with a very fast worker model, and a diffusion LLM could fill the worker role. The quoted post announces Mercury 3, claimed to be the most intelligent diffusion LLM yet, outperforming GPT-6 Luna and Haiku 5.5 on quality while running at over 1000 tokens per second.

## Background

Diffusion LLMs generate text by refining many tokens in parallel over a few steps instead of one token at a time, which can be much faster. An orchestrator-worker setup uses a strong model to plan and cheaper fast models to execute subtasks.

## What I took away

- Split agent workloads into planning and execution to use different models.
- Fast diffusion models fit execution tasks where speed matters.
- Quality comparisons are the vendor's claims.

## Reference

- Original post: [https://x.com/tri_dao/status/2112263206085554392](https://x.com/tri_dao/status/2112263206085554392)
- Author: [@tri_dao](https://x.com/tri_dao) on X, posted 2026-10-19
