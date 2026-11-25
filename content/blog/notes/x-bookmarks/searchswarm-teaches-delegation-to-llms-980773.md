---
title: "SearchSwarm Teaches Delegation to LLMs"
publishDate: "2026-06-11"
date: "2026-06-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "reinforcement-learning", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2064873386308980773-1.jpg"
excerpt: "A post on SearchSwarm, a model that splits long tasks among subagents and combines results."
---

> Source: [@HuggingPapers on X](https://x.com/HuggingPapers/status/2064873386308980773) · posted 2026-06-11

![SearchSwarm Teaches Delegation to LLMs](/imgs/blogs/x-bookmarks/2064873386308980773-1.jpg)

## Summary

@HuggingPapers describes SearchSwarm: one model decomposes long tasks, sends bounded subtasks to subagents and integrates their condensed results. It claims SOTA at 30B scale on BrowseComp, GAIA and xbench-DeepSearch.

## Background

Subagents work in separate contexts and return short summaries. Training the model to delegate is the point.

## What I took away

- Delegation is a skill to train.
- Condensed results save context.

## Reference

- Original post: [https://x.com/HuggingPapers/status/2064873386308980773](https://x.com/HuggingPapers/status/2064873386308980773)
- Author: [@HuggingPapers](https://x.com/HuggingPapers) on X, posted 2026-06-11
