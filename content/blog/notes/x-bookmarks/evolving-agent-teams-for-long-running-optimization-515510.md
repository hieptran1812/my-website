---
title: "Evolving Agent Teams for Long-Running Optimization"
publishDate: "2026-11-02"
date: "2026-11-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "ai-research", "harness-engineering"]
author: "Hiep Tran"
featured: false
excerpt: "EAT uses a supervisor that diagnoses why an agent team stalls and restructures it to escape local optima."
---

> Source: [@shiyi_c98 on X](https://x.com/shiyi_c98/status/2117240310736515510) · posted 2026-11-02

## Summary

@shiyi_c98 introduces Evolving Agent Teams (EAT), a NeurIPS'26 work. A supervisor diagnoses why an agent team stalls and restructures the team to escape local optima on long-running optimization tasks. The authors report strong results across domains, including the Open Kernel Challenge.

## Background

A local optimum is a point where small changes stop helping even though better solutions exist. Agent teams working on long tasks can get stuck repeating similar ideas. A supervisor that changes the team's structure is like a manager reorganizing a stuck project.

## What I took away

- When agents plateau, changing the team structure may help more than more steps.
- Diagnosing why progress stalled is a useful step to build into agent harnesses.
- Long-running optimization tasks like kernel tuning are a good testbed for agent teams.

## Reference

- Original post: [https://x.com/shiyi_c98/status/2117240310736515510](https://x.com/shiyi_c98/status/2117240310736515510)
- Author: [@shiyi_c98](https://x.com/shiyi_c98) on X, posted 2026-11-02
