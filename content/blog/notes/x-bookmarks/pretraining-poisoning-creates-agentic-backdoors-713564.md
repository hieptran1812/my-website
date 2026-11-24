---
title: "Pretraining Poisoning Creates Agentic Backdoors"
publishDate: "2026-11-01"
date: "2026-11-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-safety", "pretraining", "ai-agents"]
author: "Hiep Tran"
featured: false
excerpt: "An Anthropic paper shows poisoning 0.1% of pretraining data can plant an agent backdoor that survives clean post-training."
---

> Source: [@difficultyang on X](https://x.com/difficultyang/status/2116938349453713564) · posted 2026-11-01

## Summary

The quoted post describes an Anthropic paper, Agentic Backdoors via Pretraining Poisoning. It says poisoning just 0.1% of pretraining data can plant a remote-code-execution backdoor in an AI agent that survives a fully clean SFT, DPO and RL pipeline, shown end to end for the first time. @difficultyang points out that 0.1% of all human written language is not small at all.

## Background

Data poisoning means inserting crafted examples into training data so the model learns a hidden behavior. A backdoor activates only on a trigger. SFT, DPO and RL are post-training steps; the finding is that clean post-training does not remove the backdoor.

## What I took away

- Pretraining data provenance is a real security concern for agents.
- Clean post-training may not remove behaviors learned in pretraining.
- Percentages can mislead; 0.1% of a huge dataset is a lot of data.

## Reference

- Original post: [https://x.com/difficultyang/status/2116938349453713564](https://x.com/difficultyang/status/2116938349453713564)
- Author: [@difficultyang](https://x.com/difficultyang) on X, posted 2026-11-01
