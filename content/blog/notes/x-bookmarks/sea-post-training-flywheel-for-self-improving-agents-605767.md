---
title: "SEA Post-Training Flywheel for Self-Improving Agents"
publishDate: "2026-08-13"
date: "2026-08-13"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "ai-agents", "distillation"]
author: "Hiep Tran"
featured: false
excerpt: "A description of SEA's post-training flywheel that turns tasks into feedback triples to update Qwen3.6-35B-A3B."
---

> Source: [@sheriyuo on X](https://x.com/sheriyuo/status/2087762408840605767) · posted 2026-08-13

## Summary

The author describes SEA's post-training flywheel. Each task becomes a (context, decision, feedback) triple, filtered by confidence, feedback sufficiency and diversity. Retained trajectories are then used to update Qwen3.6-35B-A3B through LoRA, cross-task replay and external-teacher distillation.

## Background

A flywheel here is a loop where the agent's own experience becomes training data that improves it. LoRA is a cheap fine-tuning method that trains small added matrices instead of all weights. Replay mixes in older tasks to avoid forgetting, and teacher distillation uses a stronger model's outputs as targets.

## What I took away

- Filter self-generated data carefully; bad trajectories can make agents worse.
- Replay of older tasks helps prevent forgetting during continual updates.
- LoRA makes frequent incremental updates affordable.

## Reference

- Original post: [https://x.com/sheriyuo/status/2087762408840605767](https://x.com/sheriyuo/status/2087762408840605767)
- Author: [@sheriyuo](https://x.com/sheriyuo) on X, posted 2026-08-13
