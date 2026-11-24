---
title: "Harnesses That Generalize Through Composition"
publishDate: "2026-07-20"
date: "2026-07-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "harness-engineering", "ai-agents", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A proposal that the harness around a model, not the transformer itself, should handle generalization by composing known skills."
---

> Source: [@a1zhang on X](https://x.com/a1zhang/status/2079203524395573442) · posted 2026-07-20

## Summary

The author argues that transformers struggle with tasks they were not trained on. Their proposal for 2026 is that the **harness** should do the generalizing, by composing pieces the model already handles. They report a useful property when training RLMs on tasks that share structure; the post text cuts off before the details.

## Background

A harness is the code and loop around a model: prompts, tools, memory and control flow. RLM here refers to a model trained to work inside such a recursive or structured loop. The idea is that breaking a new task into familiar sub-tasks can get results the raw model cannot reach alone.

## What I took away

- When a model fails on a new task, try decomposing it into sub-tasks it already does well.
- Generalization can live in the system design, not only in the weights.
- Open the thread to see the shared-structure property the author describes.

## Reference

- Original post: [https://x.com/a1zhang/status/2079203524395573442](https://x.com/a1zhang/status/2079203524395573442)
- Author: [@a1zhang](https://x.com/a1zhang) on X, posted 2026-07-20
