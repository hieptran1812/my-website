---
title: "LongVideoAgent Multi-Agent Long Video Understanding"
publishDate: "2025-12-24"
date: "2025-12-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "reinforcement-learning", "audio-video"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2003860273132924945-1.jpg"
excerpt: "A paper presents a multi-agent framework for hour-long videos, with a master LLM coordinating grounding and vision agents."
---

> Source: [@HuggingPapers on X](https://x.com/HuggingPapers/status/2003860273132924945) · posted 2025-12-24

![LongVideoAgent Multi-Agent Long Video Understanding](/imgs/blogs/x-bookmarks/2003860273132924945-1.jpg)

## Summary

HuggingPapers presents LongVideoAgent. It says traditional multimodal LLMs fail on hour-long videos due to lossy compression. Its framework uses a master LLM that coordinates grounding and vision agents to gather evidence step by step, and is trained with RL to cooperate better.

## Background

Long videos produce far more frames than a model can read at once, so they are compressed and detail is lost. An agent approach lets the model look up only the relevant parts as needed. The results are the authors' claims.

## What I took away

- Retrieving evidence on demand can beat compressing everything.
- RL can train agents to cooperate.

## Reference

- Original post: [https://x.com/HuggingPapers/status/2003860273132924945](https://x.com/HuggingPapers/status/2003860273132924945)
- Author: [@HuggingPapers](https://x.com/HuggingPapers) on X, posted 2025-12-24
