---
title: "Bidirectional Evolutionary Search for Hard Reasoning"
publishDate: "2026-05-28"
date: "2026-05-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2060022930172506200-1.jpg"
excerpt: "A paper proposes a new way to sample on hard reasoning problems where direct rollouts rarely give a correct answer."
---

> Source: [@Kevin_GuoweiXu on X](https://x.com/Kevin_GuoweiXu/status/2060022930172506200) · posted 2026-05-28

![Bidirectional Evolutionary Search for Hard Reasoning](/imgs/blogs/x-bookmarks/2060022930172506200-1.jpg)

## Summary

The post asks how LLMs should sample on hard reasoning problems during post-training and inference when direct rollouts rarely succeed. It says Best-of-N (such as GRPO) and tree search share two limits: verification signals are sparse and candidates stay within the model's own distribution. The text is cut off before the solution.

## Background

In RL for LLMs, rollouts are sampled answers. If none are correct, there is little learning signal. Evolutionary search tries to mix and improve candidates to find better ones.

## What I took away

- Sparse rewards are a core problem for hard reasoning RL.
- Best-of-N alone cannot escape the model's own distribution.
- Search methods that evolve candidates are worth following.

## Reference

- Original post: [https://x.com/Kevin_GuoweiXu/status/2060022930172506200](https://x.com/Kevin_GuoweiXu/status/2060022930172506200)
- Author: [@Kevin_GuoweiXu](https://x.com/Kevin_GuoweiXu) on X, posted 2026-05-28
