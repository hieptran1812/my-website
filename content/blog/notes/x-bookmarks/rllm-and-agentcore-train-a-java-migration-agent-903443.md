---
title: "rLLM And AgentCore Train A Java Migration Agent"
publishDate: "2026-06-25"
date: "2026-06-25"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "ai-agents", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2070287479257903443-1.jpg"
excerpt: "A collaboration trains a long-horizon coding agent for Java 8 to 17 migration, lifting Pass@1 from 43% to 76%."
---

> Source: [@sijun_tan on X](https://x.com/sijun_tan/status/2070287479257903443) · posted 2026-06-25

![rLLM And AgentCore Train A Java Migration Agent (1)](/imgs/blogs/x-bookmarks/2070287479257903443-1.jpg)

![rLLM And AgentCore Train A Java Migration Agent (2)](/imgs/blogs/x-bookmarks/2070287479257903443-2.png)

## Summary

@sijun_tan announces work with AWS AgentCore on rLLM, enabling RL training for long-horizon coding agents by decoupling the agent runtime and model gateway. The quoted thread says Qwen3-Coder-30B learns to migrate real Java repos from Java 8 to 17 on MigrationBench. Pass@1 goes from 43% to 76%, per the authors.

## Background

Long-horizon agents take many steps, so RL training needs the agent environment and the model server to be separate parts. Pass@1 is the share of tasks solved on the first try. Code migration is a good RL target because tests can check results.

## What I took away

- Decoupling runtime and model gateway makes agent RL practical.
- Verifiable tasks like migrations suit RL.
- The 43% to 76% figure is the authors' reported result.

## Reference

- Original post: [https://x.com/sijun_tan/status/2070287479257903443](https://x.com/sijun_tan/status/2070287479257903443)
- Author: [@sijun_tan](https://x.com/sijun_tan) on X, posted 2026-06-25
