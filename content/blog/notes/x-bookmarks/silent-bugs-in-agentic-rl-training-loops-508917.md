---
title: "Silent Bugs in Agentic RL Training Loops"
publishDate: "2026-05-29"
date: "2026-05-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "ai-agents", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2060175330665508917-1.jpg"
excerpt: "Clement Delangue warns that many teams training agentic LLMs with RL have a silently broken loop once tools are added."
---

> Source: [@ClementDelangue on X](https://x.com/ClementDelangue/status/2060175330665508917) · posted 2026-05-29

![Silent Bugs in Agentic RL Training Loops](/imgs/blogs/x-bookmarks/2060175330665508917-1.jpg)

## Summary

The author claims most people training agentic LLMs with RL have a **silently broken training loop**. Single-turn RL works well with clean curves and sane rewards. Then adding tools so the model can act mid-rollout causes problems. The post text is cut off before the details.

## Background

With tools, a rollout includes tool outputs that the model did not generate. If the loss or token masking treats them as model output, training can quietly go wrong even when curves look fine.

## What I took away

- Healthy-looking curves do not prove the loop is correct.
- Multi-turn tool use changes what tokens should be trained on.
- Audit masking and rollout handling when you add tools to RL.

## Reference

- Original post: [https://x.com/ClementDelangue/status/2060175330665508917](https://x.com/ClementDelangue/status/2060175330665508917)
- Author: [@ClementDelangue](https://x.com/ClementDelangue) on X, posted 2026-05-29
