---
title: "Weight Space Programming Injects Facts Into Qwen 0.8B"
publishDate: "2026-11-08"
date: "2026-11-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "ai-agents", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2119416808273019037-1.jpg"
excerpt: "An agent edited a small share of Qwen 0.8B's tensors to inject facts without training."
---

> Source: [@TensorSlay on X](https://x.com/TensorSlay/status/2119416808273019037) · posted 2026-11-08

![Weight Space Programming Injects Facts Into Qwen 0.8B](/imgs/blogs/x-bookmarks/2119416808273019037-1.jpg)

## Summary

The author describes **Weight Space Programming** as a glimpse into continual learning. An agent edited Qwen 0.8B's weights to inject facts about his X profile. It changed 2% of the network (5 of 488 tensors) with no training or gradients. The result is cut off.

## Background

Instead of fine-tuning, the agent directly rewrote some weight tensors to store facts. This is knowledge editing carried out by an agent.

## What I took away

- Facts may be written directly into weights.
- Small edits can be targeted.
- Test side effects on other abilities.

## Reference

- Original post: [https://x.com/TensorSlay/status/2119416808273019037](https://x.com/TensorSlay/status/2119416808273019037)
- Author: [@TensorSlay](https://x.com/TensorSlay) on X, posted 2026-11-08
