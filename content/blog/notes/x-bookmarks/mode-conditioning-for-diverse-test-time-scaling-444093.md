---
title: "Mode-Conditioning For Diverse Test-Time Scaling"
publishDate: "2025-12-18"
date: "2025-12-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "post-training", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001668804506444093-1.jpg"
excerpt: "A thread proposes Mode-Conditioning so parallel sampling produces different solutions instead of repeating the same failure."
---

> Source: [@ChenHenryWu on X](https://x.com/ChenHenryWu/status/2001668804506444093) · posted 2025-12-18

![Mode-Conditioning For Diverse Test-Time Scaling](/imgs/blogs/x-bookmarks/2001668804506444093-1.jpg)

## Summary

The author says parallel test-time scaling such as pass@k often wastes compute because models repeat the same dominant failure. Raising temperature usually does not fix this. They propose Mode-Conditioning to generate more creative and varied solutions.

## Background

Pass@k means sampling k answers and counting success if any is correct. This only helps when the answers differ. If the model keeps making the same mistake, extra samples add little, so methods that force diversity are valuable.

## What I took away

- More samples only help if they are diverse.
- Higher temperature adds noise but not always real diversity.
- Conditioning on distinct modes is an idea to try for pass@k workloads.

## Reference

- Original post: [https://x.com/ChenHenryWu/status/2001668804506444093](https://x.com/ChenHenryWu/status/2001668804506444093)
- Author: [@ChenHenryWu](https://x.com/ChenHenryWu) on X, posted 2025-12-18
