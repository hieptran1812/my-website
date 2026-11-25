---
title: "Learning From Rich Feedback Beyond Verifiable Rewards"
publishDate: "2026-03-11"
date: "2026-03-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2031662498449928698-1.jpg"
excerpt: "tugot17 recommends a paper that turns rich feedback into dense supervision, instead of one bit per response."
---

> Source: [@tugot17 on X](https://x.com/tugot17/status/2031662498449928698) · posted 2026-03-11

![Learning From Rich Feedback Beyond Verifiable Rewards (1)](/imgs/blogs/x-bookmarks/2031662498449928698-1.jpg)

![Learning From Rich Feedback Beyond Verifiable Rewards (2)](/imgs/blogs/x-bookmarks/2031662498449928698-2.jpg)

## Summary

The author calls it a really good paper and adds some intuition. The quoted post says training LLMs with verifiable rewards gives 1 bit per generated response, hiding why the model failed, and introduces a simple algorithm that learns from any rich feedback and turns it into dense supervision.

## Background

Verifiable rewards only say pass or fail, a single bit. Rich feedback, like error messages or critiques, explains why. Turning it into dense supervision gives more learning signal per sample.

## What I took away

- Pass or fail rewards carry little information.
- Use textual feedback as training signal where available.
- Read the paper thread for the actual method.

## Reference

- Original post: [https://x.com/tugot17/status/2031662498449928698](https://x.com/tugot17/status/2031662498449928698)
- Author: [@tugot17](https://x.com/tugot17) on X, posted 2026-03-11
