---
title: "Tokens-In Tokens-Out API For LLM RL"
publishDate: "2025-12-17"
date: "2025-12-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "harness-engineering"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001242003581870337-1.jpg"
excerpt: "Taco Cohen explains why his Env API uses a tokens-in-tokens-out signature and a Trajectory class to avoid retokenization bugs in RL."
---

> Source: [@TacoCohen on X](https://x.com/TacoCohen/status/2001242003581870337) · posted 2025-12-17

![Tokens-In Tokens-Out API For LLM RL (1)](/imgs/blogs/x-bookmarks/2001242003581870337-1.jpg)

![Tokens-In Tokens-Out API For LLM RL (2)](/imgs/blogs/x-bookmarks/2001242003581870337-2.jpg)

![Tokens-In Tokens-Out API For LLM RL (3)](/imgs/blogs/x-bookmarks/2001242003581870337-3.jpg)

![Tokens-In Tokens-Out API For LLM RL (4)](/imgs/blogs/x-bookmarks/2001242003581870337-4.jpg)

## Summary

Taco Cohen says people keep hitting this bug in LLM-RL codebases, which is why his Env API uses tokens in and tokens out and a Trajectory class that keeps contexts consistent between inference and training. The quoted post says most RL frameworks are unstable: with OpenAI-style messages, parsing and retokenizing produces subtly different tokens, and this cost a lot of H100 hours in debugging.

## Background

In RL for LLMs, training must use exactly the tokens the model saw and produced during sampling. If text is decoded to messages and then tokenized again, the tokens can differ slightly, so the training signal no longer matches. Working at the token level avoids this mismatch.

## What I took away

- Store the exact token ids from rollouts, not re-tokenized text.
- Retokenization mismatches can silently break RL training.
- Design environment APIs around tokens, not chat messages.

## Reference

- Original post: [https://x.com/TacoCohen/status/2001242003581870337](https://x.com/TacoCohen/status/2001242003581870337)
- Author: [@TacoCohen](https://x.com/TacoCohen) on X, posted 2025-12-17
