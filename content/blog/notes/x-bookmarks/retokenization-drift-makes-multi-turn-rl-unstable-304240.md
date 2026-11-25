---
title: "Retokenization Drift Makes Multi-Turn RL Unstable"
publishDate: "2025-12-17"
date: "2025-12-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-agents"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001128217030304240-1.jpg"
excerpt: "The author says parsing and retokenizing chat messages in RL environments produces subtly different tokens and wasted lots of debugging time."
---

> Source: [@maxrumpf on X](https://x.com/maxrumpf/status/2001128217030304240) · posted 2025-12-17

![Retokenization Drift Makes Multi-Turn RL Unstable (1)](/imgs/blogs/x-bookmarks/2001128217030304240-1.jpg)

![Retokenization Drift Makes Multi-Turn RL Unstable (2)](/imgs/blogs/x-bookmarks/2001128217030304240-2.jpg)

![Retokenization Drift Makes Multi-Turn RL Unstable (3)](/imgs/blogs/x-bookmarks/2001128217030304240-3.jpg)

![Retokenization Drift Makes Multi-Turn RL Unstable (4)](/imgs/blogs/x-bookmarks/2001128217030304240-4.jpg)

## Summary

@maxrumpf says most RL frameworks are fundamentally unstable. When environments use OpenAI-style messages, parsing and retokenizing gives subtly different tokens, and this cost them more H100 hours than any other issue in a multi-turn, multi-environment run. The quoted post announces SID-1, a model built only for retrieval, claimed to have 1.8x better recall than embedding search alone.

## Background

In RL training the model must learn from the exact tokens it produced. If text is converted to messages and tokenized again, the tokens can differ from the original. The training signal then no longer matches what the model generated. Keeping the original token IDs avoids this.

## What I took away

- Store the original token IDs from generation rather than re-tokenizing text.
- Mismatched tokens can cause silent instability in RL runs.
- SID-1's recall numbers are the authors' claim and need independent checks.

## Reference

- Original post: [https://x.com/maxrumpf/status/2001128217030304240](https://x.com/maxrumpf/status/2001128217030304240)
- Author: [@maxrumpf](https://x.com/maxrumpf) on X, posted 2025-12-17
