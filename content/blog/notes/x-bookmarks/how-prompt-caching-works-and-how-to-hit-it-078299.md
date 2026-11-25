---
title: "How Prompt Caching Works And How To Hit It"
publishDate: "2025-11-30"
date: "2025-11-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "kv-cache", "ai-agents"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1995247669888078299-1.jpg"
excerpt: "A blog post on how prompt caching works and tips to hit the cache more consistently."
---

> Source: [@dejavucoder on X](https://x.com/dejavucoder/status/1995247669888078299) · posted 2025-11-30

![How Prompt Caching Works And How To Hit It](/imgs/blogs/x-bookmarks/1995247669888078299-1.jpg)

## Summary

@dejavucoder says prompt caching is the best bang-for-buck optimization for LLM workflows and agents. Their post covers tips for hitting the cache more consistently and how it works under the hood, and they say it is probably the first such resource.

## Background

Prompt caching reuses work already done for the start of a prompt, so repeated prefixes are cheaper and faster. It relies on keeping the stable part of the prompt first and unchanged. This matters for agents that resend long context on each step.

## What I took away

- Keep stable content at the start of the prompt and variable content at the end.
- Avoid small prompt changes that break cache hits.
- Caching can cut both cost and latency for agents.

## Reference

- Original post: [https://x.com/dejavucoder/status/1995247669888078299](https://x.com/dejavucoder/status/1995247669888078299)
- Author: [@dejavucoder](https://x.com/dejavucoder) on X, posted 2025-11-30
