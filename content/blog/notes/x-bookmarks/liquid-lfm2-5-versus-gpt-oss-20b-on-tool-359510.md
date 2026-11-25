---
title: "Liquid LFM2.5 Versus gpt-oss-20b on Tool Calling"
publishDate: "2026-05-30"
date: "2026-05-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "ai-agents", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2060783957143359510-1.jpg"
excerpt: "Atomic Chat reports a local tool-calling test where Liquid's LFM2.5-8B-A1B beat gpt-oss-20b on a MacBook Pro."
---

> Source: [@atomic_chat_hq on X](https://x.com/atomic_chat_hq/status/2060783957143359510) · posted 2026-05-30

![Liquid LFM2.5 Versus gpt-oss-20b on Tool Calling](/imgs/blogs/x-bookmarks/2060783957143359510-1.jpg)

## Summary

Atomic Chat says Liquid's **LFM2.5-8B-A1B** beat OpenAI's **gpt-oss-20b** on tool calling. Both ran locally on a MacBook Pro M5 Max with 64GB. The task was a trip-planning request that only completes if the model makes all 7 tool calls, such as weather for 3 cities and currency conversions.

## Background

Tool calling means the model asks external functions to run, like a weather lookup, and uses the results. Small models often skip or botch some calls. This is one vendor's informal test, so treat the result as a single data point.

## What I took away

- Tool-call completeness is a practical test for local agent models.
- A single task is anecdotal; run your own tasks before choosing a model.
- Small local models are getting usable for agent work.

## Reference

- Original post: [https://x.com/atomic_chat_hq/status/2060783957143359510](https://x.com/atomic_chat_hq/status/2060783957143359510)
- Author: [@atomic_chat_hq](https://x.com/atomic_chat_hq) on X, posted 2026-05-30
