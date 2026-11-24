---
title: "GPT-Live Rebuilt Voice Stack Listens While Speaking"
publishDate: "2026-08-03"
date: "2026-08-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "audio-video", "serving", "industry-news"]
author: "Hiep Tran"
featured: false
excerpt: "OpenAI says GPT-Live can listen while it speaks, after rebuilding the voice stack from client to model."
---

> Source: [@OpenAI on X](https://x.com/OpenAI/status/2084378415818579975) · posted 2026-08-03

## Summary

OpenAI says **GPT-Live** can listen while it speaks. To make this feel natural at ChatGPT scale, they rebuilt the voice stack from client to model. The new architecture keeps audio flowing continuously so deeper reasoning and tool use do not interrupt the conversation.

## Background

Listening while speaking is full-duplex voice. Older voice systems often stop listening or pause while thinking or calling tools. Keeping audio streams running while reasoning happens in parallel makes conversation feel more natural.

## What I took away

- Natural voice AI needs changes across the whole stack, not just the model.
- Decouple audio streaming from reasoning and tool calls to avoid pauses.
- Compare with other full-duplex models such as the NVIDIA one in this batch.

## Reference

- Original post: [https://x.com/OpenAI/status/2084378415818579975](https://x.com/OpenAI/status/2084378415818579975)
- Author: [@OpenAI](https://x.com/OpenAI) on X, posted 2026-08-03
