---
title: "Metronome Paper on Serving Live Voice Models"
publishDate: "2026-07-08"
date: "2026-07-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "audio-video", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "A new paper, Metronome, addresses serving voice models that hold live conversations rather than answering single prompts."
---

> Source: [@bojie_li on X](https://x.com/bojie_li/status/2074831636617736491) · posted 2026-07-08

## Summary

The author announces a paper called Metronome, written with his wife Jiaying Meng. He notes that voice models like Moshi, MiniCPM-o, Qwen-Omni and interaction models from Thinking Machines are not chatbots. A chatbot answers one prompt and frees its memory, while these models hold a live conversation where every frame matters; the text cuts off.

## Background

Full-duplex voice models listen and speak continuously, processing audio frames on a fixed clock. Serving them is different from text chat because each session holds memory for its whole length and must meet strict per-frame deadlines.

## What I took away

- Live voice models need serving systems built around steady per-frame timing.
- Long-lived sessions change memory planning compared with request-response chat.
- Read the paper for how they schedule many concurrent conversations.

## Reference

- Original post: [https://x.com/bojie_li/status/2074831636617736491](https://x.com/bojie_li/status/2074831636617736491)
- Author: [@bojie_li](https://x.com/bojie_li) on X, posted 2026-07-08
