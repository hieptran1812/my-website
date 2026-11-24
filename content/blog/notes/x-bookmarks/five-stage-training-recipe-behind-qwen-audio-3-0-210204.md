---
title: "Five-Stage Training Recipe Behind Qwen-Audio-3.0-TTS"
publishDate: "2026-08-05"
date: "2026-08-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "audio-video", "pretraining", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "A post on the progressive five-stage training paradigm used for Qwen-Audio-3.0-TTS, a new text-to-speech model."
---

> Source: [@NanaHou_ on X](https://x.com/NanaHou_/status/2085058707596210204) · posted 2026-08-05

## Summary

The author says the most impressive part of **Qwen-Audio-3.0-TTS** is its progressive five-stage training. Stage 1 pretrains the language model (LM) and flow-matching model (FM) separately for semantic planning and acoustic reconstruction, and stage 2 trains them jointly with high-quality data annealing. The quoted Qwen post announces the model in Flash (real-time) and Plus (high quality) versions, with 16 languages, natural-language style control and tags for non-verbal sounds.

## Background

Many modern TTS systems use an LM to plan what to say as tokens and a flow-matching model to turn that plan into audio. Data annealing means ending training on a smaller set of high-quality data to polish output. Staged training lets each part learn its job before they are combined.

## What I took away

- Train components separately before joining them to stabilize learning.
- Finishing with high-quality data can noticeably improve output quality.
- Offering fast and high-quality variants serves different product needs.

## Reference

- Original post: [https://x.com/NanaHou_/status/2085058707596210204](https://x.com/NanaHou_/status/2085058707596210204)
- Author: [@NanaHou_](https://x.com/NanaHou_) on X, posted 2026-08-05
