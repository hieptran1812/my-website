---
title: "Building a Realtime Multilingual Voice Assistant on Free Tiers"
publishDate: "2026-08-03"
date: "2026-08-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "audio-video", "ai-agents", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "A developer shares Aria, a realtime multilingual voice assistant using LiveKit, Groq Whisper, LLaMA 3.3 70B and Edge TTS."
---

> Source: [@justkarangupta on X](https://x.com/justkarangupta/status/2084255591430795456) · posted 2026-08-03

## Summary

The author built **Aria**, a realtime multilingual voice assistant. You pick a language, speak, and get a natural reply over LiveKit WebRTC. Speech recognition runs on Groq Whisper, replies come from LLaMA 3.3 70B, and speech output uses Microsoft Edge TTS, all on free tiers.

## Background

This is a classic cascade voice pipeline: speech to text, LLM, then text to speech. WebRTC is the browser standard for low-latency audio and video streams, and LiveKit is a platform built on it.

## What I took away

- A working voice assistant can be built entirely on free tiers.
- In cascade pipelines, latency adds up at each stage, so measure the full loop.
- Compare with end-to-end full-duplex models when interruptions matter.

## Reference

- Original post: [https://x.com/justkarangupta/status/2084255591430795456](https://x.com/justkarangupta/status/2084255591430795456)
- Author: [@justkarangupta](https://x.com/justkarangupta) on X, posted 2026-08-03
