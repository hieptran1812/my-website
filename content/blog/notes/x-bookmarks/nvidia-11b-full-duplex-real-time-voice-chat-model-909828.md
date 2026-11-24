---
title: "NVIDIA 11B Full-Duplex Real-Time Voice Chat Model"
publishDate: "2026-08-06"
date: "2026-08-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "audio-video", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "A post about NVIDIA NemotronLabs VoiceChat-11B, an end-to-end full-duplex audio model that can be interrupted mid-speech."
---

> Source: [@C_of_Creativity on X](https://x.com/C_of_Creativity/status/2085197789013909828) · posted 2026-08-06

## Summary

The post highlights an 11B end-to-end, real-time, full-duplex audio model for conversational AI, linked to the Hugging Face page for nvidia/NVIDIA-NemotronLabs-VoiceChat-11B. It jointly does streaming audio understanding and audio generation, and the author stresses that it can be interrupted. It contrasts this with traditional cascade stacks of ASR, then LLM, then TTS.

## Background

Full-duplex means the system can listen and speak at the same time, like a phone call, so users can interrupt it. A cascade stack chains speech recognition (ASR), a text LLM and text-to-speech (TTS), which adds latency at each step. End-to-end audio models handle speech in and out within one model.

## What I took away

- End-to-end audio models can cut latency compared with ASR to LLM to TTS pipelines.
- Interruptibility is a key feature for natural voice assistants.
- Check the model card for license and hardware needs before using it.

## Reference

- Original post: [https://x.com/C_of_Creativity/status/2085197789013909828](https://x.com/C_of_Creativity/status/2085197789013909828)
- Author: [@C_of_Creativity](https://x.com/C_of_Creativity) on X, posted 2026-08-06
