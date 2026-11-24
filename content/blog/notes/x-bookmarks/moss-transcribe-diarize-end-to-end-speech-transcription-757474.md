---
title: "MOSS-Transcribe-Diarize End-to-End Speech Transcription"
publishDate: "2026-07-09"
date: "2026-07-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "audio-video", "open-models", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "MOSS-Transcribe-Diarize-0.9B is an open model that transcribes and labels speakers in long audio in one pass, with vLLM support."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2075063247837757474) · posted 2026-07-09

## Summary

vLLM congratulates MosiAI on MOSS-Transcribe-Diarize-0.9B, an open end-to-end model for multi-speaker long-audio transcription with day-0 vLLM support. It notes most setups chain ASR, diarization and alignment, WhisperX-style, while this model does all three in one. The quoted post lists details: 0.9B parameters, Apache 2.0 license, 128k context, up to about 90 minutes of audio, and speaker labels plus timestamps in one output.

## Background

ASR is automatic speech recognition, turning speech into text. Diarization means figuring out who spoke when. Alignment adds word timestamps. Doing all of these in one model avoids errors from passing data between separate tools.

## What I took away

- End-to-end models can replace multi-tool audio pipelines.
- Apache 2.0 licensing makes it usable commercially.
- Test speaker labeling accuracy on your own recordings.

## Reference

- Original post: [https://x.com/vllm_project/status/2075063247837757474](https://x.com/vllm_project/status/2075063247837757474)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-07-09
