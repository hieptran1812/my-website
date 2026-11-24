---
title: "Reducing Whisper Hallucinations Without Retraining"
publishDate: "2026-09-07"
date: "2026-09-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "audio-video", "interpretability", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A method projects a non-speech subspace out of Whisper's decoder activations, cutting hallucinations without retraining."
---

> Source: [@di_zhang_fdu on X](https://x.com/di_zhang_fdu/status/2096886023695548449) · posted 2026-09-07

## Summary

The post reports Whisper hallucinations falling from 31.31% to 2.44% without retraining. The method finds a low-rank subspace from non-speech calibration audio and projects it out of decoder activations. A gated version limits the intervention but raises LibriSpeech WER by 0.33 to 4.39 points.

## Background

Whisper is a popular speech recognition model that sometimes outputs fluent text for silence or noise. Projecting out a subspace means removing activation directions linked to a behavior, a form of activation steering. WER (word error rate) measures transcription mistakes, so higher is worse.

## What I took away

- Activation editing can fix specific behaviors without retraining.
- Check the cost on normal inputs; here WER rises somewhat.
- Filtering non-speech audio is a key problem in ASR pipelines.

## Reference

- Original post: [https://x.com/di_zhang_fdu/status/2096886023695548449](https://x.com/di_zhang_fdu/status/2096886023695548449)
- Author: [@di_zhang_fdu](https://x.com/di_zhang_fdu) on X, posted 2026-09-07
