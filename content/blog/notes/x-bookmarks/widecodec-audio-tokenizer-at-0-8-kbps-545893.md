---
title: "WideCodec Audio Tokenizer at 0.8 kbps"
publishDate: "2026-07-18"
date: "2026-07-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "audio-video", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "WideCodec is released as a 44.1 kHz audio tokenizer that the author claims is best on naturalness versus bitrate."
---

> Source: [@huseinzol05 on X](https://x.com/huseinzol05/status/2078628235177545893) · posted 2026-07-18

## Summary

The author announces WideCodec, a 44.1 kHz audio tokenizer running at 0.8 kbps with one codebook and 50 tokens per second. They report a UTMOSv2 score of 2.79 against about 2.82 for ground truth. They claim it is Pareto-best on naturalness versus bitrate compared with DAC, SNAC and EnCodec on a 9,291-clip benchmark covering 400+ languages.

## Background

An audio tokenizer, or neural codec, turns sound into a small set of discrete tokens that language models can read and generate. Lower bitrate and fewer tokens per second make audio models cheaper. UTMOS is an automatic score that predicts how natural speech sounds to humans.

## What I took away

- Tokens per second matters a lot for audio LLM cost; 50 tok/s with one codebook is compact.
- Pareto-best means no compared codec is better on both axes at once.
- Automatic quality metrics should be checked against listening tests.

## Reference

- Original post: [https://x.com/huseinzol05/status/2078628235177545893](https://x.com/huseinzol05/status/2078628235177545893)
- Author: [@huseinzol05](https://x.com/huseinzol05) on X, posted 2026-07-18
