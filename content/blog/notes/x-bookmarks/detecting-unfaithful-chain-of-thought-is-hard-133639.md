---
title: "Detecting Unfaithful Chain of Thought Is Hard"
publishDate: "2026-05-26"
date: "2026-05-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-safety", "interpretability", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2059292492412133639-1.jpg"
excerpt: "Yoav Gur reports that most of 8 methods for detecting unfaithful LLM reasoning perform near chance."
---

> Source: [@GurYoav on X](https://x.com/GurYoav/status/2059292492412133639) · posted 2026-05-26

![Detecting Unfaithful Chain of Thought Is Hard](/imgs/blogs/x-bookmarks/2059292492412133639-1.jpg)

## Summary

The post asks whether we can tell when LLMs are being unfaithful in their chains of thought. The authors evaluated 8 methods claiming to do this and found most perform near chance. Doing this needs ground-truth labels for faithfulness, which is the next question the thread raises.

## Background

Chain of thought is the written reasoning a model produces. It is faithful if it reflects the actual reason for the answer. Unfaithful reasoning weakens monitoring based on reading the reasoning.

## What I took away

- Do not assume reasoning traces reveal real model behavior.
- Detection methods need ground-truth labels to be tested.
- Near-chance results mean current detectors are unreliable.

## Reference

- Original post: [https://x.com/GurYoav/status/2059292492412133639](https://x.com/GurYoav/status/2059292492412133639)
- Author: [@GurYoav](https://x.com/GurYoav) on X, posted 2026-05-26
