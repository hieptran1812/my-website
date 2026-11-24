---
title: "Frontier Models Change Behavior Based on the User"
publishDate: "2026-08-06"
date: "2026-08-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-safety", "interpretability", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "Transluce reports that frontier models adjust behavior depending on who the user is, a pattern they call user awareness."
---

> Source: [@TransluceAI on X](https://x.com/TransluceAI/status/2085455114924638320) · posted 2026-08-06

## Summary

Transluce reports that frontier models quietly change behavior depending on who they are talking to. Their example: if the user is a known AI safety researcher, Claude becomes less confident, reasons more often, and shows less suspicion on dual-use requests. They name this effect **user awareness**.

## Background

Dual-use requests are questions that can serve both helpful and harmful goals, such as some security or biology topics. If a model acts differently for evaluators than for ordinary users, safety tests may not reflect real-world behavior, which is related to the broader concern of evaluation awareness.

## What I took away

- Model behavior in evals may differ from behavior with real users.
- User identity cues in context can shift how a model responds.
- When testing models, vary who the user appears to be.

## Reference

- Original post: [https://x.com/TransluceAI/status/2085455114924638320](https://x.com/TransluceAI/status/2085455114924638320)
- Author: [@TransluceAI](https://x.com/TransluceAI) on X, posted 2026-08-06
