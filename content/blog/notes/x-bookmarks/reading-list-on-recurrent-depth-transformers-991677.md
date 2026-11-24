---
title: "Reading List on Recurrent Depth Transformers"
publishDate: "2026-09-02"
date: "2026-09-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "learning-resources", "ai-safety"]
author: "Hiep Tran"
featured: false
excerpt: "A list of papers on recurrent depth transformers, shared in response to a report about OpenAI's Astra model."
---

> Source: [@NielsRogge on X](https://x.com/NielsRogge/status/2095003071658991677) · posted 2026-09-02

## Summary

Niels Rogge shares four paper links for learning about recurrent depth transformers. The quoted post from The Information says OpenAI's **Astra** uses a reasoning approach called recurrent depth, which may help cost and performance, but some researchers worry it hides the model's thinking and makes it harder to monitor.

## Background

Recurrent depth means the model repeats a block of layers internally to think longer, instead of writing out reasoning as visible text tokens. Visible chain-of-thought lets humans read and monitor reasoning; internal loops do not, which is the safety concern.

## What I took away

- Reasoning inside hidden layers is harder to monitor than reasoning written as text.
- There is an existing paper trail on recurrent depth worth reading before forming views.
- Efficiency gains and monitorability can pull in opposite directions.

## Reference

- Original post: [https://x.com/NielsRogge/status/2095003071658991677](https://x.com/NielsRogge/status/2095003071658991677)
- Author: [@NielsRogge](https://x.com/NielsRogge) on X, posted 2026-09-02
