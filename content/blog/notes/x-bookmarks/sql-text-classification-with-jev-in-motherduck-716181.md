---
title: "SQL Text Classification With Jev in MotherDuck"
publishDate: "2026-09-22"
date: "2026-09-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "MotherDuck adds prompt_jev(), a SQL function powered by TypeSafe's Jev model, claimed to classify text far faster and cheaper than an LLM."
---

> Source: [@typesafeai on X](https://x.com/typesafeai/status/2102206611024716181) · posted 2026-09-22

## Summary

The quoted MotherDuck post says text classification is about 50x faster at about 1% of the cost using **prompt_jev()**, a SQL function powered by TypeSafe's Jev system one model. On 100k rows it took 40s and $0.50 with frontier-LLM accuracy, versus 32 minutes and $37 for an LLM. TypeSafe's own post repeats the 50x faster and 100x cheaper claims.

## Background

A system one model, borrowing from psychology, is a fast, intuitive model rather than a slow step-by-step reasoner. Putting it inside a SQL function lets analysts classify text in a database query. For bulk labeling, a small specialized model is often enough.

## What I took away

- For bulk classification, a small fast model may match an LLM at a fraction of the cost.
- Running models inside the database removes data movement overhead.
- These numbers come from the vendors; test on your own data.

## Reference

- Original post: [https://x.com/typesafeai/status/2102206611024716181](https://x.com/typesafeai/status/2102206611024716181)
- Author: [@typesafeai](https://x.com/typesafeai) on X, posted 2026-09-22
