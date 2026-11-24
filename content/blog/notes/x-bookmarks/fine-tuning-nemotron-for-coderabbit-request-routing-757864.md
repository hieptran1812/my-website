---
title: "Fine-Tuning Nemotron for CodeRabbit Request Routing"
publishDate: "2026-08-11"
date: "2026-08-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "serving", "ai-agents"]
author: "Hiep Tran"
featured: false
excerpt: "CodeRabbit and Baseten fine-tuned Nemotron 3.5 Lightning for routing decisions cheaply and quickly."
---

> Source: [@philipkiely on X](https://x.com/philipkiely/status/2087253119851757864) · posted 2026-08-11

## Summary

The author says routing is a hard problem because it is hard to know how difficult a problem is in advance, comparing it to how rarely people predict project durations correctly. The quoted CodeRabbit post says that with Baseten they fine-tuned NVIDIA Nemotron 3.5 Lightning on CodeRabbit's routing decisions in under three hours for less than $100. They report about 4% higher accuracy than their previous GPT-class model.

## Background

Routing means deciding which model or path should handle a request, for example sending easy tasks to a cheap model and hard ones to a strong model. A small fine-tuned model can make this decision quickly and cheaply, which matters because the router runs on every request.

## What I took away

- Small fine-tuned models can beat larger general models on narrow decisions like routing.
- Estimating task difficulty ahead of time is inherently hard; expect errors.
- Logging past routing decisions creates training data for a better router.

## Reference

- Original post: [https://x.com/philipkiely/status/2087253119851757864](https://x.com/philipkiely/status/2087253119851757864)
- Author: [@philipkiely](https://x.com/philipkiely) on X, posted 2026-08-11
