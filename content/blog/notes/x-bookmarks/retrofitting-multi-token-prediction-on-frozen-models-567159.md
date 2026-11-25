---
title: "Retrofitting Multi-Token Prediction On Frozen Models"
publishDate: "2026-06-26"
date: "2026-06-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "speculative-decoding", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2070579898465567159-1.jpg"
excerpt: "Google Research presents a method to add multi-token prediction to frozen production models to speed up on-device inference."
---

> Source: [@GoogleResearch on X](https://x.com/GoogleResearch/status/2070579898465567159) · posted 2026-06-26

![Retrofitting Multi-Token Prediction On Frozen Models](/imgs/blogs/x-bookmarks/2070579898465567159-1.jpg)

## Summary

Google Research introduces a blog post on retrofitting Multi-Token Prediction (MTP) onto frozen production models. It aims to speed up on-device inference without the inefficiencies of separate drafters.

## Background

Multi-token prediction lets a model propose several next tokens at once instead of one. Speculative decoding usually needs a separate small drafter model, which costs memory and complexity. Retrofitting MTP onto a frozen model avoids a second model, which matters on phones and other small devices.

## What I took away

- Drafting can be built into the main model instead of a separate drafter.
- Frozen-model methods are easier to deploy because the base weights do not change.

## Reference

- Original post: [https://x.com/GoogleResearch/status/2070579898465567159](https://x.com/GoogleResearch/status/2070579898465567159)
- Author: [@GoogleResearch](https://x.com/GoogleResearch) on X, posted 2026-06-26
