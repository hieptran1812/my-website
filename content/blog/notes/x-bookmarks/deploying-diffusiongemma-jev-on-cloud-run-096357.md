---
title: "Deploying DiffusionGemma-Jev on Cloud Run"
publishDate: "2026-09-22"
date: "2026-09-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "diffusion", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "Google Gemma shows a one-command way to deploy a Jev API-compatible DiffusionGemma-Jev endpoint on Google Cloud Run."
---

> Source: [@googlegemma on X](https://x.com/googlegemma/status/2102536894169096357) · posted 2026-09-22

## Summary

The Gemma team says deploying **DiffusionGemma-Jev (djev)** is now easier: a single command spins up a Jev API-compatible endpoint on Google Cloud Run. They report about 35 to 60 ms single-step latency and about 100 to 123 requests per second at batch size 32.

## Background

Cloud Run is Google's serverless container platform, where you deploy without managing servers. API compatibility means existing clients written for Jev can call this endpoint without changes. Single-step latency matters for diffusion-style models that may run one or a few steps per request.

## What I took away

- API-compatible endpoints make it easy to swap model backends.
- Serverless deployment lowers the barrier for trying new models.
- Check latency and throughput numbers at the batch size you actually need.

## Reference

- Original post: [https://x.com/googlegemma/status/2102536894169096357](https://x.com/googlegemma/status/2102536894169096357)
- Author: [@googlegemma](https://x.com/googlegemma) on X, posted 2026-09-22
