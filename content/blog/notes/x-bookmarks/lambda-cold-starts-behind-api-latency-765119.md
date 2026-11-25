---
title: "Lambda Cold Starts Behind API Latency"
publishDate: "2025-11-15"
date: "2025-11-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "A debugging story where 500ms of extra API Gateway latency turned out to be Lambda cold starts."
---

> Source: [@brankopetric00 on X](https://x.com/brankopetric00/status/1989821067330765119) · posted 2025-11-15

## Summary

The author says API Gateway showed 500ms of added latency, which turned out to be Lambda cold starts. The symptoms were P50 latency of 150ms (fine) but P99 of 750ms, with some requests very slow and no pattern. They had checked database query time, which was fast. The text is cut off.

## Background

A cold start happens when a serverless function must start a new instance before handling a request. This adds delay for some requests only. P50 is the median latency and P99 is the slowest 1 percent, so a bad tail with a good median is a classic clue.

## What I took away

- Check tail latency (P99), not just the median.
- Random slow requests with fast dependencies can point to cold starts.
- Rule out the database and application first, as the author did.

## Reference

- Original post: [https://x.com/brankopetric00/status/1989821067330765119](https://x.com/brankopetric00/status/1989821067330765119)
- Author: [@brankopetric00](https://x.com/brankopetric00) on X, posted 2025-11-15
