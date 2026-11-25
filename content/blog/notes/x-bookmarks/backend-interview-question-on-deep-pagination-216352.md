---
title: "Backend Interview Question On Deep Pagination"
publishDate: "2026-01-18"
date: "2026-01-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "career", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A prompt asks what a backend developer should do on seeing GET /api/v1/trades?page=8000&size=50."
---

> Source: [@Sumits_X on X](https://x.com/Sumits_X/status/2012873506028216352) · posted 2026-01-18

## Summary

@Sumits_X asks what next steps a backend dev would take upon seeing a request for page 8000 with page size 50 on a trades API. It is an open-ended system design prompt.

## Background

Page 8000 at 50 per page means skipping 400,000 rows, which is costly with offset pagination. Answers might include checking whether it is a bug or a scraper, adding rate limits, and using cursor pagination.

## What I took away

- Offset pagination gets slow at depth.
- Check who sends the request before acting.
- Rate limits and cursors are standard fixes.

## Reference

- Original post: [https://x.com/Sumits_X/status/2012873506028216352](https://x.com/Sumits_X/status/2012873506028216352)
- Author: [@Sumits_X](https://x.com/Sumits_X) on X, posted 2026-01-18
