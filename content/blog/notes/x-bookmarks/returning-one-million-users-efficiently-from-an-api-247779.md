---
title: "Returning One Million Users Efficiently From An API"
publishDate: "2025-11-15"
date: "2025-11-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "career", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "A backend interview question about returning data efficiently when the database holds 1 million users."
---

> Source: [@Sumits_X on X](https://x.com/Sumits_X/status/1989699484029247779) · posted 2025-11-15

## Summary

The author shares a backend interview question: you are implementing an API that returns users, but the database has 1,000,000 users. How do you return data to the client efficiently? No answer is given in the text.

## Background

Returning everything at once is slow and memory heavy. Common answers include pagination (offset or cursor based), filtering, limiting fields and caching. Cursor pagination is usually better for large, changing tables.

## What I took away

- Never return an unbounded list.
- Know offset versus cursor pagination.
- Consider field selection and caching.

## Reference

- Original post: [https://x.com/Sumits_X/status/1989699484029247779](https://x.com/Sumits_X/status/1989699484029247779)
- Author: [@Sumits_X](https://x.com/Sumits_X) on X, posted 2025-11-15
