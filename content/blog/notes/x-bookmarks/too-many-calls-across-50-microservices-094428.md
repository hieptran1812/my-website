---
title: "Too Many Calls Across 50 Microservices"
publishDate: "2025-11-14"
date: "2025-11-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "career", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "A system design question about a frontend making 10+ calls to 50 microservices to render one page."
---

> Source: [@asmah2107 on X](https://x.com/asmah2107/status/1989199691029094428) · posted 2025-11-14

## Summary

The author asks: with 50 microservices each having its own API, the frontend makes 10+ network calls to render a page, the app is sluggish and users complain. How do you fix it? The answer is in a later post.

## Background

Many small calls add latency and complexity on the client. Typical fixes are an API gateway, a backend-for-frontend that aggregates data, or GraphQL. This is a common system design interview question.

## What I took away

- Aggregate calls on the server side.
- Consider gateway, BFF or GraphQL.
- Measure where the latency actually comes from.

## Reference

- Original post: [https://x.com/asmah2107/status/1989199691029094428](https://x.com/asmah2107/status/1989199691029094428)
- Author: [@asmah2107](https://x.com/asmah2107) on X, posted 2025-11-14
