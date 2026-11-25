---
title: "Fixing Many Frontend Calls In Microservices"
publishDate: "2025-11-15"
date: "2025-11-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "career", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "An answer to a system design question using API Gateway, Backend for Frontend and GraphQL, with a video and article."
---

> Source: [@asmah2107 on X](https://x.com/asmah2107/status/1989595892588933626) · posted 2025-11-15

## Summary

The author gives three approaches: API Gateway, Backend for Frontend (BFF) and GraphQL. They link a YouTube video and Substack article. The quoted post asks how to fix a frontend that makes 10+ calls across 50 microservices to render one page.

## Background

An API Gateway gives one entry point to many services. A BFF is a service tailored to one frontend that combines several backend calls into one response. GraphQL lets the client ask for exactly the data it needs in a single request. All reduce the number of network round trips.

## What I took away

- Aggregate many backend calls into one for the client.
- BFF fits when different frontends need different data shapes.
- GraphQL shifts shaping of data to the client query.

## Reference

- Original post: [https://x.com/asmah2107/status/1989595892588933626](https://x.com/asmah2107/status/1989595892588933626)
- Author: [@asmah2107](https://x.com/asmah2107) on X, posted 2025-11-15
