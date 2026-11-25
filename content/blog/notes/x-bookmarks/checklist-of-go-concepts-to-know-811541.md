---
title: "Checklist Of Go Concepts To Know"
publishDate: "2025-12-06"
date: "2025-12-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "learning-resources", "career"]
author: "Hiep Tran"
featured: false
excerpt: "A post listing the Go language topics an engineer should be able to explain, quoting a similar system design list."
---

> Source: [@0xlelouch_ on X](https://x.com/0xlelouch_/status/1997403874567811541) · posted 2025-12-06

## Summary

@0xlelouch_ lists Go topics to be able to explain: goroutines, channels, context cancellation, WaitGroups, Mutex vs RWMutex, buffered vs unbuffered channels, worker pools, select, interfaces, error wrapping, generics, net/http internals, gRPC in Go, sync.Pool, io.Reader and io.Writer, and goroutine leaks. The quoted post is the same format for system design topics such as CDNs, caching, load balancers and sharding.

## Background

These are core Go concurrency and library topics. Goroutines are lightweight threads, and channels pass data between them. A goroutine leak happens when a goroutine never exits and keeps using memory. Lists like this work as study checklists.

## What I took away

- Use the list as a self-check: can you explain each item in your own words?
- Concurrency topics such as leaks and cancellation are common sources of bugs.

## Reference

- Original post: [https://x.com/0xlelouch_/status/1997403874567811541](https://x.com/0xlelouch_/status/1997403874567811541)
- Author: [@0xlelouch_](https://x.com/0xlelouch_) on X, posted 2025-12-06
