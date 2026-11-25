---
title: "Shared Memory Ringbuffer In A Chromium Fork"
publishDate: "2025-12-15"
date: "2025-12-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "audio-video"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2000608790836854925-1.jpg"
excerpt: "davidgu describes forking Chromium to add a lock-free zero-copy Rust shared memory ringbuffer for streaming raw video."
---

> Source: [@davidgu on X](https://x.com/davidgu/status/2000608790836854925) · posted 2025-12-15

![Shared Memory Ringbuffer In A Chromium Fork](/imgs/blogs/x-bookmarks/2000608790836854925-1.jpg)

## Summary

The author says they forked Chromium and added a lock-free, zero-copy, low-latency shared memory ringbuffer written in Rust. They needed to pass 100+ MB/s of raw video between processes, and Chromium's WebSocket implementation was very inefficient for it.

## Background

A ringbuffer in shared memory lets two processes exchange data without copying it through the network stack. Lock-free means they do not wait on each other with locks. This is a standard way to move high-bandwidth streams with low latency.

## What I took away

- For heavy local streams, shared memory beats WebSockets.
- Zero-copy and lock-free design cuts latency.
- Measure the default transport before assuming it is good enough.

## Reference

- Original post: [https://x.com/davidgu/status/2000608790836854925](https://x.com/davidgu/status/2000608790836854925)
- Author: [@davidgu](https://x.com/davidgu) on X, posted 2025-12-15
