---
title: "Agent Rewritten in Go for Tiny Boards"
publishDate: "2026-02-16"
date: "2026-02-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "hardware", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2023222494312935523-1.jpg"
excerpt: "A claim that a large AI assistant was rewritten in Go to run on a $9.9 board with under 10MB of memory."
---

> Source: [@BoWang87 on X](https://x.com/BoWang87/status/2023222494312935523) · posted 2026-02-16

![Agent Rewritten in Go for Tiny Boards (1)](/imgs/blogs/x-bookmarks/2023222494312935523-1.jpg)

![Agent Rewritten in Go for Tiny Boards (2)](/imgs/blogs/x-bookmarks/2023222494312935523-2.jpg)

## Summary

The author says a Chinese hardware team rewrote a 430,000-line assistant that needs a $599 Mac Mini and 1GB of RAM in Go. He claims it now runs on a $9.9 dev board with under 10MB of memory and boots in 1 second instead of 500 seconds.

## Background

An agent that calls remote model APIs does little local compute, so a lean runtime can fit on tiny hardware. Go compiles to small, fast binaries. The numbers here are the author's claims, and other posts in the batch cite similar figures.

## What I took away

- The agent shell can be much smaller than the model it calls.
- Check the repo before trusting the headline numbers.

## Reference

- Original post: [https://x.com/BoWang87/status/2023222494312935523](https://x.com/BoWang87/status/2023222494312935523)
- Author: [@BoWang87](https://x.com/BoWang87) on X, posted 2026-02-16
