---
title: "MirrorCode Benchmark Resists Cheating By Design"
publishDate: "2026-06-26"
date: "2026-06-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "ai-agents", "ai-safety"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2070647654887932315-1.jpg"
excerpt: "A quoted post describes MirrorCode, a sandboxed coding benchmark designed to block cheating and scorer hacking."
---

> Source: [@ArthurConmy on X](https://x.com/ArthurConmy/status/2070647654887932315) · posted 2026-06-26

![MirrorCode Benchmark Resists Cheating By Design](/imgs/blogs/x-bookmarks/2070647654887932315-1.jpg)

## Summary

@ArthurConmy is pleased that this benchmark actually prevents hacking. The quoted post says MirrorCode sandboxes AIs with no internet access, no access to the original source code, and no way to hack the scorer. Models never see held-out tests, so they cannot build a lookup table against the original program.

## Background

Benchmark hacking means a model scores well by exploiting the test setup instead of solving the task, for example by memorizing answers or reading the scorer. Sandboxing and hiding held-out tests close these routes. This matters because inflated scores mislead decisions about model ability.

## What I took away

- Hide held-out tests from the model while it develops.
- Remove internet access and scorer access when evaluating coding agents.
- A high score is only meaningful if cheating routes are closed.

## Reference

- Original post: [https://x.com/ArthurConmy/status/2070647654887932315](https://x.com/ArthurConmy/status/2070647654887932315)
- Author: [@ArthurConmy](https://x.com/ArthurConmy) on X, posted 2026-06-26
