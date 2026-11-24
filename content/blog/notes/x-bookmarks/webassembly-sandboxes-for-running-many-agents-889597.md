---
title: "WebAssembly Sandboxes for Running Many Agents"
publishDate: "2026-09-06"
date: "2026-09-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "tools", "harness-engineering"]
author: "Hiep Tran"
featured: false
excerpt: "Rivet built a WebAssembly and isolates-based Linux-compatible sandbox library, responding to the claim that CPUs are short for agents."
---

> Source: [@NathanFlurry on X](https://x.com/NathanFlurry/status/2096704470008889597) · posted 2026-09-06

## Summary

The quoted Cloudflare talk argues there are not enough CPUs to power all agents, so sandbox design needs a rethink using isolates. The author agrees and points to their open source project, a WebAssembly and isolates-based sandbox as a library that is Linux-compatible and POSIX-compliant. They say it is far more resource efficient and runs on more types of hardware.

## Background

Agents often need a sandbox to run code safely. Full VMs or containers use a lot of memory per agent. Isolates, used in V8 and Cloudflare Workers, are lightweight sandboxes inside one process, and WebAssembly is a portable, sandboxed binary format.

## What I took away

- Sandbox cost per agent matters when running many agents at once.
- Lightweight isolates can replace containers for some agent workloads.
- Check POSIX compatibility limits before porting tools.

## Reference

- Original post: [https://x.com/NathanFlurry/status/2096704470008889597](https://x.com/NathanFlurry/status/2096704470008889597)
- Author: [@NathanFlurry](https://x.com/NathanFlurry) on X, posted 2026-09-06
