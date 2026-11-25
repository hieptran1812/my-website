---
title: "Shirudo Lets Agents Defend Linux With eBPF"
publishDate: "2026-11-23"
date: "2026-11-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "ai-safety", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2124980590717911527-1.jpg"
excerpt: "OpenAI introduces Shirudo, an open-source Linux runtime where agents write eBPF controls to shield systems from exploits."
---

> Source: [@OpenAI on X](https://x.com/OpenAI/status/2124980590717911527) · posted 2026-11-23

![Shirudo Lets Agents Defend Linux With eBPF](/imgs/blogs/x-bookmarks/2124980590717911527-1.jpg)

## Summary

OpenAI announces Shirudo, an open-source Linux runtime that lets agents defend systems without downtime. Agents write dynamic eBPF controls to shield systems from exploits, giving defenders more time to test and roll out patches. It was built by Cisco together with OpenAI.

## Background

eBPF lets small programs run safely inside the Linux kernel, which makes it useful for monitoring and blocking behavior. Using agents to write these controls could provide a quick temporary shield while a proper patch is prepared.

## What I took away

- Agents can act as fast stopgap defenders while patches are tested.
- eBPF is a common tool for runtime security controls.
- Automatically written kernel controls still need careful review.

## Reference

- Original post: [https://x.com/OpenAI/status/2124980590717911527](https://x.com/OpenAI/status/2124980590717911527)
- Author: [@OpenAI](https://x.com/OpenAI) on X, posted 2026-11-23
