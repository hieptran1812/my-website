---
title: "Designing A Multi-Tenant Notification System"
publishDate: "2025-11-22"
date: "2025-11-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "career", "serving", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A backend interview prompt about designing a multi-tenant system that sends email, SMS and push at scale."
---

> Source: [@Kaperskyguru on X](https://x.com/Kaperskyguru/status/1992193450134151395) · posted 2025-11-22

## Summary

The post sets a backend interview question: design a multi-tenant notification delivery system that handles email, SMS and push at scale. The author says they will show how to approach it.

## Background

Multi-tenant means many customers share the same system while their data and limits stay separate. Notification systems usually need queues, retries, rate limits and per-channel providers. The post only introduces the question.

## What I took away

- Think about queues, retries and per-tenant rate limits.
- Separate channels like email, SMS and push behind one interface.
- Clarify requirements before designing in an interview.

## Reference

- Original post: [https://x.com/Kaperskyguru/status/1992193450134151395](https://x.com/Kaperskyguru/status/1992193450134151395)
- Author: [@Kaperskyguru](https://x.com/Kaperskyguru) on X, posted 2025-11-22
