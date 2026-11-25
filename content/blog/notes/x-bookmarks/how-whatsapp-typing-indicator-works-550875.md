---
title: "How WhatsApp Typing Indicator Works"
publishDate: "2026-02-20"
date: "2026-02-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "learning-resources", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "Oblivious9021 poses an interview question on how typing appears in WhatsApp."
---

> Source: [@Oblivious9021 on X](https://x.com/Oblivious9021/status/2024708676171550875) · posted 2026-02-20

## Summary

The post asks how the typing indicator works and whether each key press is sent to the server.

## Background

Typing indicators are usually sent over a persistent connection like a WebSocket, and throttled or debounced to avoid a message per key press.

## What I took away

- Throttle ephemeral events.
- Use persistent connections for realtime.

## Reference

- Original post: [https://x.com/Oblivious9021/status/2024708676171550875](https://x.com/Oblivious9021/status/2024708676171550875)
- Author: [@Oblivious9021](https://x.com/Oblivious9021) on X, posted 2026-02-20
