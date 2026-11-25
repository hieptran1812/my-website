---
title: "How Events Get Lost In Event-Driven Systems"
publishDate: "2025-12-27"
date: "2025-12-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "career"]
author: "Hiep Tran"
featured: false
excerpt: "A post explains that event-driven systems can lose data, starting with problems on the publisher side."
---

> Source: [@RaulJuncoV on X](https://x.com/RaulJuncoV/status/2004906272131612917) · posted 2025-12-27

## Summary

The author likes asynchronous communication because it is scalable, decoupled and resilient, but says losing data is their top concern. Event-driven systems look simple: one service emits an event and another reacts. The post then lists ways data slips through, starting with the publisher.

## Background

A common failure is writing to a database and publishing an event as two separate steps, so one can succeed and the other fail. Patterns such as the transactional outbox are used to make this safe. The post is cut off, so the full list is at the link.

## What I took away

- Assume events can be lost, duplicated or delayed.
- Design publishers so that the data write and the event stay consistent.

## Reference

- Original post: [https://x.com/RaulJuncoV/status/2004906272131612917](https://x.com/RaulJuncoV/status/2004906272131612917)
- Author: [@RaulJuncoV](https://x.com/RaulJuncoV) on X, posted 2025-12-27
