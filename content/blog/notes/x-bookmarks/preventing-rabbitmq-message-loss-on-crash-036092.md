---
title: "Preventing RabbitMQ Message Loss On Crash"
publishDate: "2026-02-17"
date: "2026-02-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "learning-resources", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "0xlelouch_ asks how to prevent message loss when RabbitMQ crashes."
---

> Source: [@0xlelouch_ on X](https://x.com/0xlelouch_/status/2023555585481036092) · posted 2026-02-17

## Summary

The interview scenario needs high availability and framed as a Shopify incident.

## Background

Typical answers: durable queues, persistent messages, publisher confirms, and mirrored or quorum queues across nodes.

## What I took away

- Use durable and replicated queues.
- Use confirms.

## Reference

- Original post: [https://x.com/0xlelouch_/status/2023555585481036092](https://x.com/0xlelouch_/status/2023555585481036092)
- Author: [@0xlelouch_](https://x.com/0xlelouch_) on X, posted 2026-02-17
