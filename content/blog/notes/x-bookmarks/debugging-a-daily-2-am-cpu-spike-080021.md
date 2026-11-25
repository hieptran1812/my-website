---
title: "Debugging A Daily 2 AM CPU Spike"
publishDate: "2026-02-11"
date: "2026-02-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "learning-resources", "career"]
author: "Hiep Tran"
featured: false
excerpt: "An interview-style question about why CPU hits 100% for ten minutes every day at 2 AM."
---

> Source: [@0xlelouch_ on X](https://x.com/0xlelouch_/status/2021474807293080021) · posted 2026-02-11

## Summary

The post asks what could cause an otherwise healthy application to spike to 100% CPU every day at 2 AM for exactly 10 minutes, and how to investigate. It gives no answer in the text.

## Background

A spike at a fixed time usually points to something scheduled, such as a cron job, backup, log rotation, report, cache refresh or garbage collection. Correlating the time with schedulers and profiling the process at that moment helps find the cause.

## What I took away

- A regular schedule suggests a scheduled job, not random load.
- Check cron, backups, log rotation and batch jobs first.
- Capture a profile or top output during the spike to see which process is responsible.

## Reference

- Original post: [https://x.com/0xlelouch_/status/2021474807293080021](https://x.com/0xlelouch_/status/2021474807293080021)
- Author: [@0xlelouch_](https://x.com/0xlelouch_) on X, posted 2026-02-11
