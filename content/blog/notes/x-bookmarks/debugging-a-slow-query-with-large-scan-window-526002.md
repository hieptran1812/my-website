---
title: "Debugging a Slow Query With Large Scan Window"
publishDate: "2026-03-05"
date: "2026-03-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A checklist for fixing slow database queries with a huge scan window."
---

> Source: [@ashoKumar89 on X](https://x.com/ashoKumar89/status/2029407237857526002) · posted 2026-03-05

## Summary

The post flags a huge scan window as a red flag and lists next steps: check indexes on the date column used in the filter, review the query plan to rule out a full table scan, add pagination or aggregation instead of raw rows, and partition the table by date if range queries are common.

## Background

A full table scan reads every row, which is slow on big tables. An index lets the database jump to matching rows. Partitioning splits a table by a key such as date so queries read less data.

## What I took away

- Check indexes on filter columns first.
- Read the query plan before optimizing.
- Aggregate or paginate instead of returning raw rows.

## Reference

- Original post: [https://x.com/ashoKumar89/status/2029407237857526002](https://x.com/ashoKumar89/status/2029407237857526002)
- Author: [@ashoKumar89](https://x.com/ashoKumar89) on X, posted 2026-03-05
