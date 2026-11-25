---
title: "Cloudflare Async Stale-While-Revalidate Change"
publishDate: "2026-02-26"
date: "2026-02-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "Cloudflare made cache revalidation fully asynchronous so the first visitor does not wait for the origin."
---

> Source: [@kristianfreeman on X](https://x.com/kristianfreeman/status/2027086954525790618) · posted 2026-02-26

## Summary

The author recommends following the quoted account. The quoted post says Cloudflare cache revalidation is now fully asynchronous, so the first visitor no longer waits for an origin round-trip while the asset updates in the background.

## Background

Stale-while-revalidate serves a slightly old cached copy at once and refreshes it in the background. This lowers latency at the cost of brief staleness.

## What I took away

- Async revalidation removes origin wait from the first request.
- Check the changelog for your setup.

## Reference

- Original post: [https://x.com/kristianfreeman/status/2027086954525790618](https://x.com/kristianfreeman/status/2027086954525790618)
- Author: [@kristianfreeman](https://x.com/kristianfreeman) on X, posted 2026-02-26
