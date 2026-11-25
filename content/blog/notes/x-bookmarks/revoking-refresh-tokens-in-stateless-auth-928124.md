---
title: "Revoking Refresh Tokens in Stateless Auth"
publishDate: "2026-02-16"
date: "2026-02-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "An interview-style question about revoking 30-day refresh tokens after a phone is stolen."
---

> Source: [@mysticwillz on X](https://x.com/mysticwillz/status/2023267568643928124) · posted 2026-02-16

## Summary

The post asks how to revoke refresh tokens in a stateless auth system. The case: a remember-me token lasts 30 days, a phone is stolen, and the user revokes sessions, yet the thief's app keeps refreshing.

## Background

A refresh token is a long-lived credential used to get new short access tokens. Stateless means the server stores no session, so it cannot easily cancel one. Common fixes are a revocation list, token versioning or rotating refresh tokens with reuse detection.

## What I took away

- Pure stateless design cannot revoke; you need some server-side state.
- Consider token rotation, a deny list, or a per-user token version.
- Keep access tokens short-lived.

## Reference

- Original post: [https://x.com/mysticwillz/status/2023267568643928124](https://x.com/mysticwillz/status/2023267568643928124)
- Author: [@mysticwillz](https://x.com/mysticwillz) on X, posted 2026-02-16
