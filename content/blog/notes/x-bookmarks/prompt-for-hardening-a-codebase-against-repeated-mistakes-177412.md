---
title: "Prompt for Hardening a Codebase Against Repeated Mistakes"
publishDate: "2026-11-22"
date: "2026-11-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "harness-engineering", "ai-agents", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "A post shares a prompt asking an agent to find repeated mistakes in a repo and make each one impossible."
---

> Source: [@poteto on X](https://x.com/poteto/status/2124525336994177412) · posted 2026-11-22

## Summary

The author says it is a good day to garden your codebase. The quoted post shares a prompt that tells an agent to read the git log, past bug fixes and review comments, find mistakes that agents and humans keep repeating, and make each one impossible instead of just warning about it.

## Background

Making a mistake impossible usually means adding a type, test, lint rule or check so it cannot happen. This is more reliable than documentation, especially when agents write code.

## What I took away

- Mine git history and reviews for recurring errors.
- Prefer guardrails that block mistakes over notes.
- Run it as periodic maintenance.

## Reference

- Original post: [https://x.com/poteto/status/2124525336994177412](https://x.com/poteto/status/2124525336994177412)
- Author: [@poteto](https://x.com/poteto) on X, posted 2026-11-22
