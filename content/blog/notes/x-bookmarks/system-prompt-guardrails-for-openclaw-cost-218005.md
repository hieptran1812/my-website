---
title: "System Prompt Guardrails For OpenClaw Cost"
publishDate: "2026-02-01"
date: "2026-02-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "harness-engineering", "ai-safety"]
author: "Hiep Tran"
featured: false
excerpt: "The author shares a system prompt meant to make OpenClaw more useful, cheaper, and safer."
---

> Source: [@alex_verem on X](https://x.com/alex_verem/status/2017982342854218005) · posted 2026-02-01

## Summary

The author offers a system prompt for **OpenClaw** that adds guardrails, cost awareness, and real utility. They say many people install it raw and are surprised when it burns $200 organizing a Downloads folder. The prompt itself is in the linked thread.

## Background

A system prompt is the standing instruction given to an LLM before any task. For agents that can run commands and call paid APIs, it is a place to set limits, such as spending caps and what needs confirmation. Without limits, an agent can loop and run up large bills.

## What I took away

- Give agents explicit cost limits and confirmation rules before turning them loose.
- Running an agent raw with broad access is a security and budget risk.
- Start with small, bounded tasks and watch spend.

## Reference

- Original post: [https://x.com/alex_verem/status/2017982342854218005](https://x.com/alex_verem/status/2017982342854218005)
- Author: [@alex_verem](https://x.com/alex_verem) on X, posted 2026-02-01
