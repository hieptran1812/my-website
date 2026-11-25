---
title: "Goldfish Memory In Long-Running Agents"
publishDate: "2025-11-26"
date: "2025-11-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "harness-engineering"]
author: "Hiep Tran"
featured: false
excerpt: "A humorous post explains why agents forget work across context windows, quoting Anthropic's harness post."
---

> Source: [@TheAhmadOsman on X](https://x.com/TheAhmadOsman/status/1993756423684264148) · posted 2025-11-26

## Summary

The author jokes in a greentext style that Anthropic engineers realized long-running agents have goldfish memory. Each new context window is like a new intern who forgot everything, so a project like building a ChatGPT clone ends up with half the frontend missing. The quoted post is Anthropic's engineering blog about harnesses for long-running agents.

## Background

A context window is the amount of text a model can see at once. When a task runs longer than that, the agent starts a fresh window and loses earlier details unless a harness saves progress. A harness is the code around the model that manages tools, memory and state.

## What I took away

- Agents lose memory when a new context window starts.
- A good harness saves progress notes so new sessions can pick up.
- Read Anthropic's post for the real techniques.

## Reference

- Original post: [https://x.com/TheAhmadOsman/status/1993756423684264148](https://x.com/TheAhmadOsman/status/1993756423684264148)
- Author: [@TheAhmadOsman](https://x.com/TheAhmadOsman) on X, posted 2025-11-26
