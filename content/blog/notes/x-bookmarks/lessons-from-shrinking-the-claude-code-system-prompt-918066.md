---
title: "Lessons From Shrinking the Claude Code System Prompt"
publishDate: "2026-07-24"
date: "2026-07-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "harness-engineering", "ai-agents"]
author: "Hiep Tran"
featured: false
excerpt: "An Anthropic engineer says they removed about 80% of the Claude Code system prompt for newer models and shares lessons on prompts and skills."
---

> Source: [@trq212 on X](https://x.com/trq212/status/2080710971228918066) · posted 2026-07-24

## Summary

The author says they removed about **80% of the Claude Code system prompt** for their newest models. They share what they learned about writing system prompts, skills and CLAUDE.md files for these models. The post text does not list the lessons themselves.

## Background

A system prompt is the standing instruction given to a model before user input. Skills are packaged instructions loaded when relevant, and CLAUDE.md is a project file that gives Claude Code context. Stronger models often need fewer explicit rules, and long prompts can add noise and cost.

## What I took away

- Revisit long prompts when upgrading models; many rules may be unneeded.
- Prefer loading context on demand through skills over one huge prompt.
- Read the linked write-up for the specific guidance.

## Reference

- Original post: [https://x.com/trq212/status/2080710971228918066](https://x.com/trq212/status/2080710971228918066)
- Author: [@trq212](https://x.com/trq212) on X, posted 2026-07-24
