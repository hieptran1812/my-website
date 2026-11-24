---
title: "AI-Assisted Bank Hacker Exposed by Poor OPSEC"
publishDate: "2026-10-08"
date: "2026-10-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-safety", "industry-news"]
author: "Hiep Tran"
featured: false
excerpt: "A thread on how researchers traced a reported AI-assisted hacker of South Korean banks through his own mistakes."
---

> Source: [@jukan05 on X](https://x.com/jukan05/status/2108110513033093369) · posted 2026-10-08

## Summary

The author calls the hacker an amateur and lists his mistakes: he left directory listing on his own server, letting researchers identify him, and researchers found files like CLAUDE.md and read his Claude Code sessions. The quoted post says some of South Korea's biggest banks were hit by a cyberattack and, per a CrowdStrike report, it may have been done by one person using a stack including an open-source AI penetration tool called ARTEX and DeepSeek.

## Background

Directory listing is a web server setting that shows all files in a folder to anyone who visits. CLAUDE.md is a project instruction file used by Claude Code, so finding it can reveal how the tool was used.

## What I took away

- AI tools can let a single person run attacks that used to need a team.
- Agent session logs and config files are sensitive; protect them like credentials.
- Turn off directory listing on any server you run.

## Reference

- Original post: [https://x.com/jukan05/status/2108110513033093369](https://x.com/jukan05/status/2108110513033093369)
- Author: [@jukan05](https://x.com/jukan05) on X, posted 2026-10-08
