---
title: "LLM Routers Leaking Credentials and Injecting Tool Calls"
publishDate: "2026-09-10"
date: "2026-09-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-safety", "ai-agents"]
author: "Hiep Tran"
featured: false
excerpt: "Security research shows LLM routers stealing credentials and injecting malicious tool calls, with a follow-up claiming access via bought router data."
---

> Source: [@shoucccc on X](https://x.com/shoucccc/status/2098169782541631871) · posted 2026-09-10

## Summary

The quoted post says 26 LLM routers were secretly injecting malicious tool calls and stealing credentials, with one draining a client's $500k wallet, and that researchers could poison routers to take over about 400 hosts. The author claims they bought a dataset from a top Chinese LLM router. They say 6TB of data contained SSH keys, VPN configs, cloud keys and GitLab tokens that could compromise government entities and major firms.

## Background

An LLM router is a third-party service that forwards your API calls to various model providers. Everything you send, including secrets pasted into prompts or agent context, passes through it. A malicious router can read data or alter responses, such as adding harmful tool calls for agents to run.

## What I took away

- Never send secrets through untrusted LLM proxies or routers.
- Agents that run tool calls from model output need strict allowlists.
- Treat third-party API middlemen as part of your attack surface.

## Reference

- Original post: [https://x.com/shoucccc/status/2098169782541631871](https://x.com/shoucccc/status/2098169782541631871)
- Author: [@shoucccc](https://x.com/shoucccc) on X, posted 2026-09-10
