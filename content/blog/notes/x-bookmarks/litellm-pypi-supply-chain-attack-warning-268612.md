---
title: "litellm PyPI Supply Chain Attack Warning"
publishDate: "2026-03-24"
date: "2026-03-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-safety", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "Karpathy warns that a compromised litellm release could steal credentials on install."
---

> Source: [@karpathy on X](https://x.com/karpathy/status/2036487306585268612) · posted 2026-03-24

## Summary

Karpathy calls it software horror: a simple pip install of litellm was enough to exfiltrate SSH keys, cloud credentials, Kubernetes configs, API keys and more. The quoted post says release 1.82.8 contained a .pth file that sends credentials to a remote server and self-replicates.

## Background

A supply chain attack compromises a dependency so installs run malicious code. A .pth file in Python runs on startup. Pin versions and rotate secrets if exposed.

## What I took away

- Pin and hash dependencies.
- Rotate credentials if you installed the bad version.
- Isolate dev environments from secrets.

## Reference

- Original post: [https://x.com/karpathy/status/2036487306585268612](https://x.com/karpathy/status/2036487306585268612)
- Author: [@karpathy](https://x.com/karpathy) on X, posted 2026-03-24
