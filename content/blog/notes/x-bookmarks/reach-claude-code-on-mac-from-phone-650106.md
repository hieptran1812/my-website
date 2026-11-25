---
title: "Reach Claude Code On Mac From Phone"
publishDate: "2026-01-20"
date: "2026-01-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "ai-agents"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2013606749492650106-1.jpg"
excerpt: "A three-step setup using Tailscale, Remote Login and Termius to use Claude Code on a Mac from a phone."
---

> Source: [@majidmanzarpour on X](https://x.com/majidmanzarpour/status/2013606749492650106) · posted 2026-01-20

![Reach Claude Code On Mac From Phone](/imgs/blogs/x-bookmarks/2013606749492650106-1.jpg)

## Summary

The author gives three steps to access Claude Code on a Mac from a phone. Install Tailscale on both devices, enable Remote Login in macOS Sharing settings, and install Termius on the phone to connect over SSH. They add that this also lets you reach web projects.

## Background

Tailscale creates a private network between your devices. Remote Login turns on an SSH server on the Mac. Termius is an SSH client for the phone. Together they let you open a terminal on your home machine from anywhere without exposing it to the public internet.

## What I took away

- Tailscale plus SSH is a simple, private way to reach your own machine remotely.
- Only enable Remote Login if you also secure the account with keys or strong passwords.
- Pair this with tmux so sessions survive disconnects.

## Reference

- Original post: [https://x.com/majidmanzarpour/status/2013606749492650106](https://x.com/majidmanzarpour/status/2013606749492650106)
- Author: [@majidmanzarpour](https://x.com/majidmanzarpour) on X, posted 2026-01-20
