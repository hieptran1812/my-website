---
title: "Spotify API With WebGL Texture Atlas"
publishDate: "2025-12-03"
date: "2025-12-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1996262339692114154-1.jpg"
excerpt: "A short note on a visual project using the Spotify API, WebGL and a canvas2D texture atlas."
---

> Source: [@highpfloat on X](https://x.com/highpfloat/status/1996262339692114154) · posted 2025-12-03

![Spotify API With WebGL Texture Atlas](/imgs/blogs/x-bookmarks/1996262339692114154-1.jpg)

## Summary

The author lists the pieces of a project: the Spotify API, WebGL, and a texture atlas built with canvas2D. No further detail is given in the text.

## Background

A texture atlas packs many small images into one large image so the GPU can draw them in a single pass. Drawing images to a canvas2D first is a simple way to build the atlas. This is a common trick for rendering many album covers quickly.

## What I took away

- Pack many small images into one texture to reduce draw calls.
- Canvas2D is a handy tool to build atlases in the browser.

## Reference

- Original post: [https://x.com/highpfloat/status/1996262339692114154](https://x.com/highpfloat/status/1996262339692114154)
- Author: [@highpfloat](https://x.com/highpfloat) on X, posted 2025-12-03
