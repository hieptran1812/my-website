---
title: "Open Source Rust PDF Parser pdf-inspector"
publishDate: "2026-08-02"
date: "2026-08-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "ai-agents"]
author: "Hiep Tran"
featured: false
excerpt: "Firecrawl open sources pdf-inspector, a fast Rust PDF parser that classifies PDFs and extracts markdown locally so agents avoid waiting on OCR."
---

> Source: [@firecrawl on X](https://x.com/firecrawl/status/2083759818133688517) · posted 2026-08-02

## Summary

Firecrawl says it open sourced **pdf-inspector**, which powers its /parse endpoint together with custom OCR models, and claims 0.002s per page. The quoted post says the tool classifies any PDF in about 20ms, extracts clean markdown locally, processed 200 PDFs in 2.8s, handles tables and graphs well, and is written in Rust.

## Background

OCR (optical character recognition) turns images of text into text and is slow. Many PDFs already contain a real text layer, so a tool that first detects the PDF type can skip OCR when it is not needed. This matters for AI agents that read many documents and need fast, clean input.

## What I took away

- Classify documents first and only run expensive OCR when the PDF has no usable text layer.
- Speed numbers are the vendor's claims; test on your own document mix.
- Clean markdown output is a useful format for feeding documents to LLMs.

## Reference

- Original post: [https://x.com/firecrawl/status/2083759818133688517](https://x.com/firecrawl/status/2083759818133688517)
- Author: [@firecrawl](https://x.com/firecrawl) on X, posted 2026-08-02
