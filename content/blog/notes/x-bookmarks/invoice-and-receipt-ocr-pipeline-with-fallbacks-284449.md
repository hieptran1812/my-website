---
title: "Invoice And Receipt OCR Pipeline With Fallbacks"
publishDate: "2025-12-20"
date: "2025-12-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002345525174284449-1.jpg"
excerpt: "A developer describes an AI OCR pipeline for invoices that uses Mistral with Gemini fallbacks and validation checks."
---

> Source: [@pontusab on X](https://x.com/pontusab/status/2002345525174284449) · posted 2025-12-20

![Invoice And Receipt OCR Pipeline With Fallbacks](/imgs/blogs/x-bookmarks/2002345525174284449-1.jpg)

## Summary

The author calls it the best OCR extraction pipeline for invoices and receipts, built with an AI SDK. It uses Mistral 3.0 first, falls back to Gemini 3.0-flash and then Gemini 3.0-pro, and adds quality checks, math checks and validations.

## Background

OCR extracts text and fields from scanned documents. A fallback chain tries cheaper models first and escalates when output fails checks. Math checks, such as totals matching line items, catch errors that look plausible.

## What I took away

- Chain cheaper models with stronger fallbacks.
- Validate extracted data with simple arithmetic checks.
- Best is the author's claim; test on your own documents.

## Reference

- Original post: [https://x.com/pontusab/status/2002345525174284449](https://x.com/pontusab/status/2002345525174284449)
- Author: [@pontusab](https://x.com/pontusab) on X, posted 2025-12-20
