---
title: "System Design Question Downloading The Whole Internet"
publishDate: "2025-11-29"
date: "2025-11-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "career", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "A system design interview question about how to crawl the whole internet for a search engine or chatbot."
---

> Source: [@ChShersh on X](https://x.com/ChShersh/status/1994787294637850684) · posted 2025-11-29

## Summary

The author poses an interview question: you are writing your own Google Search or ChatGPT and need to download the whole internet, so how would you do it? No answer is given in the post.

## Background

This is a web crawler design problem. Typical topics include a URL frontier queue, politeness to servers, deduplication, distributed workers, retries and storage. It is a classic system design topic.

## What I took away

- Think about queues, politeness and duplicates.
- Practice by sketching the architecture before looking up answers.

## Reference

- Original post: [https://x.com/ChShersh/status/1994787294637850684](https://x.com/ChShersh/status/1994787294637850684)
- Author: [@ChShersh](https://x.com/ChShersh) on X, posted 2025-11-29
