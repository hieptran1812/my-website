---
title: "BandPeak"
excerpt: "An AI-powered IELTS self-study platform: instant examiner-grade band scores, 1,000 full mock exams and a deep practice library across all four skills."
description: "BandPeak is a production web platform that lets learners prepare for IELTS entirely on their own. It grades speaking from your voice and essays from your draft with AI, runs full four-skill mock exams, teaches vocabulary, grammar and pronunciation through visual lessons, and wraps the whole loop in streaks, XP and an adaptive study plan — for a fraction of a tutor's cost."
category: "AI Agent"
subcategory: "Education"
technologies:
  [
    "Next.js",
    "React",
    "Large language model",
    "Whisper ASR",
    "Speech scoring",
    "Spaced repetition (SM-2)",
    "AI Agent",
  ]
status: "Production"
featured: true
publishDate: "2026-08-03"
lastUpdated: "2026-08-03"
githubUrl: null
liveUrl: "https://bandpeak.com"
image: "/imgs/projects/bandpeak.webp"
highlights:
  - "12,400+ learners worldwide"
  - "1,000 full-length mock exams"
  - "AI band scores for speaking & writing"
  - "6,254 words, 160+ grammar lessons"
difficulty: "Advanced"
---

# BandPeak

**Self-study IELTS with AI.** BandPeak is an all-in-one IELTS coach for people preparing without a tutor. It covers all four skills — reading, writing, listening and speaking — grades your work the way an examiner would, and turns daily practice into a habit you actually keep.

Live at [bandpeak.com](https://bandpeak.com) · 12,400+ learners preparing worldwide.

## The Problem

IELTS preparation has a structural gap. The two things that move your band most — speaking and writing — are exactly the two that a book, an app or a question bank cannot grade. So learners either pay a tutor by the hour for feedback, or they practise into a void: they write fifty essays and never learn which ones would have scored a 6.5 and why.

BandPeak closes that loop. Every productive-skill attempt gets a band score and criterion-level feedback within seconds, so the practice you already do starts telling you something.

## Core Product

### Instant, examiner-grade feedback

- **Speaking, graded from your voice** — answer real Part 1/2/3 cue cards out loud. Whisper transcribes the recording and the AI returns a band plus feedback on fluency, pronunciation, lexical resource and grammatical range.
- **An AI writing coach** — plan, draft and polish Task 1 and Task 2 essays with a coach that suggests the next word, tightens coherence, and explains every fix in plain language rather than just marking it wrong.
- **Pronunciation scoring** — each lesson teaches one sound, stress pattern or intonation contour, then records your attempt and scores it so you know precisely what to correct.

### Full mock exams

A bank of **1,000 full-length papers** — 700 Academic and 300 General Training — each run end to end and graded:

| Skill | How it runs | Grading |
| --- | --- | --- |
| Listening | 40 questions at exam pace, audio played once | Auto-marked on submit |
| Reading | 40 questions on original, exam-grade passages | Marked objectively server-side |
| Writing | Task 1 report + Task 2 essay | AI-graded on all four criteria |
| Speaking | Three parts recorded from your voice | AI-graded on fluency and range |

The point is not just the question count — it is that a learner can sit a complete, timed, four-skill exam and get a whole-paper result without another human being involved.

### The fundamentals, taught visually

- **Vocabulary that sticks** — 6,254 essential words across 84 topics, ranked by frequency so the most useful words come first, then locked in with spaced repetition so they reach long-term memory.
- **Grammar made visual** — ~160 illustrated lessons across 16 sections that turn rules into patterns you can see, with auto-graded practice and an AI tutor available inside the lesson.
- **Pronunciation** — 61 lessons covering individual sounds, stress and intonation.

### Practice built for variety

- **Reflex drills** — 2,140 fast, gamified micro-drills across 30 types and five CEFR levels, adapting to your target band.
- **Spoken AI roleplay** — multi-turn conversations with an AI partner across 500 scenarios in 25 everyday situations, with the whole exchange graded afterwards.
- **Dictation** — type what you hear, sentence by sentence, from the built-in audio bank or by pasting any YouTube link that has an English transcript.
- **Reading & listening** — exam-grade original passages with strategy hints and instant scoring, plus real-audio listening exercises that train the ear.

## Peak — the study buddy

**Peak** is an IELTS coach in chat that is grounded in the learner's own data rather than answering in a vacuum. It reads your current band, your study plan, your streak and your weak spots, and uses them:

- Answers questions from your actual progress, not generic advice
- Explains grammar and vocabulary in plain language, and fixes slips as they happen mid-conversation
- Points you to the exact lesson or drill to practise next

This is the difference between a chatbot bolted onto a course and a tutor who remembers you.

## A plan that fits your week

You give BandPeak a target band and the free time you actually have. It lays out a weekly schedule you can tick off — *Mon: reading, timed passage, 20m · Tue: listening, section 3, 20m · Wed: writing, Task 2 essay, 30m* — and a calendar that logs your notes and scores automatically as you go, so progress is recorded without any bookkeeping on your part.

## Motivation as a first-class feature

The hardest part of exam prep is showing up on day 19. BandPeak treats retention as a product problem:

- **Timezone-aware streaks** and a heatmap of real, measured study time
- **XP and levels** — everything you do earns XP, turning steady practice into visible progress
- **29 trophies** that unlock automatically on streaks, levels and band milestones
- **Daily challenges** — a fresh set of quick tasks each day
- **Word Vault** — save any word and review it on an SM-2 spaced-repetition schedule
- **Word of the Day** — a new word and a bite-size tip on the dashboard each morning

## Content library at a glance

| Volume | Content |
| --- | --- |
| 1,000 | Mock exams — full four-skill papers |
| 6,254 | Essential words across 84 topics |
| 800+ | Speaking questions — Parts 1, 2 & 3 |
| 500 | Roleplay conversation scenarios |
| 2,140 | Reflex drills — 30 types, 5 levels |
| 160+ | Visual grammar lessons, 16 sections |
| 61 | Pronunciation lessons, one per sound |
| Deep bank | Reading passages, listening audio, writing prompts |

All of it built, quality-checked and matched to a target band, so a learner never hits the end of the material.

## What Makes It Interesting Technically

Three problems sit at the centre of the build:

1. **Scoring reliably enough to be trusted.** A band score is only useful if it is stable — the same essay should not score 6.0 on Monday and 7.5 on Tuesday. Grading against the four public IELTS criteria, per criterion rather than as one holistic guess, is what makes the output actionable instead of decorative.
2. **The voice loop.** Speaking practice means capture → transcription → scoring → feedback, fast enough that the learner is still in the moment. Whisper handles transcription; pronunciation scoring works on the audio itself, not just the transcript, since a perfect transcript can come from badly pronounced speech.
3. **Grounding the tutor.** Peak is only worth having because it is fed the learner's band, plan and weak spots. Retrieving and injecting the right slice of a learner's history into each turn is what turns a general-purpose model into something that feels like *your* tutor.

## Positioning

Free to start, no credit card. The pitch is explicit: everything you need to level up your English, for a fraction of what a tutor costs. It is aimed at self-directed test-takers — people who are willing to do the work but cannot or would rather not pay by the hour for someone to mark it.

---

*BandPeak is an independent study platform and is not affiliated with, authorized by or endorsed by the owners of the IELTS test (British Council, IDP: IELTS Australia, and Cambridge University Press & Assessment). Band scores shown in the product are AI-generated practice estimates, not official IELTS results.*
