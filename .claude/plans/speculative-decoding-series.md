# Speculative Decoding Series — Roadmap

Series: "Speculative Decoding: Speed Up LLMs Without Losing Quality"
Folder: content/blog/machine-learning/speculative-decoding/ (NEW)
Kit: .cache/blog-writer/_speculative-decoding-series-kit.md

## Posts

| # | Slug | Status |
|---|------|--------|
| 1 | why-llms-are-slow-autoregressive-bottleneck | TODO |
| 2 | speculative-decoding-core-idea-draft-and-verify | TODO |
| 3 | speculative-decoding-token-acceptance-rejection-sampling | TODO |
| 4 | draft-models-for-speculative-decoding | TODO |
| 5 | medusa-multi-head-speculative-decoding | TODO |
| 6 | eagle-speculative-decoding-feature-alignment | TODO |
| 7 | tree-speculation-drafting-multiple-futures | TODO |
| 8 | speculative-decoding-in-production | TODO |

## Wave recipe

For each wave (or all 8 at once):
1. Spawn agents to write .md + .dsl.json files
2. Main session: for each slug:
   a. mkdir -p .cache/blog-writer/<slug>/
   b. layout-scene.mjs each *.dsl.json → *.scene.json
   c. Build manifest.json
   d. render-scene-batch.mjs manifest.json
   e. cwebp each *.png → public/imgs/blogs/<slug>-N.webp
   f. verify-post.sh <post.md> <slug> deep-dive
3. Commit only wave files: git add content/... public/imgs/... && git commit
4. Push to main
