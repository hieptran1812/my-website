# Top-20 writeup addendum: brief for research agents

Series: "Learn from Kaggle: LLM competitions" (plan: `.claude/plans/kaggle-llm-competitions-series.md`). The user requires every competition post to **synthesise the writeups of the top 20 teams**, not just the top 3 to 5. The existing banks (`.claude/plans/kaggle-llm-writeup-bank-part{1..4}.md`) cover ranks 1 to 3 in depth and some of 4 to 7. Your job: extend each assigned competition to the full top 20.

All output in English. Never read anything in `content/blog`.

## Output: one file per competition

`.claude/plans/kaggle-llm-top20/<competition-slug>.md`, written section by section as you go so partial progress survives.

1. **Header:** competition title, slug, which bank and H2 already cover ranks 1 to 3 (do not repeat their deep detail; add only what the bank lacks).
2. **Top-20 table:** private rank, team, members, private score, public rank (shake-up), writeup URL or `no writeup found`, approach in one line. Ranks and scores from the leaderboard API, never from memory. Include ties exactly as the API gives them.
3. **Per-team synthesis, ranks 4 to 20** (and ranks 1 to 3 only for facts the bank is missing): for each team WITH a writeup, 150 to 400 words paraphrased closely from their own post: data (external/synthetic, sizes), retrieval if any, models (exact names and sizes), training recipe (LoRA rank/alpha, lr, epochs, max length, hardware) when stated, inference (quantisation, vLLM, TTA, time management), ensembling, CV scheme and its CV-LB correlation, what did NOT work, the one idea that distinguishes this team. Teams with no writeup: one line, plus any detail their members posted in comments or other threads (with URL).
4. **Technique frequency matrix:** rows = techniques/choices (e.g. backbone family, external data, synthetic data, LoRA vs full FT, distillation, quantisation, pseudo-labels, TTA, ensembling size, specific trick), columns = ranks 1 to 20, cells = Y / N / ? (unknown). Then a counted summary: "N of the M teams with writeups used X".
5. **Consensus vs divergence:** what nearly every top-20 team did (the table stakes), what separated the top 5 from 6 to 20, contradictory findings (team A says X helped, team B says X hurt) with both URLs.
6. **Aggregated "what did not work"**, de-duplicated, each item with the team(s) that reported it.
7. **Shake-up note:** which teams moved most between public and private, and why if any writeup explains it.
8. **UNVERIFIED** list. Never invent ranks, scores, names, hyperparameters.

## Method

kaggle.com pages are JS-rendered. Use Kaggle's internal JSON API through the helper:
`/Users/hieptran1812/Documents/my-website/.claude/scripts/kaggle-api.sh <Service/Method> '<json>'` (read its header comment for working calls).
- `competitions.CompetitionService/GetCompetition` `{"competitionName":"<slug>"}` gives id and forumId.
- `competitions.LeaderboardService/GetLeaderboard` `{"competitionId":<id>}` gives the final board (team names live in a separate array of the response).
- `discussions.DiscussionsService/GetTopicListByForumId` `{"forumId":<id>}` lists topics; page through (try `page`/`pageToken`/`pageSize`/`sortBy` fields; an error body names the expected fields) and collect every topic whose title matches place/solution/rank/writeup/"st |nd |rd |th ". Solution posts may also live under `/competitions/<slug>/writeups/<id>`; a leaderboard team entry may link its writeup.
- `discussions.DiscussionsService/GetForumTopicById` `{"forumTopicId":<id>}` gives the full markdown.
- Calls are occasionally flaky (empty output): retry. Save raw JSON under `/private/tmp/claude-501/kaggle-top20/<slug>/` and extract with `python3` scripts; never print big JSON into your context. Batch many calls into one shell script per competition rather than one call per turn.
- WebSearch/WebFetch for GitHub repos, blogs (Zenn/Qiita/Medium) and papers by top-20 teams when the Kaggle post is thin or missing.

## Final reply

At most 8 lines: files written, per competition "writeups found X/20", main gaps. Do not paste content.
