# Learn from Kaggle: Game-Agent Competitions (rules vs search vs RL)

**Subcategory:** `learn-from-kaggle` (EXISTING folder, renders "Learn from Kaggle"; holds the AIMO-2 solutions post)
**Path:** `content/blog/machine-learning/learn-from-kaggle/<slug>.md`
**frontmatter:** `category: "machine-learning"` · `subcategory: "Learn from Kaggle"`
**Size:** **24 posts, 5 tracks (A to E) + capstone, 4 waves of 6.** **STATUS: SAVED 2026-10-27 to run later (not started). Defaults below (3–4k competition posts, English) stand unless the user changes them; start with Wave 1 in a fresh session.**
**Language:** English (repo convention, verify gate enforces it). Request came in VN: *tổng hợp writeup cho từng cuộc thi, giải thích dễ hiểu, rút ra bài học.*
**Skill:** `blog-writer` (ML engine).

### Source of truth: the writeup banks (read only YOUR competition's H2)

- `.claude/plans/kaggle-rl-writeup-bank-part1.md`: ConnectX, Halite IV (+Playground), Google Research Football, Rock Paper Scissors, Santa 2020, Hungry Geese, Lux AI S1, cross-competition arc, gaps.
- `.claude/plans/kaggle-rl-writeup-bank-part2.md`: Kore 2022 (+Beta), Lux AI S2 (+NeurIPS Stage 2, 2022 Beta), LLM 20 Questions, FIDE & Google Efficient Chess, Lux AI S3, PTCG AI Battle (+Playground), Kaggriculture, Orbit Wars, Maze Crawler, Game Arena note, synthesis.

Every rank, team name, score, team count, prize and technical claim in a post must come from the bank or from the writeup URL it cites. Drafting agents **Grep the bank for their H2 and read with offset/limit**. Never read both banks whole (15k words). Items marked UNVERIFIED in the bank are either dropped or stated as unverified in the prose; never upgraded to fact.

---

## Angle: the decade-long argument between hand-written rules and learned agents

Kaggle has run ~20 simulation competitions since 2020: you submit an agent, it plays everyone else's agent on a live ladder. The banks show a clean, under-told story:

> **2020 to early 2025: rules and search won almost everything** (Halite IV, Kore, Lux S2, Chess, 20 Questions, RPS, Santa), with deep RL winning only where the game handed it a gift (Football: one controlled player and dense shaping; Lux S1: one network for all units). **From Lux AI S3 (2025) onward, self-play RL, usually bootstrapped from imitation, won every prize-money competition** (Lux S3, Orbit Wars, PTCG, Kaggriculture). Small Playgrounds (Maze Crawler) still go to heuristics.

The series explains *why the line moved*: GPU-vectorised JAX environments, imitation bootstrap from replays, one-network-controls-all-units encodings, league training, and a recurring cast of people who carried lessons from one competition to the next (Tom Van de Wiele: rules win Halite after RL fails, then clone+RL for Football 3rd; LiamK/IsaiahP: 2nd Geese then 1st Lux S1 then Lux S3 2nd then Orbit Wars 1st; HandyRL authors: Football 5th then Geese 1st; flg: best RL in Lux S2 then PTCG 4th then Orbit Wars 6th).

Three things in every competition post, without exception:

1. **The game explained like a board game to a friend, then as an MDP.** State, action, reward, opponent, per-step time budget, submission size. Drawn first, formalised second.
2. **The scorecard and the winner, explained from intuition up.** Top placements with approach family and a Source link, then the 1st-place solution taken apart until a reader could rebuild a baby version. 2nd/3rd and the best contrasting entry (the best RL agent if rules won; the best rules agent if RL won).
3. **Lessons that transfer.** 4 to 6 lessons, each traceable to a sentence in a writeup, plus a short "If you enter a competition like this" checklist.

### Honesty rule (hard gate)

- No invented teams, ranks, scores, compute numbers or architectures. If the winner never published compute (WeKick) or deleted the post (Hungry Geese 1st), say so.
- Do not draw a figure of an architecture the bank marks UNVERIFIED as if it were known; label it "reconstructed from the 2nd-place writeup" or skip it.
- Code snippets are **illustrative toy versions** labelled as such, never presented as the team's code. Link the real repo when the bank has one.
- Do not embed Kaggle replay screenshots or writeup images (license unclear). Redraw boards and diagrams in Excalidraw.

---

## Gates

- **Competition posts (Tracks B to D): 3,000 to 4,000 prose words** (`WORD_MIN=3000 WORD_MAX=4000`, same override as the CVPR/interp series; a 2–3k post cannot cover a top-3 plus a deep dive). **Technique posts (Track E) and the capstone: default 2,000 to 3,000.**
- Figures: **>= 5 floor, target 6**, >= 1/2 pictorial (boards, unit maps, replay-style frames, payoff curves), box-flow <= 1/3 and max 2 (house rule since 2026-10-16). 0 or 1 animated figure where motion is the point (a self-play league ladder, a unit swarm under one policy, a minimax tree pruning).
- Verify: `WORD_MIN=3000 WORD_MAX=4000 bash .claude/skills/blog-writer/scripts/verify-post.sh <post.md> <slug> deep-dive` via `post-verifier`.
- No em dash anywhere (no `—`, no spaced ` – `, no ` -- `).
- LaTeX traps: brace-wrap inline math starting with a digit; `\lt ` not bare `<letter`.
- Renderer lives outside the repo (`/Users/hieptran1812/Documents/mcp_excalidraw/scripts/`); count webp files, never trust a green sharpness gate.
- Check the slug is not already on disk before the figure pipeline.
- Cross-link the existing RL series in `machine-learning/reinforcement-learning/` (PPO, self-play / multi-agent, MCTS / MuZero, reward shaping, credit assignment, imitation / offline RL posts) instead of re-explaining an algorithm from zero.

### Recurring visual language (define in A1, reuse everywhere)

1. **The approach strip.** Five boxes `Rules | Search | Imitation | RL self-play | Hybrid`, the winner's family lit. It opens every competition post, so by post 16 the reader has watched the light move from left to right.
2. **The era timeline.** 2020 to 2026, one dot per competition coloured by the winning family, with the current competition ringed.
3. **The game card.** A redrawn board/state with the action space and per-step time budget annotated, used as each post's first pictorial figure.

### Fixed post skeleton (competition posts)

1. TL;DR box (winner, family, the one lesson) + approach strip
2. The game in 60 seconds (game card figure)
3. What made it hard (the constraint that decided the competition)
4. Scorecard table: rank, team, family, key idea, Source
5. The winning solution, explained (intuition → mechanism → toy code)
6. Runners-up and the best contrasting approach
7. What did not work (from the writeups)
8. Lessons + "If you enter a competition like this" checklist
9. Sources (writeup and repo URLs from the bank)

---

## Track A: Orientation (1 post)

| # | Slug | Post |
|---|---|---|
| A1 | `kaggle-simulation-competitions-how-they-work` | How Kaggle simulation competitions work: `kaggle-environments`, the ladder and its rating (TrueSkill / skill rating), submission limits, replays as a dataset, why local win rate lies. Introduces the three recurring visuals and the series' central question. |

## Track B: Era 1, 2020 to 2021: the first experiments (7 posts)

| # | Slug | Competition | Winner's family | Hook |
|---|---|---|---|---|
| B1 | `kaggle-connectx-solutions` | ConnectX | Search / perfect solver | A solved game: why alpha-beta + opening book beats RL, and what the rolling leaderboard means |
| B2 | `kaggle-halite-iv-solutions` | Halite by Two Sigma (+Playground) | Rules | The winner spent a month on RL, then won with rules: per-unit credit assignment |
| B3 | `kaggle-google-research-football-solutions` | Google Research Football | Deep RL + league | WeKick, SaltyFish, Raw Beast: the competition RL was built for, and why |
| B4 | `kaggle-rock-paper-scissors-solutions` | Rock, Paper, Scissors | Bandit over an ensemble of bots | Opponent modelling: no Nash equilibrium wins a ladder of exploitable bots |
| B5 | `kaggle-santa-2020-candy-cane-solutions` | Santa 2020 Candy Cane | Supervised GBDT on replays | LightGBM 1521 vs RL 998: when the hidden state is recoverable from replays |
| B6 | `kaggle-hungry-geese-solutions` | Hungry Geese | RL self-play + search (HandyRL) | Lost by 1.3 rating points: RL priors vs BC + MCTS, with the 1st writeup deleted |
| B7 | `kaggle-lux-ai-season-1-solutions` | Lux AI S1 | Deep RL from scratch | One network for every unit, on a home PC with two GPUs |

## Track C: Era 2, 2022 to early 2025: rules strike back (4 posts)

| # | Slug | Competition | Winner's family | Hook |
|---|---|---|---|---|
| C1 | `kaggle-kore-2022-solutions` | Kore 2022 (+Beta) | Rules + route tree search | One Beta bot became the base of most of the top 5 |
| C2 | `kaggle-lux-ai-season-2-solutions` | Lux AI S2 (+NeurIPS Stage 2, Beta) | Rules + forward simulation | ry_andy_ wins twice; the best RL agent (flg) is 4th: why RL stalled |
| C3 | `kaggle-llm-20-questions-solutions` | LLM 20 Questions | Protocol + Bayesian search, LLM as component | The LLM was the weakest part: agreed protocols and information gain |
| C4 | `kaggle-efficient-chess-ai-solutions` | FIDE & Google Efficient Chess | Alpha-beta + tiny NNUE | Veteran engine authors under a 5 MB memory budget |

## Track D: Era 3, 2025 to 2026: RL takes over (5 posts)

| # | Slug | Competition | Winner's family | Hook |
|---|---|---|---|---|
| D1 | `kaggle-lux-ai-season-3-solutions` | Lux AI S3 (NeurIPS 2024) | Deep RL self-play | Flat Neurons and Frog Parade: what changed since S2 (JAX env, partial observability, 5-game matches) |
| D2 | `kaggle-orbit-wars-solutions` | Orbit Wars | Pure self-play RL, 200M params | Top 3 all RL: the scale moment |
| D3 | `kaggle-pokemon-tcg-ai-battle-solutions` | PTCG AI Battle (+Playground) | RL self-play | 6,807 teams, hidden information, deck building; all 8 Strategy winners used self-play. Note the open Playground (to Jan 2027) |
| D4 | `kaggle-kaggriculture-solutions` | Kaggriculture | Imitation → RL + rule patches | Markets, weather and an opponent; pure imitation 4th. **Re-check for the full 1st/2nd writeups before drafting** |
| D5 | `kaggle-maze-crawler-solutions` | Maze Crawler (Playground) | Heuristic BFS | The counterpoint: in a 2-month no-prize Playground, a BFS and a tie-break still win |

## Track E: Cross-cutting techniques (6 posts, 2–3k words)

Each technique post is a synthesis across competitions, drawing only on the banks.

| # | Slug | Post |
|---|---|---|
| E1 | `game-agents-imitation-learning-from-replays` | Behaviour cloning from top-team replays: the bootstrap that appears in Football, Geese, Lux S1/S3, Kaggriculture, Santa |
| E2 | `game-agents-self-play-leagues-and-opponent-pools` | Self-play, leagues, frozen-opponent pools, KL-to-previous-model: preventing strategy collapse |
| E3 | `game-agents-observation-encoding-multi-unit` | One network controls every unit: grid encodings, UNet/ResNet backbones, per-unit action heads |
| E4 | `game-agents-reward-design-and-credit-assignment` | Win/loss vs shaped rewards, and why per-unit credit assignment sank RL in Halite |
| E5 | `game-agents-search-under-a-time-budget` | Alpha-beta, MCTS, forward simulation and NNUE when you get one second per move |
| E6 | `game-agents-ladder-strategy-and-local-evaluation` | Reading the ladder: local arenas, rating noise, submission timing, opponent modelling (RPS, 20Q protocols) |

## Capstone (1 post)

| # | Slug | Post |
|---|---|---|
| F1 | `rules-search-or-rl-kaggle-game-agents-playbook` | The decade scorecard (all competitions on the era timeline), the recurring cast and what they carried forward, and a decision playbook: given a new game's properties (units, observability, step budget, env speed, replay availability, duration), which family to start with and when to switch |

---

## Waves (fresh session per wave; dispatch through `figure-author` / `post-verifier`)

| Wave | Posts | Bank sections |
|---|---|---|
| W1 | A1, B1, B2, B3, B4, B5 | part1 |
| W2 | B6, B7, C1, C2, C3, C4 | part1 tail + part2 §1–4 |
| W3 | D1, D2, D3, D4, D5, E1 | part2 §5–9 (re-check Kaggriculture + PTCG Tokyo final first) |
| W4 | E2, E3, E4, E5, E6, F1 | both synthesis sections + published posts' scorecards (Grep only) |

**Open gaps to re-check right before the relevant wave:** Kaggriculture full 1st-place writeup and anything from 2nd (Victor @ Tufa Labs); PTCG Tokyo final champion; code URLs for Lux S2 1st, Lux S3 1st, 20Q 2nd; Kore/Lux S3 per-step time limits; Lux S2 NeurIPS Stage 2 top-3 writeups. Hungry Geese 1st detail and WeKick compute are permanently unavailable: write around them.
