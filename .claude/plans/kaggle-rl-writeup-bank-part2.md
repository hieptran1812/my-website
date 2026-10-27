# Kaggle simulation competitions 2022 to 2026: verified writeup bank (part 2)

Compiled 2026-10-24. Scope: Kaggle simulation (agent-vs-agent) competitions launched from 2022 through Oct 2026.

## How this was verified (read first)

- **Primary source = Kaggle itself, rendered in a real browser.** Kaggle pages are JS-rendered and return empty to a plain fetch, so every leaderboard, overview page and writeup below was read in a logged-in Chrome session (claude-in-chrome), not from search snippets.
- **Complete list of Kaggle simulation competitions.** Taken from Kaggle's internal `ListCompetitions` API with `requireSimulations: true`, run from inside the page (active plus completed). This is the authoritative list behind "Other competitions discovered".
- **Team counts and prizes.** Taken from each competition's Overview sidebar ("Teams", "Prizes & Awards"). Where an organiser post gives a different number, both numbers are shown.
- **Ranks.** Taken from the final leaderboard. Ranks in writeup titles were checked against that leaderboard.
- **Code URLs.** Only listed when the URL appears as a link inside the writeup itself.
- **UNVERIFIED.** Marks anything not confirmed on a primary page. The reason is given each time.
- **Limits of the reading.** Writeups were read for their opening and technical sections, not always to the end. Kaggle rate-limited writeup fetches (`GetWriteUpBySlug`) for a while, which is why some 4th to 10th place writeups were checked by title only. Those are flagged "(title only)".

Approach-family legend: **RB** = rule-based / heuristic, **SR** = search (alpha-beta, BFS, forward simulation), **IL** = imitation learning / behaviour cloning, **RL** = deep RL self-play, **LLM** = LLM prompting, **HY** = hybrid.

---

## 1. Kore 2022

### Facts
- **URL:** https://www.kaggle.com/competitions/kore-2022
- **Dates:** Apr 13 to Jul 27, 2022.
  - Final submission deadline: Jul 12.
  - Final games ran Jul 13 to 27.
- **Teams:** 469 (537 participants, 2,261 entrants).
- **Prize:** $15,000, paid as $3,000 each to places 1 to 5.
- **Ranking:** Gaussian skill rating N(μ, σ²), μ0 = 600. Updates depend only on win/loss/draw, not on the margin. Up to 5 submissions per day.

### Rules
- **Board and goal.** 1v1 on a 21x21 board that wraps on both axes. Each player starts with 1 shipyard and 500 kore. Whoever holds the most kore after 400 turns wins, or whoever eliminates the opponent first.
- **Actions.** Each shipyard can either spawn ships (10 kore each) or launch a fleet with a flight-plan string (for example `N5E5S5W`).
  - The maximum plan length grows with fleet size: floor(2·ln(ships)) + 1.
  - A fleet of 50 or more ships can convert into a new shipyard.
- **Mining and combat.** Fleets mine ln(ships)/20 of a cell's kore per turn. Fleets that collide fight by ship count. Adjacent fleets damage each other. A large enough fleet captures an enemy shipyard.
- **Information.** Full information: every flight plan is visible to the opponent once launched.
- **Compute.** Per-step time limit and submission size limit: UNVERIFIED (not on the Overview page; the 1st place writeup mentions timing and memory problems).

### Top placements

**1st – Harm Buisman (RB + SR routing)**
- **Starting point:** the Kore Beta 1st place bot by egrehbbt (= aDg4b; see Section 9), ported by @realneuralnetwork. Harm added about 5,900 lines.
- **Board-state ledger:** for every cell and every future turn, the actual damage and the damage potential of both players.
- **Route search:** a tree search over almost all candidate routes. Its complexity is tuned to the time left, with heavy caching that persists between turns.
- **Defence:** when an attack cannot be held, an "abandon" pattern takes over.
- **Shipyard attacks:** fleets from several shipyards combine into distributed attacks.
- **Process:** the final rule change was version 127 of 146. All later versions only tuned parameters. He debugged with jmerle's visualiser.
- **Writeup:** https://www.kaggle.com/competitions/kore-2022/writeups/harm-buisman-1st-place-solution
- **Code:** none linked.

**2nd – basilisk1337 (RB)**
- **Base:** the same egrehbbt baseline.
- **Planning layer:** each action waits up to 5 moves when a better option is coming.
- **Main-shipyard strategy:** 55 to 85% of all ships are concentrated in one shipyard.
- **Endgame tactics:** `big_attack` (a 10 to 15 step macro-attack across shipyards) and `crash_ships` (attrition attacks).
- **Expansion:** fast, deliberately less safe.
- **Exploit:** in the final days, repeated tiny attacks triggered the baseline's `if incoming_hostile_fleets: continue` and froze opponents' first expansion. Worth about +150 rating.
- **Writeup:** https://www.kaggle.com/competitions/kore-2022/writeups/basilisk1337-2nd-place-solution-rule-based

**3rd – ex-Team Durrett (author bomac1) (RB + SR)**
- **Simulator:** projects the board 25 turns ahead, assuming nobody issues new orders. This works because launched flight plans are fixed and public.
- **Attack test:** estimates the opponent's defensive ships and the ships the opponent can still build before the attack lands. It attacks the opponent's *second shipyard* only when that estimate says the attack succeeds.
- **RL attempt:** spent about 3 weeks setting up RL. Quote: "many features can be computed analytically, so there seemed to be few that would benefit from using statistical methods such as reinforcement learning". Dropped after about a week of training.
- **Writeup:** https://www.kaggle.com/competitions/kore-2022/writeups/ex-team-durrett-3rd-place-solution-attack-the-oppo

**4th qihuaz and 5th 1 Musketeer**
- Both writeup titles say rule-based (title only).
- qihuaz commented on the 1st place writeup: "wanted to play with RL but soon switched to rule-based agents after discovering the Beta 1st place solution".
- Writeups: `.../writeups/qihuaz-4th-place-solution-rule-based` and `.../writeups/1-musketeer-1-musketeer-5th-place-solution-rules-b`.

**Notable RL solution in the top 10:** none found. The top-5 writeups are all rule-based.

### Who won and why
**Rule-based won.** The stated reasons are deterministic, fully observable dynamics: flight plans are public, so the future can be simulated analytically (3rd place). A strong public rule-based baseline from the Beta also let competitors iterate fast (1st and 4th place).

The 2nd place author regretted this outcome: "it's unfortunate that so many solutions were rule based".

### Transferable lessons
1. **When the future is computable, write the simulator first.** Fixed, public flight plans made a 25-turn lookahead accurate enough to decide attacks (3rd place).
2. **A strong open baseline sets the meta.** Places 1 to 4 all built on the Kore Beta winner's code (1st, 2nd and 3rd writeups).
3. **Treat the agent like a software product.** An issue list, version control and a replay visualiser drove the work; the final 19 versions only tuned parameters (1st place).
4. **Hand-written rules hide exploitable corner cases.** A single `continue` in shared baseline code was exploited across many top agents (2nd place, and the comment on the 1st place writeup).
5. **Rating convergence is noisy at the top.** The lead flipped daily in the final period (1st place).

### Sources
- https://www.kaggle.com/competitions/kore-2022/overview (rules, prizes, dates, team count)
- https://www.kaggle.com/competitions/kore-2022/leaderboard
- The three writeups linked above

---

## 2. Lux AI Season 2 (Kaggle main, NeurIPS Stage 2, 2022 Beta)

### 2a. Lux AI Season 2 (main Kaggle competition)

#### Facts
- **URL:** https://www.kaggle.com/competitions/lux-ai-season-2
- **Dates:** Jan 26 to May 9, 2023. Final submission deadline May 8.
- **Teams:** 646 (752 participants).
- **Prize:** $55,000.
  - $15k for 1st, $10k for 2nd, $5k each for 3rd to 8th.
  - Monthly "Sprint" awards.
  - QuantCo non-monetary awards for the best deep-RL and best supervised-DL solutions.
- **Ranking:** Kaggle skill rating. The NeurIPS page says "Trueskill".

#### Rules
- **Phases (from flg's writeup):** 1v1, zero-sum, on a 48x48 map.
  1. A simultaneous bidding phase decides who places first.
  2. Players take turns placing 2 to 5 factories.
  3. The main phase lasts 1,000 turns. It is made of 20 day/night cycles of 50 turns each.
- **Units and goal.** Factories refine ice into water and ore into metal, and build light or heavy robots. Robots dig rubble and resources. The player with more lichen at the end wins.
- **Power.** Power is the scarce resource. Robots execute action queues of up to 20 actions, and each queue update costs power.
- **Collisions.** Heavy robots destroy light ones. Self-collisions kill all but one of the colliding units.
- **Compute.** About 3 s per step: 2nd place writes "1000 steps of 3 seconds". The VMs are CPU-only and "mostly single-core" (4th place).
- **Submission size limit:** UNVERIFIED.

#### Top placements

**1st – ry_andy_ (RB + SR forward simulation)**
- **Forward simulation:** a full roll-out of his own (and future) units for about 2.9 s per call, which gives a 5 to 50+ step plan.
- **Roles and goals:** about 10 unit roles (antagonizer, pillager, blockade, cow, miner, ...), each running a goal state machine.
- **Action locking:** actions lock in a prioritised order across units.
- **Signature trick, "ice conflicts":** place a factory specifically to deny ice to an opponent's factory.
  - It won Sprint 2 and briefly broke matchmaking.
  - He then hid it until the final day.
  - It did not work against flg's RL agent, which used heavy robots to move ice flexibly.
- **Writeup:** https://www.kaggle.com/competitions/lux-ai-season-2/writeups/ry-andy-1st-place-solution
- **Code:** UNVERIFIED (no repository linked in the portion read).

**2nd – Tigga (RB)**
- **Bot design:** stateful rule-based bot written in TypeScript. Each robot gets a role, plus a resolver for collisions and swaps, plus a safety check that asks "can an enemy kill me before I get home?".
- **Economy:** "chain"-based, with resource couriers.
- **Bidding:** aggressive, with a final bid of 41 (others capped at 40).
- **Testing:** mostly self-play; some games took more than 20 minutes.
- **Writeup:** https://www.kaggle.com/competitions/lux-ai-season-2/writeups/tigga-yet-another-logic-bot

**3rd – SiestaGuru (RB + statistical parameter tuning)**
- **Structure:** C++ logic with about 1,500 parameters.
- **Tuning loop:**
  1. Randomise a subset of parameters in every self-play game.
  2. Log the outcome plus game statistics.
  3. Hill-climb each parameter using correlations, random forests or small neural nets.
- His framing: the method takes "some of the good parts of reinforcement learning".
- **Writeup:** https://www.kaggle.com/competitions/lux-ai-season-2/writeups/siestaguru-final-notes-and-parameter-tweaking

**4th – flg (RL, the notable RL solution)**
- **Action space:** 7 actor heads covering bid, spawn location, water amount, metal amount, factory actions, light units and heavy units. The unit heads are per-cell grid action spaces.
- **Observation:** 48x48x105.
- **Collisions:** moves that would cause a self-collision are cancelled.
- **Architecture search by imitation:** about 2,000 top-agent matches from Meta Kaggle (about 300 GB HDF5) were used only to choose the network shape. The winner was the "DoubleCone" net, which does much of its compute at 12x12.
- **Training:** self-play with V-trace, TD(λ), a teacher-KL term, and a teacher kept about 30M steps behind.
- **Rewards:** about 65M steps of heavily shaped rewards first. Then zero-sum rewards: game result, relative lichen, factories, units.
- **Rare events:** a dedicated "spawn actor" upsamples bidding and placement.
- **Main constraint:** CPU inference speed on Kaggle. A queue of 4 to 6 predicted steps was needed to save power, and that ruled out larger models.
- **Writeup:** https://www.kaggle.com/competitions/lux-ai-season-2/writeups/flg-flg-s-approach-deep-reinforcement-learning-wit

**Other top-10 entries (title only):**
- 5th Harm Buisman: no writeup found.
- 6th Philipp Kostuch: "pure logic approach".
- 7th skyramp: "IL-based solution".
- 8th loh-maa: approach not stated in the title.
- 10th Deimos: "Deimos's RL approach".

#### Who won and why
**Rule-based plus forward simulation won**, and the top 3 were all non-learning.

Why, from the writeups:
- **Inference cost.** The CPU-only, per-step time budget, together with the action-queue power cost, made large networks impractical at inference time (4th place).
- **Hidden strategic exploits.** A rules author can hold back a trick like "ice conflicts" until the last day (1st place).

RL was competitive: the best RL agent finished 4th, and its matchup cost the winner rating.

#### Transferable lessons
1. **Inference latency is a design constraint, not an afterthought.** flg picked his architecture for CPU speed and receptive field, and gave up on a larger model because of runtime (4th place).
2. **Upsample rare but decisive decisions.** Bidding and placement happen on about 1% of steps, so a dedicated actor generated more of them (4th place).
3. **Use imitation data for cheap architecture search,** even when the final agent is pure RL (4th place).
4. **Parameterise opinions, then let data tune them.** A middle path between rules and RL (3rd place).
5. **Withhold a strategic edge until the end.** Information leaks through public replays (1st place).

### 2b. Lux AI Season 2 – NeurIPS Stage 2

#### Facts
- **URL:** https://www.kaggle.com/competitions/lux-ai-season-2-neurips-stage-2
- **Dates:** Sep 21 to Nov 29, 2023.
- **Teams:** 64.
- **Prize:** swag (Lux merchandise for the top 4 not already prized), possible co-authorship on a NeurIPS 2024 benchmark paper, and $300 GCP credits for 30 participants.
- **Format:** 64x64 map. The organisers pushed large-scale RL through the Jax environment (Jux).

#### Final leaderboard
- 1st ry_andy_ (3442.3)
- 2nd SiestaGuru (3256.3)
- 3rd Bart von Meijenfeldt (1883.7)
- 4th aDg4b

**Writeups:**
- No top-3 writeup is attached on the Kaggle leaderboard.
- The only writeup attached is the 31st place "PPO using Jux" by Scott Goodfriend: PPO on JAX-vectorised Jux with 512 to 1,024 environments, curriculum 16x16, then 32x32, then 64x64, and heavy invalid-action masking.
  - Writeup: https://www.kaggle.com/competitions/lux-ai-season-2-neurips-stage-2/writeups/scott-goodfriend-ppo-using-jux-lux-ai-season-2-neu
  - Code: https://github.com/sgoodfriend/rl-algo-impls and https://github.com/sgoodfriend/jux
- NeurIPS 2023 "Presentation from Top Competitors" page: https://neurips.cc/virtual/2023/83972. The speakers and contents were not checked (UNVERIFIED).

**Who won and why:** the same two rule-based authors who finished 1st and 3rd in the main event took 1st and 2nd. The writeups give no new stated reason (UNVERIFIED beyond the leaderboard).

**Stage 1:** the Overview mentions a "NeurIPS Stage 1 competition". It does not appear in Kaggle's simulation list (UNVERIFIED where it was hosted).

### 2c. Lux AI 2022 – Beta (pre-season)
- **URL:** https://www.kaggle.com/competitions/lux-ai-2022-beta
- **Dates:** Oct 1 to Dec 22, 2022.
- **Teams:** 44.
- **Prize:** swag.
- **Top 3:** 1st loh-maa, 2nd DaveZwieback, 3rd paradite. 4th was Harm Buisman and 6th ry_andy_.
- **Writeups:** none attached.

### Sources
- https://www.kaggle.com/competitions/lux-ai-season-2/overview
- https://www.kaggle.com/competitions/lux-ai-season-2/leaderboard
- https://www.kaggle.com/competitions/lux-ai-season-2-neurips-stage-2/overview
- https://www.kaggle.com/competitions/lux-ai-season-2-neurips-stage-2/leaderboard
- https://www.kaggle.com/competitions/lux-ai-2022-beta/leaderboard
- The writeups linked above
- https://nips.cc/virtual/2023/competition/66593 (NeurIPS 2023 page, via search snippet)

---

## 3. LLM 20 Questions (2024)

### Facts
- **URL:** https://www.kaggle.com/competitions/llm-20-questions
- **Dates:** May 16 to Aug 30, 2024. Final submission deadline Aug 29.
- **Teams:** 832 (963 participants, 6,281 entrants).
- **Prize:** $50,000: $12k, $10k, $10k, $10k, $8k for places 1 to 5.
- **Ranking:** skill rating in a **cooperative 2v2** format. Your bot is randomly paired with a bot of similar rating; within each pair one bot asks and the other answers. Two active submissions per team.

### Rules
- **Game.** Each team has a questioner LLM and an answerer LLM. Over up to 20 rounds the questioner asks a yes/no question and makes a guess. The first correct guess wins for that pair. Correct guesses in the same round, or 20 rounds without one, give a tie.
- **Limits:**
  - Questions: 750 characters maximum.
  - Guesses: 100 characters maximum.
  - Time: 60 s per round, plus 300 s of overage for the whole game.
  - Any timeout ends the game. An answer other than yes/no loses.
- **Hardware:** 100 GB disk, 16 GB RAM, 1x T4 GPU.

### Top placements

**1st – c-number (HY: protocol + Bayesian search + LLM)**
- **Opening question:** asks "Is it Agent Alpha?" (an alphabetical-bisection handshake; see 3rd place).
- **Keyword prior:** 116,937 NLTK nouns, each weighted by:
  - word count;
  - English frequency;
  - a GPT-4o-mini "thing-ness" probability.

  Bisection then halves *probability mass*, not keyword count.
- **Fallback for non-alpha partners:** entropy-minimising question selection.
  - Candidates: about 35k top keywords × about 13k questions (26 letter questions, about 3k mined from winning public games, about 10k generated by GPT-4o-mini).
  - Answer probabilities p(keyword, question) are averaged over Llama-3-8B-Instruct, Phi-3-small and Gemma-7B.
  - The table was built with vLLM on a rented 8x RTX 4090; about $500 of compute.
- **Answerer:** Llama-3-8B for general questions; DeepSeek-Math for letter and counting questions, picked by a rule-based router.
- His second agent ranked at a level equivalent to 4th.
- **Writeup:** https://www.kaggle.com/competitions/llm-20-questions/writeups/c-number-1st-place-solution
- **Code:** none linked.

**2nd – Tricksy Hobbitses (HY)**
- **Search:** "dual alphabetic binary search" over 120k unigrams, filtered with LLM queries and ordered by frequency.
- **LLM extension:** proposes noun phrases from the identified first word.
- **Fallback:** a fully online Llama 3.1 8B (8-bit) asker and guesser with auto-summarised context.
- **Answerer:** regex and LLM extraction for spelling, list and alphabetical questions.
- **Result:** 89% keyword find rate in alpha matches.
- **Writeup:** https://www.kaggle.com/competitions/llm-20-questions/writeups/tricksy-hobbitses-2nd-place-solution-for-the-llm-2
- **Code:** linked as "Full code available here"; the URL was not captured (UNVERIFIED).

**3rd – agent alpha (loh-maa) (HY)**
- **Origin:** made the alphabetical-bisection approach public. He judged that open publication was the best defence against covert collusion.
- **Answerer:** a regex handler, then an alpha answerer, then Llama 3 8B with a 6-shot prompt.
- **Questioner:** Alpha bisection first, then "Omega" (an offline entropy-minimisation and Bayesian-update policy), then Llama as backup.
- **Writeup:** https://www.kaggle.com/competitions/llm-20-questions/writeups/agent-alpha-3rd-place-solution

**Other titles:** 4th Oxford Kaggle Club, 5th yukky_maru, 9th ISAKA Tsuyoshi, 10th Full Power Agent ("a 90% coverage...") – writeups exist (title only).

**Notable RL solution:** none found.

### Who won and why
**A protocol-and-search approach won, not LLM reasoning.** Because scoring was cooperative, answerers could agree on a machine-checkable alphabetical bisection, and that "finally dominated the top 20" (3rd place).

The 1st place agent won by modelling the private keyword distribution, which made each question cut probability mass rather than keyword count.

LLMs mattered mainly as answerers for non-protocol partners and for offline table-building.

### Transferable lessons
1. **In a cooperative game, the protocol is the strategy.** A shared handshake beat smarter natural-language play (1st and 3rd places).
2. **Put the prior on the hidden target, not on the candidate count.** Probability-weighted bisection beat uniform bisection (1st place: its first and 4th-equivalent agents differed only in this).
3. **Precompute the LLM, then do Bayesian search at play time.** A keyword × question probability table turns the LLM into a lookup (1st place, and 3rd place's "Omega").
4. **Route sub-tasks LLMs fail at to tools.** Letter and counting questions went to regex or a math model (1st and 3rd places).
5. **Under-specified rules invite exploits.** Disclosing one publicly can be the most robust play (3rd place).

### Sources
- https://www.kaggle.com/competitions/llm-20-questions/overview (incl. the `/overview/20-questions-rules` section)
- https://www.kaggle.com/competitions/llm-20-questions/leaderboard
- The writeups linked above

---

## 4. FIDE & Google Efficient Chess AI Challenge (2024 to 2025)

### Facts
- **URL:** https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge
- **Dates:** Nov 19, 2024 to Mar 7, 2025.
  - Final submission deadline: Feb 11, 2025.
  - Final games were extended to Mar 6.
- **Teams:** 1,120 (1,275 participants).
- **Host:** FIDE.
- **Prize:** $50,000: $15k, $10k, $10k, $10k, $5k for places 1 to 5.
- **Ranking:** Gaussian skill rating, μ0 = 600. The leaderboard was reset at the deadline.

### Rules
- **Format:** standard chess. Openings are drawn from a random set. Matches are played in colour-swapped pairs. Threefold repetition and the 50-move rule are automatic draws.
- **Time:** 10 s per game with a 0.1 s simple delay (unused time is not banked).
- **Constraints:**
  - 5 MiB RAM.
  - A single dedicated 2.20 GHz CPU core.
  - **64 KiB compressed submission size limit.**

### Top placements

**1st – linrock (SR + small NNUE)**
- **Engine:** "minifish", based on Cfish (a C port of Stockfish), stripped down to fit the size limit.
- **Network:** a tiny NNUE with standard 768 inputs, dual perspective, horizontal king mirroring and one hidden layer (the example shown has 64 neurons).
- **Compression tricks:**
  - transposing weight matrices for compressibility;
  - zeroing unused pawn and king weights;
  - comparing gzip, zopfli, bz2, xz and 7z output sizes.
- **Training (bullet):**
  - stage 1: 10B Stockfish-scored positions;
  - stage 2: 12B positions from Leela Chess Zero RL runs (T77/T79 data), with piece-count-balanced stochastic skipping.
- **Testing:** SPRT on a private fishtest instance.
- He says "much of the strength was hidden in the network weights".
- **Writeup:** https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/linrock-my-solution-cfish-nnue-data-1st
- **Code:** https://github.com/linrock/minifish

**2nd – Approvers (SR + NNUE)**
- **Team:** a Stockfish contributor and a hobby engine author.
- **Engine:** Cfish base with four short-time-control search features, including skipping odd root depths and time checks in quiescence search.
- **Network:** (768x1hm → 64)x1 → 1x8, SCReLU activation, 8 output buckets. Trained in 3 stages, then tuned with SPSA.
- **Size work:** switching from gcc to clang.
- **Testing:** about 20M games through OpenBench SPRT/SPSA.
- **Writeup:** https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/approvers-approvers-submission-2nd
- **Code:** https://github.com/peregrineshahin/Approvers

**3rd – "Fix the bugs?" (Andrew Grant, Kim Kahre; SR + NNUE)**
- **Early baseline:** a pruned version of Grant's Ethereal engine with a hand-crafted evaluation, which led for a long time.
- **Final engine:** "KaggleFish", a Cfish clone with a Grapheus-trained network.
- **Testing:** OpenBench. The 2nd place writeup cites their total as about 38M games.
- **Writeup:** https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/writeups/fix-the-bugs-fix-the-bugs-solution-write-up-3rd
- **Code:** https://github.com/AndyGrant/KaggleFish

**Other titles:** 4th nagiss, 5th A, 9th ymg_aq ("Cfish, simple SPSA") – writeups exist (title only).

**Notable RL:** no RL agent in the top 10. RL entered only indirectly: the 1st place network was trained on Leela Chess Zero self-play data.

### Who won and why
**Classical alpha-beta search plus a tiny NNUE won.** The top 3 were all professional or veteran engine developers.

The stated reasons:
- A small learned evaluation still beat hand-crafted evaluation by about 100 Elo under the 64 KiB cap (1st place).
- The rest of the edge came from engineering for this exact time control, memory and size limit, and from statistically rigorous testing (2nd place).

### Transferable lessons
1. **Under a hard size cap, data quality beats architecture.** Lc0 RL data plus weak-to-strong curriculum training mattered most (1st place).
2. **Compressibility is a model-design objective.** Weight layout and zeroing directly bought network capacity (1st place).
3. **Optimise for the deployment regime.** Short-time-control tweaks that hurt long time controls won here (2nd place).
4. **Use statistically sound A/B testing.** SPRT for decisions and SPSA for tuning, over tens of millions of games (2nd and 3rd places).
5. **Domain expertise compounds.** Mature tooling (OpenBench, fishtest, bullet, Grapheus) was the moat (all top 3).

### Sources
- https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/overview/description (rules, constraints, prizes, timeline)
- https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/leaderboard
- The writeups and GitHub repositories linked above
- https://www.fide.com/news/3312 (launch announcement, via search snippet)

---

## 5. NeurIPS 2024 – Lux AI Season 3

### Facts
- **URL:** https://www.kaggle.com/competitions/lux-ai-season-3
- **Dates:** Dec 10, 2024 to Mar 25, 2025.
  - Final submission deadline: Mar 24.
  - Final games ran Mar 10 to 24.
- **Teams:** 701 on the Overview page. The organiser's conclusion post says "704 teams and 800+ competitors"; both numbers are recorded.
- **Prize:** $50,000.
  - $10k for 1st, $8k for 2nd, $7k for 3rd, $5k each for 4th to 8th.
  - Monthly Sprint prizes.

### Rules (from 2nd place's summary)
- **Units and map:** each player has up to 16 ships on a 24x24 procedurally generated map.
- **Match structure:** a game is best-of-5 matches; first to 3 match wins takes the game. Each match lasts 100 steps and goes to whoever scores more relic-node points.
- **Information:** fog of war. Hidden game parameters are randomised per game but fixed within it, for example move cost, sap effectiveness and nebula drift.
- **Actions:** move, or "sap" (a ranged laser). Ships must manage energy and avoid asteroids and nebulae.
- **Compute:**
  - Submission size limit: 100 MB ("62MB shy of the 100MB submission file size limit", 2nd place).
  - Per-step time limit: UNVERIFIED. 1st place mentions an overtime budget.

### Top placements

**1st – Flat Neurons (TonyK, Kat-ies, Sergei Zhgirovski, Andrew Volchek) (RL)**
- **Training algorithm:** large-scale IMPALA with V-trace, UPGO and TD losses.
- **Network:**
  - more than 1,000 features per tile;
  - 24 residual blocks, then a ConvLSTM for memory, then 4 transformer blocks over the map;
  - a supervised head that predicts enemy positions under fog;
  - per-unit 15x15 patches feeding the movement head and the sap-target head.
- **Stabilisers:**
  - teacher-KL and teacher-baseline losses against a frozen best model;
  - a frozen opponent pool;
  - **dynamic reward scaling**, keeping returns in [-5, 5];
  - **dynamic entropy**, a target entropy that decays linearly.
- **Reward:** many shaped rewards were tried before settling on a sparse one: ±1 per match plus ±2.5 spread over point collection.
- **Compute:** the final model took about 1.5B environment steps over 3 to 4 days; more than 20B steps across all experiments; bf16 and `torch.compile`.
- **Test-time:** flip augmentation and greedy actions.
- **Leaderboard secrecy:** to avoid being copied by imitation learners, the submission played a weaker model 85% of the time and the strong one 15%. Later they injected noise into the logits. Win rates were tracked from logs.
- **Writeup:** https://www.kaggle.com/competitions/lux-ai-season-3/writeups/flat-neurons-1st-place-approach-by-flat-neurons
- **Code:** linked as "here"; the URL was not captured (UNVERIFIED).

**2nd – Frog Parade (Isaiah Pressman, Garrett M) (RL)**
- **Engine:** the rules engine and all feature engineering rewritten in Rust and tested test-first. Integration tests check parity with the official engine, and the tests caught bugs in the official engine itself.
- **Inferred features:** map symmetry, the location of point tiles, hidden rules, and the energy field, which can be fully deduced in more than 99% of observations.
- **Network:** a 10M-parameter squeeze-excitation ResNet with 10-frame history.
- **Training:** PPO with a teacher-KL term and a sparse win/loss reward. The value function sees both sides during training.
- **Compute:** about 300M steps over about 8 days on an RTX 3090 plus an RTX 2070 Super. The simulator runs at 110k steps/s.
- **Test-time:** stochastic sampling, which helped with blind saps, plus 3 symmetry augmentations.
- **Writeup:** https://www.kaggle.com/competitions/lux-ai-season-3/writeups/frog-parade-frog-parade-s-solution
- **Code:** https://github.com/IsaiahPressman/kaggle-lux-2024

**3rd – aDg4b (Kaggle user egrehbbt) (IL)**
- **Path:** abandoned a rule-based bot in January. Imitation learning beat it after "just a couple of hours of training".
- **Networks:** two UNets, one predicting each unit's action and one predicting the sap target.
- **Data:** replays of Frog Parade and Flat Neurons, using won matches only. His own fog/hidden-parameter tracker re-derived features from each replayed agent's point of view, and the map was mirrored so every game starts at (0, 0).
- **Final submission:** trained on Frog Parade data, then fine-tuned on Flat Neurons data.
- **Writeup:** https://www.kaggle.com/competitions/lux-ai-season-3/writeups/adg4b-imitation-learning-3rd-place-solution
- **Code:** https://github.com/w9PcJLyb/lux3-bot

**Other top-10 entries (title only):**
- 4th YumeNeko: imitation learning.
- 5th OneUpKiwi: xLSTM agent.
- 6th EcoBangBang: "yet another RL".
- 7th Vadasz.
- 8th Gregor Lied: imitation learning.
- 9th team-k.
- 10th Boey: "end-to-end JAX RL".

### Who won and why
**Deep RL self-play won 1st and 2nd.** Imitation of those two agents took 3rd.

The stated reasons:
- Partial observability, hidden per-game rules and many interacting units rewarded learned policies with memory and inference.
- Scale and stability tricks mattered: teacher KL, opponent pools, reward and entropy scheduling (1st and 2nd places).
- Public replays made imitation learning cheap and strong, hence the 1st place's anti-imitation testing scheme.

### Transferable lessons
1. **Make the environment fast and provably correct before training.** A Rust rewrite with parity tests gave 110k steps/s (2nd place).
2. **Infer hidden state explicitly, then hand it to the network.** Deduced energy fields and point tiles, or a supervised enemy-position head (2nd and 1st places).
3. **Stabilise long self-play.** Teacher KL, frozen opponent pools, and dynamic reward and entropy schedules (1st place).
4. **Imitation of the leaders is a strong, cheap baseline** when replays are public. Hide your edge while testing (3rd place; 1st place's 85/15 trick).
5. **Overly strict action masking can remove good moves.** Frog Parade's mask on blind saps cost them against Flat Neurons (2nd place).

### Sources
- https://www.kaggle.com/competitions/lux-ai-season-3/overview
- https://www.kaggle.com/competitions/lux-ai-season-3/leaderboard
- https://www.kaggle.com/competitions/lux-ai-season-3/discussion/569919 (organiser conclusion: Flat Neurons won, 704 teams)
- The writeups and GitHub repositories linked above
- https://neurips.cc/virtual/2024/competition/84798 (NeurIPS page, via search snippet)

---

## 6. The Pokémon Company – PTCG AI Battle Challenge (2026) and its Playground

The challenge has two linked Kaggle competitions plus an offline final. Prize money is attached to the **Strategy (Hackathon)** track, not to the Simulation leaderboard.

### Facts
**Simulation category**
- **URL:** https://www.kaggle.com/competitions/pokemon-tcg-ai-battle
- **Dates:** Jun 16 to Sep 1, 2026.
  - Final submission deadline: Aug 17.
  - Games continued until about Aug 31.
- **Teams:** 6,807 (8,085 participants).
- **Kaggle prize:** "Knowledge" only.
- **Ranking:** Gaussian skill rating, μ0 = 600. Only the latest 2 submissions count.

**Strategy category (hackathon)**
- **URL:** https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy
- **Dates:** Jun 16 to Sep 14, 2026.
- **Teams:** 938.
- **Prize:** $240,000 on the Kaggle page.
- **Evaluation:** a written report, judged holistically on simulation rank, deck construction, originality and explanation quality.

**Offline final ("Second Round")**
- **Qualification:** the top 8 of the Strategy category.
- **Format:** in person in Tokyo, best-of-3, same deck and code across a match, games played sequentially with logs from earlier games available.
- **Hardware:** an H100 80 GB, 256 GiB RAM and 16 vCPUs per team, with 30 min of thinking time per game (Kaggle discussion 732331).
- **Official prizes** (official site, via fetch and search): $30k to each Strategy top-8 team; $50k to the overall winner and $30k to the runner-up.

**Playground continuation**
- **URL:** https://www.kaggle.com/competitions/the-pokemon-company-ptcg-ai-battle-challenge-playground
- **Dates:** Sep 29, 2026 to Jan 8, 2027.
- **Prize:** swag.
- **Teams:** 496 when checked on Oct 24, 2026; still open.

### Rules
- **Game:** two agents play the Pokémon TCG on the "cabt" engine, a C++ simulator built for kaggle-environments.
- **Deck:** each agent brings a 60-card deck, chosen from a designated card list based on the Standard format.
- **Interface:** each turn the agent receives the game log, the visible board and a list of legal options, and returns option indices. The engine presents only legal moves.
- **Hidden information:** the opponent's hand, deck order and prizes, plus coin flips.
- **Win:** win the game under TCG rules.
- **Resources:** 2 vCPUs, 12.2 GiB RAM, 11.8 GiB disk, **197.7 MiB submission limit**.
- **Time:** a maximum of 10 minutes per player per match (official FAQ, via search snippet; UNVERIFIED on the Kaggle page itself).

### Final leaderboards
**Simulation leaderboard (top 5):**
1. Luca (1398.2)
2. palsystem
3. Unown Gradiant
4. flg
5. Petit Canard

**Strategy track winners (Kaggle "Winners" tab):**
1. Luca
2. Unown Gradiant
3. palsystem
4. flg
5. LumenLiquidity (7th in simulation)
6. KawattaTaido
7. やる気元気ミワハルキ (8th in simulation)
8. Petit Canard

### Top writeups (all in the Strategy track)

**1st – Luca, "Aim to Be a Pokémon Master" (RL)**
- **Network:** about 2.24M-parameter transformer.
  - Global token plus object tokens for cards and legal options.
  - Opponent deck and hand are inferred from the game history.
  - Multi-card choices are picked autoregressively.
- **Training:** PPO with GAE in 4 stages, narrowing from a broad top-50 deck pool to fixed-deck specialists.
- **Deck handling:**
  - decks clustered into archetypes so common archetypes do not dominate;
  - a local arena that scores checkpoint–deck pairs;
  - small card mutations to patch counter-matchups.
- **Final submission:** 2 Dragapult ex agents, chosen from the final day's metagame.
- **Infrastructure:** official C++ engine with batched calls, models in JAX; 4x RTX 5090 at 40k to 60k samples/s.
- **Writeup:** https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/writeups/new-writeup-1784257142000
- **Code:** https://github.com/yijieyuan/kaggle-pokemon-tcg

**2nd in simulation, 3rd in strategy – palsystem (RL + shallow SR features)**
- **Deck pool:** 3,341 distinct decks scraped from public episodes.
- **Generalist:** trained from random weights with PPO for about 12B steps, reward ±1. Opponents are the current policy 80% of the time and historical snapshots 20%.
- **Oracle critic:** a PerfectDou-style critic that sees hidden cards during training only.
- **Transition features:** each legal option is simulated with a limited breadth-first search through a modified official engine, and the results become features.
- **Deck choice:** the generalist ranked candidate decks against the top teams' decks. The pick was Crustle / Mega Kangaskhan ex, then fine-tuned into a 2.4M-parameter specialist.
- **Writeup:** https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/writeups/new-writeup-1788449799640

**3rd in simulation, 2nd in strategy – Unown Gradiant (RL)**
- **Network:** a single memoryless 4.4M-parameter pointer-attention model, plus an explicit tracker of revealed information. Card text is embedded with a text-embedding model and reduced by PCA.
- **Training:** PPO with terminal rewards on PufferLib's C/CUDA backend, with an oracle critic. The model plays about 500 decks.
- **Deck choice:** iterated round-robin selection against the top teams' decks.
- **Cost:** the final 6B-step run took about 4 days on a rented 8x RTX 3090 node for about $110; about $350 in total.
- **Incident:** a Muon optimizer bug collapsed a run after about 1B steps.
- **Tooling:** all code was written with Claude models.
- **Writeup:** https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/writeups/team-unown-gradiant-solution-all-decks-on-hand
- **Code:** https://github.com/dipamc/kaggle-ptcg-ai-battle

**Others in the Strategy track (title and subtitle only):**
- flg: "Teaching a Transformer to play Pokémon TCG through self-play".
- LumenLiquidity: "deck-agnostic policy trained by a league of its own exploiters".
- KawattaTaido: "self-play RL with Bayesian hidden-hand estimation and per-matchup policy switching".
- やる気元気ミワハルキ: "Generalist-to-Specialist PPO, 273 million games of GPU-accelerated self-play".
- Petit Canard: "behavioural cloning, a self-play league, per-matchup specialists, and a Gumbel search rebuilt for hidden information".
- Writeups are also attached in the Simulation track for 15th, 20th and 24th ("population-based RL ecosystem").

### Second Round (Tokyo) results
**UNVERIFIED / not found.** No champion is published on Kaggle or found via search as of Oct 24, 2026.

### Who won and why
**Deep RL self-play swept the top 8.** All eight Strategy winners describe PPO or self-play leagues, often combined with behaviour cloning or search.

The stated reasons:
- The organisers explicitly expected that "rule-based programming alone may not ensure a high ranking".
- The hosts released the official engine source on request (1st place credits this), which enabled massive self-play.
- Deck choice against the observed metagame was a co-equal lever. Every top-3 report devotes a major section to it.

### Transferable lessons
1. **In a hidden-information game, give the critic the hidden state during training only.** An "oracle critic" in the PerfectDou style (2nd and 3rd simulation places).
2. **Train a generalist first, then specialise.** Broad multi-deck self-play, then narrower fine-tunes (1st place; palsystem).
3. **Choose your "deck" (strategy) with your own policy as the evaluator** against the observed top-team meta (all top-3 reports).
4. **Cheap compute can be competitive.** Third place in simulation for about $350 of rented RTX 3090s with PufferLib (Unown Gradiant).
5. **RL bugs surface late.** Add KL and value-loss guards and long-run monitoring (Unown Gradiant's Muon bug).

### Sources
- https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/description (rules, resources, dates, team count)
- https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/leaderboard
- https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/732331 (Second Round format)
- https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy/hackathon-winners
- The writeups and GitHub repositories linked above
- https://ptcg-abc.pokemon.co.jp/?lang=en (official prize structure; the fetched copy was a pre-competition snapshot)
- https://www.oricon.co.jp/pressrelease/2995780/ (Rist press release: gold medals; via search snippet)

---

## 7. Kaggriculture (2026)

### Facts
- **URL:** https://www.kaggle.com/competitions/kaggriculture
- **Dates:** Jul 30 to Oct 15, 2026.
  - Final submission deadline: Sep 30.
  - Games continued until about Oct 15.
- **Teams:** 10,148 (11,337 participants, 21,345 registrations, 141 countries; organiser recap 751368).
- **Prize:** $50,000, paid as $5,000 to each of places 1 to 10.
- **Ranking:** skill rating during the competition, using only the latest 2 submissions. The **final leaderboard is a Bradley–Terry fit** over about 2 weeks of post-deadline games.

### Rules
- **Format:** two players on separate farms. A season is 30 days of 24 turns each, so 720 turns. The player with the most coins at the end wins; the size of the winning margin does not affect the rating.
- **What players do:**
  - plant, water, fertilise and harvest wheat, carrot, tomato, strawberry and melon;
  - buy and feed geese, cows and sheep (the animals eat wheat);
  - hire farm hands for the day;
  - buy 5x5 land quadrants;
  - trade on a **shared, price-reactive market**.
- **Failure modes:** unwatered plants become weeds after 2 days, and unfed animals escape after 2 days.
- **Compute:** 1 s per step (actTimeout) plus a 60 s overage budget per game. This is per the 3rd place writeup, which cites `kaggriculture.json` in kaggle-environments 1.32.7; not shown on the Overview page.
- **Submission size limit:** UNVERIFIED.

### Final top 5
1. M & M & P & Q
2. Victor @ Tufa Labs
3. DSM
4. Farmcore
5. 有辣条有权

### Top placements

**1st – M & M & P & Q (HY: IL + RL + heuristics)**
- **Writeup status:** only a pre-final "preview" exists (discussion 745073); the full writeup is promised.
- **Training loop:** behaviour cloning alternated with self-play PPO. The demonstrations come from public replays and from the team's own strong **heuristic planner**, which reached #1 on its own at one point.
- **Model lines:** parallel 10M and 20M models.
- **Final agent:** a 10M shared transformer for unit actions and market orders, plus a rule-based action patch at inference time. Its training lineage covers about 8.29M self-play games.
- **Preview:** https://www.kaggle.com/competitions/kaggriculture/discussion/745073
- **Code:** https://github.com/msdsm/kaggriculture-1st-solution
- **Final writeup:** UNVERIFIED / not yet published.

**2nd – Victor @ Tufa Labs:** no writeup or preview found. UNVERIFIED.

**3rd – DSM (HY: BC → PPO + post-processing)**
- **Network:** a 10.4M-parameter transformer over 266 tokens. It is end-to-end: it outputs every unit's move plus the market orders each step.
- **Initialisation:** behaviour cloning from the top teams' replays, replayed in their own C++ engine and dropped if not bit-exact. Later fine-tuned on the leading team's games.
- **Reward:** win/loss only.
- **What made PPO take off:** pre-warming the critic, clipping the ratio per decision, and an antisymmetric critic. League win rate rose from 0.359 to 0.542.
- **Compute:** up to 23 A100s; 20.1B positions over 1,883 generations. The final run was about 360M steps.
- **Post-processing:**
  - fix moves that provably have no effect, using the official rule functions;
  - a rule-timed late-game melon play.

  Together these took the win rate from 74.2% to 87.8%.
- **Writeup:** https://www.kaggle.com/competitions/kaggriculture/writeups/kaggriculture-3rd-place-dsm-solution
- **Code:** https://github.com/seven320/kaggriculture-3rd-place-solution

**4th – Farmcore (IL only)**
- **Key idea:** clone **what the engine executed**, not what the player ordered. Strong players over-order and the engine silently drops the excess.
  - Cloning raw orders: 3 of 24 games won against their previous best.
  - Cloning executed labels: 12 of 24.
  - Executed labels plus a "feed guard" rule: 23 of 24.
- **Network:** one-pass transformer that drafts the whole turn, plus a 6-layer corrector over the draft.
- **Training:** pretrain on 67 top submissions, fine-tune on one team, then average two fine-tunes.
- **Compute:** about 5.5 A100-hours end to end, and zero self-play. "Per GPU-hour, data and pretraining paid at least ten times more" than PPO.
- **Missed mechanic:** after the deadline they found the market fills orders by list position.
- **Writeup:** https://www.kaggle.com/competitions/kaggriculture/writeups/farmcore-a-one-pass-transformer-trained-on-what-t
- **Code:** https://github.com/B3tMy3x/farmcore-kaggriculture

**Other top-10 entries (title only):**
- 5th 有辣条有权: "It's SFT all the way down".
- 7th DECEM: "only behavior cloning".
- 9th: "From reinforcement learning to a final BC agent".
- 10th KawattaTaido: "BC + PPO on a single RTX 5090".

### Who won and why
**Hybrids won: behaviour cloning bootstrapped from replays, then self-play PPO, plus rule-based patches at inference.**

The stated reasons:
- The reward is sparse, arriving only after 720 steps.
- Each step bundles about 31 joint decisions.
- The shared market makes values depend on the opponent (3rd place).

These make pure RL from scratch hard, so cloning top replays is the efficient start. The top teams then needed self-play to go beyond the teachers, and hand rules to fix rare-state failures.

Pure imitation reached 4th place on a tiny budget.

### Transferable lessons
1. **Clone executed effects, not intended actions.** Logged orders can include silently dropped no-ops (4th place).
2. **Imitation leaves gaps in states the teacher never visited.** Patch them with a few mechanic-derived guard rules (4th place's feed guard; 1st and 3rd places' inference-time patches).
3. **Sparse-reward PPO needs critic engineering.** Pre-warming, per-decision ratio clipping and an antisymmetric critic (3rd place).
4. **Evaluate on paired seeds with both seats against a deterministic engine** to cut variance (3rd and 4th places).
5. **Budget reality check:** 5.5 A100-hours of imitation learning finished 4th, while the 1st and 3rd place lineages ran millions of self-play games. Data quality buys the first 90%, and scale buys the last few places (4th versus 1st and 3rd).

### Sources
- https://www.kaggle.com/competitions/kaggriculture/overview/description (rules, prizes, ranking, timeline)
- https://www.kaggle.com/competitions/kaggriculture/leaderboard
- https://www.kaggle.com/competitions/kaggriculture/discussion/751368 (organiser recap with counts)
- https://www.kaggle.com/competitions/kaggriculture/discussion/745073 (1st place preview)
- The writeups and GitHub repositories linked above

---

## 8. Orbit Wars (2026) — additional competition discovered

### Facts
- **URL:** https://www.kaggle.com/competitions/orbit-wars
- **Dates:** Apr 17 to Jul 8, 2026. Final submission deadline Jun 23.
- **Teams:** 4,729.
- **Prize:** $50,000, paid as $5k to each of places 1 to 10.
- **Ranking:** Gaussian skill rating, using only the latest 2 submissions.

### Rules
- **Board:** a remake of the 2010 Planet Wars challenge in continuous 100x100 space with a sun at the centre. Fleets that cross the sun are destroyed.
- **Planets and comets:** 20 to 40 planets, some orbiting the sun and some static, placed with 4-fold symmetry. Comets cross on elliptical paths.
- **Actions:** launch [planet, angle, ships]. Fleets fly in straight lines, and their speed grows with log(ships).
- **Format and win:** 500 turns, played as **2-player or 4-player** games. Whoever has the most ships at the end wins.
- **Compute:**
  - actTimeout of 1 s per turn plus 60 s overage.
  - 100 MiB submission limit (per the 1st place writeup).

### Top placements

**1st – Isaiah @ Tufa Labs (RL)**
- **Network:** one 200M-parameter transformer that plays both 2p and 4p. Inputs are entity tokens plus summary, plan and scratch tokens. The action space is target-based: each source picks a target by attention, and fleet size comes from a logistic mixture.
- **Training:** pure self-play PPO from scratch for 15B steps, with no imitation learning, and a KL anchor to the previous best checkpoint.
- **Environment:** rewritten in Rust and verified with replay parity tests.
- **Fitting the limits:**
  - 4-bit NormalFloat quantisation to fit inside 100 MiB, plus int8 inference;
  - a fallback 5M-parameter model when the overage budget runs low.
- **Compute:** about 2,400 B200-hours, finishing on four 8xB200 nodes.
- **Development:** fully agentic coding (Codex).
- **Thesis:** scaling the model beats feature engineering ("Bitter Lesson").
- **Writeup:** https://www.kaggle.com/competitions/orbit-wars/writeups/1st-place-solution-scaling-reinforcement-learnin
- **Code:** https://github.com/IsaiahPressman/kaggle-orbit-wars

**2nd – Hober Malloc (RL, after IL)**
- **Path:** heuristics reached the top 50. Imitation learning reached the top 10, RL fine-tuning the top 5, and RL from scratch for 10B steps was trained in the final days.
- **Network:** 4.3M-parameter ModernBERT over 1D-CNN embeddings of each body's future states for the next 19 steps, simulated with no new launches.
- **Action space:** per body, either do nothing or send everything to a nearby target.
- **Writeup:** https://www.kaggle.com/competitions/orbit-wars/writeups/2nd-place-solution-for-orbit-wars
- **Code:** https://github.com/SimJeg/orbit-wars

**3rd – Felix M Neumann (RL)**
- **Training:** pure self-play PPO with prioritised fictitious self-play (PFSP), no imitation learning.
- **Engineering:** environment and features written in JAX; 6.2M-parameter transformer.
- **"Reachability tensor":** ship count, launch angle and arrival time for every source, target and action combination.
- **Semantic actions:** send-all, sortie, hold and kill-at-arrival.
- **Biggest training knob:** the entropy schedule.
- **Writeup:** https://www.kaggle.com/competitions/orbit-wars/writeups/3rd-place-ab-in-den-orbit

**Others (title only):** 6th flg, "RL + league + search with a custom edge-attention"; 9th Boey, "end-to-end JAX PPO".

### Who won and why
**Deep RL won the top 3.**

- The 1st place states that the bitter-lesson bet paid off: "spending time scaling training ... worked better than ... a domain-specific, smaller-scale solution".
- The 3rd place found the opposite lever just as strong at small scale: semantic action design increased learning speed "by a lot".

### Transferable lessons
1. **Scale the model until the inference budget binds.** Then quantise (4-bit NormalFloat) and keep a fallback model (1st place).
2. **Action-space design is the cheapest speed-up.** Target selection instead of raw angles (1st place); semantic intents instead of ship fractions (3rd place).
3. **Precomputed rollouts of "no new action" futures are strong features** (2nd place's 19-step series; 3rd place's reachability tensor).
4. **Fast, parity-tested environment rewrites** in Rust or JAX were common to all of the top 3.

### Sources
- https://www.kaggle.com/competitions/orbit-wars/overview/description
- https://www.kaggle.com/competitions/orbit-wars/leaderboard
- The writeups and GitHub repositories linked above

---

## 9. Other Kaggle simulation competitions in scope (2022 to Oct 2026)

These come from Kaggle's `requireSimulations` competition list.

**Kore 2022 – Beta**
- **URL:** https://www.kaggle.com/competitions/kore-2022-beta
- **Dates:** Mar 17 to Apr 7, 2022.
- **Teams:** 58.
- **Prize:** swag.
- **Winner:** 1st aDg4b (= egrehbbt). Writeup: https://www.kaggle.com/competitions/kore-2022-beta/writeups/adg4b-1st-place-solution (not read). This bot became the base for most of the Kore 2022 top 5.
- **Runners-up:** 2nd More Kore!, 3rd MarkHaoxiang.

**Lux AI 2022 – Beta:** covered in Section 2c.

**Lux AI Season 2 – NeurIPS Stage 2:** covered in Section 2b.

**Maze Crawler (Playground simulation, 2026)**
- **URL:** https://www.kaggle.com/competitions/maze-crawler
- **Dates:** May 1 to Jul 1, 2026.
- **Teams:** 459.
- **Prize:** "Knowledge" (no money, no medals).
- **Game:** 1v1 on a 20x20 east/west-mirrored maze that scrolls north with fog of war. Each player has a factory plus scouts, workers and miners, and the last surviving factory wins.
- **1st – Maksim Savelev (RB + SR):** "a fully heuristic agent: no learning, no search tree".
  - A single scored first-move BFS (jump-aware) drives all movement.
  - Economy: go to mining nodes, and switch to workers when the scroll is close.
  - The deciding factor was winning the factory-collision energy tie-break.
  - Writeup: https://www.kaggle.com/competitions/maze-crawler/writeups/maze-crawler-1st-place-preliminary-solution-wr
- **Others:** 3rd Genematon has a writeup (not read). The 5th place title says "rules-based". 7th uses "Hungarian matching" for global tasks.

**Still running**
- The PTCG Playground (Section 6).
- **ConnectX:** a perpetual Getting Started simulation since 2020, outside the window.

**Not a game-agent competition**
- **"Autonomous Agent Prediction (Beta)"** (Playground, 2026, 570 teams): submit an agent that trains a model on its own. Listed for completeness.

**Kaggle Game Arena** (launched Aug 4, 2025)
- A **benchmark** in which frontier LLMs play games against each other (chess first, with an Aug 5–7 exhibition), ranked by all-play-all results.
- It is not a submission competition and has no team leaderboard to mine.
- Sources: https://blog.google/technology/ai/kaggle-game-arena/ (via search snippet) and https://siliconangle.com/2025/08/04/google-deepmind-host-ai-chess-tournament-evaluate-leading-ai-models-reasoning-skills/

---

## Cross-competition synthesis (for the series arc)

### Approach of the winner, competition by competition

| Competition | Year | Winner's approach | RL placing |
| --- | --- | --- | --- |
| Kore 2022 | 2022 | Rule-based | None in the top 5 |
| Lux S2 | 2023 | Rule-based + forward simulation | Best RL 4th |
| Lux S2 NeurIPS | 2023 | Rule-based | Leaderboard only |
| LLM 20Q | 2024 | Protocol + Bayesian search, LLM as a component | None |
| Chess | 2025 | Alpha-beta + NNUE | None |
| Lux S3 | 2025 | Deep RL | 1st and 2nd; imitation 3rd |
| Orbit Wars | 2026 | Deep RL | 1st to 3rd |
| PTCG | 2026 | Deep RL | Top 8 of Strategy all self-play |
| Kaggriculture | 2026 | Imitation → RL + rules | Pure imitation 4th |
| Maze Crawler | 2026 | Heuristic BFS | Small Playground |

### Recurring people across competitions
- **egrehbbt / aDg4b:** Kore Beta 1st, Lux S3 3rd.
- **Harm Buisman:** Kore 1st, Lux S2 5th.
- **flg:** Lux S2 4th, PTCG 4th, Orbit Wars 6th.
- **Isaiah Pressman:** Lux S3 2nd with Frog Parade, Orbit Wars 1st with Tufa Labs.
- **loh-maa:** Lux Beta 1st, 20Q 3rd.
- **ry_andy_:** Lux S2 and NeurIPS Stage 2 1st.

### Gaps to fill before drafting
1. Kaggriculture 1st place full writeup (only a preview so far) and anything from 2nd place (Victor @ Tufa Labs).
2. The PTCG Second Round (Tokyo) champion.
3. Code URLs for Lux S2 1st, Lux S3 1st (linked but not captured) and 20Q 2nd.
4. Per-step time limits for Kore and Lux S3.
5. Writeups for Lux S2 NeurIPS Stage 2's top 3.
6. 4th to 10th place details that were checked by title only.
