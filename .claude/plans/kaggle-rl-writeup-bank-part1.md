# Kaggle Simulation Competitions: Writeup Bank, Part 1 (2020-2021 era)

Built 2026-10-24. Scope: every Kaggle **simulation** (agent-vs-agent) competition that ran 2019-2021.

## How this bank was verified (read first)

- **Primary source = Kaggle's own JSON endpoints.** kaggle.com pages are JS-rendered, so the bank was built from the
  internal API the site itself calls (anonymous session + XSRF cookie):
  - `competitions.CompetitionService/GetCompetition` (dates, team counts, submission limits, reward type)
  - `competitions.PageService/ListPages` (Overview / Evaluation / Rules / Prizes / Timeline page text)
  - `competitions.LeaderboardService/GetLeaderboard` (final private LB: rank, score, team name, members, linked writeup)
  - `discussions.DiscussionsService/GetTopicListByForumId` + `GetForumTopicById` (full markdown of each writeup post)
- Every team name, rank, and score below comes from the **final leaderboard API**. Every technical claim comes
  from the **winner's own forum post**, quoted or closely paraphrased. All writeup URLs were returned by the API.
- **Competition discovery:** `ListCompetitions` filtered to simulation competitions returns exactly 8 Kaggle-hosted
  simulation competitions from 2019-2021 (listed below). The next ones are Kore 2022 (Apr 2022) and Lux AI 2022 Beta,
  both out of scope.
- **Per-step limits:** quoted from the competition's own Environment Rules page where it gives them. Halite and
  Football pages don't state a per-step timeout. For those two the value comes from the current `kaggle-environments`
  master JSON (`actTimeout`), and it is marked as such. The competition-time value may have differed.
- **Writeup URL forms:** some posts were migrated to `kaggle.com/competitions/<slug>/writeups/<id>`. The original
  `.../discussion/<id>` id is also given, because the organisers' recap threads link to it.
- **UNVERIFIED** marks anything that could not be confirmed from a primary source.

### Full list of 2019-2021 Kaggle simulation competitions (from the API)

| Slug | Title | Open → Deadline | Teams |
|---|---|---|---|
| `connectx` | Connect X (Getting Started, rolling) | 2020-01-03 → 2030-01-01 (never closes) | 252 (current rolling LB) |
| `halite` | Halite by Two Sigma (Halite IV) | 2020-06-15 → 2020-09-22 | 1,139 |
| `halite-iv-playground-edition` | Halite by Two Sigma - Playground Edition | 2020-09-23 → 2020-12-31 | 79 |
| `google-football` | Google Research Football with Manchester City F.C. | 2020-09-28 → 2020-12-11 | 1,138 |
| `rock-paper-scissors` | Rock, Paper, Scissors | 2020-11-02 → 2021-02-22 | 1,662 |
| `santa-2020` | Santa 2020 - The Candy Cane Contest | 2020-12-08 → 2021-02-22 | 792 |
| `hungry-geese` | Hungry Geese | 2021-01-26 → 2021-08-09 | 879 |
| `lux-ai-2021` | Lux AI (Season 1) | 2021-08-16 → 2021-12-20 | 1,178 |

Deadlines here are the API `deadline` field, which falls at the end of the final-games week. The last day to submit
was earlier; each section gives it.

The only "extra" competition beyond the user's list is the **Halite IV Playground Edition**, a knowledge-only
continuation (see the Halite section). No other 2019-2021 simulation competition exists on Kaggle.

### Ranking system shared by all eight (Evaluation pages)

Each submission gets a Gaussian skill rating N(μ, σ²) with **μ₀ = 600**. After a self-play validation episode, the
submission joins the pool. Kaggle matches submissions of similar rating, about **8 episodes per submission per day**
(newer submissions slightly more often). A win raises μ and lowers the loser's μ. A draw pulls the two μ values
together. The update size scales with surprise and with σ, and **the margin of victory is ignored**. Only the team's
best submission is shown. After the submission deadline, games keep running for about one more week (two weeks
for Lux) before the leaderboard is final. Late submissions are not possible.
Source: Evaluation page of each competition (e.g. hungry-geese Evaluation tab text, retrieved via `PageService/ListPages`).

All agents: max **100 MB** submission (API `submissionSizeLimitMb: 100` for all eight). Daily submission cap: 5
(ConnectX: 2).

---

## ConnectX

### Facts
- **URL:** https://www.kaggle.com/competitions/connectx
- **Type:** Kaggle's first simulation competition. Its Description page calls it a "**beta-version** of a brand-new type of ML competition called Simulations".
  It is a *Getting Started* competition with no end date (API deadline 2030-01-01).
- **Dates:** opened 2020-01-03. **Rolling leaderboard:** a submission older than two months is invalidated, and a
  team with no recent submission drops off ("Rolling Leaderboards", Addison Howard, 2020-01-03,
  https://www.kaggle.com/competitions/connectx/discussion/124405).
- **Teams:** 252 on the current rolling LB (API, 2026-10-24). This is not a historical total.
- **Prize:** knowledge only (API `reward: KNOWLEDGE`).
- **Ranking:** μ/σ skill rating as above.

### Game rules
Two players take turns dropping a checker into a 7-column × 6-row board. The first to get 4 in a row (horizontal,
vertical or diagonal) wins (+1/−1), and a full board is a draw (0). The state is fully observable and the game is
deterministic. The opponent is another submission. An agent must return a move within **2 s (60 s on the first turn)**
(Environment Rules), and the submission limit is 100 MB. CPMP pointed out on day one that Connect-4 is solved
(first player wins) (https://www.kaggle.com/competitions/connectx/discussion/124397).

### Top placements
Because the leaderboard rolls, there is **no final ranking or official winner**. Today's API top 3 (Peter Cnudde
1219.1, qq290637843 1218.5, ianeng88 1214.8) reflect 2026 submissions, not 2020.

| Who | When | Approach family | Key ideas | Source |
|---|---|---|---|---|
| Tom Van de Wiele | early 2020 | **Deep RL** (DQN-style) | Says Deep RL "briefly helped me to the first spot in the ConnectX competition". His Colab uses a simple convnet and 20-step returns and "gets remarkably good … in less than 1 hour of training". | Halite 1st writeup (https://www.kaggle.com/competitions/halite/discussion/183543); https://www.kaggle.com/competitions/connectx/discussion/129145 |
| dott1718 | 2020 | **Rule-based / search** | Named by the Raw Beast team as having "topped … ConnectX … with a rule-based approach". No writeup by dott1718 was found: UNVERIFIED beyond that statement. | https://www.kaggle.com/competitions/google-football/discussion/200709 |
| Taaha Khan | Oct 2020 ("rank 1-2", score ~1600) | **Search** | 19-ply opening book (~1 MB compressed, built with connect4.gamesolver.org scoring code), plus Negamax with alpha-beta, bitboards, iterative deepening. Deliberately plays an off-centre first move when second, to induce mistakes from imperfect players. | https://www.kaggle.com/competitions/connectx/discussion/188305 |
| (community) | 2022 | **Perfect solvers** | "I'm guessing that all of the top 6 … have perfect solutions." | https://www.kaggle.com/competitions/connectx/discussion/310075 |
| omikad | later | NN (PROBS algorithm) | "currently top 6" with a one-night-trained model | https://www.kaggle.com/competitions/connectx/discussion/532822 |

### Did RL or rules/search win?
**Search/solver.** The game is solved and the 2 s budget is enough for alpha-beta plus an opening book, so perfect play
caps the leaderboard. RL briefly led very early on (Tom Van de Wiele), before solvers appeared.

### Transferable lessons
1. **If the game is small enough to solve, solve it.** An opening book plus alpha-beta reaches perfect play, and RL
   cannot beat perfect. (188305, 310075, 124397)
2. **Against a field of imperfect players, perfect isn't enough. Choose moves that induce mistakes.** Taaha Khan
   plays a theoretically worse first move when second, to set traps. (188305)
3. **Deep RL bootstraps fast on small boards:** a convnet with n-step returns becomes strong in under an hour. (129145)

### Sources
- Competition API (`GetCompetition`, `ListPages`, `GetLeaderboard`) for `connectx`
- https://www.kaggle.com/competitions/connectx/discussion/124405 (rolling LB)
- https://www.kaggle.com/competitions/connectx/discussion/124397 (solved game)
- https://www.kaggle.com/competitions/connectx/discussion/188305 (Taaha Khan, alpha-beta + book)
- https://www.kaggle.com/competitions/connectx/discussion/129145 (Tom Van de Wiele, Deep Q-learning notebook)
- https://www.kaggle.com/competitions/connectx/discussion/310075, /532822
- https://www.kaggle.com/competitions/google-football/discussion/200709 (dott1718 claim)

---

## Halite by Two Sigma (Halite IV), plus Playground Edition

### Facts
- **URL:** https://www.kaggle.com/competitions/halite
- **Dates:** opened 2020-06-15. Entry/merger deadline 2020-09-08, final submission 2020-09-15, final games
  Sept 16-22, winners announced 2020-09-22 (Timeline page).
- **Size:** 1,139 teams / 1,291 competitors / 15,037 submissions (API). The organiser recap says 1,143 teams /
  1,295 participants before leaderboard cleaning (https://www.kaggle.com/competitions/halite/discussion/186290).
- **Prize:** "Swag will be provided to the top 4 finishers" (Prizes page). No cash. Medals and ranking points were awarded.
- **Ranking:** μ/σ skill rating. The recap calls Halite "Our first simulation competition" (186290), ConnectX being a beta Getting Started. The 8th-place team notes it had "normal ranking points and medals" (183312).
- **Playground Edition:** https://www.kaggle.com/competitions/halite-iv-playground-edition. Ran 2020-09-23 →
  2020-12-31, 79 teams, knowledge only. Kaggle launched it because late submissions weren't possible in the main
  competition (recap 186290). Its final #1 was Stanley Zheng (1639.5, API). **No writeups found** (its forum has 14 topics, none
  a solution).

### Game rules
A **4-player** free-for-all on a 21×21 wraparound grid, lasting 400 turns with simultaneous moves. Each player
starts with 5,000 halite and one ship. A ship can move N/E/S/W, stay (mining 25% of the cell's halite), or convert into a
shipyard for 500 halite. Shipyards spawn ships for 500. Cargo counts only once deposited at a shipyard. When ships
collide, the one carrying the least halite survives and takes the others' cargo; on a tie, all are destroyed. Empty cells regrow 2% per turn,
up to 500. Final reward is banked halite, and elimination ranks a player last (4th-place writeup restating the rules,
https://www.kaggle.com/competitions/halite/discussion/183727; env JSON `halite.json`). Per-step limit: `actTimeout = 3` s
in the current kaggle-environments master. **The competition-time value is UNVERIFIED** because the rules page doesn't state it.
The submission limit is 100 MB.

### Top placements (final LB from API)

| Rank | Team (members) | Score | Family | Key technical ideas | Writeup / code |
|---|---|---|---|---|---|
| 1 | **Tom Van de Wiele** | 1592.3 | **Rule-based (hand-engineered planner)** | Tried Deep RL first: "after spending a month trying to get the ships to do anything different from standing still, I decided to move on", because credit assignment over an arbitrary number of units, long episodes and a dynamic opponent pool was too hard. Final design is a 3-stage pipeline. **(1) Per-ship score maps** over all 21×21 squares for collect / return / build base / attack base, with exponentially decaying distance masks so targets stay consistent from step to step. **(2) Planner** assigns each ship a task, resolving conflicts by priority. **(3) Action translator** turns plans into moves. Specialised modules: camping/harassment of opponent bases, the "chihuahua" base defence, boxing in opponent ships with fewer than 4 ships when the next move is predictable, chase and cycle detection, rescue missions, and a "badness" hierarchy (1-step / 2-step / N-step risk). Over 11,000 lines of code. "I submitted 22 agents of which 16 would have taken the first place." | https://www.kaggle.com/competitions/halite/discussion/183543 · code https://github.com/ttvand/Halite |
| 2 | **Raine Force** (dereview) | 1524.5 | **Rule-based** | "Despite my team name … my agents are 100% rule based." Stateful, priority-sorted "Projects" (HuntProject boxes in an enemy ship with 5-8 ships, plus DefenseShipyard, Expedition/aggressive conversion, Escort, Mine) that hold ships and halite budget over several steps. Its aggressive conversion strategy happened to resist Tom's harassment, giving a ~40% win rate against Tom's best 5. | https://www.kaggle.com/competitions/halite/writeups/raine-force-raine-force-writeup-2nd-place-solution (orig. discussion/186032) · code https://github.com/viewlagoon/HaliteRaineForce |
| 3 | **convexOptimization** (HyperSeedOptimized, convexOptimization) | 1513.5 | **UNVERIFIED** | No writeup in the organiser's list or the forum. Tom credits their early-game collect strategy as "miles ahead of mine". | none found |
| 4 | **Uninstall LoL** (Lucas Schnack, Dave Frank) | 1487.8 | **Rule-based** | "Simple rule-based bot." | https://www.kaggle.com/competitions/halite/discussion/183727 · code https://github.com/0Zeta/HaliteIV-Bot |
| 5 | **Leukocyte** (Panpan Zhou) | 1484.3 | **Imitation learning (ML)** | Tried DQN / actor-critic first but "struggled in assigning proper rewards" for CONVERT and group hunting, then switched to learning from top players' winning replays. Ship-centred 9×21×21 map tensor plus global player stats, a PyTorch per-ship action classifier, and three separately trained game-phase models (steps 0-75, 75-325, 325-400). Class-weighting of the loss mattered. | https://www.kaggle.com/competitions/halite/discussion/183704 · code https://github.com/panpanzhouhouston/HaliteIV |
| 6 | mzotkiew | 1480.9 | Rule-based | A "stateless, rule-based bot". Each ship gets "intention" values for 5 end squares, and actions are assigned to maximise total satisfied intention. | https://www.kaggle.com/competitions/halite/discussion/185865 · https://github.com/mzotkiew/Halite-IV |
| 8 | KhaVo Dan Gilles Robga Tung | 1439.7 | **Hybrid: IL + rule overrides** | A centralised imitation-learning model framed as semantic segmentation, predicting every ship's action at once. Fine-tuned heuristics override it for spawning, converting and base protection. | https://www.kaggle.com/competitions/halite/discussion/183312 · https://github.com/digitalspecialists/halite4 |

No pure deep-RL agent appears in the top 10 by any writeup found.

### Did RL or rules win?
**Rules won**, with places 1, 2, 4 and 6 hand-coded. The winner says why in his own words: per-unit credit assignment
with a variable number of ships, 400-step episodes, a changing opponent pool, and limited compute (183543). The best
learned agents (5th, 8th) were **imitation** learning, not RL. The 5th-place author gave the same reason for
abandoning RL: reward design for CONVERT and group hunting (183704).

### Transferable lessons
1. **Variable-unit-count games break naive per-unit RL credit assignment.** The winner abandoned RL for exactly this
   reason. (183543)
2. **Score → plan → action decomposition lets you iterate on strategy without breaking tactics.** Tom fixed conflict
   resolution first, then iterated on the score maps. (183543)
3. **Keep target scores temporally consistent**, using decaying distance masks, or units dither between goals. (183543)
4. **Imitating the leaderboard is a cheap path to gold.** Two of the top 8 cloned top replays, adding rule overrides
   where the clone was weak. (183704, 183312)
5. **Robustness to the dominant strategy matters more than average strength.** Raine Force's 2nd place came from a
   conversion style that happened to resist the leader's harassment. (186032)

### Sources
- API: `GetCompetition`/`ListPages`/`GetLeaderboard` for `halite` and `halite-iv-playground-edition`
- Recap and curated list: https://www.kaggle.com/competitions/halite/discussion/186290; community compilation https://www.kaggle.com/competitions/halite/discussion/184028
- Writeups: /discussion/183543, /writeups/raine-force-raine-force-writeup-2nd-place-solution, /discussion/183727, /discussion/183704, /discussion/185865, /discussion/183312
- GitHub repos above (all returned HTTP 200 on 2026-10-24)

---

## Google Research Football with Manchester City F.C.

### Facts
- **URL:** https://www.kaggle.com/competitions/google-football
- **Host:** Google AI (Google Brain) with Manchester City F.C.
- **Dates:** opened 2020-09-28. Entry/merger deadline 2020-11-23, final submission 2020-11-30, final games
  Dec 1-8 (Timeline page). The leaderboard was finalised later (recap posted 2020-12-16). The API deadline reads 2020-12-11.
- **Size:** 1,138 teams / 1,288 competitors / 14,015 submissions from 77 countries (API + recap).
- **Prize:** **$6,000** (1st $3,000, 2nd $2,000, 3rd $1,000), plus a Manchester City hospitality and behind-the-scenes
  experience for the top 3 (Prizes page).
- **Ranking:** μ/σ skill rating. Margin of victory is ignored.

### Game rules
Full 11-vs-11 football in Google's GRF physics engine, with offsides and cards. The game runs 3,000 steps (two halves of 1,500)
with no side swap, and observations are always mirrored so the agent attacks left-to-right. **Each agent controls a
single player**, automatically the ball carrier in attack or the player nearest the ball in defence. The built-in AI drives the other 10.
The state is the full raw observation (positions, directions, tiredness, score, ball). The action is one of **19** discrete
actions. Reward is +1/−1 per goal, and the match result (win/draw/loss) drives the rating (Google Football Rules page, env JSON).
Per-step limit: `actTimeout = 0.5` s in the current kaggle-environments master, **competition-time value UNVERIFIED**.
Raw Beast reports its agents took 0.01-0.15 s per action (200709). The submission limit is 100 MB.

### Top placements (final LB from API)

| Rank | Team (members) | Score | Family | Key technical ideas | Writeup / code |
|---|---|---|---|---|---|
| 1 | **WeKick** (Kaiwen Zhu, ZiyangLi, fenngming; Tencent) | 1785.8 | **Deep RL self-play + league** | "We totally build our agents using reinforcement learning and selfplay." Asynchronous distributed framework after Ye et al. 2019 (arXiv 1912.09729). **PPO** (adapted OpenAI baselines) with dense 256-d layers plus an LSTM (32 steps, 256 hidden), Adam with fixed lr 1e-4. Spatial "SMM"/CNN inputs were tried and dropped as too slow. **Multi-head value**: rewards decomposed into heads with different discount factors (Ye et al. 2020). Extended the 115-d "simple115" vector with relative poses, offside flags, and sticky-action/card/tiredness features. **Zero-sum shaped rewards**: ±0.2 possession change, ±0.2 slide, ±0.001 ball holding, +0.1 per pass before a goal. **GAIL** on the replays of rival team kangaroo to imitate a counter-attack style that was hard to reward by hand. **AlphaStar-style league** of distinct styles (counter-attack, short-pass, ball-holding), with a final model trained to beat the whole league, worth about +100 Elo internally. Local Elo leaderboard correlated with the public one. Training compute: **not stated** in the writeup. | https://www.kaggle.com/competitions/google-football/discussion/202232 · code: none published (UNVERIFIED). Replay dataset (81 games): https://www.kaggle.com/competitions/google-football/discussion/203384 |
| 2 | **SaltyFish** (Niuhe, Hank, Dharmaraja Hu; NetEase Fuxi) | 1597.5 | **Deep RL (PvE curriculum → self-play)** | Distributed **IMPALA** in their in-house RLEase library. First trained against the built-in bot (difficulty 0.05, then 1.0) with **checkpoint rewards** to learn offence, then **self-play** to learn defence and passing. Multi-head vector input with "surgery" to add features mid-training (OpenAI Five style). MLP network. **Legal-action masks** per possession state sped up learning but blocked WeKick's one-touch passing; a mask-free model trained in the last week became their best submission (1597.5 vs 1597.0). Also behaviour-cloned WeKick into the opponent pool. **Compute: 1,000 CPU cores per experiment, up to 2,500 in the last two weeks, no GPUs needed** (vector inputs). | https://www.kaggle.com/competitions/google-football/writeups/saltyfish-saltyfish-current-2nd-place-solution (orig. discussion/202977) |
| 3 | **Raw Beast** (dott, Tom Van de Wiele) | 1564.3 | **Hybrid: behaviour cloning → RL** | An "AlphaStar-like" recipe. Supervised cloning of top submissions (already about 20th place on its own), then actor-critic RL (custom PyTorch **IMPALA**, using torchbeast for the policy loss) initialised from the clone, with a **KL penalty to the clone** that is re-anchored each round. Clone strength predicted post-RL strength. Separate actor and critic, with critic lr twice the actor's and the actor frozen until the critic is decent. Trust-region KL per pass over an 800k-transition replay buffer. **Simplifying the reward to goal scored/conceded is what made RL start working**, after per-mode hand-crafted rewards failed. Features for Markovness: time since last pass/shot/slide, offside flags. Ran single-machine Docker experiments; actor data rate was the bottleneck. | https://www.kaggle.com/competitions/google-football/discussion/200709 · code: not linked (UNVERIFIED) |
| 4 | zsp1197 zsp (hy Xue, zsp1197 zsp, Sijia0515) | 1483.6 | UNVERIFIED | No writeup found | none |
| 5 | **TamakEri** (kyazuki, YuriCat) | 1471.0 | **Deep RL (self-play + opponents)** | Joined about 3 weeks before the deadline. Off-policy distributed policy gradient on **HandyRL**: TD(λ) with clipped importance sampling, and value (outcome) plus return (discounted goals) heads. Early supervised learning on top replays, decayed to zero. Multi-head self-attention + CNN + GRU + MLP. Played self-play plus games against rule-based bots and the built-in AI, reaching ~90% win rate against the hard built-in AI. **Compute: ~4,000 CPUs and ~20 GPUs (T4) in the last 3 days, ~700k games total.** | https://www.kaggle.com/competitions/google-football/writeups/tamakeri-tamakeri-5th-place-solution (orig. discussion/203412) · https://github.com/YuriCat/TamakEriFever |
| 6 | liveinparis (seungeun07, Seongho Son, Taehwan Kwon) | 1466.3 | **Deep RL self-play from scratch** | No demonstrations. Checkpoint saved every ~1 h into an opponent pool, with 50% of opponents from the latest 10 checkpoints and 50% uniform over the rest. | https://www.kaggle.com/competitions/google-football/discussion/201376 · https://github.com/seungeunrho/football-paris |
| 7 | s_toppo | 1465.8 | **Imitation (GBDT)** | LightGBM/CatBoost cloning of WeKick/SaltyFish replays. Agents clustered by action distribution before pooling data. | https://www.kaggle.com/competitions/google-football/discussion/200912 |

Follow-up research: TiKick (arXiv 2110.04507) built a multi-agent offline-RL dataset by rolling out WeKick's agent.
This is a secondary source.

### Did RL or rules win?
**RL won decisively.** Every top-3 team used deep RL, and the leader beat 2nd by about 190 rating points. Raw Beast both
had won earlier simulation competitions with rules, but said a rule-based approach "quickly became clear that this was not going to scale" (200709). Raw Beast called it "the first competition that really required reinforcement learning (RL) to excel" (200709). Why RL won here: a continuous-physics, long-horizon game with a single controlled player
(so no multi-unit credit-assignment problem), cheap vector observations, and teams with industrial CPU budgets.

### Transferable lessons
1. **Diversity beats a single self-play lineage.** WeKick's league of stylistically distinct agents (shaped rewards,
   GAIL clones of rivals) plus a final "beat-the-league" model added ~100 Elo. (202232)
2. **Keep shaped rewards zero-sum in self-play**, so shaping can't be farmed cooperatively. (202232)
3. **Simplify the reward.** Raw Beast's RL only started working once they dropped per-mode hand-crafted rewards for
   plain goal ± signals. (200709)
4. **Clone first, then RL with a KL leash to the clone.** Clone quality predicts final RL quality, the AlphaStar result
   reproduced at Kaggle scale. (200709)
5. **Action masks are a double-edged sword.** They speed up learning but can forbid the strongest emergent tactic
   (one-touch passing). (202977)

### Sources
- API: `GetCompetition`/`ListPages`/`GetLeaderboard` for `google-football`
- Recap: https://www.kaggle.com/competitions/google-football/discussion/204645
- Writeups: /discussion/202232, /writeups/saltyfish-saltyfish-current-2nd-place-solution, /discussion/200709, /writeups/tamakeri-tamakeri-5th-place-solution, /discussion/201376, /discussion/200912, /discussion/203384
- Secondary: TiKick https://arxiv.org/abs/2110.04507; Ye et al. https://arxiv.org/abs/1912.09729, https://arxiv.org/abs/2011.12692 (cited by WeKick)

---

## Rock, Paper, Scissors

### Facts
- **URL:** https://www.kaggle.com/competitions/rock-paper-scissors
- **Dates:** opened 2020-11-02. Entry, merger and final submission deadline 2021-02-01, final episodes Feb 2-8
  (Timeline page). API deadline 2021-02-22.
- **Size:** 1,662 teams / 1,814 competitors / 48,236 submissions (API). The recap says it was "the most popular Kaggle
  simulation to date" (https://www.kaggle.com/competitions/rock-paper-scissors/discussion/222022).
- **Prize:** Kaggle-branded merchandise for the top 5 (Prizes page). Medals awarded.
- **Ranking:** μ/σ skill rating.

### Game rules
Two players throw rock/paper/scissors simultaneously for **1,000 rounds**. The observation is the opponent's last
move (plus step). An episode is a win only if the cumulative score difference exceeds a threshold; otherwise it is a draw.
(`tieRewardThreshold = 20` in the env JSON. The Environment Rules page says "Best of 1000 rounds wins the match".)
The time limit is **1 s per move (61 s on the first)** and the submission limit is 100 MB (Environment Rules). The Nash
equilibrium (uniform random) can't be beaten in expectation but can't win either. The whole game is about exploiting
non-random opponents.

### Top placements (final LB from API)

| Rank | Team | Score | Family | Key ideas | Writeup / code |
|---|---|---|---|---|---|
| 1 | **Where is my bag** (Boooooooooow) | 1076.9 | **Ensemble + bandit meta-selection (rule/statistical)** | "multi-armed bandit with list of rpscontest agents + 1-step & 2-steps ahead transformed agents + dirichlet distribution." The pool held public Kaggle bots (decision tree, rfind) and old rpscontest.com bots, each also in a "beater" (1-step-ahead) and 2-step-ahead version. The two design principles: beat as many public agents as possible, and "a good agent cannot easily beat itself". Replacing many if-else selection rules with **Beta, later Dirichlet, sampling** over agent performance "dominated over all of my previous agents". Local evaluation on an 8+ CPU EC2 instance, about 3,000 games/day. | https://www.kaggle.com/competitions/rock-paper-scissors/writeups/where-is-my-bag-1st-place-solution-where-is-my-bag (orig. discussion/221502) |
| 2 | **Georg Streich** | 1052.6 | **Ensemble + randomised selection** | Ensemble of the top 50 rpscontest.com agents plus strong public Kaggle bots, each with an "evil twin" (reversed perspective). Picks a group of agents that did well over a window, then picks within it by recent performance, with **scores drawn from a distribution** (Dirichlet) so the selection itself leaks no pattern. Full code in the post. | https://www.kaggle.com/competitions/rock-paper-scissors/writeups/georg-streich-2nd-place-solution (orig. discussion/221506) |
| 3 | Stas Sl | 1048.3 | UNVERIFIED | No writeup in the organiser's list or the forum | none found |
| 4 | Taaha Khan | 1021.0 | Ensemble + dynamically weighted MAB ("HydraRPS") | Large ensemble of public and archived bots, with a distribution over their votes | https://www.kaggle.com/competitions/rock-paper-scissors/writeups/taaha-khan-4th-place-solution-taaha-khan (orig. discussion/217046) · https://github.com/taaha-khan/rock-paper-scissors |
| 5 | mike h | 1016.9 | Pattern matching + defence | History pattern detectors including "meta" and "meta-meta" rotations, decayed scoring (+2 win / −1 draw / −1 loss), voting, and **random play on half the turns** as defence | https://www.kaggle.com/competitions/rock-paper-scissors/writeups/mike-h-5th-place-solution (orig. discussion/222178) |
| 12 | Kazuo Watanabe | 992.0 | Supervised ML (LightGBM) | Notable as the best ML-model writeup | https://www.kaggle.com/competitions/rock-paper-scissors/writeups/kazuo-watanabe-12th-place-solution-lightgbm-model |

No deep-RL solution with a documented top-10 finish was found. A "self-playing Transformer" proof of concept was posted
but not ranked (https://www.kaggle.com/competitions/rock-paper-scissors/discussion/224020).

### Did RL or rules win?
**Neither in the deep-RL sense.** The winners were **online-adaptive ensembles**: a bandit (Thompson-style Beta/Dirichlet sampling) over a
library of hand-written predictors. This is "RL" only in the bandit sense. The stated reason: the leaderboard is a
population of exploitable public bots, so winning means beating that population while staying random enough not to
be exploited yourself (221502, 221506).

### Transferable lessons
1. **Mine prior art.** The 1st-place jump came from importing a decade of rpscontest.com bots. (221502)
2. **Model the field, not the game.** "To achieve a higher rank, the agent should beat as much public agents as
   possible." (221502)
3. **Randomise the selector itself.** Sampling agent scores from a Dirichlet beat hand-tuned deterministic rules,
   because a deterministic meta-policy is itself a pattern. (221502, 221506)
4. **Level-k thinking:** add 1-step- and 2-step-ahead "beater" versions of every agent, including yourself. (221502)
5. **Partial randomness is cheap insurance against counter-exploitation.** (222178)

### Sources
- API: `GetCompetition`/`ListPages`/`GetLeaderboard` for `rock-paper-scissors`
- Recap with writeup list: https://www.kaggle.com/competitions/rock-paper-scissors/discussion/222022
- Writeups as linked above; /discussion/224020

---

## Santa 2020 - The Candy Cane Contest

### Facts
- **URL:** https://www.kaggle.com/competitions/santa-2020
- **Dates:** opened 2020-12-08. Entry, merger and final submission deadline 2021-02-01, final episodes about
  Feb 2-9 (Timeline page). API deadline 2021-02-22.
- **Size:** 792 teams / 919 competitors / 24,724 submissions (API).
- **Prize:** Kaggle merchandise for the top 5 (Prizes page). Medals awarded.
- **Ranking:** μ/σ skill rating.

### Game rules
A **two-player multi-armed bandit**. Both players share the same **100 vending machines (bandits)**, and each machine
pays a candy cane with probability tied to a hidden threshold. **Every pull of a machine, by either player, decays its
reward probability by 3%** (`decayRate = 0.97`). The game runs **2,000 rounds per player**. Each agent sees its own total
reward and **both players' last pulls**, but not whether the opponent got a reward. The most candy canes wins. Time limit: **0.25 s per
step + 60 s overage pool**, and the submission limit is 100 MB (Environment Rules).

### Top placements (final LB from API)

| Rank | Team | Score | Family | Key ideas | Writeup / code |
|---|---|---|---|---|---|
| 1 | **nagiss 🦌** | 1536.2 | **Supervised ML (LightGBM) + greedy** | Two LightGBM models predict each bandit's **initial threshold** every step, trained on 3,000 replays fetched via the episode API. Ten features including step, the opponent's repeat-pulls of a bandit, decay-corrected opponent pull counts, and the Gini coefficient of the opponent's choices. Model 1 is exploitation (RMSE). Model 2 trains on 1.02^threshold, an asymmetric loss that adds optimism for exploration. The blend is −0.2:1.2 at the start, moving to 1:0. Act = argmax(predicted threshold × decay). Validation on about 1,000 games against his own best agent, using **win rate rather than mean reward**. Submitted the maximum number of agents to "make the most of the luck factor". | https://www.kaggle.com/competitions/santa-2020/discussion/218453 · notebook https://www.kaggle.com/nagiss/imaginary-rudolph-prize-solution |
| 2 | **kibuna** | 1534.5 | **Supervised ML (LightGBM quantile regression)** | His own table: UCB-like rules 951.9, **RL 997.9**, LGBM regression 1442.1, **LGBM quantile regression 1521.1**. The 60th percentile before step 750 and the 65th after, trained on 20-30k episodes from the top 30 teams. | https://www.kaggle.com/competitions/santa-2020/discussion/221030 |
| 3 | **Coffee Candy ☕** (Shane, Yijia) | 1531.0 | **Hybrid: probability-estimation models (LSTM/tree) + RL (SEED RL, V-trace)** | ~45k replays downloaded. Trained both a threshold-estimation model and RL policies (SEED RL on GCP, MLP/LSTM, opponent sampling from public kernels, fitted models and recent self-play). **Which model family produced the top-scoring submission is not stated: UNVERIFIED.** | https://www.kaggle.com/competitions/santa-2020/discussion/220634 |
| 4 | **Dmitriy Guller** | 1524.1 | **Bayesian + opponent modelling** | Exact discrete posterior over each bandit from his own pulls, an **LSTM to estimate the opponent's posterior** from their actions, and **deception tactics** so the opponent can't read his knowledge. Argues the game is "more like poker" than an explore/exploit problem. | https://www.kaggle.com/competitions/santa-2020/discussion/216536 |
| 9 | toshi_k | 1449.4 | (not read in detail) | | https://www.kaggle.com/competitions/santa-2020/writeups/toshi-k-9th-place-solution (orig. discussion/221707) |
| 16 | (pure self-play RL team) | n/a | **Deep RL self-play** | "A pure self-play reinforcement learning approach" on a single home GPU. Each new agent beat all previous ones. | https://www.kaggle.com/competitions/santa-2020/discussion/217020 |

### Did RL or rules win?
**Supervised learning on leaderboard replays won**, with LightGBM at 1st and 2nd. kibuna's own comparison is the
cleanest evidence: his RL agent topped out at 997.9 while LightGBM quantile regression reached 1521.1 (221030). The
game rewards inferring hidden state from the opponent's behaviour, and the replay API supplied tens of thousands of labelled games.

### Transferable lessons
1. **Turn a game into supervised learning when replays give you labels.** The hidden threshold is recoverable from
   replays, so regression on it beat RL. (218453, 221030)
2. **Bake exploration into the loss, not the policy.** An asymmetric loss or upper quantile = optimism under
   uncertainty. (218453, 221030)
3. **The opponent's actions are information.** The strongest features were counts of the opponent's (repeat) pulls,
   and 4th place modelled the opponent's posterior with an LSTM. (218453, 216536)
4. **Hide your own information.** Deception (deliberately sub-optimal pulls) has value in adversarial bandits. (216536)
5. **Evaluate on win rate, not mean reward**, because the rating counts only wins. (218453)

### Sources
- API: `GetCompetition`/`ListPages`/`GetLeaderboard` for `santa-2020`
- Recap: https://www.kaggle.com/competitions/santa-2020/discussion/222028
- Writeups: /discussion/218453, /221030, /220634, /216536, /217020, /writeups/toshi-k-9th-place-solution

---

## Hungry Geese

### Facts
- **URL:** https://www.kaggle.com/competitions/hungry-geese
- **Dates:** start 2021-01-25/26. Entry, merger and final submission deadline 2021-07-26, final games
  Jul 27-Aug 9, winners announced 2021-08-10 (Timeline page).
- **Size:** 879 teams / 1,043 competitors / 33,296 submissions (API). The recap says 875 teams / 1,039 participants.
- **Prize:** Kaggle merchandise to the **top team each month** (7 dates, no repeat winners) (Prizes page). Medals awarded.
- **Ranking:** μ/σ skill rating. Episodes are **4-player**.

### Game rules
Four geese (snakes) move simultaneously on an **11×7 wraparound grid** for **200 steps**. Actions are N/S/E/W, and a goose
cannot reverse. At least 2 food items are always on the board. Eating grows the goose, and every **40 steps** each goose
loses a segment. Running into any body eliminates the goose whose head hit it. Reward = steps survived × (max length + 1) +
length, and placement determines the rating update. The state is fully observable. Time: **1 s per step + 60 s overage**
(exceeding it disqualifies), and the submission limit is 100 MB (Rules of Play / Evaluation pages, env JSON).

### Top placements (final LB from API)

| Rank | Team (members) | Score | Family | Key ideas | Writeup / code |
|---|---|---|---|---|---|
| 1 | **HandyRL** (kyazuki + one now-deleted account; DeNA / quantum) | 1312.8 | **Deep RL self-play (+ search at the end)** | The **1st-place post is deleted**: https://www.kaggle.com/competitions/hungry-geese/discussion/263279 returns "[Deleted Topic]" (26 messages, posted 2021-08-09). Verified facts: the team released **HandyRL** early. Their public model took **1 day on 1 GPU + 64 CPUs**, and the published config uses TD policy/value targets, γ = 0.8, forward_steps 32, λ = 0.7 (https://www.kaggle.com/competitions/hungry-geese/discussion/218190). The 2nd-place team writes that combining the policy network with a fast compiled search "is what HandyRL did in the end with excellent results" (263686). Japanese press credits Katsumi Ohwatari (quantum) and Kazuki/Ikki Tanaka (DeNA), with most training run on quantum's machines (AIsmiley, secondary). **Architecture, search type and compute of the final agent: UNVERIFIED.** | deleted post (link above) · library https://github.com/DeNA/HandyRL (its README links the 1st-place post) |
| 2 | **Goosebumps** (LiamK, IsaiahP, Vishal Singh) | 1311.5 | **Hybrid: behaviour cloning + MCTS** | **Lost by 1.3 rating points.** BC network trained daily on replays where *every* agent was above a rising rating threshold (raising it "almost always" helped). 19×7×11 input, with a torus-equivariant SE-ResNet (6 blocks × 64 ch, the best) or an attention variant. Predicts all 4 geese's policies, with zero-sum softmax values. **PUCT MCTS** at 20-50 iterations/s in pure Python. **Overage time spent dynamically** when the search policy is shifting. Ran on one 8-core, dual-GPU machine, which "limited our capacity to run traditional CPU-intensive reinforcement learning at scale". | https://www.kaggle.com/competitions/hungry-geese/writeups/goosebumps-goosebumps-solution-2nd-place (orig. discussion/263686) · https://github.com/IsaiahPressman/Kaggle_Hungry_Geese |
| 3 | **NejuMixWATORI** (Maxwell, YuyaYamamoto, NIWATORI) | 1239.1 | **Hybrid: AlphaZero-like (BC pretrain → RL → MCTS)** | BC on top-agent episodes, then HandyRL RL (UPGO, switching to V-trace at the end for calibration), then MCTS building on the public "AlphaGeese" notebook. Treating step 199 as terminal gave a big search-depth gain. | https://www.kaggle.com/competitions/hungry-geese/writeups/nejumixwatori-3rd-place-solution (orig. discussion/263735) |
| 4 | Goose luck (currypurin, Ryuichi, Shun_PI, Yoshio Sugiyama) | 1236.6 | **Deep RL (HandyRL)** | "Almost using the initial parameters of HandyRL". Rich body-disappearance and length-difference channels, plus flip test-time augmentation. | https://www.kaggle.com/competitions/hungry-geese/writeups/goose-luck-4th-place-solution (orig. discussion/263690) |
| 5 | takedarts | 1233.2 | **AlphaZero from scratch (no external data, no RL library)** | Self-play game records (~350k, generated on a Threadripper 3970X). 98×11×11 head-centred, north-rotated input. A collision-penalty trick in the shared search tree. | https://www.kaggle.com/competitions/hungry-geese/writeups/takedarts-5th-place-solution-geesezero (orig. discussion/263702) · https://github.com/takedarts/hungry-geese |
| 6 | Daewoo Myoung | 1225.2 | Deep RL (HandyRL) + ensemble | Argmax over the summed outputs of the top-3 models | https://www.kaggle.com/competitions/hungry-geese/writeups/daewoo-myoung-6th-place-solution (orig. discussion/263916) |
| 7 | CGILAB x H2O.ai | 1224.9 | SL + MCTS + heuristics | | https://www.kaggle.com/competitions/hungry-geese/discussion/263486 |
| 8 | TCOR3参加失敗組 (nagiss, KKT 89) | 1224.1 | AlphaZero with an **NNUE**-style MLP | | https://www.kaggle.com/competitions/hungry-geese/writeups/tcor3-8th-place-solution (orig. discussion/264015) · https://github.com/Lgeu/hungry_geese |
| 9 | THE GOOSE IS LOOSE | 1223.5 | RL (HandyRL) + alpha-beta lookahead + flood-fill + rules | Also detects and exploits the deterministic public HandyRL bot ("about 25% of all submissions") | https://www.kaggle.com/competitions/hungry-geese/discussion/255931 |

### Did RL or rules win?
**Learned policy/value networks + search won.** Every top-9 team used a neural network. The 1st place was RL self-play
(HandyRL) with search added at the end. 2nd was behaviour cloning + MCTS, and 3rd and 5th were AlphaZero variants. Goosebumps' stated
reason for using BC rather than RL was compute (one machine). They also hypothesise that HandyRL's self-play net gave
better search priors than their BC net, which "rarely made it to step 200 when playing against itself without
search" (263686).

### Transferable lessons
1. **Network + search beats either alone in short tactical games.** Nearly the whole top 10 is NN + MCTS or alpha-beta.
   (263686, 263735, 263702, 264015, 255931)
2. **Curate imitation data by the *weakest* player in a game, not the strongest.** Raising that threshold reliably
   improved the clone. (263686)
3. **Spend overage time adaptively, only on uncertain decisions.** (263686)
4. **Open-sourcing a strong baseline reshapes the field and creates an exploitable monoculture.** About 25% of
   submissions were the public HandyRL bot, and 9th place built a detector to exploit it. (264053, 255931)
5. **Exploit symmetry in the architecture** (torus-equivariant convs, circular padding, reflections), not just in
   augmentation. (263686)

### Sources
- API: `GetCompetition`/`ListPages`/`GetLeaderboard` for `hungry-geese`
- Recap: https://www.kaggle.com/competitions/hungry-geese/discussion/264053
- HandyRL post: https://www.kaggle.com/competitions/hungry-geese/discussion/218190; HandyRL README https://github.com/DeNA/HandyRL
- Writeups as linked above
- Secondary (1st-place team identity only): https://aismiley.co.jp/ai_news/kaggle-hungry-geese-dena-quantum/

---

## Lux AI Challenge Season 1 (Lux AI 2021)

### Facts
- **URL:** https://www.kaggle.com/competitions/lux-ai-2021
- **Host:** Lux AI Challenge (Stone Tao, Bovard), sponsored by QuantCo, J Ventures and QAImera.
- **Dates:** start 2021-08-16. Entry, merger and final submission deadline 2021-12-06, final games
  Dec 7-20, winners announced 2021-12-21 (Timeline page).
- **Size:** 1,178 teams / 1,449 competitors / 22,331 submissions (API). 5,691 registrations, 84 countries (recap).
- **Prize:** **$10,000 main pool** (1st $4,000, 2nd $2,500, 3rd $1,500, 4th $1,000, 5th $1,000), plus monthly "sprint"
  prizes, a J Ventures Innovative Strategy award and t-shirts (Prizes page).
- **Ranking:** μ/σ skill rating.

### Game rules
A **1v1** game on a square map from 12×12 to 32×32, lasting 360 turns. Each player controls workers, carts and city
tiles. Units mine wood, coal and uranium (the last two need research points) and build cities. Every 40-turn cycle has 10
turns of night, when units and cities must burn fuel or vanish. The winner has the most city tiles at the end, with
unit count as the tiebreak. The state is fully observable. The action space is one action per unit and per city tile, so a
variable-size multi-unit action. Time: **3 s per turn plus a 60 s overage pool** (exhausting it freezes the agent), and the
submission limit is 100 MB (Lux AI Specifications / Evaluation pages; 1st-place writeup's game summary).

### Top placements (final LB from API)

| Rank | Team (members) | Score | Family | Key technical ideas | Writeup / code |
|---|---|---|---|---|---|
| 1 | **Toad Brigade** (Robert Sturrock, LiamK, IsaiahP) | 2078.7 | **Deep RL self-play from scratch** | Started with rules (because rules won Halite), but "within the first month the RL approach began to beat the rules-based one", so rules were abandoned from August. **A single network commands every unit on every square at once**, avoiding per-unit credit assignment. Boards zero-padded to 32×32 with masking after every conv. Learned embeddings for discrete features. **Game-phase feature** (turn // 40) was "a crucial part of its success". **SE-ResNet, 24 blocks × 128 ch 5×5 convs, no normalisation, ~20M parameters.** Three actor heads (worker / cart / city) plus a critic. **IMPALA (torchbeast) + UPGO + TD(λ)**, with a **KL term to a frozen teacher** to stabilise and prevent strategic cycles. Shaped reward for the first 20M steps, then pure win/loss. Grew from 8 to 16 to 24 blocks, each time distilling from the previous net. Illegal actions masked. Transfers discretised to all-or-nothing. **Trained on one personal PC: 8-core/16-thread, dual GPU.** | https://www.kaggle.com/competitions/lux-ai-2021/writeups/toad-brigade-toad-brigade-s-approach-deep-reinforc (orig. discussion/294993) · https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021 |
| 2 | **RL is all you need** (Damien Mourot, Théo Cachet) | 1932.5 | **Deep RL (PPO)** | PPO written from scratch with a **hand-crafted dense reward**: the time derivative of a score combining farming, territory control (Voronoi-style reachability), surviving city tiles, and ±50 win/loss. Pure win/loss from scratch only reached about 1200 Elo. A small **ResNet of 17 blocks × 32 ch 3×3, about 300k parameters**. Features were selected via a fast behaviour-cloning pipeline, and normalisation and clipping mattered. A single action map, with transfers hard-coded to one action and invalid actions masked. A **KL penalty distils previous best models**. Opponents sampled by **Prioritized Fictitious Self-Play**. Trained on an RTX 2080 Ti + i9-9900K, with the slow game engine as the bottleneck. In hindsight they say IMPALA (off-policy) would have been better given the slow environment. | https://www.kaggle.com/competitions/lux-ai-2021/writeups/rl-is-all-you-need-rliayn-s-approach-online-deep-r (orig. discussion/300844) |
| 3 | skyramp | 1828.5 | UNVERIFIED | No writeup in the organiser's list or the forum | none found |
| 4 | Team Durrett (bomac1, tsuchhiii) | 1822.8 | **Imitation learning** | Cloned the **top 7 agents** with a shared CNN trunk and **one FC head per imitated agent**, using only the top agent's head at inference. FReLU, SE and transformer layers at 384 ch. Training for more epochs (about 1 week) helped. "We couldn't make RL work out." | https://www.kaggle.com/competitions/lux-ai-2021/writeups/team-durrett-4th-solution-team-durrett-il-from-mul (orig. discussion/296938) · https://github.com/takuwwwo/LuxAI |
| 5 | ironbar (Guillermo Barbadillo) | 1791.8 | **Imitation learning** | Conditioned U-Net. The conditioning vector encodes game state and *which agent to imitate*, trained on 82 agents (>1700 LB) and about 16k matches. He notes a ~300-point gap to Toad Brigade, and that 300 matches weren't enough to copy them well. | https://www.kaggle.com/competitions/lux-ai-2021/discussion/293911 · https://github.com/ironbar/luxai |
| 6 | nosound | 1753.2 | Imitation learning ("multiple attempts at RL did not succeed") | Fine-tune last layers on a single agent | https://www.kaggle.com/competitions/lux-ai-2021/discussion/293776 |
| 8 | A.Saito | 1743.1 | IL network + C++ MCTS | Reimplemented the game engine in C++ for search and self-play data | https://www.kaggle.com/competitions/lux-ai-2021/discussion/294603 |

### Did RL or rules win?
**RL won**, with places 1 and 2 both deep RL. The organisers called them "a great masterclass in reinforcement learning"
(recap 296519). Places 4-6 were imitation learning, and the best rule-based agent finished around 12th-15th
(https://www.kaggle.com/competitions/lux-ai-2021/discussion/293953, "Top rules-based agent?"). The winners' own reasons:
a single network controlling all units sidesteps the per-unit credit-assignment problem that sank RL in Halite, and the
agent kept improving "near-continuously" where the rules agent plateaued (294993). The 5th-place author attributes the
gap between imitators and RL to how far ahead Toad Brigade was: you can't clone your way past the best agent (293911).

### Transferable lessons
1. **Treat the whole multi-unit army as one image-in, image-out policy.** A fully convolutional action map with masking
   solves variable unit counts and coordination under a single reward. Both top teams did this. (294993, 300844)
2. **Anchor self-play with a KL to a teacher or previous best** to stop strategic cycling and forgetting. Both top
   teams did this. (294993, 300844)
3. **Reward shaping is scaffolding.** The winner shaped for the first 20M steps, then trained bigger nets on pure
   win/loss with distillation from the smaller ones. (294993)
4. **Environment speed and off-policy learning matter when simulation is slow.** RLIAYN says IMPALA would have suited
   the slow engine better than PPO. (300844)
5. **Imitation caps out below the imitated.** IL teams clustered at 4-6, a few hundred points behind RL. (293911, 296938)
6. **Compute is not the moat.** The winner trained on one home PC with two GPUs. (294993)

### Sources
- API: `GetCompetition`/`ListPages`/`GetLeaderboard` for `lux-ai-2021`
- Recap: https://www.kaggle.com/competitions/lux-ai-2021/discussion/296519; community summary https://www.kaggle.com/competitions/lux-ai-2021/discussion/294459
- Writeups as linked above; /discussion/293953 (rule-based ~12th-15th)

---

## Cross-competition arc (for the series spine)

| Year | Competition | Winner's family | Why (winner's own words, condensed) |
|---|---|---|---|
| 2020 | ConnectX | Search / perfect solver | Solved game, so perfect play is reachable |
| 2020 | Halite IV | **Rules** | Per-unit RL credit assignment too hard (183543) |
| 2020 | Google Football | **Deep RL + league** | One controlled player, dense shaping, league diversity (202232) |
| 2020-21 | Rock Paper Scissors | Bandit over an ensemble of predictors | Beat the population of public bots and stay unpredictable (221502) |
| 2020-21 | Santa 2020 | Supervised GBDT on replays | Hidden state recoverable from replays, and RL scored lower (221030) |
| 2021 | Hungry Geese | RL self-play + search (writeup deleted) | NN priors + search; HandyRL's self-play net had better priors than BC (263686) |
| 2021 | Lux AI S1 | **Deep RL from scratch** | Single network for all units; RL out-improved rules within a month (294993) |

The recurring cast is a key story thread. Tom Van de Wiele won Halite with rules after RL failed, then
finished 3rd in Football with clone + RL. LiamK and IsaiahP went from 2nd in Geese (BC + MCTS) to 1st in Lux (pure RL), explicitly
because Halite had made them expect rules to win. Théo Cachet went from the Halite RL discussions to 2nd in Lux. YuriCat and
kyazuki went from 5th in Football to 1st in Geese with the same HandyRL library.

## Gaps / UNVERIFIED summary
- Hungry Geese 1st (HandyRL): writeup deleted, so the final architecture, search and compute are unverified.
- Halite 3rd (convexOptimization), RPS 3rd (Stas Sl), Lux 3rd (skyramp), Football 4th (zsp1197 zsp): no writeups found.
- WeKick (Football 1st): training compute and code not published.
- Santa 3rd (Coffee Candy): which model family made the top submission is unstated.
- Halite and Football per-step timeouts: taken from current kaggle-environments defaults, not the 2020 rules pages.
- ConnectX: rolling leaderboard with no final standings. "Who led in 2020" rests on Tom Van de Wiele's and the Raw Beast
  team's statements only.
