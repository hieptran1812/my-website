---
title: "Emergent Behaviour in Multi-Agent Systems: From Hide-and-Seek to AlphaStar"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How a binary win/lose reward and millions of self-play games produce tool use, role specialisation, and superhuman strategy that nobody programmed — and the league-training machinery that makes it reliable."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "multi-agent",
    "self-play",
    "emergence",
    "machine-learning",
    "pytorch",
    "game-theory",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/emergent-behaviour-and-multi-agent-games-1.png"
---

In 2019 a team at OpenAI dropped a handful of simulated agents into a walled physics playground, split them into hiders and seekers, and handed them the simplest reward imaginable: hiders score `+1` for every step no seeker can see them, seekers score `+1` for every step they can. No one told the agents that boxes could be pushed, that ramps could be climbed, or that doors could be barricaded. The first two million episodes were pure noise — agents jittered in place, ran into walls, lost track of each other. Then something happened that no engineer wrote down: the hiders learned to grab boxes and wall themselves into a room. A few million episodes later the seekers learned to drag a ramp over to the wall and climb in. The hiders responded by stealing the ramps first and locking them away. The seekers responded by exploiting a physics bug — they learned to stand on a box and "surf" it across the floor by pushing against it, gliding over the hiders' barricade.

Nobody designed any of those skills. They fell out of two facts: a reward function, and an opponent who was also learning. That is the entire subject of this post — **emergent behaviour**, the complex, structured, often surprising group behaviour that arises in multi-agent reinforcement learning (MARL) when you have simple individual incentives plus rich interaction, and nobody scripts the strategy. The same phenomenon produced OpenAI Five, the five-agent team that beat the Dota 2 world champions; AlphaStar, which reached Grandmaster in StarCraft II; and AlphaGo's "move 37," a stone placement so alien that professional commentators thought it was a mistake before it won the game.

Figure 1 traces the six phases of the hide-and-seek arms race, and we'll return to it repeatedly because it is the cleanest example in all of RL of capability emerging for free. If you read the whole series, this is the post where the agent↔environment↔reward loop — the spine of everything from `what-is-reinforcement-learning` to the policy gradient theorem — gets a second agent dropped into the environment, and the environment stops holding still. That single change, **non-stationarity from a learning opponent**, is what makes emergence possible and what makes it hard to control.

By the end you will understand *why* adversarial pressure manufactures novel strategy (the Red Queen effect), *how* the production systems actually train — self-play, league training, and Policy Space Response Oracles (PSRO) — with runnable code for a self-play loop, an Elo rating system, and a league setup, and *what* the measured results were on named games. You will also know the failure modes: the cycling and catastrophic forgetting that sink naive self-play, and the trivial equilibria that make agents stop learning. Let's build it up from first principles.

![Timeline of the six emergent strategy phases in OpenAI hide-and-seek from random play through box surfing](/imgs/blogs/emergent-behaviour-and-multi-agent-games-1.png)

## 1. What "emergent behaviour" actually means

Start with a definition that does real work. **Emergent behaviour** is group-level structure or capability that (a) is not explicitly encoded in any agent's reward, policy, or training objective, and (b) arises predictably from the *interaction* of agents each optimising a simple local objective. The two clauses matter equally. Random noise is unprogrammed but not emergent because it does not arise predictably from interaction. A hard-coded finite-state-machine bot has structure but it is not emergent because a human wrote it.

The classic single-agent RL setup is a Markov Decision Process (MDP): states $s$, actions $a$, a transition kernel $P(s' \mid s, a)$, a reward $r(s, a)$, and a discount $\gamma$. The agent learns a policy $\pi(a \mid s)$ to maximise expected discounted return $\mathbb{E}\left[\sum_t \gamma^t r_t\right]$. The environment is *stationary*: $P$ does not change while you learn. Almost every algorithm in this series — Q-learning, DQN, PPO, SAC — leans on that stationarity for its convergence story.

Multi-agent RL breaks it. With $N$ agents, the effective environment seen by agent $i$ includes the other $N-1$ policies, which are themselves changing. Agent $i$'s transition becomes $P(s' \mid s, a^1, \dots, a^N)$ and its reward $r^i(s, a^1, \dots, a^N)$ depends on the joint action. From agent $i$'s perspective the world is **non-stationary**: the same action in the same state yields different outcomes over time because everyone else got better. This is the formal reason single-agent convergence guarantees evaporate, and — flipped around — it is exactly the engine of emergence. A non-stationary environment is one that keeps presenting new problems, so the agent keeps finding new solutions.

Three interaction structures span the space. In **cooperative** MARL all agents share one reward (think a robot soccer team that only scores as a unit); the challenge is credit assignment across agents. In **competitive** (zero-sum) MARL one agent's gain is another's loss (hide-and-seek, chess, StarCraft); the challenge is that there is no fixed target to converge to. In **mixed** settings (Dota, traffic, markets) agents both cooperate within teams and compete across them. Emergence shows up in all three, but it is most dramatic in competitive and mixed settings because the opponent actively *manufactures* harder problems for you. We'll see that the entire design of league training is an answer to "how do I keep manufacturing good problems without the system collapsing into a degenerate loop?"

The hallmark of MARL, then, is this: you specify the *game*, not the *strategy*. You write down who the agents are, what they observe, what actions they have, and how reward flows. The strategy — door blocking, ramp surfing, role specialisation, a never-before-seen Go opening — is discovered. That is both the promise (you get capability you could never hand-engineer) and the peril (you get capability you did not ask for and cannot easily predict).

## 2. OpenAI hide-and-seek: six phases of emergent tool use

Let's make the hide-and-seek result concrete because it is the canonical demonstration. Baker et al. (2019), "Emergent Tool Use From Multi-Agent Autocurricula," set up a physics environment (built on MuJoCo) with one to three hiders and one to three seekers, movable boxes and ramps, and static walls and rooms. Hiders get a small prep phase to set up before seekers are released. The reward is team-based and binary in spirit: hiders are rewarded when *no* hider is visible to *any* seeker, seekers when they can see *any* hider. Agents observe nearby objects and other agents via a vision cone and use a transformer-based policy trained with PPO (Proximal Policy Optimization — see `proximal-policy-optimization-ppo` for the clipped-surrogate mechanics this relied on). Crucially: there is no reward term for "use the box," "build a fort," or "climb the ramp." Those are the things that *emerged*.

The paper documents six phases, each a distinct strategy regime separated by a sharp transition in behaviour and reward. The agents progress through them roughly in this order, with each phase appearing only after enough episodes that the previous one became reliable:

1. **Random play.** Agents move and grab essentially at random. Reward hovers near chance.
2. **Chasing and fleeing.** Seekers learn to pursue; hiders learn to run. Pure pursuit, no tool use.
3. **Door blocking / fort building.** Hiders discover they can push boxes to seal off doorways and corners, building a shelter the seekers cannot enter. Hider reward jumps.
4. **Ramp use.** Seekers discover ramps: drag a ramp to the fort wall and climb over it. Seeker reward jumps; the fort is no longer safe.
5. **Ramp locking (ramp defence).** Hiders counter by grabbing the ramps during prep phase and locking them inside the fort or far away, denying the seekers their tool.
6. **Box surfing.** Seekers exploit the physics engine: they stand on top of a box and, because the agent can apply force to a held object, push the box while standing on it to *surf* across the level — over walls, over the hiders' barricade. (The paper notes this exploited under-constrained physics and that adding a tether penalty changed the dynamics.)

The reason this is the textbook example: the total reward signal is almost content-free. There is no curriculum, no demonstrations, no shaped reward for any of these behaviours. The curriculum is *automatic* — the paper's term is **autocurriculum** — generated by the agents themselves. Each team's solution becomes the next problem for the other team, and that loop, not a designer, schedules the difficulty.

#### Worked example: reading the phase transitions

Suppose you are monitoring a hide-and-seek run and you log mean hider reward per episode in windows of 100k episodes. A plausible trace (numbers approximate, consistent with the paper's qualitative curves) looks like this:

| Episode window | Mean hider reward | Dominant strategy | Interpretation |
|---|---|---|---|
| 0–2M | ≈ 0.05 | Random | No structure yet |
| 2M–10M | ≈ 0.30 | Chase/flee | Hiders learn to run |
| 10M–40M | ≈ 0.65 | Fort building | Hiders dominate — sharp rise |
| 40M–80M | ≈ 0.35 | Seekers use ramps | Hider reward *drops* — counter found |
| 80M–150M | ≈ 0.60 | Ramp locking | Hiders recover |
| 150M–400M | ≈ 0.30 | Box surfing | Seekers exploit physics |

The diagnostic insight: in single-agent RL a *falling* reward curve means your training is broken. In competitive MARL, a falling reward for one team is the *signal that the opponent just got smarter* — a phase transition, not a bug. You confirm an emergent phase by watching for these non-monotonic steps in *one team's* reward that coincide with a step *up* in the other's. If both teams' rewards are flat and high, you've hit a stalemate; if both are flat and low, you've hit a degenerate equilibrium (more on that in §12). The way this works in practice is that you watch the *gap* and its sign changes, not either curve alone.

#### Worked example: dissecting the box-surfing exploit

Walk through Phase 6 — box surfing — in full mechanical detail, because it is the cleanest case in the literature of an emergent behaviour that is simultaneously a brilliant solution and an unambiguous exploit, and seeing exactly *why* tells you how to read every reward-hacking episode that follows.

First, the **reward signal that caused it**. The game is zero-sum: the seeker's reward is the negation of the hider's, and operationally the seeker is rewarded for every timestep on which it can *see at least one hider* in its vision cone. By Phase 5 the hiders have a stable defence — they barricade themselves in a room with boxes and, during the prep phase, grab every ramp and lock it away so the seekers have no tool to climb the wall. From the seeker's point of view the reward has gone to roughly zero: the hiders are walled in, the ramps are gone, and there is no scripted action labelled "get over the wall" available. The gradient is starving. *Anything* the seeker can do that produces even a single timestep of hider-visibility is now worth more than the flat-zero status quo — and that is the pressure that drives the search into the physics engine's loopholes.

Second, **what the seekers discovered**. An agent can apply force to a box it is holding. The seekers learned to climb onto *top* of a box and then, while standing on it, push against it — and because the physics engine let an agent exert force on an object it was simultaneously standing on, the box (and the agent riding it) *slid* across the floor. The seeker had effectively built itself a vehicle: it could "surf" the box over to the hiders' barricade and across or over it, getting line-of-sight on the hiders the ramps used to provide. The hiders had locked away the ramps, but they could not lock away the boxes — the boxes were load-bearing parts of their own fort — so the very tool the hiders depended on became the seekers' route in.

Third, and most important, **exactly why this is an exploit rather than a legitimate strategy**. A legitimate strategy is one that exploits the *intended* dynamics of the world; box-surfing exploited an *unintended* one. In a faithful physics simulation, an agent standing on a box cannot push that same box and ride it — the forces should cancel or the agent should simply fall off. Box-surfing worked only because the simulator's physics were *under-constrained*: it permitted a force relationship between an agent and the object beneath it that is not physically realisable. The agent did not break any rule the designers wrote; it found a rule the designers *forgot to write*. This is the signature of a reward hack — the policy is optimising the literal reward (timesteps of visibility) using a literal affordance of the literal environment (the buggy force model), in a way that maximises the metric while violating the designer's intent. The paper notes this directly: it was a consequence of under-constrained physics, and adding a penalty (a tether constraint that discouraged moving objects far during prep, in their variant) changed the dynamics and closed the loophole.

Fourth, **how the hiders responded** — and why this still counts as healthy emergence even though it began as a bug. Once seekers could surf, the hiders' walled-room defence was no longer safe, so the *autocurriculum continued*: the hiders were pushed to lock away or otherwise neutralise the boxes too, denying the seekers their surfboard the same way they had earlier denied them the ramps. The arms race did not stop at the exploit; it absorbed the exploit as just another tool to be contested. That is the dual lesson of box-surfing in one episode: emergence is prior-free optimisation, so it will find genuinely creative solutions *and* genuine exploits with equal enthusiasm — which is exactly why §12 insists you close trivial wins and why a deployed emergent policy must always be probed for the loopholes it has surely found.

## 3. Why emergence happens: the Red Queen effect

Now the theory — the *why*. Why does adversarial interaction reliably produce escalating capability, when a single agent in a fixed environment plateaus the moment it solves the task?

The mechanism is borrowed from evolutionary biology and named after the Red Queen in *Through the Looking-Glass*, who tells Alice "it takes all the running you can do, to keep in the same place." In a coevolving system, fitness is *relative*: a hider is not trying to achieve an absolute score, it is trying to beat *this particular seeker*. When the seeker improves, the hider's effective fitness drops even if the hider did not change. So the hider must improve just to stay even. Each side runs as fast as it can and the *relative* standing barely moves — but the *absolute* capability of both ratchets upward without bound (until some hard limit like the physics engine or compute is hit).

Figure 2 draws this as an escalation tree rather than a cycle, which is deliberate and important. The naive picture of an arms race is a *cycle*: A beats B, B beats A, repeat forever. That can happen, and when it does it is the failure mode (§9, §10). But a *healthy* arms race is not a cycle — it is a *tree* that keeps branching into genuinely new strategy space. Hider "block the door" is not the same node you return to after "lock the ramp"; it is a strictly more capable behaviour built on top. The escalation is the productive case; the cycle is the pathological case. Most of the engineering in league training and PSRO is machinery to keep the system in the productive escalation regime and out of the cycle.

![Branching escalation tree where each team's counter-strategy forces the opponent to invent a new counter](/imgs/blogs/emergent-behaviour-and-multi-agent-games-2.png)

There is a clean way to see *why* the opponent manufactures good problems. Consider the seeker's optimisation target at training time $t$:

$$
\pi^{\text{seek}}_{t+1} = \arg\max_{\pi} \; \mathbb{E}_{\text{hider} \sim \pi^{\text{hide}}_t}\left[ R^{\text{seek}}(\pi, \pi^{\text{hide}}_t) \right].
$$

The seeker is computing a **best response** to the *current* hider. When the hider then computes a best response to *that* seeker, the hider is being pushed into exactly the region of strategy space where the seeker is currently weak — which is, by construction, a region the hider has not yet explored. The opponent is a problem-generator that always poses the problem you are least equipped to solve. Compare that to a fixed environment, where once you solve it the gradient goes to zero and you stop. The adversary keeps the gradient alive. This is the deep reason competitive self-play is such a strong source of open-ended learning, and it is why the same idea recurs from AlphaGo to GANs to adversarial robustness: a learning adversary is a curriculum that never runs out.

The catch is that "best response to the current opponent" is a *local* and *forgetful* objective. The seeker is not trying to be good against *all* hiders, only the current one. That is the seed of catastrophic forgetting, which we will pay for in §9.

## 4. OpenAI Five: cooperation, roles, and sacrifice in Dota 2

Hide-and-seek is competitive between two teams. OpenAI Five (2019) added a second axis: *cooperation within a team* under competitive pressure across teams — the mixed setting. Dota 2 is a 5-versus-5 game with a continuous flow of decisions, a partial-observability fog of war, roughly 20,000-dimensional observations, and a combinatorial action space (the paper reports on the order of $10^{4}$ to $10^{5}$ valid actions per step depending on encoding). Each of the five heroes was controlled by a separate LSTM policy network sharing weights, trained with PPO at enormous scale — the system played the equivalent of roughly **180 years of Dota 2 per day** against itself, accumulated over months.

What emerged, with no explicit reward for any of it:

- **Role specialisation.** Although all five networks shared parameters, the agents differentiated into recognisable Dota roles (carries that farm gold, supports that enable teammates) purely because the team reward made specialisation more profitable than five identical players.
- **Communication as timing.** The agents could not literally talk, but they coordinated ganks (coordinated kills) and team fights by *implicit* synchronisation — converging on the same target at the same moment because their shared situational features made the same play optimal for all of them simultaneously. Coordination emerged as a Schelling-point phenomenon, not a message-passing protocol.
- **Sacrifice plays.** Agents learned to trade a hero's life for map control or an objective when the discounted team return justified it — a behaviour that looks altruistic but is just the team-reward gradient.

The headline result: in April 2019, OpenAI Five beat **OG**, the reigning Dota 2 world champions, **2–0** in a best-of-three, and in a subsequent public "Arena" period won the large majority of its tens of thousands of games against human teams. The honest caveats matter: the game was played under restrictions (a fixed pool of heroes, no item drafting in some configurations), and human pros adapted within days once they could study the bot. But the qualitative point stands — coordinated, role-differentiated, sacrificial team play emerged from a team reward and self-play, with zero hand-coded tactics.

The engineering lesson is about *reward design for cooperation*. OpenAI Five used a shared team reward with a small amount of shaping (rewards for gold, experience, kills, with a "team spirit" coefficient that was *annealed from 0 toward 1* over training — early on each agent cared about its own stats, later it cared about the team's). That annealing schedule is itself a curriculum: let agents first learn to be individually competent, then gradually fuse their incentives. We'll generalise this into a takeaway about reward shaping for emergence in §12.

It is worth dwelling on the **team-spirit hyperparameter** $\tau$ because it is the single cleanest knob for cooperation in all of these systems, and it makes the cooperation-versus-emergence tension legible. Each hero $i$ received a reward of the form $r_i = (1 - \tau)\, \rho_i + \tau\, \bar{\rho}$, where $\rho_i$ is the hero's *own* shaped reward (its gold, its experience, its kills) and $\bar{\rho}$ is the *mean* reward across the five heroes. At $\tau = 0$ each agent is purely selfish — it optimises its own farm and its own kills and will happily let a teammate die to grab a last-hit. At $\tau = 1$ each agent is fully cooperative — it sees only the team-averaged reward and is utterly indifferent to which hero on the team gets the gold, as long as the *team* gets it. Neither extreme works well on its own. At $\tau = 0$ you get five greedy carries who never set up a kill for anyone else and never sacrifice; at $\tau = 1$ early in training you get a credit-assignment swamp — an agent that makes a good individual play gets the same averaged reward as its four idle teammates, so the gradient signal for *learning the play in the first place* is diluted fivefold. OpenAI's reported solution was to *anneal* $\tau$ upward over training, starting near $0$ and ending near $1$, so that what looked like a constant value of roughly $\tau \approx 0.6$ in the public discussions was really a point on a schedule: high enough that an agent will trade its own resources for a teammate's stronger position, but reached only after the agents had already learned, under lower $\tau$, what a good individual play even looks like. The deep point is that **cooperation is itself emergent under the right $\tau$ schedule** — you do not script "support the carry," you tune the degree to which each agent internalises the team's outcome and let the support role fall out of the gradient.

Two more concrete emergent behaviours are worth naming because they show how literal the optimisation is. First, the agents discovered a **courier-management** strategy that surprised the developers: the courier is a fragile unit that ferries items from base to heroes, and OpenAI Five learned to use it *early and aggressively* — sending it out to deliver regeneration items at moments human players would consider too risky — and even learned to route couriers *to the side lanes* to keep them out of the enemy's line of fire while still delivering. Nobody rewarded "protect the courier" or "use the courier early"; it fell out of the value of the items it carried under the discounted team return. Second, and more notoriously, the agents found a **"hand-in" exploit**: in some configurations of the reward, certain actions were worth more to the optimiser than the developers intended, and the agents learned to exploit a quirk in how an item or buyback was credited — a small reward-hacking episode that the team had to patch out, in the same family as hide-and-seek's box-surfing. The lesson recurs: at $180$ years of self-play *per day*, accumulated over many months into thousands of subjective years of experience, the policy will find every loophole in the literal reward long before a human would think to close it. Scale is not just a route to strength; it is a relentless search for the cheapest path to reward, intended or not.

## 5. AlphaStar and the invention of league training

StarCraft II is harder than Dota in a specific way that forced a genuinely new training idea. It is a 1-versus-1 real-time strategy game with imperfect information, a vast action space, and — critically — a famous **strategic cycle**: a rush beats a greedy economy, an economy beats a slow turtle, a turtle beats a rush. This is essentially rock-paper-scissors at the strategy level. Naive self-play *cannot* solve rock-paper-scissors, and understanding why is the key to this whole section.

In naive self-play, agent version $t+1$ trains only against version $t$ (or a recent average). If version $t$ plays "rock," version $t+1$ learns "paper." But then version $t+2$ trains against the "paper" agent and learns "scissors," forgetting that scissors loses to rock. Version $t+3$ relearns rock. The population *cycles* and never converges, and worse, an agent that forgets how to beat rock is *exploitable* the moment anyone plays rock again. This is **strategic catastrophic forgetting**: the policy improves against the current opponent by *un-learning* the response to a past opponent.

DeepMind's AlphaStar (Vinyals et al., 2019, "Grandmaster level in StarCraft II using multi-agent reinforcement learning") solved this with **league training**. Instead of one self-play opponent, you maintain a growing *league* — an archived population of past agents — and you sample opponents from the whole league, with a prioritised distribution that favours opponents the current agent is *losing* to. Crucially, the league contains three distinct *types* of agent, each plugging a different hole:

Figure 4 stacks the three roles. **Main agents** are the ones you actually ship; they maximise win rate against the entire league (and against past versions of themselves), so they cannot afford to forget any counter. **Main exploiters** train *only* against the current main agents, searching aggressively for any weakness in them — they are a dedicated red team. **League exploiters** train against the *entire* league to find systemic weaknesses that any main agent shares. The exploiters' job is not to be shipped; it is to *generate hard, diverse opponents* so the main agents never get comfortable. Periodically, snapshots of all three types are frozen and added permanently to the league, so old strategies are never lost.

![Stacked architecture of AlphaStar league roles main agents main exploiters and league exploiters feeding a frozen archive](/imgs/blogs/emergent-behaviour-and-multi-agent-games-4.png)

The result: AlphaStar reached **Grandmaster** level — above 99.8% of ranked human players on the official Battle.net ladder — in all three races (Terran, Protoss, Zerg), playing under camera-view and action-rate restrictions designed to keep it within human physical limits. The emergent behaviour here is subtler than hide-and-seek's tool use: it is *strategic robustness*. The main agent does not just play one strong strategy; it plays a *distribution* of strategies and switches based on what it scouts, because the league punished any agent that was a one-trick pony. Robustness against a diverse population is the emergent capability that league training is engineered to produce.

#### Worked example: why the three roles each earn their keep

Imagine a league with only main agents (plain prioritised self-play). The main agents collectively drift toward a strong "meta" strategy, say a particular Protoss build. Because every main agent plays roughly that build, none of them has any incentive to learn the counter to a weird cheese rush — nobody in the league does the cheese rush. A human ladder opponent who *does* cheese rush wins. The league has a *blind spot* it cannot see because no member exposes it.

Now add a **main exploiter** whose entire reward is beating the current main agents. It will find the cheese rush, because that is the path of least resistance to a win against the current meta. Once it starts winning with the cheese, the prioritised sampler feeds that exploiter to the main agents as a frequent opponent, and the main agents are *forced* to learn the defence. The exploiter then has to find a *new* hole. The **league exploiter** does the same but against the whole archive, catching weaknesses that are systemic rather than specific to the latest main agent. Concretely, DeepMind reported that the league grew to hundreds of distinct agents over training, and ablations removing the exploiters produced agents that were strong on average but *exploitable* by specific strategies — exactly the blind-spot failure. The three-role stack converts "strong on average" into "hard to exploit," which is what Grandmaster-level play against adaptive humans actually requires.

### AlphaStar technical details

It is worth opening the hood on AlphaStar's agent itself, because the *scale and shape* of the problem explain why a simple policy network would never have worked and why league training was necessary rather than merely convenient.

Start with the **action space**, which is where StarCraft's difficulty is most vivid. At each step the agent must choose *what* to do (an action type — move, build, attack, train a unit), *which* of its own units to do it with (selecting from up to dozens of units, often as a group), *where* on the map to do it (a target point on a roughly $256 \times 256$ spatial grid), and *when* to act next. Multiplying these together — action types, the combinatorial choice of which units to select, and the spatial target — DeepMind estimated on the order of $10^{26}$ possible actions at *every single decision point*. For comparison, Go has about $10^{170}$ possible games but only a few hundred legal moves per position; StarCraft's branching factor *per step* dwarfs Go's, and the agent makes hundreds of such decisions per game under a real-time clock. You cannot enumerate this action space, you cannot tree-search it the way AlphaGo searched Go, and you certainly cannot hand-script a policy over it. It has to be learned, and the architecture has to *factorise* the choice so the network is not forced to score $10^{26}$ outputs directly.

That factorisation is the heart of the **architecture**, which is a hybrid stack rather than a single network. Raw observations come in three flavours — scalar features (minerals, gas, supply, the game clock), a set of entities (every visible unit and building, each with its own feature vector), and the spatial minimap — and each gets its own encoder. The **entity list is processed by a transformer**, so the agent can attend over all its units and the enemy's relationally (which units threaten which) rather than treating them as an unordered bag. The **minimap is processed by a convolutional residual network** to preserve spatial structure. These encodings are fused and fed into a **deep LSTM core**, which is what gives the agent memory across time — essential under imperfect information, where you must remember that you scouted an enemy expansion thirty seconds ago even though it is now back under fog of war. The LSTM's state is then decoded *auto-regressively*: the network first picks the action type, then conditions on that to pick the units, then conditions on both to pick the spatial target, and so on. Critically, the unit and target selection uses a **pointer network** — instead of choosing from a fixed output vocabulary, the network produces a query and *points at* one of the variable-length set of entities it encoded, which is exactly how you select "that specific marine" out of an army whose size changes every step. This auto-regressive, pointer-based decoder is what tames the $10^{26}$ action space into a sequence of tractable conditional choices.

On the **league's three agent types and their meta-game roles**, it is worth being precise about how the three differ in their opponent distribution and their reset behaviour, because that is what makes each one earn its keep. **Main agents** are trained against a prioritised mixture of the *entire* league plus 35% self-play against their own past copies; they are the agents that get shipped, and their job is to be robust against everything that has ever existed in the league. They are never reset — they accumulate strategy continuously. **Main exploiters** train *only* against the current main agents (and themselves), with the sole objective of finding any exploit in the current shipped policy; crucially, they are periodically *reset* to a supervised-learning checkpoint once they have either found an exploit or run long enough, so they keep searching fresh holes rather than slowly converging into another main agent. **League exploiters** train against the *whole* league to surface *systemic* weaknesses shared across many agents, and they too are reset on a schedule. The meta-game role is a division of labour: main agents play to be un-exploitable, main exploiters generate point-attacks on the current meta, and league exploiters generate broad pressure that no single main agent can dodge. Snapshots of all three are frozen into the permanent archive on a schedule, which is what guarantees no past strategy is ever truly forgotten.

The **headline result** deserves its precise form. AlphaStar was rated by Battle.net's own matchmaking ranking, the same ladder humans climb, and it reached **Grandmaster** — the top league — across *all three races*: it achieved Grandmaster as Protoss, as Terran, and as Zerg, an "**11/11**" in the sense that it was evaluated and ranked Grandmaster on every race-and-ladder combination DeepMind reported, placing it above 99.8% of the active ranked human population. It played under deliberate handicaps designed to keep the comparison fair: a limited camera view (it had to *move the camera* to see different parts of the map, rather than reading the whole game state at once) and a capped action rate with human-like reaction delays, so its advantage came from *decision quality*, not from inhuman clicking speed. The emergent capability, as in the worked example above, was not a single dominant build but *strategic robustness* — a learned distribution over openings and responses, switched on the basis of what the agent scouted, because the league had punished every one-trick pony out of existence.

## 6. AlphaGo and the move nobody expected

AlphaGo predates the others (the Lee Sedol match was March 2016) but it is the purest demonstration that *self-play discovers strategy humans never found*. Go has perfect information and a single agent playing a symmetric zero-sum game against a copy of itself, so it is the simplest competitive-emergence setting. AlphaGo combined a policy network and a value network with Monte Carlo Tree Search; AlphaGo Zero (2017) removed human game data entirely and learned **purely from self-play starting from random**, which is the version that matters for our argument.

Starting from random play, self-play in Go climbs through a curriculum it generates for itself — the same autocurriculum logic as hide-and-seek, but in a board game. Early self-play games are terrible, so the bar to beat is low; as both copies improve, the bar rises in lockstep (the Red Queen again). AlphaGo Zero surpassed the version that beat Lee Sedol in three days of self-play, and reached superhuman strength without ever seeing a human game.

The emblem of emergence here is **move 37** in game 2 against Lee Sedol — a shoulder hit on the fifth line that violated centuries of Go orthodoxy. Commentators initially thought it was a mistake; AlphaGo's own estimate was that a human would play it with probability about 1 in 10,000. It turned out to be the move that defined the game. The point is not that one move was good; it is that *self-play has no human prior to anchor it*. It explores strategy space according to what *wins*, not what looks reasonable, so it routinely finds regions humans pruned away for cultural rather than game-theoretic reasons. Emergence is, in part, the absence of a human prior. This is also a warning: the same lack of prior is why hide-and-seek found a physics exploit and why reward hacking is a constant risk — the agent optimises the literal objective, not the intended one.

## 7. Self-play, formally: the training loop you can actually run

Time to write code. The core of all of these systems is a self-play loop. Strip away the scale and it is short. Below is a runnable self-play training skeleton for a simple competitive game using PyTorch and Gymnasium-style conventions; it shows the *structure* — sample an opponent, play, update — that AlphaStar and OpenAI Five elaborate.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)  # logits

    def act(self, obs):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a.item(), dist.log_prob(a)

def play_episode(env, learner, opponent):
    """One self-play episode. Returns learner trajectory and final reward."""
    obs, _ = env.reset()
    log_probs, rewards = [], []
    done = False
    while not done:
        # learner moves on even plies, opponent on odd plies
        if env.to_move == 0:
            a, lp = learner.act(torch.as_tensor(obs, dtype=torch.float32))
            log_probs.append(lp)
        else:
            with torch.no_grad():
                a, _ = opponent.act(torch.as_tensor(obs, dtype=torch.float32))
        obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        if env.to_move == 1:        # reward credited after learner's move resolves
            rewards.append(r)
    return log_probs, sum(rewards)
```

That is a pure-REINFORCE learner; in production you would use PPO's clipped surrogate (the variance-reduction story is in `proximal-policy-optimization-ppo`) and a value baseline, but the *self-play scaffolding* is identical. The interesting decisions are not in the gradient step — they are in **which opponent you sample**. Naive self-play sets `opponent = copy_of(learner)`. That single line is the one that causes cycling. The rest of this post is about replacing it.

Here is the training driver that makes the opponent choice explicit:

```python
def self_play_train(env, obs_dim, n_actions, episodes=200_000, lr=3e-4):
    learner = PolicyNet(obs_dim, n_actions)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)
    league = deque(maxlen=50)              # archive of frozen past policies
    league.append(snapshot(learner))

    for ep in range(episodes):
        opponent = sample_opponent(league, learner)   # <-- the key design choice
        log_probs, ret = play_episode(env, learner, opponent)

        # REINFORCE update on the learner's actions
        opt.zero_grad()
        loss = -(ret * torch.stack(log_probs).sum())
        loss.backward()
        opt.step()

        if ep % 2000 == 0:                  # periodically freeze a checkpoint
            league.append(snapshot(learner))
    return learner, league

def snapshot(net):
    frozen = PolicyNet(net.net[0].in_features, net.net[-1].out_features)
    frozen.load_state_dict(net.state_dict())
    for p in frozen.parameters():
        p.requires_grad_(False)
    return frozen
```

The `sample_opponent` function is where naive self-play, fictitious play, and league training diverge — and where you either get healthy escalation or pathological cycling.

## 8. Rating agents: Elo and why you need it

Before we can build a *good* `sample_opponent`, we need a way to *measure* whether agent B is actually stronger than agent A. In a single-agent task you read the reward off the environment. In a competitive multi-agent task there is no absolute reward — strength is *relative to the population*. The standard tool is the **Elo rating system** (from chess), and it does double duty: it tells you if you are making progress, and it tells you which opponents are worth sampling.

Elo models the probability that player A beats player B as a logistic function of their rating difference:

$$
P(A \text{ beats } B) = \frac{1}{1 + 10^{(R_B - R_A)/400}}.
$$

A 400-point gap means the stronger player wins about 10 times as often. After a game, ratings update toward the surprise:

$$
R_A \leftarrow R_A + K\,(S_A - E_A),
$$

where $S_A \in \{0, 0.5, 1\}$ is the actual result, $E_A$ is the expected result from the formula above, and $K$ (typically 16–32) controls update speed. Here is a clean implementation:

```python
def expected_score(r_a, r_b):
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def update_elo(r_a, r_b, score_a, k=32):
    """score_a: 1.0 win, 0.5 draw, 0.0 loss for player A."""
    e_a = expected_score(r_a, r_b)
    r_a_new = r_a + k * (score_a - e_a)
    r_b_new = r_b + k * ((1.0 - score_a) - (1.0 - e_a))
    return r_a_new, r_b_new

# Example: a 1500 agent beats a 1700 agent (an upset)
ra, rb = update_elo(1500, 1700, score_a=1.0)
print(round(ra, 1), round(rb, 1))   # 1500 jumps, 1700 drops sharply
```

Running that prints approximately `1524.5 1675.5`: the underdog gains 24.5 points for the upset, the favourite loses the same. Elo is how AlphaStar and OpenAI Five reported progress (OpenAI used TrueSkill, a Bayesian cousin that tracks rating *uncertainty* too, which matters when agents play few games). The practical reason you compute Elo *during* training, not just at evaluation, is that it gives you the prioritised sampling distribution: sample opponents you are *expected to lose to* (lower win probability) more often, because those are the ones with something to teach you.

#### Worked example: detecting a cycle with Elo

Here is how Elo exposes the cycling failure mode quantitatively. Suppose you run naive self-play and snapshot agents every 10k episodes: call them $g_1, g_2, g_3, \dots$. You then run a round-robin tournament among the snapshots and compute Elo from the results. In a *healthy* run, Elo should be roughly monotonic in snapshot index: $g_5$ beats $g_1$, $g_{10}$ beats $g_5$, and so on — later is stronger. In a *cycling* run you instead see a rock-paper-scissors pattern in the head-to-head matrix: $g_3$ beats $g_2$, $g_2$ beats $g_1$, but $g_1$ beats $g_3$. The Elo numbers come out nearly *flat* across all snapshots (everyone is roughly 1500) even though each beats the previous one — because Elo measures *transitive* strength and a cycle has none.

Concretely, if your latest agent has Elo ≈ 1500 against the *whole archive* but Elo ≈ 1900 against *only the previous snapshot*, you are cycling, not improving. That gap — strong against the immediate predecessor, mediocre against the archive — is the single most useful diagnostic for self-play health. The fix is to make the agent train against the archive in the first place, which is exactly what league training does.

## 9. Naive self-play vs league training: the diversity problem

Now we can state the contrast cleanly. Figure 3 puts naive self-play and league training side by side, and the difference is entirely about *who is in the opponent set*.

![Before-after comparison of naive self-play cycling and forgetting versus league training with a diverse population](/imgs/blogs/emergent-behaviour-and-multi-agent-games-3.png)

**Naive self-play** trains against one opponent — the current self (or a short moving average). Its objective is "beat the latest me." This has three linked pathologies. First, **cycling**: in any game with a non-transitive strategy structure (rock-paper-scissors, most real RTS games), it chases its own tail forever. Second, **catastrophic forgetting**: improving against the current opponent means the gradient happily discards the response to past opponents, because those are not in the loss. Third, **low diversity**: the whole population collapses toward one strategy, so the agent is brittle — anyone playing an off-meta strategy can exploit it.

**League training** (and its relatives, fictitious self-play and PSRO) train against a *diverse, archived population*. The objective becomes "beat a distribution of past and adversarial opponents." This directly fixes all three: cycles cannot form because old strategies stay in the pool and keep punishing any agent that forgets their counter; forgetting is prevented because the archived agents re-test old skills; diversity is maintained by design and amplified by dedicated exploiter agents.

The mechanism that makes this principled goes back to game theory. **Fictitious self-play** (FSP) trains a best response not against the latest opponent but against the *historical average* of all opponents. The classical result (Brown, 1951) is that in two-player zero-sum games, the time-average of fictitious play *converges to a Nash equilibrium*. Intuitively, by responding to the average you are forced to be good against *everything the opponent has ever done*, which kills the cycle. League training is FSP plus the engineering insight that a *uniform* average is wasteful — you should oversample the opponents you are losing to (prioritised FSP) and add dedicated exploiters to actively probe for holes. Here is prioritised opponent sampling, the heart of a real league:

```python
import numpy as np

def sample_opponent(league, learner, ratings):
    """Prioritised FSP: oversample opponents the learner is expected to lose to."""
    learner_elo = ratings["learner"]
    # win prob of learner vs each archived opponent
    win_probs = np.array([
        1.0 / (1.0 + 10 ** ((ratings[id(o)] - learner_elo) / 400.0))
        for o in league
    ])
    # weight by (1 - p_win)^2 : focus on hard opponents, never zero out easy ones
    weights = (1.0 - win_probs) ** 2 + 1e-3
    weights /= weights.sum()
    idx = np.random.choice(len(league), p=weights)
    return league[idx]
```

The `(1 - p_win)**2` weighting is the prioritisation: an opponent you beat 95% of the time gets weight ≈ 0.0025, one you beat 50% of the time gets 0.25 — a hundredfold preference for the instructive opponent — while the `+1e-3` floor guarantees you still occasionally re-test the easy ones so you don't forget them. This one function is the difference between Figure 3's left and right panels.

Figure 6 shows the complete self-play training loop that wraps the sampling function above. Each iteration samples an opponent from the league, plays a game, updates ELO ratings, and conditionally archives the agent if it clears a strength threshold — turning the loop into a self-curating curriculum.

![Pipeline of the self-play training loop sampling an opponent playing a game updating ELO and adding strong agents to the league](/imgs/blogs/emergent-behaviour-and-multi-agent-games-6.png)

## 10. PSRO: the principled framework underneath league training

League training is, frankly, a pile of well-chosen heuristics. **Policy Space Response Oracles (PSRO)** (Lanctot et al., 2017) is the clean theory that explains why those heuristics work and generalises them. It unifies self-play, fictitious play, and the double-oracle algorithm under one umbrella, and it is worth understanding because it tells you *what* the league is approximating.

PSRO maintains a finite set of policies $\Pi = \{\pi_1, \dots, \pi_k\}$ for each player and an **empirical payoff matrix** (the "meta-game") whose entry $(i, j)$ is the expected outcome when policy $\pi_i$ plays $\pi_j$. The loop, shown in Figure 8, is:

1. **Solve the meta-game.** Compute a Nash equilibrium (or another solution concept) over the *current* finite strategy set. This yields a mixture $\sigma$ — a probability distribution over which of your existing policies to play.
2. **Call the oracle.** Train a *new* best-response policy $\pi_{k+1}$ against opponents sampled from the meta-Nash mixture $\sigma$. The "oracle" is just an RL algorithm (PPO, say) doing approximate best-response.
3. **Grow the set.** Add $\pi_{k+1}$ to $\Pi$, fill in its row/column of the payoff matrix by playing it against everyone, and repeat.

![Graph of the PSRO double oracle loop where an oracle adds best responses and the meta-game Nash is recomputed](/imgs/blogs/emergent-behaviour-and-multi-agent-games-8.png)

The strategy set grows monotonically, the meta-Nash is recomputed on the larger set each round, and under standard assumptions the meta-Nash converges to a Nash equilibrium of the *full* game. The beautiful part is that the special cases fall out by choosing the meta-solver:

- **Self-play** = PSRO where the meta-distribution puts all mass on the *latest* policy. (Best-response to the latest self — and now you can see precisely why it cycles: the meta-distribution has no memory.)
- **Fictitious play** = PSRO where the meta-distribution is *uniform* over all past policies.
- **Double Oracle** = PSRO with an exact Nash meta-solver and exact best-response oracle (the classical algorithm).
- **League training** = PSRO with a prioritised meta-distribution and multiple oracle *types* (the exploiters).

Here is a minimal PSRO outer loop, with the meta-game solved by a simple iterative Nash solver for the zero-sum case:

```python
import numpy as np

def solve_zero_sum_nash(payoff, iters=2000):
    """Approximate Nash of a zero-sum matrix game via fictitious play on the meta-game."""
    n, m = payoff.shape
    row_counts = np.ones(n)
    col_counts = np.ones(m)
    for _ in range(iters):
        col_strat = col_counts / col_counts.sum()
        best_row = np.argmax(payoff @ col_strat)
        row_counts[best_row] += 1
        row_strat = row_counts / row_counts.sum()
        best_col = np.argmin(row_strat @ payoff)
        col_counts[best_col] += 1
    return row_counts / row_counts.sum(), col_counts / col_counts.sum()

def psro(initial_policy, oracle_best_response, eval_match, rounds=20):
    pop = [initial_policy]
    payoff = np.zeros((1, 1))
    for r in range(rounds):
        sigma_row, sigma_col = solve_zero_sum_nash(payoff)   # 1. meta-Nash
        new_pi = oracle_best_response(pop, sigma_col)         # 2. oracle: BR to mixture
        pop.append(new_pi)                                    # 3. grow set
        # extend payoff matrix with the new policy's row and column
        k = len(pop)
        new_payoff = np.zeros((k, k))
        new_payoff[:k-1, :k-1] = payoff
        for i in range(k):
            new_payoff[i, k-1] = eval_match(pop[i], pop[k-1])
            new_payoff[k-1, i] = -new_payoff[i, k-1]          # zero-sum
        payoff = new_payoff
    return pop, payoff
```

The key line is `oracle_best_response(pop, sigma_col)`: the new policy is a best response to the *Nash mixture* over the whole population, not to a single opponent. That is what stops cycling. PSRO is the answer to "what is league training actually approximating?" — it is approximating the meta-Nash of the game, refined one best-response at a time.

## 11. Emergent language and communication

A different and more speculative flavour of emergence: do agents develop *language*? In cooperative MARL with a communication channel, agents can learn to send signals that coordinate behaviour. The cleanest testbed is the **Lewis signalling game**: a "sender" sees a target and emits a discrete symbol; a "receiver" sees only the symbol and must pick the target out of distractors. Both are rewarded only if the receiver is correct, so any symbol-meaning mapping must be *invented* — there is no dictionary.

Two influential architectures made this trainable end to end. **DIAL** (Differentiable Inter-Agent Learning; Foerster et al., 2016) lets agents send *continuous-valued* messages during training so gradients flow *through* the communication channel from receiver back to sender — the receiver's loss directly shapes what the sender says — then discretises messages at execution time. **CommNet** (Sukhbaatar et al., 2016) lets a set of agents communicate by *averaging* their hidden states each step, a learned continuous broadcast channel. Both reliably learn protocols that solve referential and coordination tasks better than no-communication baselines.

The honest assessment is where this gets interesting. Emergent "languages" in these systems are real protocols — agents genuinely transmit task-relevant information — but they are usually **not language-like** in the ways that matter. Studies repeatedly find that emergent codes (a) lack *compositionality* (the symbol for "red square" is not built from a "red" part and a "square" part), (b) are highly sensitive to architecture and seed, and (c) often exploit degenerate channels (encoding information in message *length* or *timing* rather than symbol identity). The research frontier — work on compositionality pressure, population-level communication, and ease-of-teaching objectives — is about *forcing* the emergent protocol to have the structure of language rather than just being any-old-correlated-code. So: yes, communication emerges; no, it is not yet a language; and getting from one to the other is an open problem. This is a good place to be sceptical of breathless claims — an emergent protocol that solves one task is not evidence of an emergent language.

## 12. Open-endedness, coevolution, and when systems make progress

We've seen emergence work spectacularly (hide-and-seek, AlphaStar) and we've seen the way it fails (cycling). The unifying question is: **will a multi-agent system keep improving indefinitely, or will it stall?** This is the question of **open-endedness**.

Open-ended learning asks for a process that keeps generating novel, increasingly complex, learnable challenges forever — the way evolution did on Earth. Pure coevolution does *not* guarantee this. Coevolving predator-prey or hider-seeker systems can fall into three regimes: **escalation** (both sides keep improving — the good case, Figure 2's tree), **cycling** (the rock-paper-scissors trap from §9), or **disengagement** (one side gets so far ahead that the other can no longer learn — if seekers can *never* find hiders, the seeker reward is always zero, the gradient vanishes, and *both* sides stop improving). Disengagement is the quiet killer: it looks like convergence but it is actually a dead gradient.

**POET** (Paired Open-Ended Trailblazer; Wang et al., 2019) attacks this directly by coevolving *environments alongside agents*. POET maintains a population of (environment, agent) pairs, periodically mutates environments to create new challenges, transfers agents between environments (a policy good at one terrain might bootstrap learning on another), and — the crucial filter — only keeps new environments that are *neither too easy nor too hard* for the current population (a "minimal criterion" / learnability filter). That filter is what keeps the system in the escalation regime: it manufactures challenges that are exactly at the frontier of solvability, which is the formal version of the autocurriculum hide-and-seek got for free. The lesson generalises to any curriculum design: **the useful challenge is the one just past your current ability**, and a system that can detect and generate that band keeps improving; one that cannot, stalls.

This reframes reward and environment design for emergence. The practical rules:

- **Keep the reward simple but ensure engagement.** A binary win/lose reward is *ideal* for emergence (it does not bias the strategy) but only if both sides can occasionally win. If one side never scores, you get disengagement. Add the *minimum* shaping needed to keep both sides in the game, then anneal it away (OpenAI Five's team-spirit annealing is exactly this).
- **Avoid trivial equilibria.** If there is a way to win that requires no skill (camp one spot, exploit one bug), agents find it and stop. Hide-and-seek's box-surfing was an *interesting* exploit, but the same dynamic produces *boring* exploits that kill learning. Constrain the environment to close trivial wins.
- **Maintain diversity explicitly.** Diversity does not maintain itself; left alone, populations collapse to one strategy. League exploiters, prioritised sampling, and quality-diversity methods (MAP-Elites) are all diversity-maintenance machinery.

#### Worked example: choosing your training scheme

You're building a MARL system and need to pick a scheme. Walk Figure 7's decision tree with three concrete projects.

*Project A — two drones playing pursuit-evasion (competitive, suspected non-transitive).* Competitive: yes. Cycling likely: yes (pursuit-evasion has rock-paper-scissors structure — fast-straight beats juke, juke beats fast-straight). So plain self-play will cycle; you want **league training or PSRO**. Budget a population of, say, 20–50 archived agents and prioritised sampling.

*Project B — a fleet of warehouse robots that must not collide while maximising throughput (purely cooperative).* Competitive: no, cooperative: yes. There is no opponent to cycle against; the hard part is multi-agent credit assignment. Use a **cooperative MARL method with centralised training and decentralised execution — QMIX or MADDPG** (see `maddpg-centralised-training-decentralised-execution` once it ships in this series). No league needed.

*Project C — a market-making simulator where your agents compete for fills but a regulator agent constrains all of them (mixed).* Both competitive and cooperative elements. You want a **mixed league**: self-play/PSRO for the competitive layer, CTDE for any cooperating sub-teams. This is the hardest case and the one most prone to surprising emergence — and to reward hacking, so monitor for trivial equilibria aggressively.

![Decision tree for choosing between simple self-play league training and cooperative MARL based on game structure](/imgs/blogs/emergent-behaviour-and-multi-agent-games-7.png)

## 13. The simplicity bias in emergence

There is a counterintuitive empirical regularity running through every system in this post, and it deserves to be stated as a principle in its own right: **the simpler the reward, the more complex the emergent behaviour tends to be.** Hide-and-seek used a binary visibility signal and produced six phases of tool use; AlphaGo Zero used a single bit (win or lose) and invented openings that overturned centuries of human theory; AlphaStar's reward was win/lose plus only light statistical shaping. The systems with the *least* prescriptive rewards produced the *most* surprising strategy. This is not a coincidence, and understanding why is essential to designing for emergence rather than against it.

The mechanism is that **a sparse, outcome-only reward leaves the strategy space unconstrained**, while a dense, shaped reward implicitly encodes a human's guess about *how* to win — and that guess is almost always a constraint the agent would have been better off without. Consider hide-and-seek again. The reward says only "no seeker can see you." It says nothing about *how* to stay unseen. So the gradient is indifferent between running, hiding behind a wall, building a fort, or surfing a box — whichever empirically yields more unseen steps wins, and the agent is free to discover that forts beat running and box-surfing beats forts. Now suppose you had "helpfully" shaped the reward with a term like "+0.1 for each step you spend behind a wall." You would have *told* the agent that walls are the answer, and it would have learned to cower behind the nearest wall and stopped there — because that shaped term makes wall-cowering locally optimal and short-circuits the search that would otherwise have found forts and surfing. **Reward shaping can kill emergence by shortcutting the discovery process**: it hands the agent the first plausible strategy a human thought of, and the agent, being an optimiser, takes the free reward and never explores past it.

This sits in direct tension with the **reward hypothesis** — Sutton's claim that any goal can be expressed as the maximisation of expected cumulative scalar reward. The hypothesis is true and it is the foundation of all of RL, but in MARL it has a sharp edge: *which* scalar reward you choose determines not just what the agent optimises but *what it is capable of discovering*. A dense reward and a sparse reward can have the same optimum in principle (both maximised by winning) and yet produce radically different behaviour in practice, because the dense reward's intermediate gradient steers the search into the basin the designer anticipated. The reward hypothesis tells you a scalar *suffices*; it does not tell you that any reward-encoding the same goal is equally good for *learning*. In single-agent RL we lean on shaping precisely to make hard-exploration problems tractable — and there it is often the right call. In emergent MARL the calculus flips, because the *opponent* is your exploration engine: the autocurriculum supplies the intermediate difficulty that shaping would otherwise have to, so shaping becomes redundant at best and a discovery-killing constraint at worst.

The practical resolution is the rule we already met in §12, now with its justification: **shape the reward by the minimum needed to keep both sides engaged, and anneal even that away.** OpenAI Five's gold/experience shaping existed only because Dota's true reward (win the game, tens of minutes away) is far too sparse to bootstrap *any* competent play — without it, agents would flail for the entire early game and never get the engagement that drives the autocurriculum. But the team-spirit annealing and the heavy reliance on a near-binary win signal at the top level kept the *strategic* layer unconstrained, which is where the interesting emergence lived. The design heuristic, then: use shaping as *scaffolding for engagement*, never as *guidance for strategy*. Reward the conditions that keep the game alive (both sides occasionally winning, agents staying in contact) and stay silent about how to win — then let the Red Queen do the rest. Every gram of "how" you put in the reward is a gram of discovery you take out.

## 14. Measuring emergence

If strategy is emergent, how do you even know it happened, and how do you know it is progress and not noise? Four families of metric, used together:

**Relative skill (Elo / TrueSkill).** As in §8, run round-robin tournaments among snapshots and check for *transitive* improvement. The diagnostic from §8 — strong against the previous snapshot, weak against the archive — flags cycling. This is your primary "are we improving" signal.

**Behavioural diversity.** Count or cluster distinct strategies in the population. Cheap proxies: the entropy of the action distribution, the number of distinct "build orders" or opening sequences, or the *coverage* of a behaviour space (MAP-Elites literally maintains a grid of behaviour descriptors and measures how many cells are filled). Collapsing diversity is an early warning that you are heading toward a trivial equilibrium.

It is worth making the **behavioural diversity (BD) metric** precise, because "diversity" is easy to wave at and hard to measure. The core move, shared by all quality-diversity methods, is to first define a low-dimensional **behaviour descriptor** $b(\pi) \in \mathcal{B}$ — a vector that summarises *what an agent does* rather than *how well it does it*. For hide-and-seek a descriptor might be (fraction of time spent holding a box, fraction of time spent near a wall, mean distance from teammates); for StarCraft it might be (army composition ratio, expansion timing, fraction of resources spent on tech). Two agents with very different win rates can have the same descriptor (both turtle, one just turtles better), and two agents with the same win rate can have wildly different descriptors (one rushes, one booms) — which is exactly the information Elo throws away. Novelty of a new agent is then the mean distance from its descriptor to its $k$ nearest neighbours among all past agents, $\text{novelty}(\pi) = \frac{1}{k}\sum_{j \in \text{kNN}} \lVert b(\pi) - b(\pi_j) \rVert$; a high value means the agent occupies a region of behaviour space no prior agent reached.

**MAP-Elites** operationalises this with an **archive**: it discretises the behaviour space $\mathcal{B}$ into a grid of cells and keeps, in each cell, only the single highest-performing agent whose descriptor falls in that cell (the cell's "elite"). The archive is the object you measure diversity on — *coverage* is the fraction of cells that are filled, and *QD-score* is the sum of the elites' fitnesses across all filled cells, so a method scores well only if it is both diverse (many cells) and good (high fitness per cell). The reason this matters for emergence is that the archive *protects* rare strategies from being out-competed: a niche rush build that loses on average but is the only thing in its cell survives in the archive and remains available to seed future learning, exactly the way league training's frozen snapshots keep old counters alive. An archive that stops gaining new filled cells is the quantitative signature of a population that has stopped discovering — diversity coverage flattening is an earlier, more sensitive warning of stalled emergence than Elo, which can keep ticking up for a while even as the *kinds* of strategy stop changing.

**Why Elo is only a proxy, and where it lies.** Elo (and TrueSkill) measure one thing well — *transitive relative skill* — and they are the right primary signal for "is this agent stronger." But emergence is about behavioural *complexity*, which Elo does not see, and the two can come apart in both directions. Most importantly, **Elo can stay nearly flat while behaviours become genuinely more complex**. In a non-transitive game the population can be churning through ever-more-intricate strategies — each beating the last, each requiring more scouting and adaptation — while the round-robin Elo numbers cluster near a constant because no strategy is *transitively* dominant (the rock-paper-scissors structure of §8). Flat Elo there is not stalled learning; it is the *signature of a rich non-transitive strategy space*, and you only see the progress if you also track behavioural diversity and novelty. The converse failure also bites: Elo can keep *rising* against a stale archive while the agent is actually collapsing to one over-fit strategy — it climbs because the archive stopped adapting, not because the agent got more robust, which is precisely why §8's "strong against the previous snapshot, weak against a *fresh* exploiter" diagnostic matters. The discipline is to never read Elo alone: pair it with coverage (are we still finding new behaviour cells?) and exploitability (can a fresh best-response still crush us?). Elo tells you who wins; it does not tell you whether anything new is being invented.

**Behavioural complexity / novelty.** Detect *new* behaviour by measuring how surprising an agent's trajectory is relative to past agents — e.g. the prediction error of a model trained on past behaviour, or the appearance of states no prior agent visited (the box-surfing states never occurred until phase 6). Spikes in novelty coincide with phase transitions.

**Exploitability.** The gold standard for competitive games: how much can a *dedicated best-response* (a fresh exploiter trained against your agent) beat it? Low exploitability means your agent is near-Nash; high exploitability means it has blind spots regardless of its average win rate. AlphaStar's exploiter agents *are* this metric operationalised into training. Here is a sketch of measuring it:

```python
def measure_exploitability(target_policy, oracle_train, eval_match, train_steps=200_000):
    """Train a fresh best-response against target and report its win rate over it."""
    exploiter = oracle_train(opponent=target_policy, steps=train_steps)
    wins = sum(eval_match(exploiter, target_policy) > 0 for _ in range(1000))
    return wins / 1000.0   # 0.5 = unexploitable (near-Nash); 0.9 = badly exploitable
```

If `measure_exploitability` returns 0.85, your "strong" agent loses to a purpose-built counter 85% of the time — it is a one-trick pony and league training is the fix. If it returns ~0.5, the exploiter can do no better than a coin flip, which is what near-Nash play looks like.

## Case studies

Figure 5 lays the four landmark systems side by side. The single most important pattern across all of them: the reward stayed *simple* and the capability scaled with *interaction and compute*, not with hand-engineered skills.

![Matrix comparing hide-and-seek OpenAI Five AlphaStar and AlphaGo across reward agents emergent phenomenon and result](/imgs/blogs/emergent-behaviour-and-multi-agent-games-5.png)

**OpenAI hide-and-seek (Baker et al., 2019).** Reward: essentially binary team visibility. Agents: two teams, 1–3 each. Result: six emergent tool-use phases (fort building, ramp use, ramp locking, box surfing) over hundreds of millions of episodes, with *zero* reward for tool use. The autocurriculum — opponent-generated difficulty — is the headline contribution. The box-surfing physics exploit is also the textbook example of an agent optimising the literal reward in an unintended way.

**OpenAI Five (Dota 2, 2019).** Reward: shaped per-agent rewards with an annealed team-spirit term. Scale: ≈180 years of self-play per day for months. Result: beat world champions OG 2–0; emergent role specialisation, implicit coordination, and sacrifice plays from a shared team reward. Caveat: hero-pool restrictions, and humans adapted within days — a reminder that "superhuman" is contextual.

**AlphaStar (StarCraft II, 2019).** Reward: win/lose plus light statistical shaping. Method: league training with main agents, main exploiters, and league exploiters over a population that grew into the hundreds. Result: Grandmaster (top 0.2%) in all three races under human-like action constraints. The contribution is the *training method* — league training is the answer to non-transitive strategy cycles, and ablating the exploiters produced strong-but-exploitable agents.

**AlphaGo / AlphaGo Zero (2016–2017).** Reward: binary win/lose. Method: self-play from random (Zero) plus MCTS. Result: defeated Lee Sedol 4–1 (2016); AlphaGo Zero surpassed that version after three days of pure self-play. Move 37 is the emblem of strategy discovered without a human prior. The simplest setting and the clearest proof that self-play *invents* rather than imitates.

A fifth, honest data point for contrast: plenty of MARL projects produce *no* interesting emergence — they cycle, disengage, or collapse to a trivial equilibrium. The four above are famous precisely because they got the environment design, the scale, and the diversity machinery right. Emergence is not automatic; it is engineered conditions under which discovery becomes likely.

## When to use this (and when not to)

Self-play and league training are powerful but expensive and finicky. Be decisive about when they earn their cost.

**Use self-play / league training when:** the task is *competitive or mixed*, you can *simulate games cheaply* (millions to billions of them), and you want strategy you cannot hand-engineer. The economics only work when a game is fast to simulate — AlphaStar and OpenAI Five spent enormous compute precisely because each game is cheap relative to the value of the result. If your "game" is a slow real-world process, self-play is usually infeasible.

**Use plain self-play (no league) when:** the game is *transitive* — strictly better strategies exist and beat worse ones with no rock-paper-scissors loops (Go and chess are nearly transitive, which is why AlphaGo's plain-ish self-play worked). Check transitivity empirically with the round-robin Elo test from §8 *before* investing in league infrastructure. If snapshots improve monotonically, you don't need a league.

**Use league training / PSRO when:** you observe *cycling* (the Elo gap diagnostic) or your agent is *exploitable* despite a high average win rate. This is the StarCraft regime. The cost is real — you maintain and play against a large archived population — but it is the only thing that reliably kills non-transitive cycles.

**Use cooperative MARL (QMIX, MADDPG) instead when:** the setting is *purely cooperative*. There is no opponent, so there is nothing to self-play against; your problem is multi-agent credit assignment, and CTDE methods are the right tool.

**Do not reach for any of this when:** a single-agent formulation suffices. If the "other agents" are static (fixed bots, fixed market data), treat them as part of a stationary environment and use ordinary single-agent RL (PPO, SAC, DQN) — it is far simpler and you keep your convergence guarantees. And if you can *plan* with a known model, classical game-tree search or planning may beat learning entirely. Multi-agent learning is the tool for when the opponent is *adaptive* and the strategy space is *too large to enumerate*. Anything less and you are paying for machinery you don't need.

A final honest trade-off: emergence is double-edged. The same lack of a human prior that finds move 37 also finds box-surfing physics exploits and reward hacks. If you deploy an emergent policy, you *must* test it against adversarial probing (the exploitability metric), because it has optimised the literal reward, not your intent — and in a multi-agent system, the literal reward almost always has a loophole someone will discover.

## Key takeaways

1. **Emergence = simple individual rewards + rich interaction + no scripted strategy.** You specify the game; the strategy is discovered. Hide-and-seek's six tool-use phases came from a binary visibility reward and nothing else.
2. **Non-stationarity is the engine.** A learning opponent makes the environment non-stationary, which breaks single-agent convergence guarantees but keeps the gradient alive — the opponent is a curriculum that never runs out.
3. **The Red Queen effect drives escalation.** Because fitness is relative, each side must keep improving just to stay even; healthy arms races branch into new strategy space (a tree), pathological ones cycle (rock-paper-scissors).
4. **Naive self-play cycles and forgets.** Training only against the latest self causes catastrophic strategic forgetting and collapse to one brittle strategy in any non-transitive game.
5. **League training fixes it with a diverse archive plus exploiters.** Main agents stay un-exploitable, main exploiters red-team them, league exploiters find systemic holes, and frozen snapshots prevent forgetting — this is how AlphaStar reached Grandmaster.
6. **PSRO is the theory underneath.** Self-play, fictitious play, and double-oracle are all PSRO with different meta-distributions; the new policy should be a best-response to the *Nash mixture*, not to a single opponent.
7. **Measure with Elo, diversity, novelty, and exploitability — together.** The decisive diagnostic for self-play health is the Elo gap: strong against the previous snapshot but weak against the archive means you are cycling, not improving.
8. **Disengagement is the quiet killer.** If one side can never win, the gradient dies on both sides; keep both sides engaged with minimal shaping you then anneal away, and generate challenges just past current ability (POET's learnability filter).
9. **Emergent communication is real but not yet language.** Agents learn protocols, but they typically lack compositionality and exploit degenerate channels; treat claims of "emergent language" sceptically.
10. **Emergence is double-edged.** The same prior-free optimisation that finds brilliant novel strategy also finds reward hacks and physics exploits — always probe a deployed policy with a dedicated best-response.

## Further reading

- Baker, Kanitscheider, Markov, Wu, Powell, McGrew, Mordatch (2019), "Emergent Tool Use From Multi-Agent Autocurricula" — the hide-and-seek paper; read it for the autocurriculum framing.
- Vinyals et al. (2019), "Grandmaster level in StarCraft II using multi-agent reinforcement learning," *Nature* — AlphaStar; the definitive treatment of league training.
- OpenAI et al. (2019), "Dota 2 with Large Scale Deep Reinforcement Learning" — OpenAI Five; scale, team-spirit annealing, and emergent coordination.
- Silver et al. (2017), "Mastering the game of Go without human knowledge," *Nature* — AlphaGo Zero; self-play from random.
- Lanctot et al. (2017), "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning" (NeurIPS) — PSRO; the framework unifying self-play, fictitious play, and double-oracle.
- Foerster et al. (2016), "Learning to Communicate with Deep Multi-Agent Reinforcement Learning" — DIAL and emergent communication.
- Wang et al. (2019), "Paired Open-Ended Trailblazer (POET)" — coevolving environments and agents for open-ended learning.
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — for the single-agent foundations these multi-agent methods extend.
- Within this series: `what-is-reinforcement-learning` for the agent↔environment↔reward loop these systems generalise; `proximal-policy-optimization-ppo` for the optimiser nearly all of them used; `exploration-vs-exploitation-the-core-tension` for why a learning opponent is a powerful exploration driver; `muzero-mastering-games-without-rules` for the planning-plus-self-play lineage; and the forthcoming unified map and capstone for where multi-agent RL sits in the full taxonomy.
