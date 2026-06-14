---
title: "Logic puzzles for quant interviews: weighings, hats, prisoners, and switches"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Quant interviewers do not actually care whether you have seen the 12-coin puzzle before. They care whether you can reason cleanly under constraints. This deep dive builds the four techniques that crack the classic logic puzzles from zero, then solves the famous ones in full."
tags:
  [
    "quant-interviews",
    "logic-puzzles",
    "brain-teasers",
    "information-theory",
    "invariants",
    "induction",
    "pigeonhole-principle",
    "weighing-puzzles",
    "interview-prep",
    "quantitative-finance",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 46
---

> [!important]
> **TL;DR** — Logic puzzles in a quant interview are not memory tests. They are graded on *how you reason under constraints*, and four reusable techniques crack nearly all of them.
>
> - **Counting / information.** Each question pays a fixed amount of information: a yes/no answer is worth 1 *bit*, a three-way balance is worth $\log_2 3 \approx 1.585$ bits. That bound tells you the minimum number of questions before you even start. It is why the 12-coin puzzle needs *exactly* three weighings: $3^3 = 27 \ge 24$ cases.
> - **Invariants and parity.** Find a quantity that never changes (an *invariant*) — often just whether a count is odd or even (*parity*). The 100-hats puzzle hands one parity bit from the back of the line and saves 99 of 100 prisoners for sure.
> - **Induction.** Solve the size-$n$ problem by assuming you have already solved size $n-1$. The blue-eyed islanders all leave on exactly night 100 because the logic unwinds one night per islander.
> - **Working backwards.** Reason from the goal state. The 100-prisoners-and-boxes puzzle, where guessing wins with probability $(1/2)^{100} \approx 0$, jumps to about **31%** once each prisoner follows the permutation cycle that contains their own number.
> - The single most common failure is *guessing* — blurting an answer — instead of *deducing*. Interviewers want the deduction, narrated out loud, with assumptions stated.
> - This is educational material about reasoning techniques, not financial advice.

Here is a question a trader at Optiver, SIG, or Jane Street might open with, sometimes with a small wager attached to make it concrete: *"You have twelve coins that look identical. Eleven weigh the same; one is counterfeit and a slightly different weight — you don't know if it is heavier or lighter. You have a balance scale and may use it three times. I'll pay you \$100 if you can guarantee finding the fake and saying whether it is heavy or light. Go."*

Most people freeze, then start weighing coins more or less at random, narrating a tangle of cases until time runs out. The candidate who gets the offer does something different. Before touching a coin, they say: *"Each weighing has three possible results — left side down, right side down, or balanced. So three weighings can distinguish at most $3^3 = 27$ outcomes. I need to identify which of twelve coins is fake **and** whether it is heavy or light, which is $12 \times 2 = 24$ distinct situations. Since $24 \le 27$, three weighings is at least possible. Now let me design the splits so each weighing carves the suspects roughly into thirds."*

That candidate has not solved the puzzle yet. But they have already shown the interviewer the thing being tested: the instinct to *count the information* before flailing. The puzzle is a probe for structured reasoning under a hard constraint, and there is a small kit of techniques — counting and information, invariants and parity, induction, and working backwards — that turns the famous puzzles from "did you happen to memorize this one" into "of course, here is the method."

![Diagram titled the four techniques that crack the classic puzzles, a four-row matrix mapping counting and information, invariant and parity, induction, and working backwards to what each does, its classic puzzle, and a key number such as three weighings, parity is one shared bit, all 100 leave on night 100, and the cycle strategy wins about 31 percent.](/imgs/blogs/logic-puzzles-deduction-quant-interviews-1.png)

The matrix above is the mental model for this whole post. Read each row as a tool: a name, what it does, the famous puzzle it cracks, and the one number to remember. We will build each tool from first principles, define every term the moment it appears, and then solve the classics in full — the 12-coin weighing, the 100 hats, the 100 prisoners and the boxes, the three switches, the blue-eyed islanders — each with real numbers and a clearly marked walkthrough. By the end you should be able to recognize, within ten seconds of hearing a puzzle, which of the four tools to reach for.

## Foundations: how interviewers actually grade a logic puzzle

Before any technique, it helps to know what the person across the table is scoring. Quant firms — the market makers like Optiver, SIG, IMC, and Jane Street, and the systematic funds like Citadel, Two Sigma, and Jump — do not ask logic puzzles because trading involves weighing coins. They ask because the daily job *is* reasoning correctly under uncertainty and constraints, fast, while saying your reasoning out loud so a teammate can check it. A puzzle is a five-minute simulation of that.

Here is what they are watching for, in roughly the order it matters:

- **Do you state your assumptions?** "I'll assume the fake is *exactly one* coin, the scale is accurate, and a balanced result means truly equal." Naming assumptions is half the job on a trading desk, because a model is only as good as the conditions you assumed. A candidate who silently assumes the fake is heavier — when the problem never said so — has already lost points.
- **Do you reason out loud?** Silent thinking, then a blurted answer, is nearly worthless even when the answer is right, because the interviewer cannot tell whether you deduced it or guessed it. Narrate. "If the left pan drops, the fake is among these four and it is heavy, *or* among those four and it is light." That sentence is worth more than the final answer.
- **Do you find structure, or do you brute-force?** The strong candidate looks for the *invariant*, the *count*, the *recursion* — the lever that collapses a messy case analysis into a clean argument. Brute force occasionally works but does not scale, and the interviewer is explicitly probing whether you see the lever.
- **Do you handle the adversarial case?** Many puzzles have a hidden "what if the universe is out to get you" wrinkle. The 12-coin fake could be heavy *or* light; the hat colors could be assigned by an adversary; the box permutation is random. A guaranteed strategy must work in the *worst* case, not the lucky case. Confusing "usually works" with "always works" is a classic stumble.

Two terms we will lean on throughout, defined now from zero:

- A **bit** (short for *binary digit*) is one yes/no answer's worth of information. If a question has two equally likely answers, learning the answer is exactly one bit. Two yes/no questions distinguish $2^2 = 4$ possibilities; $k$ of them distinguish $2^k$. The word comes from information theory, and the precise statement is that distinguishing $N$ equally likely cases requires at least $\log_2 N$ bits.
- **Parity** is simply whether a whole number is *even* or *odd*. "The parity of the number of red hats" means: is that count even or odd? Parity is the workhorse invariant, because it is a single bit of information that an entire group can share and that many operations leave unchanged.

With the grading rubric and those two words in hand, we can build the toolkit.

## Information and counting: each question has a fixed price

The first and most powerful technique is also the most underused by candidates: *count the information before you act.* Every question, measurement, or observation you are allowed has a fixed **information capacity** — a hard ceiling on how many cases its answer can possibly distinguish. If you need to separate more cases than your questions can pay for, no cleverness saves you; you simply need more questions. And if the count says a solution is possible, it usually also tells you how to build one, because the optimal strategy is the one where every question is "maximally informative" — it splits the remaining possibilities as evenly as the answer's branches allow.

![Diagram titled each question pays out a fixed amount of information, a three-row matrix comparing a yes/no question, a balance weighing, and a die roll across three columns: number of outcomes, information per question in bits, and how many cases k such questions resolve, showing 1 bit and 2 to the k for yes/no, about 1.585 bits and 3 to the k for the balance, and about 2.585 bits and 6 to the k for the die.](/imgs/blogs/logic-puzzles-deduction-quant-interviews-2.png)

The table above is the entire idea. A question with $b$ equally likely outcomes carries $\log_2 b$ bits, and $k$ such questions can in principle distinguish $b^k$ cases. A yes/no question has $b = 2$ outcomes, so it carries $\log_2 2 = 1$ bit and $k$ of them resolve $2^k$ cases. A *balance scale* — the kind with two pans — has $b = 3$ outcomes (left pan down, right pan down, or balanced), so each weighing carries $\log_2 3 \approx 1.585$ bits and $k$ weighings resolve up to $3^k$ cases. That single fact, that a three-pan-outcome balance is worth a base-3 digit (a *trit*) rather than a bit, is the key to every weighing puzzle.

#### Worked example: the information lower bound on a guessing game

Suppose an interviewer thinks of a whole number between 1 and 1,000 and offers you \$50 if you can name it, charging you one yes/no question at a time. How many questions do you need to *guarantee* a win? Counting answers it instantly. You must distinguish 1,000 equally likely cases, and each yes/no question pays one bit, distinguishing a factor of two. So you need $k$ with $2^k \ge 1000$. Since $2^9 = 512 < 1000$ and $2^{10} = 1024 \ge 1000$, the answer is **ten questions**, and the strategy that achieves it is binary search: each question should ask "is it above the midpoint of the remaining range?" so that whichever way the answer falls, the surviving range is halved. After ten halvings, $1000 \to 500 \to 250 \to 125 \to 63 \to 32 \to 16 \to 8 \to 4 \to 2 \to 1$, you are down to a single number.

The one-sentence intuition: *the minimum number of questions is the number of times you must halve (or third, or sixth) the case count to reach one, and the best strategy makes every question split the survivors as evenly as the answer allows.*

The same counting logic flips into an *upper* bound on what is achievable. If you only get three weighings, you simply cannot distinguish more than $3^3 = 27$ cases, full stop — so any puzzle that demands separating 28 or more is impossible in three weighings, no matter how clever you are. This is the move that lets a strong candidate say "this is impossible" with total confidence, which is itself a valued answer when it is correct.

## The weighing puzzles: spending information optimally

Weighing puzzles are the purest test of the counting technique, because the balance's three outcomes map so cleanly onto base-3 arithmetic. Let us warm up on the easy version before the famous hard one.

### The easy case: one fake among many, known direction

Suppose you have nine coins, one of which is *heavier* than the rest (you are told the direction), and a balance. How many weighings guarantee finding it? Counting: nine cases, each weighing worth a trit, so you need $k$ with $3^k \ge 9$, giving $k = 2$. And the strategy is forced by the "split into thirds" principle: weigh three coins against three, leaving three aside.

#### Worked example: nine coins, one heavy, in two weighings

Number the coins 1 through 9. **Weighing 1:** put $\{1,2,3\}$ on the left pan and $\{4,5,6\}$ on the right, with $\{7,8,9\}$ set aside. Three outcomes:

- Left pan drops: the heavy coin is in $\{1,2,3\}$.
- Right pan drops: it is in $\{4,5,6\}$.
- Balanced: both groups are all-normal, so the heavy coin is in the untouched $\{7,8,9\}$.

Either way, you are down to a group of three. **Weighing 2:** take that group, put one coin on the left, one on the right, one aside. If the left drops, it is the left coin; if the right drops, the right coin; if balanced, the one you set aside. Done in two weighings, exactly matching the $3^2 = 9$ bound. The intuition: *because the direction is known, each weighing cleanly thirds the suspects, and two thirds take you from nine to one.*

### The 12-coin problem: one fake, unknown direction, three weighings

Now the famous one, the version with the \$100 attached. Twelve coins, exactly one is fake, you do *not* know whether it is heavier or lighter, and you have three weighings. You must identify the fake *and* its direction. The counting check we did in the intro: $12 \times 2 = 24$ cases (each coin could be the heavy fake or the light fake), and $3^3 = 27 \ge 24$, so it is just barely possible — the slack is only three outcomes, which is why the design is delicate. The reason the direction-unknown version is so much harder than the known-direction one is that a "balanced" result and an "unbalanced" result now carry different kinds of information: balanced tells you a whole group is *clean*, while unbalanced tells you the fake is present but leaves its direction ambiguous.

![Diagram titled the 12-coin weighing first split is three-way, a decision tree whose root weighs coins one through four against five through eight, branching into left-heavy meaning the fake is in coins one to eight, balanced meaning the fake is in coins nine to twelve, and right-heavy mirroring the left, with each branch leading to a second weighing and then to leaf cases that pin down individual coins.](/imgs/blogs/logic-puzzles-deduction-quant-interviews-3.png)

The tree above shows the all-important first split and how the three outcomes route to very different follow-ups. Let us walk the whole strategy.

#### Worked example: solving 12 coins in three weighings

Label the coins 1 through 12. The trick is to *rotate* coins between pans across weighings so that each coin's pattern of "left, right, or off" across the three weighings encodes its identity in base 3.

**Weighing 1:** $\{1,2,3,4\}$ versus $\{5,6,7,8\}$, with $\{9,10,11,12\}$ aside.

*Case A — balanced.* Then 1 through 8 are all genuine, and the fake is among $\{9,10,11,12\}$, with direction still unknown. We have two weighings left for four coins and two directions — eight cases, and $3^2 = 9 \ge 8$, comfortable.

- **Weighing 2:** $\{9, 10\}$ versus $\{11, 1\}$. Coin 1 is known-genuine, used as a trusted reference.
  - If balanced: the fake is 12. **Weighing 3:** weigh 12 against any genuine coin to learn if it is heavy or light. Done.
  - If $\{9,10\}$ drops: either 9 or 10 is heavy, or 11 is light. **Weighing 3:** weigh 9 against 10. If they balance, 11 is the light fake; if one drops, that one is the heavy fake.
  - If $\{11,1\}$ drops: by the mirror argument either 9 or 10 is light, or 11 is heavy. **Weighing 3:** weigh 9 against 10 to resolve it the same way.

*Case B — left pan $\{1,2,3,4\}$ drops.* Now coins 9 through 12 are genuine (they were aside and we got an unbalanced result among 1–8). The fake is in 1–8: *either* one of $\{1,2,3,4\}$ is heavy *or* one of $\{5,6,7,8\}$ is light. That is eight cases again.

- **Weighing 2:** rearrange so the new result separates those eight. Put $\{1,2,5\}$ on the left and $\{3,4,6\}$ on the right, with $\{7,8\}$ and a genuine coin aside.
  - If balanced: the fake is in the set-aside $\{7,8\}$, and from Case B it must be *light* (since 7, 8 were on the originally-heavy-suspect side… actually they were on the right, the light-suspect side). **Weighing 3:** weigh 7 against 8; the lighter pan is the light fake.
  - If the left $\{1,2,5\}$ drops: the heavy-suspect coins on this side are 1 and 2, and the light-suspect coin on the other side is 6. So either 1 or 2 is heavy, or 6 is light. **Weighing 3:** weigh 1 against 2 — if one drops it is the heavy fake; if balanced, 6 is the light fake.
  - If the right $\{3,4,6\}$ drops: symmetrically either 3 or 4 is heavy, or 5 is light. **Weighing 3:** weigh 3 against 4; if balanced, 5 is the light fake.

*Case C — right pan drops.* This is the mirror image of Case B with the roles of the two groups swapped; the same construction resolves it in the remaining two weighings.

Every path ends with the fake identified and its direction known, in exactly three weighings, claiming the \$100. The reason it works is the counting we did up front: with 24 cases and 27 outcomes, a correctly balanced design wastes almost no information. The intuition to carry: *when outcomes come in threes, think in base 3, rotate items between the pans so each one gets a distinct three-weighing signature, and design every weighing to split the live cases as evenly as a three-way answer permits.*

![Diagram titled why three weighings the information bound grows as 3 to the k, a bar chart with bars for 3 to the 1 equals 3, 3 to the 2 equals 9, 3 to the 3 equals 27, and 3 to the 4 equals 81, a dashed line marking the 24 cases needed for twelve coins, the first two bars red because they fall short and the last two green because they clear it, with a callout that k equals 3 is the smallest sufficient choice.](/imgs/blogs/logic-puzzles-deduction-quant-interviews-10.png)

The bar chart makes the bound visceral: the red bars ($3^1 = 3$ and $3^2 = 9$) cannot reach the 24-case line, while the green bars ($3^3 = 27$ and $3^4 = 81$) clear it, so three is the smallest number of weighings that can possibly work. If an interviewer changes the problem to thirteen coins, the same chart tells you the answer immediately: $13 \times 2 = 26 \le 27$, so three weighings is still *barely* enough in principle — though now the design has only one outcome of slack and a clean general solution requires a small twist (you cannot always determine the direction for the last coin without a known-good reference, so the standard answer is that 13 coins are solvable in three weighings only if you are given one extra coin known to be genuine).

## Hat problems: turning one sacrifice into a shared parity bit

Hat puzzles are the canonical home of the *parity invariant*. The setup: a row of prisoners, each wearing a hat of some color, each able to see the hats *in front* of them but not their own or those behind. They must each guess their own hat color; the interviewer (acting as warden) wants to know the best guaranteed strategy, often with a payout attached for the team.

### The two-color line of 100

A hundred prisoners stand in a line, each wearing a red or blue hat assigned adversarially. Starting from the back (the prisoner who can see all 99 hats ahead), each prisoner in turn says a single word — a color guess — that everyone hears. A correct guess frees that prisoner; a wrong one does not. The warden offers a \$1,000 team payout if at least 99 of the 100 walk free. What strategy guarantees it, and how many are saved?

The naive read is hopeless: how can prisoner 50 possibly know their own hat color? The lever is that the prisoners may *agree on a protocol in advance*, and the very first guess can be spent not on saving that prisoner but on broadcasting one bit of shared information — the *parity* of the red hats the back prisoner can see.

![Diagram titled the hat-parity scheme one sacrifice saves the rest, a left-to-right pipeline of prisoner 100 counting red hats ahead and announcing parity, passing a parity bit to prisoner 99 who hears the parity and sees 98 ahead and deduces, then to prisoner 98 who updates the running parity, and on to prisoner 1 who tracks all calls for a certain answer.](/imgs/blogs/logic-puzzles-deduction-quant-interviews-4.png)

The pipeline above is the protocol. One prisoner gambles; everyone else deduces with certainty.

#### Worked example: the parity strategy that saves 99 of 100

The agreed rule: **"red" means I see (or have deduced is consistent with) an even number of red hats ahead under the running count; "blue" means odd."** Concretely:

1. **Prisoner 100** (at the back, sees all 99 ahead) counts the red hats in front. Suppose there are 37 — an *odd* number. By the agreed code, an odd count is announced as "blue." This prisoner has a 50/50 chance of being right about their *own* hat (they gambled it to send the parity), so the worst case loses exactly one prisoner.
2. **Prisoner 99** heard "blue," meaning the total reds among prisoners 1 through 99 is *odd*. Prisoner 99 looks ahead and counts the reds among prisoners 1 through 98 — say 37, odd. For the count among 1–99 to be odd while the count among 1–98 is already odd, prisoner 99's own hat must contribute an *even* number of reds, i.e. zero reds: their hat is **blue**. They say "blue" and walk free with certainty.
3. **Prisoner 98** now knows the original parity *and* prisoner 99's color (which was announced), so they can subtract 99's contribution and recompute the parity of reds among 1 through 98. They compare it to the reds they see among 1 through 97 and deduce their own hat with certainty. And so on down the line: every prisoner tracks the running parity, subtracting off the colors already called out ahead of and behind the relevant window, and nails their own hat.

Because prisoners 1 through 99 all deduce with certainty, **99 are guaranteed saved**; only prisoner 100 is at a coin-flip, and in the worst case is the single casualty. The team clears the "at least 99" bar every time and collects the \$1,000. The intuition: *parity is a single bit that the whole group can share; spending the first, hopeless guess to broadcast it converts everyone else's impossible problem into a certain one.*

This generalizes beautifully. With $c$ colors instead of two, replace parity (sum mod 2) with the sum of color-indices *mod $c$*; the back prisoner announces that sum, sacrificing one guess, and the other 99 each deduce their color exactly. The save rate is always "all but the first," regardless of the number of colors — a result that surprises almost everyone the first time, and a favorite follow-up precisely because the candidate who understood the mod-2 case can usually generalize it on the spot.

## The 100 prisoners and the boxes: working backwards to cycles

This is the puzzle that most cleanly separates "I will compute the obvious probability" from "I will reason about the *structure*." It also delivers the single most counterintuitive number in the interview canon.

The setup: 100 prisoners are numbered 1 to 100. In a room are 100 closed boxes, also numbered 1 to 100; inside each box is a slip of paper with a prisoner's number, placed by the warden in some *random* arrangement (a random *permutation* — a one-to-one shuffling of the numbers 1 through 100). One at a time, each prisoner enters the room and may open *up to 50* boxes, looking for the slip with their own number. If every one of the 100 prisoners finds their own number, the whole team is freed; if even one fails, everyone is executed. Prisoners cannot communicate or rearrange anything once the process starts. The warden, feeling generous, offers a \$1,000,000 freedom fund if they all succeed. What is the best strategy, and what is the chance it works?

If each prisoner opens 50 random boxes, their individual chance of finding their number is 50/100 = 1/2, and since the prisoners' searches are independent, the team's chance is $(1/2)^{100}$ — about $1$ in $10^{30}$, so small that you could run the experiment every second since the Big Bang and never expect a single success. It looks utterly hopeless, and most candidates stop here and declare it impossible. The astonishing fact is that a clever strategy lifts the team's success probability to about **31%**.

![Diagram titled prisoners and boxes follow the cycle to your own number, with a top row showing a short safe cycle where box 1 holds slip 3, box 3 holds slip 5, and box 5 holds slip 1, the arrows closing back to box 1, and a bottom row showing a long failing cycle of more than 50 boxes chaining box 2 to box 7 to box 4 and onward until it returns to box 2, labeled as the only way the strategy fails.](/imgs/blogs/logic-puzzles-deduction-quant-interviews-5.png)

The figure above shows the whole idea: treat the arrangement as a set of *cycles* and have each prisoner follow their own cycle.

#### Worked example: the cycle-following strategy and its 31%

The strategy: **prisoner $k$ opens box $k$ first. Whatever number is inside, that becomes the next box to open. Repeat — each slip points to the next box — until you find your own number or run out of your 50 opens.**

Why does this work so well? Because the warden's random arrangement is a permutation, and every permutation decomposes uniquely into *cycles*. Following the strategy, prisoner $k$ traces exactly the cycle that contains the number $k$. Walk the top row of the figure: box 1 holds slip 3, box 3 holds slip 5, box 5 holds slip 1. Prisoner 1 opens box 1 (finds 3), then box 3 (finds 5), then box 5 (finds 1) — *their own number*. The cycle closed after three steps, well within 50, so prisoner 1 succeeds. Crucially, every prisoner whose number lies on that same short cycle (prisoners 1, 3, and 5) also succeeds, because they each trace the *same* loop and reach their own number in at most three opens.

So when does a prisoner fail? Only if their cycle has length greater than 50 — then they run out of opens before looping back. And here is the decisive observation: **a permutation of 100 elements can contain at most one cycle longer than 50** (two cycles of length 51+ would need more than 100 elements). Therefore *all 100 prisoners succeed if and only if the permutation's longest cycle has length at most 50.* The team's fate hinges on a single yes/no question about the random arrangement: is there a long cycle or not?

Now we compute that probability. The chance a random permutation of 100 has a cycle of length exactly $\ell > 50$ turns out to be exactly $1/\ell$ (a clean fact: there are $\binom{100}{\ell}(\ell-1)!$ ways to choose and arrange such a cycle, times $(100-\ell)!$ for the rest, over $100!$, which simplifies to $1/\ell$). Since at most one long cycle can exist, the events for different $\ell$ are disjoint, so

$$
P(\text{some cycle} > 50) = \sum_{\ell=51}^{100} \frac{1}{\ell} = \frac{1}{51} + \frac{1}{52} + \cdots + \frac{1}{100}.
$$

That sum is the difference of harmonic numbers $H_{100} - H_{50}$, which is very close to $\ln(100) - \ln(50) = \ln 2 \approx 0.693$. So the chance of failure is about 69.3%, and the chance *all 100 prisoners succeed* is about $1 - 0.693 = 0.307$, or roughly **31%**.

![Diagram titled boxes strategy success means no cycle longer than 50, a bar chart comparing an almost invisible red bar at about 0 percent for everyone guessing randomly against a green bar at about 31 percent for the cycle-following strategy, with a dashed 50 percent reference line and side cards explaining that the team wins if and only if the longest cycle is at most 50, giving one minus the sum of reciprocals from 51 to 100, about one minus the natural log of 2, around 0.3118.](/imgs/blogs/logic-puzzles-deduction-quant-interviews-12.png)

The chart drives home the contrast: random guessing wins essentially never, while the cycle strategy wins almost a third of the time — a swing of roughly thirty orders of magnitude from one structural insight. As the number of prisoners grows, the success probability does *not* shrink to zero; it converges to $1 - \ln 2 \approx 0.3069$ and stays there. The one-sentence intuition: *the strategy couples every prisoner's fate to the same global event — the existence of a long cycle — so 100 independent coin-flips collapse into a single, far more favorable question.*

## Light-switch and rooms puzzles: finding a second channel

Switch puzzles test whether you can find a *second observable channel* when the obvious one is too narrow. The classic: there are three on/off switches outside a closed room, and exactly one of them controls a single incandescent light bulb inside. The door is shut and you cannot see the bulb from outside. You may flip the switches as much as you like, but you may enter the room *only once*. Which switch controls the bulb?

The naive analysis says it is impossible: when you enter, the bulb is either on or off — two states — but there are three switches to distinguish, and two states cannot label three switches. The counting bound from earlier seems to forbid it. The escape is to notice that a bulb has *more than one* observable property. It emits light, yes, but an incandescent bulb that has been on also gets *warm*. Heat is a second channel, and two binary channels give $2 \times 2 = 4$ states — more than enough to label three switches.

![Diagram titled three switches one bulb read heat and light together, a decision tree that starts at enter the room and observe the bulb, branching into bulb is on which maps to switch B that is on right now, bulb is off but warm which maps to switch A that was on then turned off, and bulb is off and cold which maps to switch C that was never touched.](/imgs/blogs/logic-puzzles-deduction-quant-interviews-6.png)

The tree above is the full solution, reading the bulb's two channels — light and heat — together.

#### Worked example: identifying the switch in one entry

The procedure:

1. Flip **switch A on** and leave it for about ten minutes, long enough to warm the bulb if A is the controlling switch.
2. **Turn switch A off**, then immediately **flip switch B on**.
3. **Enter the room** and feel and look at the bulb.

Now read the two channels:

- **On (lit):** the bulb is currently powered, so **switch B** controls it — B is the one that is on right now.
- **Off but warm:** the bulb is not powered now, but it was recently and still holds heat, so **switch A** controls it — A was the one on for ten minutes before you turned it off.
- **Off and cold:** the bulb has been off the whole time and never warmed, so **switch C** controls it — C is the one you never touched.

Three switches, one entry, fully determined. The intuition: *when one observable channel is too small to carry the information you need, look for a second one the problem quietly provides — here, the bulb's temperature doubles your state space from two to four.*

A close cousin is the "100 light bulbs and 100 passes" puzzle, also a desk favorite: 100 bulbs start off; on pass $k$ you toggle every $k$-th bulb (pass 1 toggles all, pass 2 toggles 2,4,6,…, and so on through pass 100). Which bulbs are on at the end? Bulb $n$ is toggled once for each *divisor* of $n$, so it ends *on* exactly when $n$ has an *odd* number of divisors — and the only numbers with an odd divisor count are perfect squares (because divisors pair up $d \leftrightarrow n/d$ unless $d = n/d$). So bulbs 1, 4, 9, 16, 25, 36, 49, 64, 81, and 100 are on: exactly **ten** bulbs, the perfect squares up to 100. This is parity again — the parity of the divisor count — wearing a different costume.

## Induction puzzles: the blue-eyed islanders

Induction is the technique of solving the size-$n$ problem by leaning on the already-solved size-$(n-1)$ problem. The most famous and most argued-about induction puzzle is the blue-eyed islanders, and it is a magnificent test of careful reasoning about *common knowledge* — the recursive state of "everyone knows that everyone knows that everyone knows…"

The setup: on an island live 100 perfectly logical people, all with blue eyes, though none of them knows their own eye color and there are no mirrors or reflective surfaces, and they never discuss eyes. Their religion forbids knowing your own eye color: if you ever *deduce* that your eyes are blue, you must leave the island at dawn the next day, publicly. They all see each other every day. One day a trusted outsider visits and says, to the whole assembled group, a single true sentence: *"At least one of you has blue eyes."* What happens?

The maddening part is that the outsider seems to have said *nothing new* — every islander can already see 99 pairs of blue eyes, so "at least one" is obviously, visibly true. Yet the announcement triggers an inexorable countdown, and on the 100th dawn after it, *all 100 blue-eyed islanders leave the island together.* The resolution is that the announcement does add something: it makes "at least one has blue eyes" *common knowledge*, the shared recursive certainty that lets the induction run.

![Diagram titled blue-eyed islanders the induction ladder unwinds over nights, a timeline from night 1 where if only one islander were blue-eyed he would see zero blue and leave at once, to night 2 where two blue-eyed islanders each expect the other to leave and when nobody does both go, to night 3, to night 99 where 99 still present proves 100 blue eyes exist, to night 100 where all 100 blue-eyed islanders leave together.](/imgs/blogs/logic-puzzles-deduction-quant-interviews-7.png)

The timeline above shows the ladder unwinding one rung — one night — per islander.

#### Worked example: walking the induction from 1 to 100

Build it up from the smallest case, which is exactly what induction demands.

- **Base case, 1 blue-eyed islander.** Suppose there were just one blue-eyed person. They look around, see *zero* other blue eyes, and hear "at least one of you has blue eyes." The only possible source of that blue is themselves. So they deduce their own blue eyes immediately and leave on the **first dawn** (night 1).
- **Two blue-eyed islanders, A and B.** Each sees exactly one other blue-eyed person. A reasons: "If I am not blue-eyed, then B is the only blue, and by the base case B will leave on night 1." A waits and watches. Night 1 passes and B does *not* leave — which can only mean B *also* sees a blue-eyed person, and the only candidate is A. So on **night 2**, A deduces their own blue eyes; by the identical reasoning so does B, and both leave together on night 2.
- **Three blue-eyed islanders.** Each sees two others and reasons: "If I am not blue, the other two are the only blues, and by the two-person case they will both leave on night 2." When night 2 passes with nobody leaving, each of the three learns there must be a *third* blue — themselves — and all three leave on **night 3**.
- **The induction step.** In general, if there are $n$ blue-eyed islanders, each one sees $n-1$ others and reasons: "If I am not blue, there are only $n-1$ blues, and by the case-$(n-1)$ result they will all leave on night $n-1$." When night $n-1$ passes with no departure, each of the $n$ concludes they themselves must be the extra blue, and all $n$ leave together on **night $n$**.

With 100 blue-eyed islanders, this runs all the way up: nights 1 through 99 pass with nobody leaving (each non-departure is itself a deduction, ruling out the hypothesis of fewer blues), and on **night 100** all 100 leave at once. The role of the outsider is subtle but essential: although everyone already *saw* blue eyes, nobody knew that everyone knew that everyone knew… and so on, 100 levels deep. The announcement establishes that infinite tower of shared knowledge — *common knowledge* — which is precisely the fuel the induction burns. The intuition: *each night with no departure is one rung of an induction ladder; the count of nights it takes equals the number of blue-eyed islanders, and the outsider's role is to light the first rung by making the base fact common knowledge.*

## Invariants and parity: trapping a quantity that cannot change

We have already used parity twice (hats, light bulbs); now we name the general technique. An **invariant** is a quantity or property that *every legal move in the puzzle leaves unchanged*. Once you find one, it becomes a wall: any goal state that has a different invariant value than the start is *unreachable*, no matter how the moves are sequenced. This turns "can I get from here to there?" from an infinite search into a one-line check — compute the invariant at both ends and compare.

![Diagram titled the mutilated chessboard a coloring invariant kills the tiling, an eight by eight grid with two opposite corners removed and marked with a red X, the squares colored in a checkerboard of lavender dark squares and gray light squares, alongside a panel noting 32 dark squares, 30 light squares, that each domino covers one dark and one light cell, and that 31 dominoes would need 31 of each so the tiling cannot exist.](/imgs/blogs/logic-puzzles-deduction-quant-interviews-8.png)

The classic demonstration, shown above, is the *mutilated chessboard*.

#### Worked example: why you cannot tile the mutilated chessboard

Take a standard 8×8 chessboard — 64 squares, colored in the usual alternating pattern — and remove two diagonally *opposite* corners. Both removed corners are the same color (opposite corners of a chessboard always match), so you are left with 62 squares: **32 of one color and 30 of the other.** Now ask: can you tile these 62 squares exactly with 31 dominoes, each domino covering two adjacent squares?

Here is the invariant. *Every* domino, placed anywhere on the board, covers two *adjacent* squares — and adjacent squares are always *different* colors. So each domino covers exactly one dark and one light square, no exceptions. Therefore any arrangement of 31 dominoes covers exactly 31 dark and 31 light squares. But our mutilated board has 32 of one color and only 30 of the other. Since $31 \ne 32$ and $31 \ne 30$, *no tiling can possibly exist* — the color balance is an invariant that the goal violates. Notice what we did *not* do: we never tried to place a single domino or search through arrangements. The invariant settled an infinite search in one comparison.

This pattern recurs constantly. *"Can a knight starting on a1 visit every square and return to a1 in an odd number of moves?"* No — a knight alternates colors each move, so returning to the start (same color) always takes an even number of moves; the parity of the move count is an invariant. *"Fifteen-puzzle: can you swap just two adjacent tiles and leave the rest?"* No — every legal slide changes the permutation's parity in lockstep with the blank's position, and a single transposition violates that coupled invariant. The intuition: *when a puzzle asks whether a target state is reachable, hunt for a quantity that every move preserves; if the start and target disagree on it, the answer is "impossible" and you have proved it, not guessed it.*

#### Worked example: the chameleon parity puzzle with a payout

Here is one an interviewer might pose with a wager. On an island there are chameleons of three colors: 13 red, 15 green, and 17 blue (45 total). Whenever two chameleons of *different* colors meet, they both change to the *third* color. (Two reds meeting a... no — a red and a green meeting both turn blue, for instance.) An interviewer bets you \$20 that you cannot make all 45 chameleons the same color. Can you?

Look for an invariant in the color counts modulo 3. Start: red = 13, green = 15, blue = 17. Consider the *differences* between counts modulo 3. When a red and a green meet and both become blue, red drops by 1, green drops by 1, blue rises by 2. Track, say, (red − green) mod 3: that goes from $13 - 15 = -2$ to $12 - 14 = -2$ — unchanged. Track (green − blue) mod 3 and (red − blue) mod 3 across each of the three meeting types and you find each pairwise difference mod 3 is invariant. Initially the three counts modulo 3 are $13 \equiv 1$, $15 \equiv 0$, $17 \equiv 2$ — all *different* residues. To make all chameleons one color, two of the counts must reach zero, which would force two of the residues to be equal (both $\equiv 0$). But the pairwise differences mod 3 never change, so two residues that start unequal stay unequal forever. The goal is unreachable; you keep your \$20. (Had the starting counts shared a residue mod 3 for two colors, it *would* be possible.) The intuition: *the counts modulo 3 are frozen by the meeting rule, so the reachable configurations are exactly those with the same residue pattern as the start.*

## The pigeonhole principle: more items than boxes

The **pigeonhole principle** is the simplest deduction tool and one of the most quietly powerful. Its statement: *if you put $n+1$ items into $n$ boxes, at least one box must contain two or more items.* More generally, with $k$ boxes and more than $k$ items, some box holds at least $\lceil \text{items}/k \rceil$ of them. It sounds too obvious to be useful, yet it cracks a whole genre of "prove that there must exist…" questions where you do not need to find the example, only argue it cannot fail to exist.

![Diagram titled the pigeonhole principle 13 pigeons 12 holes one shared, showing twelve month boxes January through December each labeled one person except March which is labeled two people in amber, with explanation cards noting that the first twelve people can fill the twelve months one each, the thirteenth person has no empty month and forces a repeat, and the general rule that k boxes with more than k items force one box to hold at least the ceiling of items over k.](/imgs/blogs/logic-puzzles-deduction-quant-interviews-9.png)

The figure above is the principle in its most relatable form: thirteen people, twelve months, a guaranteed shared birth month.

#### Worked example: guaranteed shared birthday month, and a hair-count claim

First the gentle version. In any group of **13 people**, at least two share a *birth month*. Why? There are only 12 months — twelve "boxes." The first twelve people could, in the best case, each land in a different month, filling all twelve boxes one apiece. But the 13th person must land in *some* month, and every month is already taken, so they necessarily share with someone. There is no escape and no need to know anything about the actual people; 13 into 12 forces a collision.

Now a version that startles. *Claim: somewhere in a city of a million people, at least two have exactly the same number of hairs on their head.* A human head has at most a few hundred thousand hairs — call it under 200,000 as a safe ceiling. So "number of hairs" is a box label with at most ~200,000 possible values. We have 1,000,000 people — items — to drop into ~200,000 boxes. Since $1{,}000{,}000 > 200{,}000$, the pigeonhole principle forces at least one box to contain more than one person; in fact $\lceil 1{,}000{,}000 / 200{,}000 \rceil = 5$ people must share *some* exact hair count. We have proven a collision exists without examining a single scalp. The intuition: *when the number of items strictly exceeds the number of possible distinct values, a repeat is unavoidable — and pigeonhole lets you assert it with certainty while knowing nothing about the specifics.*

A slicker interview use: *prove that among any 5 points placed inside a 1×1 square, two of them are within $\frac{\sqrt 2}{2} \approx 0.707$ of each other.* Chop the square into four $\frac12 \times \frac12$ sub-squares (four boxes); five points into four boxes forces two points into the same sub-square (pigeonhole), and two points in a $\frac12 \times \frac12$ square are at most its diagonal $\frac{\sqrt2}{2}$ apart. Done — the construction of the boxes *is* the whole solution.

## In the interview room: five fully solved problems

Technique sections teach the tools in isolation; a real interview hands you a problem cold and watches you pick the tool. Here are five problems in the style and difficulty of an on-site round, each solved out loud the way you would actually want to narrate it. Notice that the first move in every one is to name the technique.

#### Worked example: the 10 bags of coins (counting, single weighing)

*"You have 10 bags of coins. Nine bags contain genuine coins weighing 10 grams each; one bag contains counterfeits weighing 11 grams each. You have a digital scale that reads exact weight — but you may use it only* once*. There's a \$500 bonus if you can identify the counterfeit bag in that single weighing. How?"*

State the technique: this is a *counting / encoding* problem, but the scale here gives a *number*, not just three outcomes, so a single weighing can carry far more information than a balance — we just need to encode the bag's identity into the weight. Take **1 coin from bag 1, 2 coins from bag 2, 3 from bag 3, …, 10 from bag 10**, for $1+2+\cdots+10 = 55$ coins total. If all were genuine, they would weigh $55 \times 10 = 550$ grams. Each counterfeit coin adds exactly 1 extra gram. So if bag $k$ is the fake one, you took $k$ counterfeit coins from it, and the scale reads $550 + k$ grams. Read the excess: a reading of 553 means $k = 3$, bag 3 is counterfeit; 557 means bag 7. The single number on the scale *encodes the bag identity*, claiming the \$500. The lesson: when a measurement returns a rich value rather than a coarse outcome, encode identities so the value decodes them.

#### Worked example: two eggs and a 100-floor building (working backwards / optimization)

*"You have two identical eggs and a 100-story building. An egg breaks if dropped from floor $T$ or above and survives below $T$; you want to find $T$. A dropped egg that breaks is gone. Minimize the number of drops in the* worst *case. What is the minimum, and the strategy?"*

Name it: this is *working backwards from the worst case* with a counting flavor. With one egg you would have to go floor by floor from the bottom (a break with no spare egg leaves you unable to test anything new safely), so one egg costs up to 100 drops. The two-egg trick: use the first egg to take *big jumps*, and once it breaks, use the second egg to walk up *one floor at a time* through the narrow band you have isolated. The clever part is making the worst case equal across all outcomes. If the first egg's drops are at floors $d, d + (d-1), d + (d-1) + (d-2), \ldots$ — decreasing step sizes — then each time the first egg survives you have "used up" one drop but the remaining band you might have to walk shrinks by one, keeping the total worst case constant. You want the smallest $d$ with $d + (d-1) + (d-2) + \cdots + 1 = \frac{d(d+1)}{2} \ge 100$. Solving, $d = 14$ gives $\frac{14 \cdot 15}{2} = 105 \ge 100$, while $d = 13$ gives only 91. So the answer is **14 drops** in the worst case: drop the first egg from floor 14, then 27, then 39, …, each jump one smaller; when it breaks, walk the second egg up from just above the previous safe floor. The lesson: to minimize a worst case, *balance the outcomes* so no single branch is worse than the others.

#### Worked example: the 100 hats with a twist (parity under adversary)

*"Ten prisoners, each given a black or white hat by an adversary, stand so each sees all the others' hats but not their own. At a signal they* simultaneously *write down a guess for their own hat color. They win a \$10,000 pot if at least one guess is correct. They may strategize beforehand but cannot communicate after hats go on. Guarantee a win."*

The earlier hat puzzle let prisoners hear each other; here guesses are simultaneous, so we cannot pass a parity bit down a line. The fix is still parity, used differently. Assign black = 0, white = 1. Agree in advance that **prisoner $i$ assumes the total of all 10 hat-values is congruent to $i \bmod 10$**, and guesses their own hat as whatever value would make the visible sum match that assumption. The actual total has *some* residue $r$ mod 10, and prisoner $r$ is the one whose assumption is correct — so prisoner $r$'s guess is exactly right. Since *exactly one* residue is the true one, *at least one* prisoner (prisoner $r$) always guesses correctly, and the pot is won every time. The lesson: when you cannot transmit information, *partition the assumptions* so that whatever the hidden truth is, somebody assumed it.

#### Worked example: the burning ropes timer (working backwards from the goal)

*"You have two ropes and a lighter. Each rope takes exactly 60 minutes to burn end to end, but burns* unevenly *— you cannot assume the halfway point is 30 minutes. Using only these ropes and the lighter, measure exactly 45 minutes."*

Work backwards from "45 = 60 − 15" and "15 = 30/2," and the trick to *halving a burn time* is to light a rope at *both ends at once*: two flames consume the whole 60-minute rope in 30 minutes regardless of unevenness, because the two burning fronts together always finish the total length in half the time. The construction: at $t = 0$, light **rope A at both ends** and **rope B at one end**. Rope A burns out at exactly $t = 30$ minutes. At that instant, rope B has 30 minutes of burn *left* (it has been burning from one end for 30 minutes). Now **light rope B's other end** too; with both ends burning, its remaining 30-minutes-worth of rope is consumed in 15 minutes, finishing at $t = 30 + 15 = 45$ minutes. The moment rope B is fully burned marks exactly 45 minutes. The lesson: lighting both ends *halves* a duration without needing to know the rope's profile — find the operation that gives you the fraction you need, then chain operations backwards from the target.

#### Worked example: a 1,000-bottle wine cellar with one poisoned bottle (information encoding)

*"A king has 1,000 bottles of wine; exactly one is poisoned. The poison kills, with no other symptom, after exactly 24 hours. He has prisoners he can use as testers and a royal feast in 24 hours. What is the minimum number of prisoners needed to identify the poisoned bottle in time? There's a \$1,000 reward for the tightest answer."*

This is *pure information encoding*. Each prisoner is a single bit: after 24 hours they are either dead (1) or alive (0). With $p$ prisoners you get a $p$-bit string of outcomes, which can encode $2^p$ distinct patterns. You must distinguish 1,000 possibilities (which bottle), so you need $2^p \ge 1000$, giving $p = 10$ since $2^{10} = 1024 \ge 1000 > 512 = 2^9$. So **10 prisoners** suffice. The construction makes it concrete: number the bottles 0 to 999 and write each number in **10-bit binary**; prisoner $j$ drinks a sip from *every bottle whose $j$-th binary digit is 1*. After 24 hours, read off which prisoners died as a 10-bit binary number — that number *is* the poisoned bottle's index, because only the poisoned bottle's bit-pattern matches the exact set of prisoners who drank from it and died. Ten yes/no channels, $2^{10}$ patterns, 1,000 bottles distinguished, reward claimed. The lesson: when each tester is one bit and you may run them in parallel, $p$ testers resolve $2^p$ cases — assign each case a distinct binary codeword and let the outcome string decode it.

## Common misconceptions

A handful of wrong instincts sink candidates who actually know the math. Naming them is half the cure.

- **Guessing the answer instead of deducing it.** The single most common failure is to *recognize* a puzzle ("oh, it's the one where the answer is 31%") and blurt the number without rebuilding the argument. Interviewers can tell, and a remembered number with no derivation scores near zero — sometimes worse than a wrong answer honestly reasoned, because it signals you optimize for *recognition* rather than *thinking*. Always reconstruct the logic out loud, even on a puzzle you have seen, as if discovering it fresh.
- **Forgetting the information bound, so you over- or under-claim.** Candidates routinely attempt to do in two weighings what provably needs three, or declare three weighings impossible for a problem that fits in $3^3 = 27$. Compute the bound *first*: it tells you the minimum number of questions and whether the task is even possible. The bound is not optional scaffolding; it is the answer to half the puzzle.
- **Solving the lucky case instead of the worst case.** A guaranteed strategy must work against an *adversary*. "If the fake happens to be in the first group, then…" is not a solution if the fake could be anywhere. The 12-coin design has to handle all 24 cases; the egg-drop strategy is judged by its *worst* outcome, not its best. Whenever a puzzle says "guarantee," translate it to "survive the worst arrangement nature can choose."
- **Believing the obvious probability is the achievable one.** The prisoners-and-boxes puzzle traps everyone who computes $(1/2)^{100}$ and stops. The naive independent-events probability is a *lower bound* under the worst strategy, not a ceiling — a clever strategy that *correlates* the outcomes can do dramatically better. When independent failure looks hopeless, ask whether a strategy can couple the events.
- **Thinking the outsider's announcement told the islanders nothing.** "Everyone could already see blue eyes" is true but irrelevant; what the announcement adds is *common knowledge* — the infinite tower of "I know that you know that I know…" — which is strictly more than each person privately seeing blue eyes. The distinction between *mutual* knowledge (everyone knows) and *common* knowledge (everyone knows that everyone knows, forever) is the whole puzzle, and it shows up on real desks whenever a fact becomes *public*.
- **Assuming a balance scale gives a yes/no answer.** A two-pan balance has *three* outcomes, not two, and forgetting the "balanced" outcome is the most common 12-coin error — it throws away a third of your information. Count the outcomes the instrument actually produces.

## How structured deduction shows up on a real desk

It would be easy to dismiss these puzzles as hazing rituals, but each technique maps onto something a quant or trader does for real. The puzzles are a compressed proxy, and the firms ask them because the proxy correlates with the job.

**Counting information before acting** is exactly how a desk decides whether a signal *can* possibly work before building it. If you want to predict a daily return and you have only 60 trading days of history, no model — however fancy — can reliably estimate 40 free parameters from 60 noisy points; the information simply is not there, and a researcher who computes that bound first saves weeks of overfitting. This is the same instinct as "three weighings can resolve at most 27 cases." The discipline of asking *how much information do I actually have* underlies sound work in [backtesting](/blog/trading/quantitative-finance/backtesting-done-right-quant-research) and in fighting [overfitting with purged cross-validation](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe).

**Invariants and conservation arguments** are the backbone of no-arbitrage reasoning. The statement "you cannot construct a riskless profit from nothing" is an invariant: a portfolio with zero cost and zero risk must have zero payoff, or someone would have already traded the difference away. The entire edifice of [put-call parity and no-arbitrage pricing](/blog/trading/quantitative-finance/put-call-parity-no-arbitrage-quant-interviews) is "find the quantity that cannot change and pin the price to it" — the same move as the mutilated-chessboard coloring. When a trader says "that price is impossible," they are usually pointing at a violated invariant.

**Parity and worst-case thinking** show up in risk management. A risk manager does not ask "what happens on an average day"; they ask "what happens in the *worst* plausible arrangement of correlations" — the adversarial case, exactly like a guaranteed weighing strategy. Stress tests are the desk's version of "your strategy must survive whatever nature throws at it." The habit of separating "usually fine" from "always fine" is what keeps a book solvent through a crisis.

**Working backwards from a goal** is how exotic derivatives get priced and hedged. To value a [structured product](/blog/trading/quantitative-finance/exotic-derivatives), you start from its payoff at expiry — the goal state — and discount and replicate backwards to today, exactly as the egg-drop and burning-rope solutions reason from the target. Dynamic programming, the engine behind pricing American options and optimal execution, is "work backwards from the end" formalized.

**Common knowledge and recursive belief** — the blue-eyes lesson — drive how markets react to *public* information. A piece of news that everyone privately knew can still move prices violently the moment it becomes *public and common knowledge*, because now everyone knows that everyone knows, and can act on others acting. Central-bank communication is a deliberate exercise in managing common knowledge: the announcement's power is not the private information it reveals but the shared certainty it creates. A trader who understands the islanders understands why a "priced-in" fact can still jolt the tape when it is finally said out loud.

Two real episodes make it concrete. In the lead-up to a Federal Reserve rate decision, the *direction* is often near-universally expected — every desk privately believes a hike is coming — yet the market still lurches at the announcement, because the move from "I believe" to "it is now common knowledge that everyone believes and the committee confirmed it" is itself information, precisely the blue-eyes effect. And in the 1998 collapse of the hedge fund Long-Term Capital Management, the firm's models were sound *on an average day* but catastrophically wrong in the *worst-case correlation arrangement*, when normally-independent positions all moved against them at once — a failure to respect the adversarial case that any guaranteed-strategy puzzle would have flagged. Structured deduction is not a parlor trick; it is the cognitive habit the job is built on.

## When this matters and further reading

If you are preparing for quant interviews, the highest-leverage move is not memorizing more puzzles — it is internalizing the *four techniques* so that a novel puzzle triggers a tool rather than a panic. When you hear a new problem, run the checklist below before you say anything.

![Diagram titled choosing a technique a decision flow for the puzzle, a branching graph that starts from read the puzzle what is asked and splits into limited questions or weighings leading to counting and information, some quantity stays constant leading to invariant and parity, and reduces to a smaller version leading to either induction or working backwards.](/imgs/blogs/logic-puzzles-deduction-quant-interviews-11.png)

The decision flow above is the triage routine. Ask, in order:

- **Is there a question budget — a limited number of weighings, questions, or tests?** Reach for **counting / information**: compute $2^k$ or $3^k$ (or $2^p$ for parallel testers) against the case count, find the minimum, and design each question to split the survivors evenly.
- **Does some quantity stay constant under every move?** Reach for an **invariant or parity**: name the conserved quantity, evaluate it at the start and the goal, and if they differ, the goal is impossible — proven, not guessed.
- **Does the problem reduce to a smaller copy of itself?** Reach for **induction** (build up from the base case) or **working backwards** (start at the goal and undo moves). Recursion and "smaller version" are the tells.

Then narrate everything, state your assumptions, and handle the worst case, not the lucky one.

To go deeper, the puzzles here connect directly to the probabilistic toolkit. The prisoners-and-boxes cycle argument is really a fact about random permutations, which sits alongside the [classic quant probability problem set](/blog/trading/quantitative-finance/classic-quant-probability-problems) and the [counting and combinatorics](/blog/trading/quantitative-finance/counting-combinatorics-quant-interviews) techniques. The information-bound logic is the discrete cousin of the [estimation and Fermi problems](/blog/trading/quantitative-finance/estimation-fermi-problems-quant-interviews) that test the same "how much can I actually pin down" instinct. The worst-case and adversarial thinking carries straight into [decision making under uncertainty](/blog/trading/quantitative-finance/decision-making-under-uncertainty-quant-interviews) and the [market-making games](/blog/trading/quantitative-finance/market-making-games-quant-interviews) where you must price under an opponent who knows your strategy.

The puzzles are finite, but the four techniques are not — they are the load-bearing habits of thinking clearly under constraints, which is the whole job. Practice the *reasoning*, narrated out loud, and the specific puzzles take care of themselves. This material is educational, aimed at sharpening reasoning for interviews and beyond, and is not financial advice.
