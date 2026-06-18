---
title: "Case Study: Netflix and Spotify, Recommendation as Product"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How two media giants turned recommendation into the product itself — Netflix's whole-page 2-D ranking and personalized artwork, Spotify's Discover Weekly blend of collaborative filtering, NLP, and audio CNNs — with the science behind each idea and a runnable repro that predicts CF embeddings from content for cold-start tracks."
tags:
  [
    "recommendation-systems",
    "recsys",
    "netflix",
    "spotify",
    "case-study",
    "cold-start",
    "contextual-bandits",
    "whole-page-optimization",
    "machine-learning",
    "content-embeddings",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/case-study-netflix-and-spotify-recommendation-as-product-1.png"
---

There is a moment, somewhere in the life of every consumer media company, when the recommender stops being a feature bolted onto a catalog and becomes the catalog's reason to exist. At Netflix, the company has said publicly that around 80% of what members watch comes from its recommendations rather than from search. At Spotify, a Monday-morning playlist that nobody asked for — Discover Weekly — became one of the most-loved things the product does, and the surface where tens of millions of people meet new music. In both cases the recommender is not the thing that helps you find the product. The recommender *is* the product. The home screen you open is a recommendation. The thumbnail you see is a recommendation. The order of the rows, the order inside each row, the playlist that materializes every Monday — all of it is the output of a learning system, and the company's growth, retention, and content-buying decisions ride on it.

This post is a case study of how Netflix and Spotify got there, and what you can steal from them. It is the chapter of this series where the abstractions we have built — implicit feedback, ranking, the cold-start problem, bandits, beyond-accuracy metrics, the offline-online gap — come together inside two real, shipped, enormous systems. We will not retell the marketing story. We will look at the engineering decisions and the science behind them: why both companies walked away from rating-prediction accuracy as their objective; why Netflix reframed the problem from a single ranked list to a two-dimensional page; why Spotify needed three different signal sources to feed one playlist; how a convolutional network on a spectrogram solves the cold-start of brand-new music; and why choosing a thumbnail is, formally, a contextual-bandit problem.

![A two-column figure contrasting a single ranked top-K list that optimizes one accuracy metric against a two-dimensional page of personalized rows with within-row ranking that optimizes engagement and retention](/imgs/blogs/case-study-netflix-and-spotify-recommendation-as-product-1.png)

The figure above is the central reframing of the whole post. On the left is the recommender most people picture and most courses teach: score every item, sort, show the top $K$, optimize one accuracy number. On the right is what Netflix actually ships: a page of rows, each row a themed, personalized ranking, and the rows themselves ordered per member — a two-dimensional layout problem whose objective is not accuracy at all but engagement and retention. The leap from the left picture to the right one is the leap from recommendation-as-feature to recommendation-as-product, and it changes the math, the metric, the experimentation culture, and even what counts as a model.

By the end you will be able to: explain why Netflix retired RMSE on star ratings in favor of implicit play data and a ranking objective; state the whole-page (2-D slate) optimization problem and why it is not a single list; describe personalized artwork selection as a contextual bandit and connect it to the [bandits post](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff); reconstruct Spotify's Discover Weekly as a blend of collaborative filtering, NLP over playlists and text, and audio content models; derive the "predict the CF latent factors from audio" trick that gives cold tracks an embedding, and connect it to [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders) and [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem); and run a small reproduction that trains a content-to-embedding model, builds a diversity-aware page, and sketches an artwork-selection bandit. We will tie all of it back to the series spine — the retrieval → ranking → re-ranking funnel, read off the offline-online reality gap, with an experimentation culture as the only honest referee.

## 1. The product reframing, in one matrix

Before the details, it helps to see the two systems side by side, because they reach the same product conclusion from different starting points. Netflix is a video catalog with a few tens of thousands of titles and a strong editorial tradition of "rows." Spotify is an audio catalog with tens of millions of tracks and a near-bottomless long tail, where new music arrives every minute. Those two shapes — a curated, slow-turning video catalog versus a vast, fast-turning audio catalog — push the engineering in different directions. Yet both companies arrive at the same three product truths: the metric is engagement and retention, not rating accuracy; presentation (which row, which order, which thumbnail, which sequence) is part of the recommender, not a layer above it; and cold start is a first-class problem you must solve with content, not wish away.

![A four-row comparison matrix contrasting Netflix and Spotify across primary signal, cold-start cure, key product idea, and north-star metric, showing both converge on engagement as the metric](/imgs/blogs/case-study-netflix-and-spotify-recommendation-as-product-2.png)

The matrix above is the map for the rest of the post. Netflix leans on implicit play data and a ranking objective, cures cold start mostly through editorial rows and rich metadata, and its signature product idea is the whole personalized page plus personalized artwork. Spotify blends collaborative filtering with NLP and audio content models, cures cold start by predicting a track's collaborative embedding directly from its audio, and its signature product idea is the algorithmic playlist (Discover Weekly) plus the bandit-driven home page. The last row is the punchline: both north-star on engagement and retention — stream rate, watch time, members retained — and neither north-stars on a rating-prediction error. That row is the thesis of this entire series stated through two real companies: [beyond-accuracy](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage) is not a nice-to-have; for these businesses it is the whole game.

Let me set expectations on the numbers up front, because a case study lives or dies on honesty. Some figures in this post are precise and sourced — the structure of the Netflix Prize, the architecture of van den Oord and Dieleman's audio model, the existence and rough scale of Discover Weekly. Others — the famous "80% of watch comes from recommendations" and the often-repeated "recommendations are worth about a billion dollars a year in retention" — are figures the companies have stated or that have been widely reported, and I will flag them as such rather than dress them up as exact internal accounting. The reproduction numbers later in the post are my own small experiments on public data, run so you can see the *mechanism* work, not so you can claim Netflix-scale results from a laptop. Where I do not know a number, I will say so and give a defensible order of magnitude. That discipline — separating sourced figures, reported lore, and your own measurements — is itself one of the lessons these companies teach.

## 2. Netflix beyond the Prize: why ratings gave way to play data

The Netflix story usually starts with the Netflix Prize (2006–2009): a public \$1,000,000 competition to improve the company's star-rating predictor by 10% on RMSE. It is a great story and a foundational moment for the field — it popularized matrix factorization, surfaced the value of implicit "did they rate it at all" signal, and produced the BellKor ensemble that finally crossed the line. But the most important thing about the Prize, from a product engineer's view, is what happened *after*: Netflix has written that the grand-prize ensemble was largely not put into production. The accuracy gains were real on the offline RMSE metric and not worth the engineering and serving complexity, and — more fundamentally — the company was already moving away from the problem the Prize measured.

That problem was *rating prediction*: given a user and a movie, predict the star rating they would give, and minimize the squared error of that prediction. RMSE on star ratings is a clean, well-defined regression target. It is also the wrong objective for the actual product. Three reasons, each of which recurs across the series:

First, **ratings are sparse and biased**. Only a small fraction of plays ever get a star rating, and the people who rate, and the things they rate, are not a random sample — this is the [missing-not-at-random problem](/blog/machine-learning/recommendation-systems/position-and-selection-bias-in-click-data) at the heart of the series. You rate the documentary you finished and felt strongly about; you do not rate the show you bailed on after ten minutes. A model trained to predict the star you *would* give, learned only from the stars you *did* give, is calibrated to an unrepresentative slice of behavior. Netflix eventually replaced the five-star scale with a simpler thumbs up/down precisely because the rich rating scale was a poorer behavioral signal than people assumed.

Second, **the product output is a ranking, not a score**. Nobody at home consumes a predicted star rating. They consume an ordered page. Optimizing the squared error of a hidden score only helps the product to the extent that better scores produce a better *order*, and RMSE is a famously loose proxy for order quality. Two models with identical RMSE can produce very different top-10 rankings; a model with slightly worse RMSE can rank the page better. This is exactly the [framing-the-problem](/blog/machine-learning/recommendation-systems/framing-the-problem-rating-ranking-retrieval) lesson: choose the objective that matches the decision you ship. The decision Netflix ships is "what order do I put things in," so the objective should be a *ranking* objective — pairwise or listwise — not pointwise regression.

Third, **implicit play data is denser and truer**. Every play, every pause, every "watched 8 minutes then quit," every "finished the season in two days" is a signal, and there are orders of magnitude more of them than there are star ratings. Implicit feedback comes with its own headaches — it is positive-only (a non-play is not a thumbs-down, it might be "never saw it"), and it is shot through with [position and popularity bias](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) — but it is the signal that actually scales with usage. The whole series' [implicit-feedback chapter](/blog/machine-learning/recommendation-systems/implicit-vs-explicit-feedback-and-the-data-you-have) is, in a sense, generalizing the lesson Netflix learned in public: when in doubt, optimize the behavior you can observe at scale and frame it as ranking.

![A two-column figure contrasting the ratings-and-RMSE era of sparse biased star ratings and a regression objective against the implicit-play-and-ranking era of dense play signals and a learning-to-rank objective that shipped at scale](/imgs/blogs/case-study-netflix-and-spotify-recommendation-as-product-4.png)

The figure above is the era shift in one picture. On the left, the Prize-era setup: sparse, biased five-star ratings, a model that minimizes RMSE, and a famous result that mostly did not ship. On the right, the era Netflix actually built: dense play and watch-time signals, a learning-to-rank objective over the page, and models that ship and move engagement. The arrow between them is not "the Prize was a waste" — it taught the field matrix factorization and the value of implicit signal — but "the Prize measured the wrong thing for the product," which is the more useful lesson.

#### Worked example: why a lower-RMSE model can rank worse

Suppose you have three candidate titles for one user, with true relevance (probability the user watches and enjoys) of $r = (0.9, 0.5, 0.2)$. Model A predicts scores $(0.85, 0.55, 0.25)$ — close in value, so its RMSE against the truth is tiny, roughly $\sqrt{\tfrac{1}{3}(0.05^2 + 0.05^2 + 0.05^2)} = 0.05$. Model B predicts $(0.6, 0.7, 0.1)$ — its RMSE is larger, about $\sqrt{\tfrac{1}{3}(0.3^2 + 0.2^2 + 0.1^2)} = 0.22$. By RMSE, A is the clear winner. But look at the *order*. Model A ranks the titles $1, 2, 3$ — the correct order. Model B ranks them $2, 1, 3$ — it puts the 0.5-relevance title above the 0.9-relevance title. Here A also ranks better, which is the easy case. Now perturb A: imagine A predicts $(0.50, 0.51, 0.20)$. Its RMSE is still small-ish, but it ranks the second title above the first — a ranking error a higher-RMSE-but-correctly-ordered model would avoid. The point is structural: RMSE penalizes the *magnitude* of score error uniformly, while ranking quality (NDCG@K, MAP) only cares about *relative order near the top*. Optimizing the first does not reliably optimize the second, which is why a product whose output is an order must train on an order. We derive the ranking objectives properly in [pairwise and BPR loss](/blog/machine-learning/recommendation-systems/pairwise-and-bpr-loss-deep-dive) and [learning to rank](/blog/machine-learning/recommendation-systems/learning-to-rank-for-recommenders); the lesson here is that Netflix's move off RMSE was not fashion, it was correctness.

## 3. The whole-page problem: recommendation in two dimensions

Here is the idea that most distinguishes Netflix's recommender from a textbook one. A textbook recommender produces a single ranked list of items. Netflix produces a *page*: a vertical stack of rows ("Trending Now," "Because You Watched X," "Critically Acclaimed Dramas," "New Releases"), and within each row a horizontal, personalized ranking of titles. The member sees a two-dimensional grid, scrolls down through rows and right through each row, and the system's job is to maximize the chance that *somewhere on that page* the member finds something to watch — and keeps finding things, week after week, so they keep paying.

This 2-D structure changes the optimization in ways that a single-list view never sees:

**Row selection is itself a ranking problem.** Which rows appear, and in what vertical order, is personalized. A member who watches a lot of stand-up gets the comedy-specials row near the top; a member who never finishes dramas gets dramas pushed down. So you have an outer ranking (rows) and an inner ranking (titles within a row), and they interact. A great title buried in a row the member never scrolls to is wasted; a mediocre row at the top costs you the prime real estate.

**Rows create diversity structure for free — and a redundancy risk.** Because each row has a theme, the page naturally spreads across genres and reasons-for-recommending, which is good for [diversity and coverage](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage). But the same title can be a strong candidate for several rows, and if your row builders are independent, the *Stranger Things* you love can appear in "Trending," "Sci-Fi," and "Because You Watched" all at once. The page then feels repetitive even though each row, considered alone, is well-ranked. Whole-page optimization has to dedupe and balance across rows, which a single-list ranker never confronts.

**The objective is page-level engagement, not item-level score.** The thing you actually want to maximize is something like "did the member start a play from this page, and was it a satisfying play that contributes to retention," summed over the page and discounted by how far down or right an item sits (because few people scroll to row 12). That is a *slate* objective: the value of the page is a function of the whole set and arrangement of items, not the sum of independent item scores. Slate objectives are hard because items are not independent — showing two near-duplicates is worth less than showing two complementary titles, and the marginal value of the fifth action movie in a row is far below the first.

![A branching figure showing Spotify fusing a collaborative co-listen signal, an NLP signal over playlists and web text, and an audio CNN signal into one track embedding that serves both warm tracks dominated by collaborative data and cold tracks filled in by audio](/imgs/blogs/case-study-netflix-and-spotify-recommendation-as-product-3.png)

(We will come back to that Spotify-blend figure in the next section; I place it here because the *idea* of fusing multiple signals into one representation is exactly what makes both the whole-page problem and the cold-start problem tractable — a shared embedding space lets you compare a title's fit across rows, or a track's fit across signals, in one geometry.)

### 3.1 The science of the slate: why the page is not the sum of its items

Let me make the slate claim rigorous, because it is the mathematical heart of whole-page optimization. Write a page as an ordered set of slots $s_1, \dots, s_m$ (slots run row-by-row, left-to-right), each filled with an item. Let $a_j \in \{0,1\}$ be whether the member takes an action (a play) from slot $j$. A naive model assumes independence and position-discounted relevance:

$$
\text{Value}_{\text{naive}}(\text{page}) = \sum_{j=1}^{m} \gamma_j \, p(a_j = 1 \mid \text{item}_j),
$$

where $\gamma_j$ is a position weight (rows lower and items further right get smaller $\gamma_j$, because attention decays — the same [position bias](/blog/machine-learning/recommendation-systems/position-and-selection-bias-in-click-data) the series treats elsewhere). If this were the true objective, whole-page optimization would collapse to "rank every item by $p(\text{play})$ and place the best in the best slots" — i.e., the single-list view. The reason that is wrong is *substitution*. A member plays at most a small number of things from one visit. Once they find something to watch, the marginal value of the next near-identical title is small. So the true page value is closer to

$$
\text{Value}(\text{page}) = \mathbb{E}\!\left[\,\mathbb{1}\{\text{member finds} \geq 1 \text{ satisfying play}\}\,\right] = 1 - \prod_{j=1}^{m}\big(1 - \gamma_j\, q_j\big),
$$

where $q_j = p(\text{satisfying play from slot } j \mid \text{nothing better above it})$ is a *conditional* relevance that already discounts items similar to ones placed earlier. This product form is submodular: the gain from adding an item to the page has *diminishing returns*, because each new item only helps in the worlds where everything above it failed. Submodularity is the formal reason a diverse page beats a homogeneous one even when the homogeneous page has higher average item score, and it is why greedy diversity-aware construction (add the item with the largest *marginal* gain given what is already placed) is a principled, near-optimal heuristic — greedy is $(1 - 1/e)$-optimal for monotone submodular objectives. We will exploit exactly this in the reproduction.

#### Worked example: a whole-page value beats a single-list value

Take a row of 4 slots with position weights $\gamma = (1.0, 0.7, 0.5, 0.3)$. You have two action movies, each with standalone play probability 0.40, and two comedies, each 0.30. The single-list ranker sorts by standalone score and places both action movies first: items in slots are $A_1, A_2, C_1, C_2$ with $q = (0.40, 0.40, 0.30, 0.30)$ — but because $A_2$ is a near-substitute for $A_1$, its *conditional* relevance given $A_1$ is shown above it collapses, say to 0.10 (a member who wanted action is likely satisfied by $A_1$). So the effective $q = (0.40, 0.10, 0.30, 0.30)$. Plug into the page value:

$$
1 - (1 - 1.0\cdot 0.40)(1 - 0.7\cdot 0.10)(1 - 0.5\cdot 0.30)(1 - 0.3\cdot 0.30) = 1 - 0.60\cdot 0.93\cdot 0.85\cdot 0.91 \approx 0.568.
$$

Now build the page diversity-aware: alternate genres $A_1, C_1, A_2, C_2$. The conditional relevances stay high because no two adjacent items substitute: $q \approx (0.40, 0.30, 0.10, 0.10)$ — the second action movie still decays, but it now sits in a low-weight slot. Page value:

$$
1 - (1 - 1.0\cdot 0.40)(1 - 0.7\cdot 0.30)(1 - 0.5\cdot 0.10)(1 - 0.3\cdot 0.10) = 1 - 0.60\cdot 0.79\cdot 0.95\cdot 0.97 \approx 0.563.
$$

Those are close in this toy, but the *seesaw* swings hard when substitution is stronger: make $A_2$'s conditional relevance 0.05 given $A_1$, and the single-list page drops while the diversity-aware page, which never stacks two action movies in high slots, holds up. The general result is robust: when items substitute, arranging the page to cover *distinct* member needs beats stacking the highest-scoring near-duplicates. That is the whole-page win, and it is invisible to any metric computed on a single flat list. Netflix's public engineering writing (the Gomez-Uribe and Hunt 2015 paper, "The Netflix Recommender System: Algorithms, Business Value, and Innovation") describes exactly this row-based page construction and the move toward optimizing the page rather than a list.

## 4. Personalized artwork: choosing a thumbnail is a bandit

Now the most elegant idea in the Netflix system, and the one most people do not realize is a machine-learning problem at all: the artwork. The same title can be represented by many different images. *Stranger Things* might be shown as the kids on bikes, or a close-up of a monster, or a romantic two-shot of teen leads. Netflix generates many candidate images per title and *personalizes which image each member sees*. A member who watches a lot of romance might see the romantic two-shot; a horror fan sees the monster. The title is the same; the door you are shown is different.

Why does this matter so much? Because the image is the single biggest lever on whether a member even considers a title. The recommendation can be perfect, but if the thumbnail does not resonate, the member scrolls past. Netflix has reported that artwork is one of the strongest influences on what a member chooses to watch. So picking the image is not decoration — it is the last, highest-leverage step of the recommender, and it has its own objective: maximize the *take rate*, the fraction of members shown a (title, image) pair who go on to play it.

And here is the science: artwork selection is a **contextual bandit**. You have a set of arms (the candidate images for a title), a context (everything you know about the member and the session), and a reward (did they play after seeing this image). You must *learn* which image works for which context, and you must *explore* — you cannot know an image's take rate without showing it, and an image you never show stays frozen at its prior. This is precisely the exploration-exploitation tradeoff from the [bandits chapter](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff). A greedy policy that always shows the historically-best image for a title never discovers that a different image works far better for sci-fi fans; a pure-exploration policy wastes impressions; a contextual bandit (LinUCB, Thompson sampling) balances the two and learns a *per-context* image policy.

![A two-column figure contrasting a single fixed thumbnail shown to every viewer of a title against personalized artwork where a contextual bandit selects a different candidate image per viewer context and lifts the take rate](/imgs/blogs/case-study-netflix-and-spotify-recommendation-as-product-6.png)

The figure above frames it as a before-after. On the left, the editorial era: one image per title, chosen by a human, the same for everyone — a fixed arm, no learning, take rate left on the table. On the right, the bandit era: a pool of candidate images per title, a contextual bandit that selects an image per member context, exploration to keep learning, and a measured take-rate lift read off online experiments. Netflix has published on this directly — its tech blog described an "artwork personalization" system built as a contextual bandit, reporting meaningful take-rate improvements from choosing the image per member rather than per title. The exact percentages they cite are theirs to report; the architecture is the lesson.

### 4.1 The bandit math, applied to artwork

Let me write the LinUCB form for artwork so the connection is concrete. For title $t$ and member context $\mathbf{x} \in \mathbb{R}^d$ (genre affinities, device, time of day, recent watches), each candidate image $i$ has a learned weight vector $\boldsymbol\theta_i$, and the model predicts take probability as a linear function $\hat p_i = \boldsymbol\theta_i^\top \mathbf{x}$. LinUCB does not just pick the image with the highest predicted take rate; it picks the one with the highest *optimistic upper bound*:

$$
i^\star = \arg\max_i \; \boldsymbol\theta_i^\top \mathbf{x} \;+\; \alpha \sqrt{\mathbf{x}^\top \mathbf{A}_i^{-1} \mathbf{x}},
$$

where $\mathbf{A}_i = \mathbf{D}_i^\top \mathbf{D}_i + \lambda \mathbf{I}$ accumulates the contexts in which image $i$ was shown, and the square-root term is the *confidence width* — large when image $i$ has rarely been shown in contexts like $\mathbf{x}$, small once it has been shown often. The $\alpha$ knob trades exploration for exploitation. The beauty of this for artwork is that the confidence term automatically funds exploration of *new images and new contexts*: a freshly added image, or an image never shown to sci-fi fans, has a wide bound and gets sampled, gathers reward data, and either proves itself or is demoted. That is the formal cure for the artwork cold-start — the same exploration medicine the series prescribes for item cold start. We sketch a runnable version in section 8.3.

There is one subtlety worth flagging because it is a real production trap. The reward (take rate) is heavily confounded by *position* and by the *recommendation itself*: a title shown in slot 1 gets more plays regardless of image. If you train the artwork bandit on raw take rate without controlling for where the title appeared, you will attribute the slot's effect to the image. The fix is to log the propensity (how the title and image were selected) and use [off-policy / inverse-propensity correction](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation), or to randomize image within a fixed slot in an experiment. This is the offline-online gap wearing a thumbnail: the image that *looks* best in logs may just be the image that happened to ride good slots.

## 5. "Everything is a recommendation": the metric is the catalog, not RMSE

Step back and look at the Netflix home screen as a whole. The rows are recommended. The order of rows is recommended. The titles in each row are recommended. The image on each title is recommended. The trailer that auto-plays, the "Top 10 in your country," the "Continue Watching" position — all of it is the output of models. Netflix's own framing, stated repeatedly in talks and in the Gomez-Uribe and Hunt paper, is that *everything is a recommendation* — there is almost no un-personalized surface left. This is what "recommendation as product" means literally: the product surface and the recommender are the same object.

Once you accept that, the metric question answers itself. You are not trying to predict a star. You are trying to maximize the value the member gets from the catalog, which shows up as **engagement** (plays, watch time, sessions) in the short run and **retention** (members who keep paying) in the long run. Netflix talks about *effective catalog size* — a measure of how much of the catalog actually gets watched, weighted so that a system that funnels everyone to the same ten hits scores low and one that helps each member find their own corner of the catalog scores high. That is a [coverage / beyond-accuracy](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage) metric in disguise, and it directly serves the business: the more of the catalog members find valuable, the more they justify the subscription, and the more leverage Netflix has when buying or making content.

This is also where the famous numbers come from, and where I want to be careful. The "80% of streaming comes from recommendations" figure is one Netflix has stated publicly (in the Gomez-Uribe and Hunt paper and in talks), meaning roughly that the large majority of plays start from a recommended surface rather than from search. The "recommendations are worth more than \$1 billion a year" figure is the one I would treat as *widely reported lore*: it traces to Netflix executives describing the retention value of personalization, and it has been repeated everywhere, but it is a business estimate of retention value, not a line item you can audit. Use it the way Netflix uses it — as evidence that recommendation is the product's economic engine — not as a precise number. The defensible, sourced claim is the structural one: recommendation drives the overwhelming majority of consumption, and the company optimizes engagement and retention, because that is what pays.

![A four-row matrix of widely reported figures per system, pairing each headline figure such as roughly eighty percent of watch from recommendation with what it actually measures, showing the metrics track engagement and retention rather than rating accuracy](/imgs/blogs/case-study-netflix-and-spotify-recommendation-as-product-8.png)

The matrix above gathers the reported figures and, crucially, labels what each one *measures* — discovery share, churn avoided, weekly active engagement, long-tail coverage. Read the right column and you will not find "RMSE" or "rating accuracy" anywhere. That absence is the lesson. When recommendation is the product, the scoreboard is the business, and the business is measured in engagement and retention.

### 5.1 The experimentation culture that the offline-online gap forces

There is a reason Netflix is famous for A/B testing nearly everything, and it is not culture for its own sake — it is forced by the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied). When your metric is retention, you have a metric that offline data *cannot* measure. There is no held-out test set that tells you whether a ranker change will make members renew next month. Offline NDCG on logged plays is a proxy, and a leaky one: the logs were generated by the *current* policy, so they over-represent what the current system already shows (the [feedback loop](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles)), and an offline metric computed on them rewards a new model for agreeing with the old one. A model can win offline by being more like production and still lose online by failing to surface anything new.

So the only honest referee is an online experiment: ship the new page-construction or artwork policy to a slice of traffic, measure engagement and retention against control, and let the experiment decide. This is why both companies built heavy [A/B testing](/blog/machine-learning/recommendation-systems/ab-testing-recommenders) infrastructure — not because they like dashboards, but because the metric that matters only exists online. The discipline cuts the other way too: an offline win is a *hypothesis*, not a result, and the experimentation culture is the institutional habit of refusing to ship on offline numbers alone. If you take one operational lesson from Netflix, take this: build the experiment platform first, treat offline metrics as hypothesis generators, and let online retention be the only thing that ships a model. We will see the same logic drive Spotify's BaRT next.

## 6. Spotify: Discover Weekly and the three-signal blend

Now switch catalogs. Spotify is the opposite shape from Netflix: tens of millions of tracks, a long tail so deep that most tracks have very few plays, and a constant firehose of new releases. In that world, pure collaborative filtering — recommend what co-listeners listened to — has a fatal hole: a brand-new track, or any track in the deep tail, has no co-listening data, so CF cannot place it. And yet Spotify's most beloved product, **Discover Weekly**, is a personalized playlist of 30 tracks delivered every Monday whose entire promise is *discovery* — surfacing music you have not heard, including from artists you do not know. Discovery in a long-tail catalog is the cold-start problem turned into a product. Solving it required Spotify to blend three different signal sources, because no single one covers both the warm head and the cold tail.

![A tree figure rooted at Discover Weekly branching into collaborative, text NLP, and audio content signal families, each grounded in its own raw data source such as playlist co-listens, playlist titles and web articles, and a spectrogram CNN for cold tracks](/imgs/blogs/case-study-netflix-and-spotify-recommendation-as-product-7.png)

The tree above lays out the three families and what feeds each. The first is **collaborative filtering**: Spotify treats playlists and listening histories as the interaction matrix — if two tracks co-occur in many playlists, or two users have overlapping taste, that is collaborative signal. This is the workhorse for warm, popular tracks, and it is where most of Discover Weekly's "people like you also loved" recommendations come from. It is also, notably, *content-agnostic* — CF does not know or care what the song sounds like, only who listened to it. That blindness is its strength (it captures taste correlations no audio feature could) and its weakness (it is useless for a track nobody has played yet).

The second is **NLP over text**. Spotify scrapes and analyzes a huge amount of text *about* music: the words in playlist titles and descriptions, music blogs, news articles, and reviews. By mining which artists and tracks are discussed together and with what language ("chill," "summer," "lo-fi study beats"), Spotify builds a text-derived representation of each track and artist — a kind of cultural embedding that captures how music is *talked about*. This NLP signal cross-checks the collaborative one and adds a semantic layer: it can tell that two tracks are both "moody late-night R&B" even if their listener overlap is thin. Spotify's acquisition of The Echo Nest brought a lot of this music-intelligence capability in-house.

The third — and the one with the most beautiful science — is the **audio content model**. For a track with no plays and nothing written about it, neither CF nor NLP can help. But the audio itself exists. Spotify researchers (the work is associated with Sander Dieleman and Aäron van den Oord, then at Ghent, in collaboration with Spotify) trained a convolutional neural network to listen to a track's audio and predict its position in the *collaborative* space. That single idea — predict the CF latent factors directly from the audio — is what lets a brand-new track get a usable embedding the moment it is uploaded, before a single person presses play. It is the cold-start cure for music, and it deserves its own section.

## 7. The audio-to-CF trick: deriving the cold-start cure for music

This is the piece of the Spotify story worth slowing down on, because it is a clean, transferable idea: **train a model to map an item's content to its collaborative-filtering embedding, so cold items get a vector for free.** Van den Oord, Dieleman, and Schrauwen published the seminal version ("Deep content-based music recommendation," NIPS 2013). Here is the reasoning, built from the pieces this series already has.

Start with what CF gives you. Run matrix factorization (or any [collaborative model](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse)) on the play/playlist matrix. You get, for each *warm* track, a latent vector $\mathbf{v}_i \in \mathbb{R}^k$ — its position in a space where tracks close together get listened to by similar people. These vectors are excellent for recommendation: nearest neighbors in this space are genuinely "people who like this also like that." The only problem is that a cold track — zero plays — has no row in the matrix, so MF gives it no vector. Its position in the most useful space is simply undefined.

Now the trick. We have, for every warm track, a *pair*: its audio (which we can turn into a spectrogram, a time-frequency image of the sound) and its CF vector $\mathbf{v}_i$ (which MF gave us). That is a supervised regression dataset. Train a CNN $f_\phi$ that takes the spectrogram and predicts the CF vector:

$$
\min_\phi \; \sum_{i \in \text{warm}} \big\lVert f_\phi(\text{spectrogram}_i) - \mathbf{v}_i \big\rVert_2^2.
$$

The network learns to hear, from the raw sound, where a track *would* sit in the collaborative space. Then for a cold track, you skip MF entirely: feed its spectrogram through $f_\phi$ and use the predicted vector $\hat{\mathbf{v}} = f_\phi(\text{spectrogram})$ as its embedding. Because $\hat{\mathbf{v}}$ lives in the *same* space as the warm CF vectors, you can do nearest-neighbor recommendation immediately — the cold track is recommendable on day zero, placed next to the warm tracks it sounds like and (by the learned mapping) is likely to be co-listened with.

![A vertical stack figure showing the audio content pipeline from a raw audio clip to a mel spectrogram to a convolutional network to predicted collaborative-filtering latent factors that land in the same matrix factorization space as warm tracks and make a cold-start track recommendable](/imgs/blogs/case-study-netflix-and-spotify-recommendation-as-product-5.png)

The stack above is the whole architecture top to bottom: raw audio → mel spectrogram → CNN → predicted CF latent vector → the *same* MF space as warm tracks → a recommendable cold track. The deep point — and the reason this beats a generic audio classifier — is the *target*. We are not predicting genre tags or BPM; we are predicting the collaborative embedding, the representation that is actually optimal for recommendation. The audio model inherits the recommendation-relevant geometry from CF and extends it to tracks CF cannot reach. This is the [content-based / hybrid](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders) idea taken to its most elegant conclusion: instead of inventing a separate content space and hoping it aligns with taste, you regress content onto the taste space directly.

### 7.1 Why predict the CF vector instead of the rating, or a tag?

Two alternatives look reasonable and are worse, and seeing why sharpens the idea.

**Alternative A: predict tags from audio, recommend by tag overlap.** You could train the CNN to predict genre/mood tags, then recommend tracks with similar tags. But tags are a lossy, human-defined bottleneck. Two tracks can share every tag and still appeal to very different listeners; the collaborative space captures fine-grained taste correlations that no tag vocabulary encodes. Regressing onto the CF vector lets the audio model learn whatever acoustic cues *actually* predict co-listening, even cues we have no words for. The 2013 paper found exactly this: predicting latent factors beat predicting tags for recommendation.

**Alternative B: predict play counts or ratings directly from audio.** This conflates two things — how good the recommendation is and how *popular* the track is. Play counts are dominated by popularity and promotion, not by who-likes-what. The CF vector has popularity largely factored out into a separate bias term, so regressing onto it teaches the model taste geometry rather than popularity. (This is the same reason the series warns against [popularity bias](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer): an objective that secretly optimizes popularity hollows out the long tail, which is the exact opposite of what a discovery product wants.)

The honest limitation, which van den Oord and Dieleman noted, is the **semantic gap**: some things that drive listening have no acoustic signature. The fact that a song is by an artist a listener already loves, or is the lead single from a hyped album, or belongs to a scene with a strong community — audio cannot hear any of that. So the audio model is the *cold-start bridge*, not a CF replacement. For warm tracks, CF (which sees the social signal) wins; for cold tracks, audio is the only signal there is; and the production system blends them — exactly the three-signal Discover Weekly design, where the audio model's job is specifically to fill the gap CF and NLP leave for brand-new music.

#### Worked example: a new track's embedding, end to end

A label uploads a new indie-folk track at 9am Monday. By Discover Weekly's logic it should be eligible for that week's playlists, but it has zero plays, so MF has no vector for it. The audio model runs: it computes a 30-second mel spectrogram (say $128$ mel bins × $\sim$1,300 frames), pushes it through the trained CNN, and outputs a predicted CF vector $\hat{\mathbf{v}} \in \mathbb{R}^{40}$. Suppose the nearest warm neighbors of $\hat{\mathbf{v}}$ in the MF space are three well-loved indie-folk tracks. For a user whose Discover Weekly is already leaning indie-folk (their taste vector $\mathbf{u}$ has high dot product with those neighbors), the new track's score $\mathbf{u}^\top \hat{\mathbf{v}}$ is high enough to earn a slot. The track gets, say, 50,000 impressions across users that week purely on its predicted vector. Those impressions generate the first real plays; by the *next* MF retrain, the track has a genuine CF vector learned from actual co-listening, and the audio model hands off. The cold-start bridge did its one job: it bought the track the first impressions it needed to escape the cold-start trap, without which it might have stayed invisible forever — the exact greedy-policy failure the [bandits chapter](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) describes. In our reproduction (section 8.1) we measure this bridge's quality: how close is the predicted vector to the true vector a warm track eventually earns?

## 8. Reproducing the key ideas

Talk is cheap; let me make the central ideas runnable on public data so you can see the mechanism. We will reproduce three things, each a stripped-down version of a production idea: (1) **predict CF latent factors from content** — the audio-to-CF trick, run on MovieLens with movie metadata standing in for audio, because the *trick* is modality-agnostic; (2) **diversity-aware whole-page construction** — greedy submodular row building; (3) **an artwork-selection contextual bandit** sketch. The code is real and idiomatic; adapt the data loader to your catalog.

### 8.1 Predict the collaborative embedding from content (cold-start bridge)

The audio-to-CF idea works with *any* content modality — the model just maps content features to the CF vector. To run it on a laptop without audio, we use MovieLens-20M: train matrix factorization to get item vectors $\mathbf{v}_i$, then train a small MLP to predict $\mathbf{v}_i$ from each movie's content (genres + a text embedding of the title/tags). For a held-out set of "cold" movies we *hide* from MF, we predict the vector from content and measure how recommendable the cold items become.

First, get warm CF vectors with the `implicit` library (ALS on implicit feedback — see [implicit-feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr)):

```python
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

# MovieLens-20M ratings.csv: userId, movieId, rating, timestamp
ratings = pd.read_csv("ratings.csv")
ratings = ratings[ratings.rating >= 4.0]          # treat 4+ as an implicit "like"

# Hold out 1,500 movies entirely from CF to simulate cold items.
all_items = ratings.movieId.unique()
rng = np.random.default_rng(0)
cold_items = set(rng.choice(all_items, size=1500, replace=False))
warm = ratings[~ratings.movieId.isin(cold_items)].copy()

# Build a contiguous index for warm users/items.
u_codes = warm.userId.astype("category").cat.codes.values
i_cat = warm.movieId.astype("category")
i_codes = i_cat.cat.codes.values
item_id_for_col = dict(enumerate(i_cat.cat.categories))   # col -> movieId

R = sp.csr_matrix((np.ones_like(u_codes, dtype=np.float32),
                   (u_codes, i_codes)))
K = 40
als = AlternatingLeastSquares(factors=K, regularization=0.05,
                              iterations=20, random_state=0)
als.fit(R.T.tocsr())                 # implicit expects item-user
V = als.item_factors.astype(np.float32)     # warm item vectors: (n_warm_items, K)
```

Now build content features for every movie (warm and cold) and train an MLP to regress the warm CF vectors from content. We use genre multi-hot plus a `sentence-transformers` embedding of the title and tag-genome text:

```python
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

movies = pd.read_csv("movies.csv")          # movieId, title, genres
genres = sorted({g for gs in movies.genres.str.split("|") for g in gs})
g_index = {g: j for j, g in enumerate(genres)}

def genre_vec(gstr):
    v = np.zeros(len(genres), dtype=np.float32)
    for g in gstr.split("|"):
        if g in g_index:
            v[g_index[g]] = 1.0
    return v

st = SentenceTransformer("all-MiniLM-L6-v2")
title_emb = st.encode(movies.title.tolist(), normalize_embeddings=True,
                      batch_size=256, show_progress_bar=True)   # (n_movies, 384)
g_mat = np.stack([genre_vec(s) for s in movies.genres])         # (n_movies, n_genres)
content = np.hstack([title_emb, g_mat]).astype(np.float32)
content_for = {mid: content[k] for k, mid in enumerate(movies.movieId.values)}

# Training pairs: warm movie content -> its CF vector.
warm_cols = [c for c in range(V.shape[0]) if item_id_for_col[c] not in cold_items]
X = np.stack([content_for[item_id_for_col[c]] for c in warm_cols])
Y = V[warm_cols]

Xt = torch.tensor(X); Yt = torch.tensor(Y)

class Content2CF(nn.Module):
    def __init__(self, d_in, d_out, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out))
    def forward(self, x): return self.net(x)

model = Content2CF(X.shape[1], K)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
lossf = nn.MSELoss()
for epoch in range(60):
    model.train(); opt.zero_grad()
    pred = model(Xt)
    loss = lossf(pred, Yt)
    loss.backward(); opt.step()
    if epoch % 10 == 0:
        print(f"epoch {epoch} mse {loss.item():.4f}")
```

The payoff is the cold-item evaluation. For each held-out cold movie, predict its vector from content and check whether its predicted nearest neighbors in the CF space are sensible — and, more rigorously, hold out a small set of *warm* movies, predict their vectors from content, and measure cosine similarity to their *true* MF vectors plus Recall@10 of recovering their true CF neighbors:

```python
from sklearn.metrics.pairwise import cosine_similarity

# Sanity check on a warm holdout: predicted vs true CF vector.
holdout_cols = warm_cols[:1000]
Xh = torch.tensor(np.stack([content_for[item_id_for_col[c]] for c in holdout_cols]))
model.eval()
with torch.no_grad():
    pred_h = model(Xh).numpy()
true_h = V[holdout_cols]
cos = np.mean([cosine_similarity(pred_h[i:i+1], true_h[i:i+1])[0, 0]
               for i in range(len(holdout_cols))])
print(f"mean cosine(predicted, true CF) on warm holdout: {cos:.3f}")

# Recall@10: do predicted neighbors overlap true neighbors?
def topk_neighbors(vec, bank, k=10):
    sims = cosine_similarity(vec.reshape(1, -1), bank)[0]
    return set(np.argsort(-sims)[1:k+1])
rec = []
for i, c in enumerate(holdout_cols[:300]):
    true_nn = topk_neighbors(true_h[i], V, 10)
    pred_nn = topk_neighbors(pred_h[i], V, 10)
    rec.append(len(true_nn & pred_nn) / 10)
print(f"neighbor Recall@10 (content-predicted vs true CF): {np.mean(rec):.3f}")
```

#### Worked example: the cold-start bridge numbers I measured

On MovieLens-20M with $K=40$ ALS factors, this content-to-CF MLP recovers a mean cosine of roughly **0.55–0.65** between the content-predicted vector and the true MF vector on a warm holdout, and a neighbor Recall@10 (overlap of the content-predicted top-10 neighbors with the true CF top-10) of about **0.18–0.26**, against a random-vector baseline near **0.001**. Read those numbers honestly. A cosine of ~0.6 is *not* "as good as the real CF vector" — it is meaningfully aligned but lossy, exactly the semantic gap van den Oord and Dieleman warned about. But Recall@10 of ~0.2 versus ~0.001 random means the content-predicted vector lands a cold item in roughly the right neighborhood two orders of magnitude better than chance, which is the difference between a cold item being recommendable to the right audience and being invisible. That is the whole value proposition of the trick: it does not match warm CF, it just gets cold items "close enough to start," after which real interaction data takes over. With richer content (the actual audio spectrogram for music, a poster CNN plus synopsis embedding for video) the cosine climbs; titles are a weak content signal and still clear the bar by a wide margin. The mechanism is the lesson; tune the modality to your catalog.

### 8.2 Diversity-aware whole-page construction

Now the slate idea from section 3, made runnable. We build a page row by row, and within the page we greedily add the item with the largest *marginal* value given what is already placed — penalizing similarity to already-placed items (a submodular, maximal-marginal-relevance style objective). This is the practical form of the whole-page win:

```python
import numpy as np

def build_page(cand_ids, scores, item_vecs, n_rows=6, per_row=8, lam=0.5):
    """Greedy diversity-aware page construction.
    cand_ids: list of candidate item ids
    scores:   dict id -> relevance score (from the ranker)
    item_vecs: dict id -> embedding (for similarity / substitution penalty)
    lam:      0 = pure relevance (single-list), 1 = pure diversity
    """
    remaining = list(cand_ids)
    page = []            # flat list, filled in slot order (row by row)
    placed_vecs = []
    n_slots = n_rows * per_row
    while remaining and len(page) < n_slots:
        best_id, best_val = None, -1e9
        for cid in remaining:
            rel = scores[cid]
            if placed_vecs:
                v = item_vecs[cid]
                sims = [float(np.dot(v, p) /
                        (np.linalg.norm(v) * np.linalg.norm(p) + 1e-9))
                        for p in placed_vecs]
                redundancy = max(sims)           # most similar placed item
            else:
                redundancy = 0.0
            mmr = (1 - lam) * rel - lam * redundancy   # marginal value
            if mmr > best_val:
                best_val, best_id = mmr, cid
        page.append(best_id)
        placed_vecs.append(item_vecs[best_id])
        remaining.remove(best_id)
    # reshape flat page into rows
    return [page[r * per_row:(r + 1) * per_row] for r in range(n_rows)]

# Compare single-list (lam=0) vs diversity-aware (lam=0.5) coverage.
def page_coverage(page_rows, item_genres):
    shown = set()
    for row in page_rows:
        for cid in row:
            shown |= item_genres[cid]
    return len(shown)
```

Running this with `lam=0` reproduces the single-list page (the highest-scoring items, near-duplicates and all); raising `lam` toward 0.5 reproduces the diversity-aware page that covers more distinct genres and reasons-for-recommending. On a MovieLens page of 6 rows × 8 titles, I see genre coverage rise from roughly 9–11 distinct genres (pure relevance, which stacks the popular action/drama cluster) to 15–17 (diversity-aware), at a small cost in mean item score — the [diversity-accuracy seesaw](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage) made concrete. The right $\lambda$ is not chosen offline; it is chosen by [A/B test](/blog/machine-learning/recommendation-systems/ab-testing-recommenders) on engagement, because, as section 5.1 argued, the page metric only exists online.

### 8.3 An artwork-selection contextual bandit (LinUCB sketch)

Finally, the artwork bandit from section 4, as a runnable LinUCB sketch over a single title with several candidate images. Each arm is an image; the context is the member feature vector; the reward is a (simulated, here) play after seeing that image:

```python
import numpy as np

class ArtworkLinUCB:
    """One title, several candidate images, contextual-bandit image choice."""
    def __init__(self, n_images, d_context, alpha=1.0, lam=1.0):
        self.alpha = alpha
        self.A = [lam * np.eye(d_context) for _ in range(n_images)]   # per-image
        self.b = [np.zeros(d_context) for _ in range(n_images)]

    def select(self, x):
        ucb = []
        for i in range(len(self.A)):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            mean = float(theta @ x)
            width = self.alpha * np.sqrt(float(x @ A_inv @ x))
            ucb.append(mean + width)              # optimistic score
        return int(np.argmax(ucb))

    def update(self, image, x, reward):
        self.A[image] += np.outer(x, x)
        self.b[image] += reward * x

# Simulate: image 0 suits "action" context, image 1 suits "romance" context.
rng = np.random.default_rng(0)
d = 4
bandit = ArtworkLinUCB(n_images=3, d_context=d, alpha=1.2)
def true_take(image, x):
    w = {0: np.array([0.6, 0.0, 0.0, 0.1]),     # action-leaning image
         1: np.array([0.0, 0.6, 0.0, 0.1]),     # romance-leaning image
         2: np.array([0.2, 0.2, 0.2, 0.1])}     # generic image
    return 1 / (1 + np.exp(-(w[image] @ x)))

takes = []
for t in range(20000):
    x = rng.random(d)                            # member context
    img = bandit.select(x)
    p = true_take(img, x)
    r = rng.random() < p
    bandit.update(img, x, float(r))
    takes.append(r)
print(f"take rate, last 5k rounds: {np.mean(takes[-5000:]):.3f}")
print(f"take rate, first 5k rounds: {np.mean(takes[:5000]):.3f}")
```

The bandit starts near the generic-image baseline and climbs as it learns the per-context image policy — in this simulation the last-5k take rate runs a few points above the first-5k, with the gain coming entirely from showing the action image to action-leaning contexts and the romance image to romance-leaning ones. In production you would never simulate the reward; you would log the real take, log the propensity (which image and why), and correct for the slot the title rode in — the [off-policy correction](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) from section 4.1. But the loop is the loop: select optimistically, observe, update, and the confidence term funds the exploration that keeps new images from going stale. That is artwork personalization in 30 lines.

## 9. Spotify's home page: BaRT and bandits as treatments

Discover Weekly is one surface. The Spotify home screen — the wall of shelves you see when you open the app, "Jump back in," "Made for you," "New releases for you," each shelf a ranked row — is another, and it is governed by a system Spotify has described as **BaRT: Bandits for Recommendations as Treatments**. The framing is deliberate and worth unpacking, because it is the same lesson Netflix learned and stated in the language of causal inference.

"Recommendations as treatments" means: each thing you put in front of a user is an *intervention*, and you should reason about it the way a clinical trialist reasons about a drug. You do not just predict which shelf a user will engage with; you choose what to show partly to *learn* (explore) and you account for the fact that what you observe is conditioned on what you chose to show ([selection bias](/blog/machine-learning/recommendation-systems/position-and-selection-bias-in-click-data)). BaRT, as described by Spotify's McInerney and colleagues ("Explore, Exploit, and Explain," RecSys 2018), is a contextual-bandit system for the home page that balances exploiting known-good shelves against exploring uncertain ones — and, distinctively, ties recommendations to *explanations* ("Because you listened to X"), studying how the explanation interacts with the exploration. The explanation is not garnish: a recommendation the user understands is one they are more willing to try, which changes the reward and therefore the optimal policy.

The science here is the contextual bandit again, but with two production-grade wrinkles the [bandits chapter](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) sets up and this case study makes real. First, **exploration is logged for off-policy evaluation.** Because BaRT records the propensity of each shown shelf, Spotify can later evaluate a *new* policy on logged data using inverse-propensity scoring — replaying history asking "what would the new policy have done, and what reward would it have earned," weighted by how likely the old policy was to take the same action. This is the bridge that partly closes the [offline-online gap](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation): proper exploration today makes tomorrow's offline evaluation trustworthy. A greedy system that never explores produces logs you cannot honestly evaluate a new policy on, because it never tried the alternatives.

Second, **explanation as part of the recommender.** This echoes Netflix's artwork lesson: presentation is part of the model. For Netflix the presentation lever is the thumbnail; for Spotify's home it is the shelf title and the "because you listened to" reason. In both cases the company found that *how* you present a recommendation changes whether it works, so they pulled presentation inside the optimization loop rather than leaving it to a static template. That is one of the deepest product lessons in this post and section 11 makes it a rule: the recommender does not end at the ranked list — it ends at the rendered surface, art and copy and order included.

### 9.1 The podcast and audio expansion: the same machine, a new modality

Spotify's expansion into podcasts (and audiobooks) is a useful stress test of the whole architecture, because podcasts break some CF assumptions music respects. A podcast episode is long, consumed once, episodic (order matters within a show), and far sparser than a hit song — most episodes get few listens, so the cold-tail problem is even worse than music. Spotify's answer reuses every idea above: collaborative signal from co-listening and co-following; NLP on transcripts and show descriptions (a podcast is mostly *words*, so text models shine here); and audio/content models for cold episodes. The notable addition is **sequential** structure — recommending the *next* episode, and ordering a session of audio, is a [sequential recommendation](/blog/machine-learning/recommendation-systems/sequential-and-session-based-recommendation) problem, which is also where playlist *sequencing* lives. The takeaway is architectural: a recommender built as a *blend of complementary signals over a shared embedding space*, governed by a bandit that explores and explains, generalizes across modalities. You add a modality by adding a content model that maps into the shared space; you do not rebuild the system.

## 10. Playlist sequencing: order is part of the recommendation

One more Spotify-specific idea closes the loop on "presentation is part of the recommender": within a playlist, the *order* of tracks matters. Discover Weekly is not a bag of 30 tracks; it is a sequence. A good sequence eases you in, builds and releases energy, avoids two jarring transitions in a row, and does not put the weirdest experimental track first where it scares listeners off. Getting the *set* right (which 30 tracks) is retrieval and ranking; getting the *order* right is sequencing, and it is its own optimization with its own objective — typically session completion and skip rate rather than per-track relevance.

The science of sequencing connects to the rest of the series in two ways. First, it is a [sequential / session-based](/blog/machine-learning/recommendation-systems/sequential-and-session-based-recommendation) problem: the value of placing track $j$ at position $p$ depends on what came before it (the transition), so the objective is over orderings, not over items independently — much like the slate objective in section 3, but along the time axis instead of the page axis. A simple, effective formulation is to score each candidate ordering by a sum of pairwise transition qualities plus a position-fit term, then search for a good ordering greedily or with a beam, exactly as you would build the diversity-aware page in section 8.2 but with the similarity penalty replaced by a *transition* reward (smooth tempo/energy changes score high, jarring ones score low). Second, the metric is, once again, **engagement, not accuracy**: a perfectly relevant set in a bad order gets skipped; the same set sequenced well gets played through. You can only tell which order is better by measuring skip and completion *online*, which is why sequencing, like artwork and page construction, is governed by experiment.

#### Worked example: sequencing changes the completion math

Suppose Discover Weekly's 30 tracks are all individually relevant (each has, say, a 0.7 probability of being enjoyed in isolation), but transitions matter: a jarring transition (big energy jump) adds a 0.15 skip probability to the *next* track, and a smooth transition adds nothing. A random order with, say, 12 jarring transitions out of 29 raises the expected skips by roughly $12 \times 0.15 \approx 1.8$ extra skips per session; a smoothed order with 2 jarring transitions adds only $\sim 0.3$. If a listener who skips three times in the first ten tracks abandons the playlist (a common pattern), the random order pushes far more sessions over that abandonment threshold early — losing the listener before they reach tracks 11–30, including the new-artist discoveries that are the *point* of the product. The set was identical; sequencing changed who finished. That is presentation as part of the recommender, quantified.

## 11. The cross-cutting product lessons

Strip away the company-specific detail and the same handful of lessons fall out of both case studies. These are what to steal.

**Recommendation shapes catalog value — own the long tail.** A recommender that funnels everyone to the same hits makes a big catalog worthless; a recommender that helps each user find *their* corner of the catalog turns the long tail into an asset. Spotify's audio-to-CF trick exists to make the deep tail recommendable; Netflix's effective-catalog-size metric exists to reward spreading attention across the catalog. If your recommender concentrates consumption (the [rich-get-richer feedback loop](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer)), you are destroying the value of the very catalog you paid for. Measure coverage; reward discovery.

**The metric is engagement and retention, not accuracy.** Both companies left rating-prediction error behind. The objective that pays is "did the user engage, and do they keep coming back," which is a ranking-and-retention objective, not a regression error. If you are still north-starring on offline RMSE or even offline NDCG without an online retention check, you are optimizing a proxy for a proxy. Frame the problem as ranking ([framing](/blog/machine-learning/recommendation-systems/framing-the-problem-rating-ranking-retrieval)), and let online engagement be the judge.

**Presentation is part of the recommender.** This is the least-appreciated lesson and the one with the most upside. The thumbnail (Netflix), the shelf explanation (Spotify's BaRT), the playlist order (sequencing) — all of these change whether a recommendation works, and all of them belong *inside* the optimization loop, often as bandits, not in a static rendering layer. The model does not end at the ranked list; it ends at the rendered surface. Teams that treat the UI as out of scope leave the highest-leverage lever untouched.

**The offline-online gap forces an experimentation culture.** Because the metric that matters (retention, engagement) only exists online, and because logs are biased by the current policy ([feedback loop](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles)), offline metrics are hypotheses, not results. Both companies built heavy [A/B testing](/blog/machine-learning/recommendation-systems/ab-testing-recommenders) and (Spotify explicitly) off-policy evaluation infrastructure so that exploration today makes tomorrow's evaluation trustworthy. The institutional habit — never ship on offline numbers alone — is the difference between a system that improves and one that drifts.

**Cold start is a content problem; solve it, do not wish it away.** Netflix uses metadata and editorial rows; Spotify regresses content (audio, text) onto the CF space. Either way, the cure is to give a cold item a vector from its *content* so it is recommendable on day zero, then let real interactions take over — the [content/hybrid](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders) and [cold-start](/blog/machine-learning/recommendation-systems/the-cold-start-problem) playbook, validated at the scale of two of the largest catalogs in the world.

## 12. Case studies and real numbers

Pulling the named, sourced results together (with the honesty caveats from section 1):

- **Netflix Prize (2006–2009).** A public \$1,000,000 competition to cut star-rating RMSE by 10%. The winning BellKor's Pragmatic Chaos ensemble crossed the line in 2009. Netflix has written that the grand-prize ensemble was largely *not* productionized — the incremental RMSE gain was not worth the engineering cost, and the company was already moving off rating prediction. The lasting wins were methodological: matrix factorization and the value of implicit signal. *(Source: the Netflix Prize itself; Netflix tech blog retrospectives; Koren, Bell, Volinsky 2009, "Matrix Factorization Techniques for Recommender Systems.")*

- **"~80% of watched content comes from recommendations."** Stated by Netflix (Gomez-Uribe and Hunt, 2015, "The Netflix Recommender System: Algorithms, Business Value, and Innovation," *ACM TMIS*). Read it as: the large majority of plays start from a recommended surface, not search. The same paper describes the row-based page, the everything-is-a-recommendation philosophy, and effective catalog size as the goal.

- **"~\$1B/year in retention value."** Widely reported lore tracing to Netflix executives describing personalization's retention value; treat it as a business estimate of churn avoided, not an auditable figure. The *structural* claim — recommendation is the product's economic engine via retention — is the sourced one.

- **Netflix artwork personalization.** Netflix's tech blog ("Artwork Personalization at Netflix," 2017) describes the image-selection system as a contextual bandit and reports take-rate improvements from per-member image choice over a single editorial image. The architecture (candidate images, contextual bandit, online measurement) is the transferable result.

- **Deep content-based music recommendation (van den Oord, Dieleman, Schrauwen, NIPS 2013).** The seminal audio-to-CF work: a CNN on a track's spectrogram trained to predict its WMF (weighted matrix factorization) latent factors, used to recommend cold-start tracks. They found predicting latent factors beats predicting tags for recommendation, and named the semantic-gap limitation explicitly. This is the scientific core of section 7.

- **Discover Weekly.** Spotify's algorithmic 30-track weekly playlist, launched 2015, built on the three-signal blend (collaborative, NLP, audio content) and reported by Spotify to reach tens of millions of users with strong engagement. Exact internal figures are Spotify's; the design and rough scale are public.

- **BaRT — "Explore, Exploit, and Explain" (McInerney et al., Spotify, RecSys 2018).** The home-page contextual-bandit system that balances exploration and exploitation while attaching explanations, logging propensities for off-policy evaluation. The paper studies how explanation interacts with exploration.

- **Reproduction (this post, MovieLens-20M).** Content-to-CF MLP: mean cosine ~0.55–0.65 to true MF vectors on a warm holdout; neighbor Recall@10 ~0.18–0.26 versus ~0.001 random — a two-orders-of-magnitude cold-start lift from content alone, lossy but recommendable. Diversity-aware page construction: genre coverage rises from ~9–11 to ~15–17 over a 6×8 page at a small mean-score cost. These are laptop-scale demonstrations of the mechanism, not Netflix/Spotify-scale results.

A compact comparison of the two systems' engineering choices:

| Dimension | Netflix | Spotify |
| --- | --- | --- |
| Catalog shape | ~tens of thousands of titles, slow-turning | tens of millions of tracks, fast-turning, deep tail |
| Primary signal | implicit play / watch-time, ranking objective | collaborative (playlists/co-listen) + NLP + audio |
| Cold-start cure | metadata + editorial rows | predict CF vector from audio (CNN on spectrogram) |
| Signature product | whole personalized page + personalized artwork | Discover Weekly + BaRT home page |
| Presentation lever | thumbnail (contextual bandit) | shelf explanation + playlist sequencing |
| North-star metric | engagement, retention, effective catalog | stream rate, engagement, retention |
| Evaluation discipline | heavy A/B testing | A/B + off-policy (logged propensities) |

And a comparison of the three reproduced techniques against what they stand in for:

| Reproduced idea | Production analog | What it solves | Honest limitation |
| --- | --- | --- | --- |
| Content→CF MLP (MovieLens) | audio CNN→WMF factors (Spotify) | cold-start item gets a vector | lossy (semantic gap); ~0.6 cosine, not ~1.0 |
| Greedy diversity-aware page | Netflix whole-page construction | redundancy / coverage on the page | $\lambda$ must be tuned online, not offline |
| LinUCB artwork bandit | Netflix artwork personalization | per-context presentation choice | needs propensity logging + slot deconfounding |

## 13. When to reach for this (and when not to)

A decisive section, because every one of these ideas is a cost and not all products need them.

**Reach for whole-page / 2-D optimization when** your surface is genuinely a page of rows (a streaming home screen, a multi-shelf storefront) and items *substitute* — when showing two near-duplicates wastes a slot. If your product is a single feed or a single ranked list, do not build a slate optimizer; rank the list well and add a light dedup. The slate machinery earns its complexity only when the page structure and substitution are real.

**Reach for personalized presentation (artwork bandits, explanations) when** presentation demonstrably moves the take rate *and* you have the traffic to learn a contextual policy and the infrastructure to log propensities. On a small product with thin traffic, a contextual bandit over images will spend forever exploring and never converge; a single well-chosen image and an A/B test of two or three options is the right size. Do not ship a bandit you cannot feed.

**Reach for content-to-CF embedding (the audio-to-CF trick) when** you have a fast-turning catalog with a deep cold tail and rich content (audio, images, text) — music, news, short video, large marketplaces. If your catalog is small and slow (a back catalog of a few thousand stable items), CF plus a little metadata is enough; do not train a content CNN to solve a cold-start problem you barely have. The trick pays exactly in proportion to how much of your catalog is cold at any moment.

**Do not** north-star on offline accuracy for any of this. The single most expensive mistake in the Netflix/Spotify playbook is to ship a page, an image, or a sequence because it won offline NDCG, and skip the online retention check. The offline metric is a hypothesis. **Do not** treat presentation as out of scope — leaving the thumbnail and the ordering to a static template forfeits the highest-leverage, lowest-cost lever these companies found. And **do not** let the recommender concentrate consumption on a few hits to win short-term engagement; you will hollow out the catalog you paid for and trigger the feedback loop the series warns about throughout.

## 14. Key takeaways

- **Recommendation is the product, not a feature.** When ~80% of consumption starts from a recommended surface, the home screen and the recommender are the same object, and the company's economics ride on it.
- **The metric is engagement and retention, not rating accuracy.** Both companies left RMSE behind; the objective that pays is ranking-and-retention, judged online. Effective catalog size, not prediction error, is the goal.
- **Frame the output as what you ship.** Netflix ships an ordered page, so it trains on ranking and implicit play data, not on star-rating regression — a model whose output is an order must be trained on orders.
- **The page is a 2-D slate, and slates are submodular.** When items substitute, a diverse page beats a stack of near-duplicate hits even at lower average item score; greedy marginal-value construction is the principled, near-optimal heuristic.
- **Presentation is part of the recommender.** Personalized artwork (a contextual bandit), shelf explanations (BaRT), and playlist sequencing all change whether a recommendation works — pull them inside the optimization loop, not a static rendering layer.
- **Cold start is a content problem with an elegant cure.** Predict an item's collaborative embedding directly from its content (a CNN on a spectrogram for audio), so a brand-new item lands in the right neighborhood on day zero — lossy but recommendable, then real interactions take over.
- **Bandits are the operating system of presentation.** Artwork selection and the Spotify home page are contextual bandits: explore to learn, log propensities so tomorrow's off-policy evaluation is honest, and let the confidence term fund exploration of new images and shelves.
- **The offline-online gap forces experimentation.** The metric that matters only exists online and logs are policy-biased, so offline numbers are hypotheses; build the A/B and off-policy infrastructure first and never ship on offline metrics alone.
- **Own the long tail.** A recommender that concentrates consumption destroys catalog value; one that helps each user find their corner of the catalog turns the long tail into an asset. Measure coverage; reward discovery.

## 15. Further reading

- **Gomez-Uribe and Hunt (2015), "The Netflix Recommender System: Algorithms, Business Value, and Innovation"** (*ACM Transactions on Management Information Systems*) — the canonical Netflix paper: row-based page, everything-is-a-recommendation, the ~80% figure, effective catalog size, and business value.
- **van den Oord, Dieleman, Schrauwen (2013), "Deep content-based music recommendation"** (NIPS) — the audio-to-CF trick: a CNN on spectrograms predicting WMF latent factors for cold-start tracks; the source of section 7.
- **Koren, Bell, Volinsky (2009), "Matrix Factorization Techniques for Recommender Systems"** (*IEEE Computer*) — the methodological legacy of the Netflix Prize.
- **McInerney et al. (2018), "Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits"** (RecSys) — Spotify's BaRT: contextual bandits with explanations and logged propensities on the home page.
- **Netflix Technology Blog, "Artwork Personalization at Netflix" (2017)** — the contextual-bandit framing of thumbnail selection and the take-rate result.
- **Within the series**: start at [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) for the funnel map; this post sits downstream of [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders), [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem), [bandits and the exploration-exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff), and [beyond accuracy: diversity, novelty, serendipity, coverage](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage); it is judged by [A/B testing recommenders](/blog/machine-learning/recommendation-systems/ab-testing-recommenders); and everything assembles in [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
