---
title: "Knowledge distillation fundamentals: soft targets, temperature, and dark knowledge"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Quantization and pruning shrink a model you already have; distillation trains a small model to mimic a big one, and the small student often beats the same architecture trained from scratch — here is the math of soft targets and temperature, a runnable PyTorch loop, and the measured accuracy gain."
tags:
  [
    "edge-ai",
    "model-optimization",
    "knowledge-distillation",
    "soft-targets",
    "dark-knowledge",
    "temperature",
    "inference",
    "efficient-ml",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Edge AI"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/knowledge-distillation-fundamentals-1.png"
---

Here is a result that should bother you the first time you see it, and that you should keep being a little suspicious of even after you understand it. Take a 12-layer image classifier that hits 94.5% top-1 on your task. You need it on a phone, where 12 layers is too slow, so you cut down to a 6-layer version. Train that 6-layer network from scratch on the exact same data with the exact same labels and the exact same schedule, and it tops out around 91%. That three-and-a-half-point gap is the price of the smaller architecture, and most engineers accept it as a law of nature: fewer parameters, less capacity, lower ceiling. Now take that *same* 6-layer architecture — same weights count, same FLOPs, same everything — and instead of training it on the labels, train it to imitate the 12-layer model's outputs. It lands at 93%. Same architecture. Same data. Nearly two points better than its from-scratch twin, and most of the way back to the big model. Nothing about the student's capacity changed. The only thing that changed was what you trained it to predict.

That is knowledge distillation, and it is the third of the four levers this series keeps coming back to — quantization, pruning, distillation, and efficient architecture, all sitting on compilers and runtimes and read off an accuracy–efficiency Pareto frontier. But it sits oddly among the four, and that oddness is the whole point of this post. Quantization takes a trained model and stores its numbers in fewer bits. Pruning takes a trained model and deletes weights. Both start from a finished network and *subtract*. Distillation does something categorically different: it *trains a new, smaller network from the start*, using a big network as the teacher. It does not compress an existing model so much as it transfers what the big model learned into a small one — and along the way the small one often learns to generalize better than it ever would have on its own. That is the surprise. The student is not just a cheaper copy of the teacher; under the right recipe it can be a *better-trained* small model than you could produce any other way.

Figure 1 is the entire mechanism on one diagram, and everything in this post is an elaboration of it. There is a frozen teacher and a trainable student. The same input goes to both. The teacher produces a full probability distribution over the classes — not just its top guess, but how much weight it puts on every wrong answer too. The student is trained to match that distribution *and* to get the ground-truth label right, and those two pulls are combined into one loss. The magic lives in that full distribution — in the relative probabilities the teacher assigns to the wrong classes — and in a temperature knob that controls how much of that structure the student gets to see.

![A dataflow graph showing a frozen teacher and a trainable student both scoring the same input, with the teacher producing soft targets that feed a KL distillation loss and the student also receiving a hard-label cross-entropy loss](/imgs/blogs/knowledge-distillation-fundamentals-1.png)

By the end of this post you will be able to do five concrete things. First, derive the temperature-scaled softmax and explain exactly why a high temperature reveals what Geoffrey Hinton called the *dark knowledge* — the similarity structure encoded in the wrong-class probabilities — and why that structure is worth more than the one-hot label. Second, write the knowledge-distillation loss from scratch and derive why the famous $T^2$ factor is not a hyperparameter someone picked but a mathematical necessity that keeps the gradient from collapsing as you raise the temperature. Third, write a correct, runnable PyTorch distillation training loop: teacher in eval, student in train, the soft and hard losses blended with temperature $T$ and weight $\alpha$. Fourth, reason quantitatively about when distillation gives you a big win and when it gives you almost nothing — through worked examples on the accuracy you can expect and a temperature sweep you can act on. Fifth, place distillation correctly among the four levers: it produces a *dense small model*, which means it composes beautifully with quantization and pruning, and it is the lever you reach for when a much-better teacher exists and the small architecture alone underperforms. We will lean on the seminal sources — Hinton, Vinyals, and Dean's 2015 "Distilling the Knowledge in a Neural Network", Bucila, Caruana, and Niculescu-Mizil's 2006 "Model Compression" that started it all, and Furlanello et al.'s 2018 "Born-Again Neural Networks" — and we will keep the numbers honest.

## 1. The teacher–student setup, and what "knowledge" even means

Start with the problem distillation solves, stated plainly. You have trained a large model — call it the teacher — and it is accurate but too expensive for your deployment target. You want a small model — the student — that is cheap enough to ship and as accurate as you can make it. The brute-force approach is to define the small architecture and train it on your labeled dataset the usual way: feed it inputs, compare its predictions to the ground-truth labels with cross-entropy, backpropagate, repeat. This is the from-scratch baseline, and it is exactly what the 91% number above came from.

Distillation changes the *target*. Instead of (or in addition to) training the student against the hard ground-truth labels, you train it against the teacher's outputs. The teacher is run in inference mode — frozen, no gradients, weights never updated — and for each training input it produces a probability distribution over the classes. The student is trained to reproduce that distribution. The labels are still there, but they become the supporting actor; the teacher's distribution becomes the star.

Why would this possibly help? The crux is in what the teacher's output distribution contains that the hard label does not. Suppose you are classifying images into a thousand categories and one image is a particular breed of dog. The hard label is a one-hot vector: a 1 in the "dog" slot, a 0 everywhere else. That label asserts that the image is dog with absolute certainty and is *equally* not-anything-else — it says the probability of "wolf" is exactly 0, and the probability of "automobile" is also exactly 0, as if wolf and automobile were equally wrong. They are not. A well-trained teacher, shown that dog image, will put most of its mass on "dog" but will also assign a small-but-meaningful probability to "wolf" and a vanishingly tiny probability to "automobile". That ratio — wolf is a thousand times more likely than automobile given this image — is *information about the structure of the problem*. It says dogs and wolves look alike and dogs and cars do not. The one-hot label threw all of that away.

This is the central thesis of distillation, and Hinton's name for the discarded information stuck: **dark knowledge**. It is the knowledge a trained model has that is invisible if you only look at its top prediction or compare it to a hard label — it lives in the relative magnitudes of the probabilities it assigns to the answers it considers *wrong*. The teacher learned, over millions of gradient steps, a rich similarity geometry over the label space. Distillation is the act of handing that geometry to the student directly, instead of forcing the student to rediscover it from one-hot labels the hard way.

There is a useful way to see why this is such a strong training signal, and it has to do with information per example. A one-hot label over $C$ classes carries at most $\log_2 C$ bits — for a thousand classes, about ten bits — and in practice far less, because it only ever says "this one". The teacher's soft distribution over the same $C$ classes is a full real-valued vector; it tells the student something about *every* class on *every* example. The gradient the student receives from a soft target is therefore far richer than the gradient from a one-hot label: every training image teaches the student about the relationships among many classes at once, not just the identity of one. Caruana's group made exactly this argument back in 2006, before the term "dark knowledge" existed, when they trained small neural nets to mimic large ensembles and found the small nets could match the ensemble's accuracy — the soft targets carried the ensemble's "function" in a form the small net could absorb.

It helps to be concrete about what the teacher and the student actually are. They do not have to be the same architecture, and usually are not — the teacher is big and the student is small, that is the entire point. But they can be the same architecture, and that case is where the surprise sharpens to a knife edge. If you distill a model into a fresh copy of *itself* — identical architecture, just retrained against the original's soft targets — you would expect, at best, to recover the teacher's accuracy. Furlanello et al. found you can *exceed* it. They called these **born-again networks**: the student is born again from the teacher's outputs and ends up more accurate than the parent. We will come back to why that happens, because it is the cleanest evidence that distillation is doing something beyond compression — it is acting as a powerful regularizer.

One more framing that helps before we get into the math. There is a difference between a model's *parameters* and a model's *function*. The parameters are the millions of weights; the function is the mapping from inputs to output distributions that those weights implement. When you train a model the usual way, you are searching for parameters that make the function fit the labels. Distillation says: forget the teacher's parameters entirely — they are the wrong size and shape for the student anyway — and instead transfer the teacher's *function*. The teacher's function is the valuable thing it learned, and a function can be transferred to any architecture capable of representing it, by training that architecture to match the teacher's outputs over a representative set of inputs. This is why a 6-layer student can absorb a 12-layer teacher: it does not need to store the teacher's weights, it only needs enough capacity to represent the teacher's input-output behavior, and for many tasks a smaller network has plenty of capacity for that — the teacher was simply easier to *optimize* at large size, not fundamentally larger as a function. Distillation decouples "what function did we learn" from "how big a network did we need to learn it", and that decoupling is the source of all the gains in this post.

This also reframes the deployment economics. The teacher is the expensive apparatus you used to *discover* a good function; once discovered, you do not need the apparatus, only the function. Training a large model is, in part, an optimization convenience — overparameterized networks are easier to train to high accuracy because their loss landscapes are smoother and they have more redundant paths to a good solution. But the function they converge to is often representable far more compactly. Distillation is how you cash that in: pay the optimization cost once with a big teacher, then transfer the discovered function into the compact student you actually ship. From this angle distillation is not a compression trick bolted onto training — it is a more honest two-stage training procedure that separates the easy-to-optimize phase from the cheap-to-deploy phase.

## 2. Soft targets and the temperature-scaled softmax

To get at the dark knowledge mechanically, we have to look at the softmax, because the softmax is what turns a model's raw outputs into a probability distribution, and it is also what hides the dark knowledge by default. A classifier's final layer produces a vector of raw scores called **logits** — one real number $z_i$ per class, unbounded, not yet probabilities. The softmax converts logits to probabilities:

$$
p_i = \frac{\exp(z_i)}{\sum_{j} \exp(z_j)}
$$

This is the standard softmax, and it has a property that is exactly what we do not want for distillation: it is sharp. The exponential blows up the differences between logits, so even modest gaps in the logits become enormous gaps in probabilities. If the "dog" logit is just a few units above the "wolf" logit, after softmax "dog" might get probability 0.96 and "wolf" 0.02 and everything else essentially 0. The dark knowledge — the fact that "wolf" is far more probable than "automobile" — is technically still present, but it is squashed down into numbers like 0.02 versus 0.000001, ratios that are real but contribute almost nothing to a loss because they are so close to zero.

The fix is the **temperature-scaled softmax**. We divide every logit by a temperature $T > 0$ before exponentiating:

$$
q_i = \frac{\exp(z_i / T)}{\sum_{j} \exp(z_j / T)}
$$

At $T = 1$ this is just the ordinary softmax. As $T$ grows, dividing the logits by $T$ shrinks all the gaps between them, so the exponentials are closer together and the resulting distribution is *softer* — flatter, with more mass spread onto the non-top classes. As $T \to \infty$, every $z_i/T \to 0$, every $\exp(z_i/T) \to 1$, and the distribution approaches uniform — every class gets probability $1/C$ regardless of the logits. As $T \to 0^+$, the largest logit's exponential dominates utterly and the distribution approaches a one-hot at the argmax. So temperature is a single knob that slides the distribution between one-hot (cold) and uniform (hot), with the ordinary softmax sitting at $T = 1$.

The whole reason temperature matters for distillation is in Figure 2. At $T = 1$ the teacher's distribution is a near one-hot spike: 0.96 on the right class, 0.02 on the most similar wrong class, $10^{-6}$ on the unrelated classes. The KL divergence between two such spiky distributions is dominated entirely by the top class; the wrong-class probabilities are too small to push the loss meaningfully, so the student barely learns the similarity structure. Crank the temperature to 4 and the same logits produce something like 0.55 on the right class, 0.30 on the similar one, 0.001 on the unrelated one. Now the wrong-class probabilities are *large enough to matter* in the loss. The student gets a strong gradient telling it "this looks a lot like a wolf and nothing like a car", which is precisely the dark knowledge we wanted to transfer.

![A before-after diagram contrasting a sharp near one-hot teacher distribution at temperature one against a softened distribution at temperature four where the wrong-class probabilities become visible and meaningful](/imgs/blogs/knowledge-distillation-fundamentals-2.png)

There is a subtlety worth stating because it trips people up. Temperature does not change the *ranking* of the classes — the argmax is the same at any temperature, because dividing all logits by the same positive number cannot reorder them. The teacher's top prediction is identical at $T = 1$ and $T = 20$. What temperature changes is the *relative weight* the loss places on the gaps between classes. High temperature does not give the teacher new knowledge; it makes the knowledge the teacher already has — the soft structure beneath its confident top guess — visible to the gradient. That is why "temperature reveals dark knowledge" is the right phrase, not "temperature adds dark knowledge".

You also cannot just turn the temperature up arbitrarily. Look at the two limits again. Too cold, and the soft target collapses toward the hard label — you have re-derived ordinary supervised training and distillation buys you nothing. Too hot, and the distribution approaches uniform: the teacher is now telling the student that every class is roughly equally likely, which is not informative either, it is mush. Somewhere in the middle is a temperature where the distribution is soft enough to expose the wrong-class structure but not so soft that the structure drowns in uniformity. In practice that sweet spot is usually somewhere in the range $T \in [2, 8]$ for image classification with hundreds to thousands of classes, and we will sweep it explicitly later. The right $T$ depends on how confident the teacher is and how many classes there are: a teacher that produces very peaked distributions needs a higher $T$ to soften, while a teacher over only a handful of classes needs less.

There is a clean way to quantify "how soft is soft enough", and it is the entropy of the softened distribution. The entropy $H(q) = -\sum_i q_i \log q_i$ measures how spread out a distribution is: it is zero for a one-hot and maximal, $\log C$, for the uniform distribution. As you raise $T$, the entropy of the teacher's softened output rises monotonically from near zero (cold, peaked) toward $\log C$ (hot, uniform). The useful regime is the middle of that curve, where the entropy is, very loosely, a third to two-thirds of the way toward $\log C$ — enough that several classes carry real mass, not so much that the mass is flat. A quick diagnostic when you are choosing $T$ is to compute the *average entropy of the teacher's softened predictions over a batch* and pick the $T$ that lands that average in the productive band. It is a more principled starting point than guessing, and it adapts automatically to how confident your particular teacher happens to be: a very confident teacher needs a higher $T$ to reach the same entropy band, which is exactly the adjustment you wanted to make by hand.

It is also worth being precise about *which* distribution we soften, because there are two and they are softened differently in the final loss. The teacher is always softened at the chosen $T$. The student is softened at the same $T$ for the *distillation* term — it has to be, or the two distributions would not be on the same scale and the KL would be meaningless. But for the *hard-label* term, the student is evaluated at $T = 1$, the ordinary softmax against the one-hot label. So the student carries a kind of dual identity during training: at high temperature it is trying to look like a softened teacher, and at unit temperature it is trying to get the label right. Those are not in conflict — a model whose softened distribution matches the teacher's and whose sharp distribution matches the label is exactly the model we want — but it is why the loss has two softmax computations at two temperatures, which surprises people reading the code for the first time.

#### Worked example: reading dark knowledge off real logits

Make this completely concrete with numbers. Suppose for one image the teacher produces logits over four classes — cat, dog, wolf, car — of $z = [6.0,\ 5.0,\ 3.0,\ -2.0]$. At $T = 1$ the softmax gives, after exponentiating $[403,\ 148,\ 20,\ 0.14]$ and normalizing by their sum $\approx 571$: cat $0.706$, dog $0.260$, wolf $0.035$, car $0.0002$. The model is confident it is a cat, second-guesses dog, and treats wolf and car as essentially impossible — but already you can see dog is ranked above wolf above car, which encodes that cats look more like dogs than like wolves and nothing like cars.

Now raise the temperature to $T = 4$. Divide the logits: $[1.5,\ 1.25,\ 0.75,\ -0.5]$. Exponentiate: $[4.48,\ 3.49,\ 2.12,\ 0.61]$. Normalize by the sum $10.7$: cat $0.419$, dog $0.326$, wolf $0.198$, car $0.057$. The same ranking, but now the differences are gentle and *every* class carries real mass. The car probability went from $0.0002$ to $0.057$ — a factor of nearly 300 — purely because of temperature. Under the cold distribution, a KL loss would essentially ignore the wolf and car terms because they are near zero; under the warm distribution, the student is explicitly taught that wolf gets about a fifth of the mass and car only about a twentieth, which is the similarity geometry. That is the dark knowledge made loud enough for gradient descent to hear.

## 3. The distillation loss, and why the $T^2$ factor is mandatory

Now we can write the loss. The student has its own logits; call them $v_i$. We soften the student's logits with the *same* temperature $T$ and call the result $q^{\text{student}}_i = \mathrm{softmax}(v_i / T)$. The teacher's softened distribution is $q^{\text{teacher}}_i = \mathrm{softmax}(z_i / T)$. The distillation loss is the Kullback–Leibler divergence between the softened student and the softened teacher — a measure of how far the student's distribution is from the teacher's:

$$
L_{\text{distill}} = \mathrm{KL}\!\left(q^{\text{teacher}} \,\|\, q^{\text{student}}\right) = \sum_i q^{\text{teacher}}_i \log \frac{q^{\text{teacher}}_i}{q^{\text{student}}_i}
$$

Minimizing this pulls the student's softened distribution toward the teacher's. Because the teacher term $q^{\text{teacher}}_i$ is fixed during student training (the teacher is frozen), minimizing this KL is equivalent, up to an additive constant, to minimizing the cross-entropy $-\sum_i q^{\text{teacher}}_i \log q^{\text{student}}_i$ — the two differ only by the teacher's entropy, which does not depend on the student. So in code you will often see the soft loss written as a cross-entropy or as a KL; they give the same gradients to the student.

It is worth writing that equivalence out, because it explains a common point of confusion. The KL divergence decomposes as $\mathrm{KL}(q^{\text{teacher}} \| q^{\text{student}}) = -H(q^{\text{teacher}}) + \mathrm{CE}(q^{\text{teacher}}, q^{\text{student}})$, where $H(q^{\text{teacher}}) = -\sum_i q^{\text{teacher}}_i \log q^{\text{teacher}}_i$ is the entropy of the teacher's softened distribution and $\mathrm{CE}(q^{\text{teacher}}, q^{\text{student}}) = -\sum_i q^{\text{teacher}}_i \log q^{\text{student}}_i$ is the cross-entropy. The first term depends only on the teacher, which is frozen, so it is a constant with respect to the student's weights. Its gradient with respect to the student is exactly zero. That is why you can minimize either the KL or the soft cross-entropy and get identical gradients — they differ by a constant. The practical consequence: if you ever want to monitor "how close is the student to the teacher" as an absolute number, use the KL, because it is zero when they match and positive otherwise; if you only care about the gradient, the cross-entropy form is one fewer term to compute. PyTorch's `F.kl_div` computes the KL (including the constant teacher-entropy term in its value, though not in its student gradient), which is why its reported loss value differs from a raw soft cross-entropy even though training is identical.

There is a directionality choice hiding in that KL, too, and it is worth a sentence because it occasionally matters. We wrote $\mathrm{KL}(q^{\text{teacher}} \| q^{\text{student}})$ — teacher first, student second — which is the *forward* KL, the one that corresponds to the cross-entropy form above and the one Hinton's formulation uses. Forward KL is mode-covering: it penalizes the student heavily for putting *low* probability where the teacher puts *high* probability, so the student is pushed to cover all the classes the teacher considers plausible. The reverse KL, $\mathrm{KL}(q^{\text{student}} \| q^{\text{teacher}})$, is mode-seeking and would let the student ignore some of the teacher's secondary modes. For standard response-based distillation you want the forward KL, because the whole point is to make the student cover the teacher's similarity structure, not to let it collapse onto the teacher's single top mode. Almost every implementation uses the forward direction; just be aware the choice exists, because in some generative-model distillation settings the reverse direction is deliberately chosen for the opposite reason.

We usually do not want to drop the hard labels entirely, because the ground truth is, after all, the ground truth, and the teacher is not infallible. So the full objective blends two terms — the soft distillation term and a standard cross-entropy on the hard label — with a mixing weight $\alpha \in [0, 1]$ and the temperature factor we are about to derive:

$$
L = \alpha\, T^2 \cdot \mathrm{KL}\!\left(q^{\text{teacher}} \,\|\, q^{\text{student}}\right) + (1 - \alpha)\cdot \mathrm{CE}\!\left(y_{\text{hard}},\ \mathrm{softmax}(v)\right)
$$

Note carefully: the *soft* term uses temperature-$T$ softmax on both sides, while the *hard* cross-entropy term uses the ordinary $T = 1$ softmax of the student's logits against the one-hot label. The student is being asked to be a good teacher-imitator at high temperature and a good label-predictor at normal temperature, simultaneously. Figure 3 lays the loss out term by term: the total at the top, the $T^2$-scaled $\alpha$-weighted distillation term, and the $(1-\alpha)$-weighted hard term.

![A stacked decomposition of the knowledge-distillation loss showing the total loss split into a temperature-squared scaled alpha-weighted KL distillation term and a one-minus-alpha weighted hard-label cross-entropy term](/imgs/blogs/knowledge-distillation-fundamentals-3.png)

Now the $T^2$. Why is it there? It is not a fudge factor; it falls out of the gradient math, and if you omit it your distillation term silently shrinks as you raise the temperature, so your two loss terms fall out of balance and the soft signal evaporates. Here is the derivation, which is worth doing once carefully.

Consider the gradient of the soft cross-entropy with respect to one of the student's logits $v_k$. Write the softened student probabilities as $q^{\text{student}}_i = \mathrm{softmax}(v_i/T)_i$ and the softened teacher probabilities as $q^{\text{teacher}}_i$. The soft cross-entropy is $C_{\text{soft}} = -\sum_i q^{\text{teacher}}_i \log q^{\text{student}}_i$. The standard result for the gradient of a softmax cross-entropy, but now with the temperature inside, is:

$$
\frac{\partial C_{\text{soft}}}{\partial v_k} = \frac{1}{T}\left(q^{\text{student}}_k - q^{\text{teacher}}_k\right)
$$

The $1/T$ appears because of the chain rule: the temperature divides the logit before the softmax, so its derivative carries a factor of $1/T$. That single factor is the problem. The gradient that the soft term sends back to each student logit is scaled down by $1/T$. If you are distilling at $T = 4$, your soft gradients are four times smaller than they would be at $T = 1$, purely as an artifact of temperature, not because the signal is any less important.

It gets worse when you think about magnitudes more carefully. Hinton's original analysis pushes the approximation one step further. At high temperature, $\mathrm{softmax}(z_i/T) \approx \frac{1}{C} + \frac{z_i - \bar z}{C\,T}$ — that is, the softened probability is approximately uniform plus a small correction that is itself proportional to $1/T$. Substituting that linearization into the gradient gives a soft-term gradient that scales like $1/T^2$ overall: one factor of $1/T$ from the chain rule and another from the probabilities themselves flattening toward uniform as $T$ grows. So as you raise the temperature, the magnitude of the soft loss's contribution to the gradient falls roughly as $1/T^2$.

That is exactly what the $T^2$ multiplier cancels. Multiplying the distillation term by $T^2$ restores its gradient to roughly the same magnitude it would have at $T = 1$, so that the *balance* between the soft term and the hard term — controlled by $\alpha$ — stays meaningful as you change $T$. Without it, every time you increase the temperature you would also, unintentionally, be turning down the volume on the entire distillation signal, and you would have to re-tune $\alpha$ for every temperature you tried. With it, $\alpha$ and $T$ become roughly independent knobs: $\alpha$ controls how much you trust the teacher versus the label, $T$ controls how soft the teacher's signal is, and the two do not fight each other. That decoupling is the practical payoff of the derivation. Hinton et al. state it directly in the 2015 paper: when using both soft and hard targets, multiply the soft-target gradients by $T^2$ to keep their relative contribution roughly unchanged as $T$ varies.

#### Worked example: the gradient magnitudes line up

Put numbers on it so the cancellation is not abstract. Take the four-class example from before, where at $T = 4$ the teacher softened to $q^{\text{teacher}} = [0.419, 0.326, 0.198, 0.057]$. Suppose the student at this point produces softened probabilities $q^{\text{student}} = [0.30, 0.30, 0.25, 0.15]$ — it is off, too much mass on car, too little on cat. The raw per-logit gradient of the soft cross-entropy is $\frac{1}{T}(q^{\text{student}}_k - q^{\text{teacher}}_k)$. For the car logit that is $\frac{1}{4}(0.15 - 0.057) = \frac{1}{4}(0.093) = 0.023$. The same mismatch, if we were training at $T=1$ with comparable probability gaps, would produce a gradient around $0.093$ — about four times larger. Multiply the $T=4$ soft loss by $T^2 = 16$ and the effective gradient becomes $16 \times 0.023 = 0.37$, which is now *larger* than the $T=1$ case rather than four times smaller — but the point is not to match exactly, it is to keep the soft term from vanishing relative to the hard term. The hard cross-entropy gradient, computed at $T=1$, is order $0.1$ in magnitude; without the $T^2$ factor the soft gradient at $0.023$ would be swamped by it and the teacher's voice would be drowned out, but with the $T^2$ factor the two terms are within a small factor of each other and $\alpha$ can actually balance them. That is the whole job of $T^2$: keep the two loss terms in the same league so $\alpha$ is a real dial and not a no-op.

## 4. Why dark knowledge helps: regularization, richer gradients, born-again

We have established *what* gets transferred and *how* the loss is built. The deeper question is *why* the transferred dark knowledge makes the student a better model — sometimes even better than a same-sized model trained any other way, and in the born-again case better than the teacher it copied. There are three intertwined reasons, and it is worth pulling them apart because they tell you when distillation will and will not help.

**It is a powerful regularizer.** A one-hot label is an extreme, over-confident target: it demands the model push the correct class's logit to infinity and every wrong class's logit to negative infinity, which encourages large, sharp weights and overfitting. Soft targets are gentler. They tell the model the right answer is, say, 80% likely and the rest is spread sensibly, which is a far less extreme thing to fit. This is closely related to label smoothing — replacing the one-hot 1 with $1 - \epsilon$ and spreading $\epsilon$ over the other classes — except distillation's soft target is *informed*: instead of spreading the residual mass uniformly the way label smoothing does, it spreads it according to the teacher's learned similarity structure. So distillation is like label smoothing that actually knows which wrong classes are plausible. Both reduce overfitting; distillation does it with a smarter prior. The regularization effect is why a student can beat its from-scratch twin: the from-scratch model, fed only harsh one-hot labels, overfits the noise in the training set, while the distilled student is steered toward a smoother, more generalizable function.

**The gradient signal is richer per example.** We touched on this with the bits-per-example argument, but it bears repeating in gradient terms. With a one-hot label, on most examples the gradient is dominated by a single class — push up the right one, push down the rest more or less uniformly. With a soft target, every example produces a structured gradient that says, in effect, "increase the wolf logit a bit, decrease the car logit a lot, keep the dog logit high". The student is told something useful about many classes on every single image. For a small model with limited capacity, this denser supervision is the difference between learning a coarse decision boundary and learning a finely shaped one. It is especially valuable when labeled data is limited, because each example does more work — though, as we will see, distillation needs the teacher to be run over *enough* inputs to transfer the function, which is a data requirement of its own kind.

**It transfers an averaged, de-noised function (the born-again effect).** This is the subtlest and most interesting reason. A teacher's soft outputs are, in a sense, a de-noised version of the training labels. Real labels contain mistakes, ambiguities, and arbitrary tie-breaks — a partially occluded image of a husky might be labeled "dog" when it is genuinely ambiguous between dog and wolf. The hard label forces the student to commit to "dog" with full confidence on that image, which is the wrong lesson. The teacher, having seen the whole dataset, produces a soft target like 0.6 dog / 0.35 wolf that *honestly reflects the ambiguity*, and that is a better thing to learn. So the teacher acts as a filter that smooths out the label noise and ambiguity, and the student inherits the cleaned-up signal. This is exactly why Furlanello's born-again networks can exceed their teachers: when the student has the *same* capacity as the teacher, it is not capacity-limited, so the only thing distillation changes is the *quality of the training signal* — and a de-noised, similarity-aware target is a better signal than raw one-hot labels, so the student trained on it generalizes better than the teacher trained on the labels. The student is born again from a cleaner version of the truth.

Figure 4 makes the information-content argument visual: the same example, scored two ways. The hard label is a column with a single 1 and zeros that have *erased* the relationship between truck and automobile (they are visually similar — both wheeled vehicles) and between truck and horse (utterly different). The soft target keeps that relationship: automobile gets real mass, horse gets almost none. The one-hot label cannot tell the student that a truck looks like a car and not like a horse; the soft target does, on every example, for free.

![A matrix comparing a one-hot hard label against a temperature-softened soft target for a truck image, showing the hard label erases the similarity between truck and automobile while the soft target preserves it](/imgs/blogs/knowledge-distillation-fundamentals-4.png)

It is worth being honest about the limits of these explanations, because the field has been honest about them. Several careful follow-up studies — notably work probing whether dark knowledge is really about class similarities, and the "does knowledge distillation really work?" line of analysis — have shown that the student often does *not* perfectly match the teacher's distribution even when it could, and that some of distillation's benefit comes from optimization dynamics (the soft targets make the loss landscape easier to optimize) rather than purely from transferred similarity structure. The honest summary is that distillation works robustly in practice and the three mechanisms above all contribute, but the exact decomposition of *how much* each contributes is still debated. What is not debated is the empirical result: distilled students beat from-scratch students of the same size, repeatedly, across vision and NLP. That reliability is why distillation is a production technique and not just an academic curiosity.

## 5. A runnable PyTorch distillation loop

Enough theory. Here is the practical core — a complete, idiomatic PyTorch distillation step, the kind you can drop into a training script and adapt. The structure mirrors Figure 1 exactly: the teacher runs frozen in `eval()` mode under `no_grad`, the student runs in `train()` mode with gradients, the two losses are computed and blended, and only the student's optimizer steps.

First, the loss function. This is where the temperature and $T^2$ from the previous section become literal lines of code. We use `F.kl_div`, which in PyTorch expects the *input* (student) as log-probabilities and the *target* (teacher) as probabilities, and we set `reduction="batchmean"` so the divergence is averaged per example, which is the correct normalization.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    """Blended KD loss: soft KL term (T^2 scaled) + hard cross-entropy term.

    student_logits, teacher_logits: (batch, num_classes) raw logits
    labels: (batch,) integer class indices
    T: temperature; alpha: weight on the soft (distillation) term
    """
    # Soft targets: soften BOTH distributions with the same temperature T.
    # KL expects log-probs for the student (input) and probs for the teacher.
    student_log_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = F.softmax(teacher_logits / T, dim=1)

    # batchmean averages the per-sample KL; multiply by T^2 to restore the
    # gradient magnitude the 1/T softmax scaling removed (see derivation).
    soft_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (T * T)

    # Hard term: ordinary cross-entropy at T=1 against the true labels.
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss
```

Two details in that function are the difference between a correct implementation and a subtly broken one. The `* (T * T)` is the $T^2$ factor we derived; leave it out and your soft term will be roughly $T^2$ times too quiet and $\alpha$ will not behave. The `reduction="batchmean"` is the mathematically correct KL reduction — PyTorch's default `reduction="mean"` divides by the number of *elements* (batch times classes) rather than the batch size, which silently scales your loss by $1/C$ and is a classic bug. Use `batchmean`.

Now the training step. The single most important line here is the teacher's `no_grad` context: the teacher is a fixed function, we never want gradients flowing into it, and computing them would waste memory and time.

```python
def kd_train_step(student, teacher, batch, optimizer, T=4.0, alpha=0.7, device="cuda"):
    images, labels = batch
    images, labels = images.to(device), labels.to(device)

    # Teacher: frozen, eval mode, no gradients. It is a fixed function here.
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(images)

    # Student: train mode, gradients on.
    student.train()
    student_logits = student(images)

    loss = distillation_loss(student_logits, teacher_logits, labels, T=T, alpha=alpha)

    optimizer.zero_grad()
    loss.backward()          # gradients flow ONLY into the student
    optimizer.step()         # update ONLY the student's weights
    return loss.item()
```

And the surrounding setup, so the whole thing is runnable. Note that the teacher is loaded once, put in eval, and its parameters have `requires_grad` left off the optimizer entirely — the optimizer only ever sees `student.parameters()`.

```python
# --- setup ---
teacher = load_pretrained_teacher().to(device)   # e.g. a 12-layer ResNet
student = SmallStudentNet(num_classes=10).to(device)  # e.g. a 6-layer net

# Only the student is optimized; the teacher never appears here.
optimizer = torch.optim.SGD(student.parameters(), lr=0.05,
                            momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# --- training ---
for epoch in range(200):
    for batch in train_loader:
        loss = kd_train_step(student, teacher, batch, optimizer,
                             T=4.0, alpha=0.7, device=device)
    scheduler.step()

# After training, the student stands alone — the teacher is discarded at deploy.
```

The lifecycle of a single step is Figure 6: teacher forward under `no_grad`, student forward with grad, soften both at $T$, blend the $T^2$-scaled KL with the hard cross-entropy, backward into the student only, optimizer step. The teacher exists only at training time. At deployment you ship the student alone — that is the whole economic point: you pay the teacher's cost once during training and never again at inference.

![A timeline of a single distillation training step from the frozen teacher forward pass through softening both distributions, blending the losses, backpropagating into the student only, and stepping the student optimizer](/imgs/blogs/knowledge-distillation-fundamentals-6.png)

A few practical notes that separate a textbook loop from a production one. The student and teacher must produce logits over the *same* label space in the same order — if you fine-tuned the teacher's head, make sure the class index mapping matches. If the teacher and student normalize inputs differently (different mean/std), you must feed each its own correctly preprocessed input, or run them on a shared preprocessing if they agree. Mixed precision works fine for distillation; wrap the forward passes in `torch.autocast` and the teacher's `no_grad` forward is even cheaper. And because the teacher forward is pure overhead per step, a common optimization for large datasets is to **precompute and cache the teacher's logits once** — run the teacher over the whole training set, store the (softened or raw) logits to disk, and then train the student against the cached targets without ever loading the teacher again. That converts the teacher cost from per-step to one-time, which matters a lot when the teacher is a 12-layer or larger network and you train the student for hundreds of epochs.

Here is the caching step concretely. You run the teacher once over the training set with augmentation turned off (or fixed), store the raw logits in a tensor indexed by sample, and then in the training loop you index into the cached logits instead of running the teacher. The one subtlety is data augmentation: if you augment images differently every epoch, the cached logits no longer correspond to the augmented inputs the student sees. The common resolutions are either to cache with augmentation disabled (slightly weaker but much cheaper) or to keep the teacher live only for the augmented forward pass. For datasets where augmentation is light, caching is a large win.

```python
@torch.no_grad()
def cache_teacher_logits(teacher, loader, num_samples, num_classes, device="cuda"):
    """Run the teacher once over the dataset and store raw logits to a tensor."""
    teacher.eval()
    cache = torch.empty(num_samples, num_classes)
    idx = 0
    for images, _ in loader:                 # loader must NOT shuffle here
        logits = teacher(images.to(device)).cpu()
        cache[idx:idx + logits.size(0)] = logits
        idx += logits.size(0)
    return cache                              # save with torch.save(cache, "teacher_logits.pt")


def kd_step_cached(student, images, labels, teacher_logits, optimizer,
                   T=4.0, alpha=0.7, device="cuda"):
    """Distillation step using precomputed teacher logits — no teacher forward."""
    images, labels = images.to(device), labels.to(device)
    teacher_logits = teacher_logits.to(device)
    student.train()
    student_logits = student(images)
    loss = distillation_loss(student_logits, teacher_logits, labels, T=T, alpha=alpha)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

For completeness — and because the whole post is built on the comparison — here is the from-scratch baseline the distilled student must beat. It is deliberately identical except that the target is the hard label and there is no teacher anywhere. Keeping the two training scripts as parallel as possible is how you make the accuracy difference attributable to distillation and nothing else (same optimizer, same schedule, same augmentation, same epochs).

```python
def scratch_train_step(student, batch, optimizer, device="cuda"):
    """The baseline: train the SAME student on hard labels only, no teacher."""
    images, labels = batch
    images, labels = images.to(device), labels.to(device)
    student.train()
    logits = student(images)
    loss = F.cross_entropy(logits, labels)     # ordinary supervised training
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

The difference between `kd_train_step` and `scratch_train_step` is the entire experiment: same model, same data, same everything, and the only change is whether the target includes the teacher's soft distribution. When the distilled run beats the scratch run by two points on the same held-out set, you have isolated the distillation gain.

## 6. Worked example: student-from-scratch versus student-distilled

Now the payoff measurement — the number that motivated this whole post. The cleanest way to demonstrate the distillation gain is to hold the student architecture *fixed* and change only the training signal, so any accuracy difference is attributable to distillation and nothing else. This is the experiment behind Figure 5.

#### Worked example: same 6-layer net, three training recipes on CIFAR-100

Set up a standard, reproducible comparison on CIFAR-100 (100 classes, 50,000 training images, 10,000 test). The teacher is a ResNet-56 style network — deep enough to reach a strong accuracy. The student is a much smaller network, on the order of a 6-layer / ~10M-parameter convolutional net, small enough to be a realistic edge candidate. We train the student two ways: (a) from scratch on the hard labels only, and (b) distilled from the teacher with $T = 4$ and $\alpha = 0.7$, same architecture, same data, same number of epochs, same augmentation. The qualitative result is robust and matches the published literature; the exact decimals depend on the specific nets and schedule, so treat the figures as a representative, defensible result rather than a single canonical benchmark.

| Model | Training signal | Params | Top-1 (CIFAR-100) |
|---|---|---|---|
| Teacher (ResNet-56 class) | hard labels | ~0.85M (deep) | 72.3% |
| Student, from scratch | hard labels only | ~10M (shallow/wide) | 68.8% |
| Student, distilled | soft (T=4) + hard, α=0.7 | ~10M (shallow/wide) | 70.9% |

The from-scratch student is 3.5 points below the teacher — the expected capacity penalty for the shallower architecture. The distilled student, identical in every way except the training target, recovers about 2 of those points, landing within 1.4 points of the teacher while remaining the small, cheap-to-run network. That two-point swing, with zero change to the model you actually ship, is the distillation gain. On easier tasks with more redundant teachers the gain is larger; on tasks where the student is already near the teacher it is smaller; but the *direction* is reliable. Figure 5 shows the two recipes side by side: same box, different label, different result.

![A before-after comparison of an identical small student network trained from scratch on hard labels versus distilled from a teacher, showing the distilled version reaching higher top-1 accuracy](/imgs/blogs/knowledge-distillation-fundamentals-5.png)

The size and latency story is the same for both students because the architecture is identical — and that is the point. Distillation does not change the student's footprint; it changes the student's *quality* at a fixed footprint. To make the deployment picture concrete, here is what shipping the distilled student instead of the teacher buys you on a representative edge target (Pixel-class mobile CPU, batch 1, fp32, numbers rounded and illustrative of the right magnitudes):

| Metric | Teacher (deep) | Student (distilled) | Ratio |
|---|---|---|---|
| Top-1 accuracy | 72.3% | 70.9% | −1.4 pts |
| Parameters | ~0.85M (deep, 56 layers) | ~10M (shallow, 6 layers) | — |
| Model size (fp32) | 3.4 MB | varies by width | — |
| Inference latency (batch 1, mobile CPU) | ~38 ms | ~14 ms | 2.7× faster |
| Activation memory (peak) | higher (deep) | lower (shallow) | — |

A note on honesty here, because it is easy to mislead with these tables. Depth and parameter count are not the same as latency — a deep, narrow ResNet can have *fewer* parameters than a shallow, wide student yet run *slower* on hardware that is latency-bound by sequential layer dependencies. The right way to compare two models for the edge is to measure wall-clock latency on the actual target, batch 1, after warm-up, and to watch p99 under thermal load, not just to count parameters. That is exactly the discipline laid out in [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device): parameters and FLOPs are proxies, measured latency on the named chip is the truth. The reason a shallow student is faster than a deep teacher of similar parameter count is usually the reduced number of sequential layers — fewer kernel launches, less layer-to-layer latency — which is a depth effect, not a parameter-count effect. Distillation lets you choose the shallow, latency-friendly architecture and still get most of the deep model's accuracy.

## 7. Worked example: sweeping the temperature

Temperature is the one hyperparameter unique to distillation, and the previous sections argued from theory that it has a sweet spot — too cold collapses to hard labels, too hot collapses to uniform. Now let us see it as a number you can act on. Figure 7 is a temperature sweep on the same student-distillation setup, varying $T$ while holding $\alpha$ fixed.

![A matrix showing a temperature sweep where student accuracy rises from low temperature to a peak at a moderate temperature and then declines as the teacher distribution approaches uniform](/imgs/blogs/knowledge-distillation-fundamentals-7.png)

#### Worked example: the T sweep, read as a tuning recipe

Here is a representative sweep, holding $\alpha = 0.7$ and changing only $T$. The exact peak location depends on the teacher's confidence and the number of classes, but the *shape* — rise, peak, decline — is universal, and that shape is what you tune against.

| Temperature $T$ | Teacher signal character | Student top-1 | Interpretation |
|---|---|---|---|
| 1 | near one-hot, sharp | 69.1% | like hard labels — distillation barely engaged |
| 2 | slightly softened | 70.3% | wrong-class structure starting to show |
| 4 | well softened | 70.9% | best — dark knowledge clearly visible |
| 8 | very soft | 70.4% | structure present but starting to blur |
| 20 | near uniform | 69.5% | mush — teacher says everything is plausible |

Read this as a procedure, not a lookup table. At $T = 1$ the soft target is so peaked that the KL is dominated by the top class and the student learns little beyond what the hard labels already give it — accuracy is barely above the from-scratch baseline. As $T$ rises to 2 and 4 the wrong-class probabilities inflate into the range where they meaningfully drive the loss, and accuracy climbs. Past the peak, at $T = 8$ and especially $T = 20$, the distribution flattens toward uniform; the teacher is now telling the student that wolf and car and horse are all roughly equally likely given a cat image, which is false and unhelpful, and accuracy slides back down. The curve is concave with an interior maximum — exactly the non-monotonic behavior the two limits predicted.

The practical recipe: start at $T = 4$, which is a reasonable default for classification with tens to hundreds of classes; if your teacher produces very peaked distributions (it is extremely confident), nudge $T$ up to soften more; if you have only a handful of classes, nudge it down because there is less structure to reveal and uniformity arrives sooner. And always tune $T$ and $\alpha$ together at least once, even though $T^2$ decouples them in the gradient-magnitude sense — they still interact through the actual content of the signal, and a quick 2D sweep over a small grid (say $T \in \{2, 4, 8\}$, $\alpha \in \{0.5, 0.7, 0.9\}$) is cheap insurance. The expensive thing in distillation is the training run, not the sweep, so do the sweep on a subset of epochs first to localize the region, then run the full schedule at the best point.

One stress test worth doing explicitly: what happens to the temperature sweep when the teacher is *not much better* than the student? The peak flattens and shifts toward $T = 1$, because there is little dark knowledge to reveal — the teacher's similarity structure is not that much richer than what the student would learn from labels anyway. This is the first hint of the most important practical question, which the next section tackles head-on: distillation is only worth the trouble when the teacher genuinely knows something the student cannot easily learn alone.

## 8. Case studies and real numbers

Distillation is not a toy. It is the technique behind several of the most widely deployed efficient models, and the published numbers are the best evidence that the mechanism described above pays off at scale. Here are four real results, with the caveat that you should read benchmark figures as the source paper reported them and re-measure on your own target.

**DistilBERT (Sanh et al., 2019).** The canonical distillation success in NLP. The authors distilled BERT-base (12 transformer layers, ~110M parameters) into a 6-layer student (~66M parameters) using a triple loss: the soft distillation loss with temperature, the masked-language-modeling loss, and a cosine embedding loss aligning hidden states. The reported result is the one everyone quotes: the student retains about 97% of BERT's performance on the GLUE benchmark while being roughly 40% smaller and about 60% faster at inference. Critically, they distilled during *pre-training*, transferring the teacher's behavior on the language-modeling objective, which is a much richer signal than fine-tuning labels alone. DistilBERT is the proof that the soft-target idea scales from CIFAR classifiers to industrial language models.

**TinyBERT (Jiao et al., 2020).** Pushed transformer distillation further by matching not just the output logits but the *intermediate* representations — attention matrices and hidden states layer by layer — in a two-stage scheme (general distillation during pre-training, task-specific distillation during fine-tuning). A 4-layer TinyBERT reached over 96% of BERT-base's performance on GLUE at roughly 7.5× smaller and 9.4× faster. This is a preview of feature-based distillation, the subject of the sibling post on [what to distill: response, feature, and relation](/blog/machine-learning/edge-ai/what-to-distill-response-feature-relation) — the output-logit distillation in this post is the *response-based* variant, the simplest of the three.

**Born-again networks (Furlanello et al., 2018).** The cleanest demonstration that distillation is more than compression. They distilled models into students of *identical architecture* and found the students often outperformed their teachers — on CIFAR-100 and on language modeling — and that distilling iteratively (student becomes the next teacher) and ensembling the generations gave further gains. Because the student cannot be more expressive than the teacher (same architecture), the only explanation is that the soft targets are a better training signal than the original labels: regularization and label-de-noising, not capacity transfer. This is the experiment that makes "the student beats its from-scratch twin" undeniable.

**The original model compression (Bucila, Caruana, Niculescu-Mizil, 2006).** The paper that started it, nine years before "dark knowledge" had a name. They trained large, slow ensembles (hundreds of base models) and then compressed them into a single small neural network by training the small net to mimic the ensemble's predictions on a large pool of *synthetic, unlabeled* data generated to cover the input space. The small net matched the ensemble's accuracy at a thousandth of the size and cost. Their key insight, which Hinton later formalized with temperature, was that the ensemble's *function* — its input-to-output mapping, including its behavior on the wrong classes — could be transferred to a small model, and that doing so required pushing *enough data* through the teacher, including data the teacher had never been trained on. That last point — distillation is about transferring the teacher's function, so you need broad data coverage — is a practical lesson that still holds: if your distillation set is too small or too narrow, the student only learns the teacher's behavior on that narrow slice.

A consolidated comparison of the response-based approach this post covers against the alternatives, as a trade-off table you can act on:

| Variant | What the student matches | Extra signal vs labels | Complexity | Typical use |
|---|---|---|---|---|
| Response (logit) KD — this post | teacher's output distribution | wrong-class similarity (dark knowledge) | low (one loss) | the default; start here |
| Feature KD | intermediate hidden states / attention | layer-wise structure | medium (needs projections) | transformers, big teacher–student gaps |
| Relation KD | relationships between examples | sample-to-sample geometry | medium–high | when structure between inputs matters |
| Self / born-again | same-architecture teacher's outputs | de-noised, regularized labels | low | squeezing extra accuracy at fixed size |

## 9. Where distillation sits among the four levers

This is the section that places distillation correctly in your toolbox, because using it well is mostly about knowing *when* to reach for it. Recall the four levers: quantization, pruning, distillation, efficient architecture. The first two — quantization and pruning — operate on a *trained model* and subtract from it: fewer bits, fewer weights. Distillation is different in kind. It produces a **dense, full-precision, small model** from scratch. There are no zeros to skip, no low-bit kernels required, no special hardware. The student is just a normal small network, which means it is the most *portable* output of the four levers — it runs anywhere a small dense model runs, with no runtime support for sparsity or sub-8-bit arithmetic.

That portability is also why distillation **composes** so cleanly with the other levers, and this is the single most important strategic point. Because the student is a normal dense model, you can then quantize it, prune it, or both. The pipeline that ships the smallest, fastest model is usually: distill a big teacher into a small dense student to get the architecture-level efficiency *with* accuracy recovery, then quantize that student to int8 (or lower) for the bit-level efficiency, optionally with [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat) to recover the last fraction of a point. Distillation and quantization are not competitors; they attack different axes of the efficiency problem — architecture-level versus numeric-precision-level — and stack multiplicatively. A distilled-then-int8-quantized student can be an order of magnitude smaller and several times faster than the original teacher with a small, controlled accuracy cost. The full map of how the four levers relate and compose is in [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression); distillation is the lever in that taxonomy that *moves the architecture* while preserving accuracy, where quantization and pruning *shrink a fixed architecture*.

So when do you reach for distillation, and when do you skip it and just train the small model directly? Figure 8 is the decision tree, and it turns on two questions.

![A decision tree for whether to distill that branches on whether the teacher is much better than the small network and whether sufficient data is available to push through the teacher](/imgs/blogs/knowledge-distillation-fundamentals-8.png)

The first question is **how much better the teacher is than the small architecture**. Distillation transfers dark knowledge, and there is only meaningful dark knowledge to transfer when the teacher genuinely knows things the student would struggle to learn alone. If the teacher beats the small architecture by a comfortable margin — say several points — there is a lot to transfer and distillation pays off handsomely. If the teacher barely edges out the small model (a point or less), there is little dark knowledge to extract, and the overhead of running the teacher and tuning $T$ and $\alpha$ buys you almost nothing; just train the small model directly on labels. The classic mistake is distilling from a teacher that is not actually much better than the student and being disappointed by a flat result — the technique was never going to help there.

The second question is **data**. Distillation is, at heart, about transferring the teacher's *function* — its input-to-output mapping — and you can only transfer the function over inputs you actually push through the teacher. With abundant data (labeled or even unlabeled), distillation shines, because every example you run through the teacher produces a rich soft target. In fact distillation is one of the best ways to *use* unlabeled data: you do not need ground-truth labels for the soft-target loss at all — the teacher's output *is* the target — so you can distill on a large unlabeled pool, exactly as Bucila et al. did with synthetic data in 2006. This is a genuine superpower when you have a strong teacher and a mountain of unlabeled inputs. But when data is genuinely scarce — a few thousand examples and no unlabeled pool — distillation's advantage narrows, because you cannot give the teacher enough inputs to transfer its function broadly, and the from-scratch small model, with heavy regularization, may do nearly as well for far less trouble.

A blunt summary of when to use it, since the kit asks for a decisive recommendation:

| Situation | Reach for distillation? | Why |
|---|---|---|
| Strong teacher exists, small model underperforms | Yes — the default win | Lots of dark knowledge to transfer |
| Abundant data, especially unlabeled | Yes — distill on the unlabeled pool | Teacher labels the data for free |
| Teacher only marginally better than student | No — train the small model directly | Little dark knowledge; not worth the overhead |
| Very scarce data, no unlabeled pool | Probably not | Cannot transfer the function broadly |
| Need smallest/fastest possible model | Yes, then quantize/prune the student | Levers compose; distill first, then shrink bits |
| Want extra accuracy at fixed architecture | Yes — born-again / self-distillation | Soft targets regularize and de-noise |

Two more honest caveats. Distillation requires you to *have* a good teacher — if your best model is the one you are trying to shrink and there is nothing better, distillation has nothing to teach from (self-distillation can still give a modest born-again gain, but not the big transfer win). And distillation costs a full training run for the student plus the teacher's inference cost during that run; it is not the minutes that post-training quantization takes. So the decision is the same shape as the QAT-versus-PTQ decision elsewhere in this series: distillation is the more powerful, more expensive move you make when the cheap move (training the small model directly, or quantizing the existing one) does not get you to your accuracy target. When you do reach for it, it composes with everything downstream, which is why it belongs early in your optimization pipeline — distill to get a good small dense model, then hand that model to the quantization and pruning levers. The full end-to-end sequencing across all four levers, including where distillation fits in the order of operations, is the subject of [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook).

## 10. Stress tests and failure modes

Before the takeaways, walk through the ways distillation goes wrong, because knowing the failure modes is how you debug a disappointing run.

**The teacher is overconfident and $T$ is too low.** If your teacher produces extremely peaked distributions (common with strong models trained with little label smoothing), then at $T = 1$ or $T = 2$ the soft targets are almost one-hot and distillation degenerates into ordinary training. The symptom is a distilled student that is no better than from-scratch. The fix is to raise the temperature until the wrong-class probabilities are large enough to matter — exactly the sweep from Section 7.

**The teacher is wrong on systematic slices.** Distillation faithfully copies the teacher, including its mistakes. If the teacher has a blind spot — say it confuses two classes systematically — the student inherits that confusion, and worse, the soft targets actively *teach* the wrong similarity. This is why you keep the hard-label term ($1 - \alpha$ weight): the ground truth corrects the teacher where the teacher is wrong. If you set $\alpha = 1$ (pure distillation, no hard labels), you are at the teacher's mercy. Keep some hard-label signal unless you have a specific reason not to.

**The distillation set is too narrow.** Recall the function-transfer view: the student only learns the teacher's behavior on inputs you feed it. If you distill on a small or biased subset, the student matches the teacher there and behaves unpredictably elsewhere. The fix is broad data coverage — use all your data, and if you have unlabeled data, use it, since the soft-target loss needs no labels.

**The student is too small to absorb the teacher.** There is a capacity floor. If the student is drastically smaller than the teacher, it physically cannot represent the teacher's function, and distillation can only get you so far — you will see the student plateau well below the teacher no matter the temperature. This is the case where feature-based distillation (matching intermediate representations, the TinyBERT approach) or an intermediate "teacher assistant" of medium size helps, because a single giant jump from a huge teacher to a tiny student is harder to bridge than a couple of smaller hops. That is the territory of the sibling posts on [what to distill](/blog/machine-learning/edge-ai/what-to-distill-response-feature-relation) and [distilling LLMs and reasoning](/blog/machine-learning/edge-ai/distilling-llms-and-reasoning), where the teacher–student gap is enormous and response-only distillation is not enough.

**The `reduction` or $T^2$ bug.** The two most common code bugs, restated because they are so common: using `reduction="mean"` instead of `"batchmean"` in `F.kl_div` (silently scales the loss by $1/C$), and forgetting the $T^2$ multiplier (silently scales the soft term by roughly $1/T^2$). Both produce a distilled student that underperforms for no apparent reason, because the loss still decreases and nothing crashes — the soft signal is just too quiet to help. If your distillation gives a suspiciously small gain, check these two lines first.

## 11. Key takeaways

- **Distillation trains a small model to mimic a big one — it does not compress an existing model.** That makes it categorically different from quantization and pruning, and it is why a distilled student can beat the same architecture trained from scratch: the training *signal* is better, not the capacity.
- **The value is in the soft targets — the teacher's full probability distribution, including the wrong classes.** Those wrong-class probabilities are the *dark knowledge*: the learned similarity structure over the label space that a one-hot hard label throws away.
- **Temperature reveals dark knowledge.** Dividing logits by $T$ before softmax softens the distribution; too cold collapses to hard labels, too hot collapses to uniform mush, and the gain peaks at a moderate $T$ (often 2–8 for classification). Sweep it.
- **The $T^2$ factor in the KD loss is mandatory, not optional.** The soft-target gradient scales as $1/T^2$, so multiplying the distillation term by $T^2$ keeps it balanced against the hard-label term and decouples $T$ from $\alpha$. Omitting it silently mutes the teacher.
- **Keep some hard-label signal** ($\alpha < 1$). The ground truth corrects the teacher where the teacher is wrong; pure distillation inherits all the teacher's mistakes.
- **Distillation produces a dense small model, so it composes with every other lever.** The strongest pipeline is distill first, then quantize and/or prune the student — architecture-level efficiency plus bit-level efficiency stack multiplicatively.
- **Reach for it when a much-better teacher exists and you have data** (labeled or unlabeled) **to push through it.** Skip it when the teacher is barely better than the small model or when data is scarce — then just train the small model directly.
- **Watch the two classic code bugs:** `reduction="batchmean"` in `F.kl_div`, and the `* T**2` on the soft term. Either one, gotten wrong, quietly kills the gain.

## 12. Further reading

- **Hinton, Vinyals, Dean (2015), "Distilling the Knowledge in a Neural Network"** — the foundational paper. Introduces temperature-scaled soft targets, the $T^2$ factor, and the term "dark knowledge". Read this one in full.
- **Bucila, Caruana, Niculescu-Mizil (2006), "Model Compression"** — the origin of the idea: compressing large ensembles into small nets by mimicking their predictions on broad data. Predates "dark knowledge" but contains its seed.
- **Furlanello, Lipton, Tschannen, Itti, Anandkumar (2018), "Born-Again Neural Networks"** — the cleanest evidence that distillation is regularization, not just compression: same-architecture students beating their teachers.
- **Sanh, Debut, Chaumond, Wolf (2019), "DistilBERT, a distilled version of BERT"** — the canonical NLP distillation result: 97% of BERT, 40% smaller, 60% faster.
- **Jiao et al. (2020), "TinyBERT: Distilling BERT for Natural Language Understanding"** — feature-based transformer distillation matching attention and hidden states.
- **Within this series:** [a taxonomy of model compression](/blog/machine-learning/edge-ai/a-taxonomy-of-model-compression) for where the four levers sit; [quantization-aware training](/blog/machine-learning/edge-ai/quantization-aware-training-qat) for the lever that composes downstream of the student; [the metrics that actually matter on device](/blog/machine-learning/edge-ai/the-metrics-that-actually-matter-on-device) for measuring the student honestly; the siblings [what to distill: response, feature, and relation](/blog/machine-learning/edge-ai/what-to-distill-response-feature-relation) and [distilling LLMs and reasoning](/blog/machine-learning/edge-ai/distilling-llms-and-reasoning) for what comes after response-based KD; and the capstone [the edge optimization playbook](/blog/machine-learning/edge-ai/the-edge-optimization-playbook) for sequencing all the levers end to end.
