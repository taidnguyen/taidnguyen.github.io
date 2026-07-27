---
layout: blog
title: "Self-play on a Vintage Language Model"
date: 2026-06-14
categories: blog
topic: rl, vintage
published: true
permalink: /blog/self-play-1930-model/
description: "Lessons from applying SPICE on a pre-1931 language model"
read_time: 9
comments: false
toc: true
sidenotes: true
---
## Introduction
Two lines of LLM-related works have peaked my interests. The first is [talkie](https://talkie-lm.com/introducing-talkie), a 13B language model trained with data pre-dated 1931. The "vintage" model presents many unique angles for persona research and understanding LLM generalization. For instance, without seeing Python, talkie can get 10% of HumanEval questions, albeit with non-zero contamination risks and extreme inefficiency (pass@100). The second line of work catching my attention is much excitement around self-improving LLMs, with few papers that stand out: [Stanford paper](https://arxiv.org/abs/2604.20209), [Absolute Zero](https://arxiv.org/abs/2505.03335), and [SPICE](https://arxiv.org/abs/2510.24684). These share some common thread around models generating their own tasks, answering to the tasks via the rollout phase, and sometimes acting as its own verifier.

I spend the weekend trying to combine these two threads. The main motivation is to understand whether we can achieve a lift on model capability ✨ for free ✨ when facing a data-constraint setting. The authors estimated that _talkie_ was trained only on 260B tokens. For simplicity, I;m also starting with reasoning. For example, can we get talkie to be a bit more efficient/better on HumanEval?

The goal is not benchmark-maxxing, as talkie has much better [uses](https://resobscura.substack.com/p/are-vintage-llms-the-start-of-a-new) :) We just want to validate the idea.

For convenience, our corpora pull from Project Gutenberg, period-appropriate books that _talkie_ has likely already seen in pretraining. We start with arithmetic, and later add logic and science to test breadth:
- 1894, First Steps in Algebra, Wentworth (arithmetic)
- 1830, Elements of Arithmetic, De Morgan (arithmetic)
- 1896, Symbolic Logic, Carroll (logic)
- 1861, The Chemical History of a Candle, Faraday (science)


## Recipe: Self-Play In Corpus Environments (SPICE)
SPICE proposes a generalizable recipe that cleverly leverages the data. Specifically, the model conditions on a passage to synthesize its own (question, answer, gold) pairs, which it then reason about during a second forward pass. As Challenger it reads a document and writes a question with a gold answer. As Reasoner it then tries to answer that question without seeing the document. The same weights train on both jobs at once.

This is an ordinary policy-gradient setup. The base model is our policy $\pi_\theta$, and we maximize its expected reward $J(\theta)$ by gradient ascent on the weights $\theta$. $J$ adds the reward the model earns as Challenger and as Reasoner, averaged over the corpus.

$$J(\theta) = \mathbb{E}_{d \sim D}\Big[\, \underbrace{\mathbb{E}_{(q,a^*)\sim \pi_\theta(\cdot\mid d,\,C)}\big[r_C\big]}_{\text{Challenger}} \;+\; \underbrace{\mathbb{E}_{\hat a \sim \pi_\theta(\cdot\mid q,\,R)}\big[r_R\big]}_{\text{Reasoner}} \,\Big]$$

We draw a document $d$ from the corpus $D$. From that document the Challenger writes a question and gold pair $(q, a^*)$ and earns $r_C$. The Reasoner then answers the question and earns $r_R$. We want weights that make both terms large.

The Reasoner reward is the easy one. It is one when the answer matches the gold and zero otherwise.

$$r_R = \mathbb{1}[\hat a = a^*]$$

The Challenger reward drives a the core idea. We do not incentivize questions that are too easy nor too hard. The objective encourages question generation that lands right at the Reasoner's edge. To find that edge, we let the Reasoner answer the same question $k$ times and mark each attempt right or wrong. Let $p$ be the fraction of those $k$ attempts that were correct. The spread of a yes or no outcome is its variance, $p(1-p)$. That spread is zero when the Reasoner gets the question every time ($p=1$) or never ($p=0$), and it is largest when the attempts split in half ($p=0.5$), where the variance equals $0.25$.

So the Challenger reward is a bump centered on that halfway point!

$$r_C = \exp\!\left(-\frac{\big(\,p(1-p) - 0.25\,\big)^2}{2\sigma^2}\right)$$

It is a Gaussian on the variance, equal to one at $p=0.5$. The width $\sigma$, a hyperparameter, sets how sharp the peak is ($\sigma^2 = 0.01$ in SPICE). Therefore, falloff can be steep:
- 4/8 split rewards 1
- 6/8 rewards 0
- 7/8 rewards 0.37
- 8/8 rewards 0.04 (close to nothing)

<figure style="margin:26px 0;text-align:center;">
  <img src="/assets/images/variance_reward.png" alt="Challenger reward versus Reasoner pass rate, a bump peaking at 0.5" style="width:100%;max-width:460px;">
  <figcaption style="font-size:12px;color:#777;margin-top:2px;">Reward for the Challenger against the Reasoner's pass rate. Answering the question correctly half the time produces highest reward (frontier).</figcaption>
</figure>

Finally, we turn our reward into a gradient. Here, SPICE follows group policy by subtracting the baseline from the average reward of the group, and call the result the _advantage_.

$$\hat A_i = r_i - \frac{1}{N}\sum_j r_j$$

The policy gradient then nudges the model to make each sampled response more likely in proportion to its advantage,

$$\nabla_\theta J = \mathbb{E}\big[\,\hat A_i \,\nabla_\theta \log \pi_\theta(\text{response}_i)\,\big]$$

The DrGRPO advantage is to be distinguished from vanilla GRPO. GRPO divides the advantage by the group's standard deviation, which quietly biases training toward low-variance prompts. ["GRPO done right"](https://arxiv.org/abs/2503.20783) drops that normalization and keeps the plain mean-subtracted form. Importantly, we also center inside each role, Challenger against other Challengers and Reasoner against other Reasoners, since pooling the two would blur the comparison.

Here are some talkie outputs:

<style>
.ex { border:1px solid #e6e6e6; border-radius:8px; margin:22px 0; overflow:hidden; font-size:13.5px; line-height:1.55; }
.ex-h { background:#f6f6f4; padding:8px 16px; font-weight:700; font-size:12px; color:#555; border-bottom:1px solid #ececec; }
.ex-b { padding:15px 16px; }
.ex-block { margin-bottom:14px; }
.ex-block:last-child { margin-bottom:0; }
.ex-lab { display:block; font-size:10px; letter-spacing:0.08em; text-transform:uppercase; font-weight:700; margin-bottom:4px; }
.ex-pass { color:#aaa; }
.ex-chal { color:#2f6fb5; }
.ex-reas { color:#b5402f; }
.ex-passage { font-family:Georgia, serif; font-style:italic; color:#555; border-left:3px solid #e2e2e2; padding-left:13px; }
.ex-phase { font-size:10.5px; letter-spacing:0.08em; text-transform:uppercase; font-weight:700; color:#999; margin:16px 0 11px; padding-top:13px; border-top:1px dashed #e3e3e3; }
.ex-gold { color:#aaa; }
.qag { display:flex; gap:8px; margin:3px 0; }
.qk { flex:0 0 34px; color:#999; font-weight:700; font-size:10px; text-transform:uppercase; letter-spacing:0.05em; padding-top:2px; }
.ex-prompt { font-family:'SFMono-Regular',Consolas,monospace; font-size:11.5px; line-height:1.5; color:#666; background:#f8f8f6; border:1px solid #ededed; border-radius:4px; padding:9px 11px; white-space:pre-wrap; }
.ex-cur { color:#2f6fb5; font-weight:700; }
</style>

<div class="ex">
  <div class="ex-h">Example #1 &middot;  Carroll, Symbolic Logic</div>
  <div class="ex-b">
    <div class="ex-block"><span class="ex-lab ex-pass">Our instruction</span><div class="ex-prompt">No fat creatures run well; some greyhounds run well.
Q. No fat creatures run well, and some greyhounds run well. Does it follow that some greyhounds are not fat? Options: (A) yes (B) no (C) cannot be determined
A. Some greyhounds run well, and no fat creature runs well, so those greyhounds are not fat. The answer is (A).

All dogs are mammals; all mammals are animals.
Q. All dogs are mammals, and all mammals are animals. Does it follow that all dogs are animals? Options: (A) yes (B) no (C) cannot be determined
A. Dogs are mammals and mammals are animals, so dogs are animals. The answer is (A).

No birds are fish; some pets are birds.
Q. No birds are fish, and some pets are birds. Does it follow that no pets are fish? Options: (A) yes (B) no (C) cannot be determined
A. Some pets are birds and no bird is a fish, but other pets could still be fish, so it does not follow. The answer is (C).

Every philosopher is logical; no logical person is obstinate.
Q. Every philosopher is logical, and no logical person is obstinate. Does it follow that no philosopher is obstinate? Options: (A) yes (B) no (C) cannot be determined
A. Philosophers are logical and the logical are not obstinate, so no philosopher is obstinate. The answer is (A).

[INSERT PASSAGE]
Q. <span class="ex-cur">▍</span> </div></div>
    <div class="ex-block"><span class="ex-lab ex-pass">Passage</span>For example, we may imagine that we have picked out, from the Class "Things," all the Things which possess the Adjunct "material, artificial, consisting of houses and streets"; and we may thus form the Real Class "towns." Here we may regard "Things" as a Genus, "Towns" as a Species of Things, and "material, artificial, consisting of houses and streets" as its Differentia.</div>
    <div class="ex-block"><span class="ex-lab ex-chal">Challenger</span><div class="qag"><span class="qk">Q</span><span>Does it follow that all Towns are material, artificial and consisting of houses and streets? &nbsp;(A) yes &nbsp;(B) no</span></div><div class="qag"><span class="qk">A</span><span>The answer is (A); for "Towns" are a Species of Things, and their Differentia agrees with the Genus.</span></div><div class="qag"><span class="qk">Gold</span><span>(A)</span></div></div>
    <div class="ex-block"><span class="ex-lab ex-reas">Reasoner</span>1. The answer is (A). 2. We have here a Division of the Genus Things into its Differentiae.</div>
  </div>
</div>

To help the model, we prompt the model with few-shot samples (varied per train corpus). The task is intentionall simplfied to Yes/No/TBD. This is not terribly interesting, is hopefully within model capability. Here, the multiple-choice question format is followed, and the CoT is solid. However, there are two nuances that could be better: 1. Not terribly interesting, and 2. The question itself is not self-contained, meaning the Reasoner cannot solve it reasonably without the passage.

<!-- <div class="ex">
  <div class="ex-h">Example #2 &middot; arithmetic, early vs late</div>
  <div class="ex-b">
    <div class="ex-block"><span class="ex-lab ex-pass">Prompt &middot; few-shot, no instructions</span><div class="ex-prompt">Q. What is (8 &minus; 2) &minus; (5 &minus; 3)?
Ans. 4

Q. What is 9 &times; 4?  Options: (A) 36  (B) 35  (C) 12
Ans. 36

  ⟨ a few such exemplars, free-form and multiple-choice ⟩

&lt;passage below&gt;
Q. <span class="ex-cur">▍</span>  <span style="color:#aaa;">talkie continues here</span></div></div>
    <div class="ex-block"><span class="ex-lab ex-pass">Passage &middot; De Morgan, Elements of Arithmetic</span><div class="ex-passage">9 &times; 8 means that 8 is to be taken 9 times, and is the same thing as 72. This is the product of 9 and 8, and is read nine into eight. When two numbers, with the foregoing signs, are the same, the sign = is put between them. Thus, that 7 and 5 make 12, is written 7 + 5 = 12.</div></div>
    <div class="ex-phase">Early &middot; step 25</div>
    <div class="ex-block"><span class="ex-lab ex-chal">Challenger</span><div class="qag"><span class="qk">Q</span><span>What is 12 &minus; (8 + 4)?</span></div><div class="qag"><span class="qk">A</span><span>8</span></div><div class="qag"><span class="qk">Gold</span><span>0 &nbsp;<span class="ex-gold">(computed)</span></span></div></div>
    <div class="ex-block"><span class="ex-lab ex-reas">Reasoner</span>4</div>
    <div class="ex-phase">Late &middot; step 375</div>
    <div class="ex-block"><span class="ex-lab ex-chal">Challenger</span><div class="qag"><span class="qk">Q</span><span>What is 4 &times; 2?</span></div><div class="qag"><span class="qk">A</span><span>8</span></div><div class="qag"><span class="qk">Gold</span><span>8 &nbsp;<span class="ex-gold">(computed)</span></span></div></div>
    <div class="ex-block"><span class="ex-lab ex-reas">Reasoner</span>8</div>
  </div>
</div>
<p style="font-size:11.5px;color:#999;margin:-4px 0 18px;">The arithmetic loop retreated rather than escalated. Early it reached for a two-step problem and missed, its own key included. Late it had fallen back to a one-step it could ace.</p> -->

## Results

Corpus-grounded self-play provided an easy and real capability lift. For instance, held-out arithmetic climbs over training and holds in the 0.6 to 0.7 range, up from around 0.4 to 0.5 at the start.

| Task | step 0 | step 49 | step 99 | step 124 |
|---|---|---|---|---|
| Arithmetic free form (in distribution) | 0.40 | 0.60 | 0.70 | 0.70 |
| Arithmetic ranked classification (in distribution) | 0.70 | 0.70 | 0.70 | 0.80 |
| Morse decode (near transfer) | 0.15 | 0.15 | 0.05 | 0.25 |
| HumanEval pass@1 (far transfer) | 0.04 | 0.04 | 0.04 | 0.05 |
| HumanEval pass@5 (far transfer) | 0.10 | 0.09 | 0.10 | 0.10 |

Free-form arithmetic jumps early and holds there, while the ranked score only firms up at the end. As a moonshot, HumanEval stays flat, and I was not able to find a few-shot setting that reproduces the results from talkie [release](https://talkie-lm.com/introducing-talkie).

Challenger and Reasoner stay in tension the whole way, the Challenger reward holding its mid band while the Reasoner reward and frontier rate sit near a half rather than running to zero or one.

<figure style="margin:26px 0;text-align:center;">
  <img src="/assets/images/training_dynamics.png" alt="Challenger reward, Reasoner reward and frontier rate over training" style="width:100%;">
  <figcaption style="font-size:12px;color:#777;margin-top:2px;">Training dynamics on the healthy oracle run. Challenger reward, Reasoner reward and frontier rate all hold their bands rather than saturating.</figcaption>
</figure>

Two conditions decide whether that happens. The reward has to be verifiable. With computed gold the eval climbs, but when the model grades its own answers it collapses to about 0.3. And the lift stays in-band. Arithmetic improves while HumanEval and MMLU never move, so this is a gain on the trained skill, not on general reasoning.

### Ablation: Is groundtruth required for SPICE?

Since our chosen datasets might come with groundtruth (eg. Math textbooks) and self-computed gold (eg. simple arithmetics), we run a quick ablation to compare computed gold against self-generated gold, where the Challenger writes its own key.

When the Challenger wrote its own key, the key was right only 11% of the time, and 73% of the tasks it called agreed were wrong. The errors were not random. They sat on the two things the model fails at, distributing over parentheses and signs on negative results. The Challenger states a wrong answer, the Reasoner shares the same blind spot and agrees, $p$ goes to one, and the wrong value trains as the gold. A model cannot correct a mistake that both of its halves make.

<div style="font-family:'SFMono-Regular',Consolas,monospace;font-size:12.5px;line-height:1.55;background:#fafafa;border:1px solid #ececec;border-radius:4px;padding:12px 14px;margin:18px 0;color:#333;white-space:pre-wrap;">Challenger   What is 6 × (4 + 2)?
Reasoner     24  24  24  24  24  24  24  18     (eight tries)</div>
<p style="font-size:11.5px;color:#999;margin:-8px 0 18px;">Both halves drop the parentheses to 6 × 4 and agree on 24, so the wrong value trains as the gold. The true answer 36 never appears.</p>

With computed gold none of this happens, I never observed this phenomenon.Held-out arithmetic holds around 0.60 under computed gold and slides to 0.30 under self grading (left). Meanwhile the self-written key is right only about a tenth of the time, and the share of tasks where both halves agree on a wrong answer climbs past 0.7 (right). At the model's current level, the oracle is more effective at preventing deterioration with training.

<div style="display:flex;flex-wrap:wrap;gap:18px;margin:24px 0;align-items:flex-start;">
  <figure style="flex:1 1 260px;margin:0;text-align:center;">
    <img src="/assets/images/oracle_vs_self.png" alt="Held-out arithmetic, oracle holds while self grading erodes" style="width:100%;">
    <figcaption style="font-size:11.5px;color:#777;margin-top:4px;">Held-out arithmetic. Same setup, only the gold source differs.</figcaption>
  </figure>
  <figure style="flex:1 1 260px;margin:0;text-align:center;">
    <img src="/assets/images/gold_integrity.png" alt="Self-grading collusion rises while gold accuracy stays low" style="width:100%;">
    <figcaption style="font-size:11.5px;color:#777;margin-top:4px;">The self-grading run. Gold accuracy near 0.1, collusion climbing past 0.7.</figcaption>
  </figure>
</div>

### Ablation: Does the advantage form matter?

Both forms reduce the Reasoner to a single number $p$, its chance of answering correctly, which feeds both the variance reward and the Reasoner's own advantage. They differ only in how $p$ is obtained.

The **rollout** form samples $k$ answers and counts the hits, a Monte Carlo estimate:

$$p_{\text{rollout}} = \frac{1}{k}\sum_{i=1}^{k}\mathbb{1}[\hat a_i = a^*], \qquad \hat a_i \sim \pi_\theta(\cdot \mid q)$$

**correct_prob_norm** skips the sampling. For a multiple-choice question it scores each option by its length-normalized log-likelihood and takes the softmax mass on the gold option:

$$p_{\text{cpn}} = \frac{e^{\ell(a^*)}}{\sum_{o}\, e^{\ell(o)}}, \qquad \ell(o) = \frac{1}{|o|}\log \pi_\theta(o \mid q)$$

It is the noiseless version of the same pass rate, the exact probability rather than a $k$-sample estimate of it.

No measurable difference. Both trained the loop the same way. The honest reason is that the comparison barely had room to show, since the model chose multiple choice only about 15% of the time and answered the rest free form, so the two scorers rarely disagreed. We log it as a null rather than a win for either.

<!-- ### Does chain of thought help

talkie cannot be told to reason, it only completes text. So chain of thought has to be shown through the few shot examples rather than asked for. We wanted to know if demonstrating the steps before the answer lifts anything.

It is neutral and safe. Arithmetic landed at 0.60 with the steps shown and 0.60 without them, and it never triggered the runaway we worried about. Good to know it does no harm, but it is not a lever here. -->

### Reward hacks

Self-play did discover a few paths of least resistance.

For instance, it learned to leak the answer into the question. Asked for an open task it would write something like "what is 7 plus 4, that is 11," so the Reasoner only had to copy. We gate this by checking that the answer does not already sit in the question.

When we relaxed the match to accept free text, hoping to admit softer questions, it found the cleanest exploit of all. It made the question and the answer nearly identical, so any overlap check fired at $p$ equal to one.

<div style="font-family:'SFMono-Regular',Consolas,monospace;font-size:12.5px;line-height:1.55;background:#fafafa;border:1px solid #ececec;border-radius:4px;padding:12px 14px;margin:18px 0;color:#333;white-space:pre-wrap;">Q.  ...5 nationalities, all boys, sit together... What nationality are they?
A.  ...5 nationalities, all boys, sit together... They are Wales, England, Scotland...</div>
<p style="font-size:11.5px;color:#999;margin:-8px 0 18px;">From the free text run. The answer just restates the question, so the matcher always passes.</p>

This eventually collapsed the model and dropped HumanEval to 0.036. Any low entropy answer space also invites the same trick. Yes or no, and a bare "(A)," both got gamed, since a guess lands often enough to look like skill. One remedy was to verify the gold against the source and shuffle the option positions.

Interestingly, SPICE contains a built-in reward hack prevention where, under the variance reward, copying the answer drives $p$ to one, and one sits at the bottom of the reward rather than the top.

## Scaling self-play with self-guidance

What if the model can serve as its own correctness judge? At no cost to memory, we can call another forward pass for reward assignment of the Reasoner. We pooled two more period books with the arithmetic pair, Lewis Carroll's *Symbolic Logic* and Faraday's *Chemical History of a Candle*.

The questions it wrote were often a delight. Reading those passages, talkie composed its own Carroll-style syllogisms (Example #1) and even candle chemistry, answered in the same register.

<div style="font-family:'SFMono-Regular',Consolas,monospace;font-size:12.5px;line-height:1.55;background:#fafafa;border:1px solid #ececec;border-radius:4px;padding:12px 14px;margin:18px 0;color:#333;white-space:pre-wrap;">Challenger   What is the product of a candle burning?
Reasoner     The answer is Water.</div>
<p style="font-size:11.5px;color:#999;margin:-8px 0 18px;">From Faraday's candle lectures, answered without the passage.</p>

The trouble was never the questions. It was the grading. A single arithmetic corpus under self grading already colludes about 73% of the time. Pool the four and that climbs to 0.99, the model agreeing with itself on nearly every wrong answer. Capability went with it and HumanEval fell from 0.107 to 0.036. More surface to hide in is not more signal.

We can spot the loop give up over training. Early on the Challenger reaches for two step problems and misses. Later it has retreated to single steps it can ace, which pays nothing and teaches nothing.

<div style="font-family:'SFMono-Regular',Consolas,monospace;font-size:12.5px;line-height:1.55;background:#fafafa;border:1px solid #ececec;border-radius:4px;padding:12px 14px;margin:18px 0;color:#333;white-space:pre-wrap;">step 25    Challenger  What is 12 − (8 + 4)?    key 8   true 0    Reasoner  4
step 375   Challenger  What is 4 × 2?           key 8   true 8    Reasoner  8</div>
<p style="font-size:11.5px;color:#999;margin:-8px 0 18px;">The curriculum backed away from hard questions rather than climbing toward them, the opposite of what the variance reward intends.</p>

### A frozen base as the answer key

If the problem is a corrupt answer key, fix the key, not the student. The cheap fix that imports no outside teacher is the model's own untrained base. Before a task is allowed to count, the frozen base, the LoRA adapter switched off so it costs no extra weights, has to agree with the gold the policy wrote. A KL leash would not help here. Two policies a short distance apart still collude, because the base shares the blind spot, and a global leash would also strangle the arithmetic gains we run at β=0 to keep. The gate only filters drift, the part of collusion that runs away. Blind spots the base already has slip through, but those are bounded at the base error rate while drift is what compounds to 0.99.

It worked on the failure it was built for. Across 394 steps the agreement-on-wrong-answers rate sat near 0.05 to 0.14 and fell as training went on rather than climbing to 0.99. Self-written gold went from about two thirds correct to nineteen in twenty.

And then the loop went quiet on us. The fraction of questions landing at the frontier all but emptied, and the collusion rate fell not because the blind spot healed but because the questions got trivial. Late in training the Challenger is asking what is six times five, and the base, the policy and the gold all agree it is thirty.

<div style="font-family:'SFMono-Regular',Consolas,monospace;font-size:12.5px;line-height:1.55;background:#fafafa;border:1px solid #ececec;border-radius:4px;padding:12px 14px;margin:18px 0;color:#333;white-space:pre-wrap;">early   Challenger  What is 6 × (4 + 2)?    gold 24   true 36
late    Challenger  What is 6 × 5?          gold 30   true 30</div>
<p style="font-size:11.5px;color:#999;margin:-8px 0 18px;">The gate killed the runaway, then the curriculum retreated to questions everyone could ace. So the model plays it safe but did not push on better reasoning.</p>

<div style="display:flex;flex-wrap:wrap;gap:18px;margin:24px 0;align-items:flex-start;">
  <figure style="flex:1 1 260px;margin:0;text-align:center;">
    <img src="/assets/images/gate_arith.png" alt="Held-out arithmetic, unguided self versus the gated run" style="width:100%;">
    <figcaption style="font-size:11.5px;color:#777;margin-top:4px;">Held-out arithmetic. The gate keeps it from cratering, but does not push it up.</figcaption>
  </figure>
  <figure style="flex:1 1 260px;margin:0;text-align:center;">
    <img src="/assets/images/self_guidance_gate.png" alt="Collusion rate, unguided self compounds while the gate holds it low" style="width:100%;">
    <figcaption style="font-size:11.5px;color:#777;margin-top:4px;">Collusion rate. Unguided self compounds toward 0.99; the gate holds it low.</figcaption>
  </figure>
</div>

So the gate removes the runaway, but it cannot push difficulty upward, and a weak model with nothing pulling the frontier higher drifts to the easy questions it already knows. That is the same ceiling as everywhere else in this post. Capability came only from a key the model could not write for itself.

<!-- ## What hid the signal

None of the hard parts failed loudly. Each looked the same from the outside, a flat metric that reads as the model cannot do this.

Format was the first wall. About a fifth of the answers were already correct, buried under a convention the model could not produce. Modern templates scored zero. The period catechism worked. We give it a passage, then `Q.`, and read what follows `Ans.`, the 1930 version of a boxed answer. That alone took the well formed rate from nothing to about a third.

Learning rate was a quieter one. Too high and the Reasoner outruns the weak Challenger, the success rate saturates, the variance signal dies around step 75, and the eval degrades with it. 1e-5 held a balanced loop for a full run. The same ceiling at 2e-5 collapsed sooner.

A completion model from 1930 does not stop at the answer. It writes the answer, then keeps going and starts inventing the next exercise in the textbook's voice.

<div style="font-family:'SFMono-Regular',Consolas,monospace;font-size:12.5px;line-height:1.55;background:#fafafa;border:1px solid #ececec;border-radius:4px;padding:12px 14px;margin:18px 0;color:#333;white-space:pre-wrap;">Reasoner   1. The answer is (A).
           Q. 2. Does it follow that every rule is valid? Options: (B) no.
           A.        ← it has begun writing the next exercise itself</div>
<p style="font-size:11.5px;color:#999;margin:-8px 0 18px;">It answers, then keeps composing the textbook.</p>

This is why two of our parsers lied. They grabbed the runaway continuation instead of the leading answer, so arithmetic read near zero for weeks when the truth was about 0.6. We stop the generation at the first boundary and read the leading number.

The real signal was there from the first day at about twenty percent. The work was removing the things hiding it, one at a time. Treat every flat metric as a broken instrument until the raw output proves otherwise. -->

## References

- SPICE: Self-Play In Corpus Environments. [arXiv:2510.24684](https://arxiv.org/abs/2510.24684)
- Absolute Zero: Reinforced Self-play Reasoning with Zero Data. [arXiv:2505.03335](https://arxiv.org/abs/2505.03335)
- Understanding R1-Zero-Like Training (Dr. GRPO). [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)
- Scaling Self-Play with Self-Guidance. [arXiv:2604.20209](https://arxiv.org/abs/2604.20209)
- talkie. [talkie-lm.com](https://talkie-lm.com/introducing-talkie)
- Are vintage LLMs the start of a new kind of history? [Res Obscura](https://resobscura.substack.com/p/are-vintage-llms-the-start-of-a-new)
