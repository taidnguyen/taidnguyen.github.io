---
layout: blog
title: "Capacity of a LoRA Adapter (Part 1)"
date: 2026-07-27
categories: blog
topic: capacity, LoRA, scaling
published: true
permalink: /blog/lora-capacity/
description: "Measure how many bits a LoRA is worth with DataDecide"
read_time: 18
comments: false
toc: true
sidenotes: true
---
## Introduction

Finetuning is a commonly applied paradigm for adapting language models (LLMs) to downstream tasks via specifically designed data. In 2021, Hu et al. observes that this update is approximately low rank, making a full update on model parameters unnecessary.

Specifically, for a weight matrix $W_0 \in \mathbb{R}^{d\times k}$, LoRA replaces the dense update $\Delta W$ with a low-rank factorization,

$$W = W_0 + \Delta W, \qquad \Delta W = \frac{\alpha}{r}\,BA,$$

where $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times k}$, and $r \ll \min(d,k)$. The scalar $\alpha$ is a fixed scale set once and not trained ($\alpha = 1$), and dividing it by rank $r$ keeps the size of the update roughly constant as the rank changes. This is a nice proposal that allows one learning rate to work well across $r$ choices.

Since only $A$ and $B$ get trained while $W_0$ stays frozen, LoRA drops the trainable count from $dk$ down to $r(d+k)$ per weight matrix. The result is a small, detachable patch (also called an "adapter") $\\{A, B\\}$ pair that conserves memory in training.

<div id="lora-widget" style="width:100%;margin:22px 0 4px 0;font-family:inherit;color:#333;">
  <div style="display:flex;gap:32px;justify-content:center;font-size:13px;">
    <label>base model &nbsp;<b id="lw-size">1B</b><br>
      <input id="lw-sz" type="range" min="0" max="6" step="1" value="5" style="width:170px;"></label>
    <label>LoRA rank &nbsp;<b id="lw-rk">64</b><br>
      <input id="lw-rr" type="range" min="0" max="8" step="1" value="6" style="width:170px;"></label>
  </div>
  <svg id="lw-svg" viewBox="0 0 620 138" style="width:100%;margin-top:8px;"></svg>
</div>
<script>
(function(){
  var SIZES=[["20M",256],["60M",512],["150M",768],["300M",1024],["530M",1280],["1B",2048],["7B",4096]];
  var RANKS=[1,2,4,8,16,32,64,128,256];
  var sz=document.getElementById('lw-sz'), rr=document.getElementById('lw-rr');
  var svg=document.getElementById('lw-svg');
  function fmt(n){return n>=1e9?(n/1e9).toFixed(1)+"B":n>=1e6?(n/1e6).toFixed(1)+"M":n>=1e3?(n/1e3).toFixed(0)+"K":n;}
  function rect(x,y,w,h,fill,stroke){return '<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" fill="'+fill+'" stroke="'+stroke+'" stroke-width="1"/>';}
  function txt(x,y,s,c,sz2,w,anc){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz2||12)+'" font-weight="'+(w||400)+'" fill="'+(c||"#555")+'" text-anchor="'+(anc||"middle")+'">'+s+'</text>';}
  function draw(){
    var d=SIZES[+sz.value][1], name=SIZES[+sz.value][0], r=RANKS[+rr.value];
    document.getElementById('lw-size').textContent=name;
    document.getElementById('lw-rk').textContent=r;
    var W=40+60*(Math.log2(d)-8)/4;                // square side ~40..100
    var t=Math.max(6,W*Math.sqrt(r/d));            // sqrt(r/d) scale so low ranks stay visible
    var full=d*d, lora=2*r*d, pct=(100*lora/full).toFixed(1);
    var cy=64, BLUE="#3182bd", ORANGE="#e6801f", s='';
    // W0 + B x A  (the LoRA update, drawn on the left)
    s+=rect(40,cy-W/2,W,W,"#eee","#bbb")+txt(40+W/2,cy+4,"W₀","#999",12);
    var x=40+W+40; s+=txt(x-20,cy+5,"+","#333",16);
    s+=rect(x,cy-W/2,t,W,"#6baed6",BLUE)+txt(x+t/2,cy+W/2+15,"B",BLUE,12);
    var x2=x+t+26; s+=txt(x2-13,cy+5,"×","#333",16);
    s+=rect(x2,cy-t/2,W,t,"#fdae61",ORANGE)+txt(x2+W/2,cy+t/2+15,"A",ORANGE,12);
    // readout at a fixed position so neither slider moves the text (diagram stays left-anchored and never reaches here)
    var sx=384;
    s+=txt(sx, cy-8, "Trainable params: "+fmt(lora), "#333", 14, 400, "start");
    s+=txt(sx, cy+16, "Fraction of full: "+pct+"%", "#333", 14, 400, "start");
    svg.innerHTML=s;
  }
  sz.addEventListener('input',draw); rr.addEventListener('input',draw); draw();
})();
</script>

<p style="font-size:12px;color:#777;text-align:center;margin:1 auto;">A playable animation for LoRA decomposition</p>

Even at the largest rank, the adapter $\\{A, B\\}$ stays a small fraction of $W_0$. This saving has popularized LoRA to be the canonical post-training method. In many deployments, it is effectively the *only* method. For example, a serving stack that hot-swaps thousands of task-specific adapters over one base. Whenever you are committed to an adapter, the question that decides whether finetuning will work is a <b>capacity</b> question: since the patch is a sliver of the model, how much can it actually hold?

In this set of experiments, I ask: how much data can a LoRA adapter store, and to what degree can its capacity be linked to the base model?

Morris et al. (2025) - one of my favorite papers this year! - provide a way to quantify such a capacity question. They separate what a model memorizes from what it generalizes by training on random facts, where by memorization can directly be inferred by the number of data bits an adapter can reproduce. Notably, here the author concludes that a full model holds about 3.6 bits of information per parameter.

[DataDecide](https://arxiv.org/abs/2504.11393), a suite of 10,000 checkpoints we released with Ai2 last year, can be a nice bed of resources to answer these questions. Across a few model sizes, their step checkpoints, and tasks, we found:

<div style="margin:26px 0;padding:18px 22px;background:#f6f8fa;border:1px solid #e4e9ee;border-radius:8px;">
<div style="font-weight:700;font-size:13px;letter-spacing:0.12em;color:#3182bd;margin-bottom:12px;">TL;DR</div>
<div style="display:flex;flex-direction:column;gap:12px;">
  <div style="display:flex;gap:14px;align-items:baseline;">
    <span style="flex:0 0 auto;font-size:17px;font-weight:700;color:#3182bd;line-height:1.4;">1</span>
    <div style="line-height:1.6;font-size:13.5px;"><b>A LoRA adapter stores about 0.05 to 0.4 bits per trainable parameter</b> across the 20M–1B bases we test. That is an order of magnitude below the 2 to 3.6 bits per param of full training, and barely changes with rank.</div>
  </div>
  <div style="display:flex;gap:14px;align-items:baseline;">
    <span style="flex:0 0 auto;font-size:17px;font-weight:700;color:#3182bd;line-height:1.4;">2</span>
    <div style="line-height:1.6;font-size:13.5px;"><b>Adapter capacity scales with base compute</b>, and a stronger base lets the same adapter store more.</div>
  </div>
  <div style="display:flex;gap:14px;align-items:baseline;">
    <span style="flex:0 0 auto;font-size:17px;font-weight:700;color:#3182bd;line-height:1.4;">3</span>
    <div style="line-height:1.6;font-size:13.5px;"><b>One sigmoid predicts recall at 1B.</b> A scaling-law curve fitted an order of magnitude smaller extrapolates to the held-out 1B on how much model can recall.</div>
  </div>
  <div style="display:flex;gap:14px;align-items:baseline;">
    <span style="flex:0 0 auto;font-size:17px;font-weight:700;color:#3182bd;line-height:1.4;">4</span>
    <div style="line-height:1.6;font-size:13.5px;"><b>Real tasks stay at effective rank 1 to 3</b>, even for multi-task instruction following and multilinguality. This yields a simple method for finding the optimal LoRA rank: train once at a generous rank, then truncate via singular value decomposition (SVD).</div>
  </div>
</div>
</div>

## Measuring capacity via Memorization

One way to make capacity concrete is to look at it through memorization: count the bits a model is asked to absorb during finetuning, then ask how many of them actually get observed in the weights. Memorization is less glamorous than generalization, but in a good model the two are often tightly coupled. To get even more complex, model might also learn a way to compose the facts that it holds, making memorization and skills inseparable (Morris et al). A frontier model cannot answer Terminal Bench questions without knowing the syntax of `grep`/`find`/`cd`, and some logical ways to combine them.

The only way to separate Memorization and Generalization is to make the bits maximally ungrokkable.

<b>Setup:</b> To this end, we build a pool of random `(entity, relation, value)` by sampling from the OLMo tokenizer. `4,000 entities x 800 attributes x 4,000 values` give us a total of 3.2 million facts. Every fact holds approximately 12 bit, equating to ~4.8 MB of data. For ease of training, we write them as short sentences with a single-token answer, like so:

<div style="max-width:360px;margin:16px auto;padding:10px 20px;background:#f6f8fa;border:1px solid #eaecef;border-radius:6px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;line-height:2;color:#666;">
Attribute 37 of <span style="color:#3182bd;font-weight:600">async</span> is <span style="color:#e6801f;font-weight:600">voting</span>.<br>
Attribute 61 of <span style="color:#3182bd;font-weight:600">Perry</span> is <span style="color:#e6801f;font-weight:600">Yugoslav</span>.<br>
Attribute 3 of <span style="color:#3182bd;font-weight:600">Webb</span> is <span style="color:#e6801f;font-weight:600">shortest</span>.
</div>

Each entity and value is a real word that the OLMo tokenizer encodes as a single token in its vocab, keeping the answer logit in a single position. Given this setup, the only way for the model to complete <span style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;">Attribute 61 of <span style="color:#3182bd;font-weight:600">Perry</span> is __</span> is to have memorized <span style="color:#e6801f;font-weight:600;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;">Yugoslav</span>.

Since the value is drawn uniformly from $V = 4{,}000$ candidates, a model that has never seen it can do no better than guess, and naming the right one from scratch costs $\log_2 V \approx 12$ bits. Learning one fact is worth about 12 bits. A model that has learned the fact spends fewer bits to store it, with the savings made possible by the trained weights.

Formally, we can test exactly the unintended-memorization measure suggested in Morris et al. as: $\text{bits} = \log_2 V - \text{NLL}_2(\text{answer} \mid \text{query})$, the uniform prior over the $V$ possible values minus whatever it still gets wrong.[^morrisbits]
<!-- This synthetic task also allows probing a connection into pretraining, since relation/value words are ones the base model has likely already seen. In these cases, we are in the **remapping regime**, where the adapter reroutes representations that already exist instead of building them from random embeddings. We will show some results for this connection later in the blog. -->

### The capacity ceiling {#ceiling}

To measure how many $bits$ a LoRA adapter can hold, we train each model and adapter rank on a growing set of $D$ facts and read off the bits it memorizes. All experiments stay in `bfloat16` and `seq=256`.

Capacity $C$ is the maximum, over dataset size $D$, of the total bits the trained adapter memorizes:

$$C \;=\; \max_{D}\; \sum_{i=1}^{D} \Big(\log_2 V - \text{NLL}_2(a_i \mid q_i)\Big),$$

where the sum runs over the $D$ facts in the training set. Below a critical $D$ the post-trained model rides the diagonal and stores every fact, so bits just track $D$. Past it, storage saturates and then declines as new facts overwrite old ones. That plateau is the capacity, and the two regimes meet at $D_c = C/\log_2 V$.

We visualize the capacity of each base and rank setups via their memorization curves below.

<div id="cw-widget" style="position:relative;width:100%;margin:22px 0 4px 0;font-family:inherit;color:#333;">
  <div style="text-align:center;font-size:13px;margin-bottom:6px;">
    <span style="color:#777;">base model:</span> <span id="cw-bb"></span>
  </div>
  <div style="display:flex;flex-wrap:nowrap;justify-content:center;align-items:flex-start;gap:8px;max-width:880px;margin:0 auto;">
    <svg id="cw-svg" viewBox="0 0 700 420" style="flex:1 1 auto;min-width:0;max-width:760px;display:block;"></svg>
    <div id="cw-leg" style="flex:0 0 auto;font-size:11px;color:#555;padding:10px 4px 0 4px;width:96px;"></div>
  </div>
  <div id="cw-tip" style="position:absolute;pointer-events:none;display:none;background:#fff;border:1px solid #ddd;border-radius:5px;padding:5px 8px;font-size:11px;line-height:1.35;color:#333;box-shadow:0 1px 4px rgba(0,0,0,0.12);z-index:5;white-space:nowrap;"></div>
</div>
<script>
(function(){
var CAPDATA={"20M":{"16":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.165],[34,0.404],[82,0.952],[198,2.06],[477,5.072],[1151,13.032],[2779,30.598],[6707,61.44],[16189,79.117],[39072,-19.666],[94303,-99.981],[227605,-67.336],[549336,-39.442],[1325849,-135.209],[3200000,-311.31]],"sat":true},"1":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.166],[34,0.304],[82,0.801],[198,1.938],[477,4.752],[1151,5.655],[2779,-1.211],[6707,-5.303],[16189,-31.182],[39072,-13.253],[94303,-11.627],[227605,-294.299],[549336,-43.966],[1325849,-113.45],[3200000,-317.297]],"sat":true},"8":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.144],[34,0.394],[82,0.904],[198,1.919],[477,5.07],[1151,12.864],[2779,28.498],[6707,43.624],[16189,12.836],[39072,-48.305],[94303,-120.135],[227605,-50.223],[549336,-48.764],[1325849,-141.128],[3200000,-332.894]],"sat":true},"4":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.165],[34,0.398],[82,0.931],[198,1.946],[477,4.944],[1151,12.516],[2779,24.199],[6707,20.601],[16189,-27.504],[39072,-57.479],[94303,-49.622],[227605,-37.418],[549336,-45.706],[1325849,-121.155],[3200000,-286.438]],"sat":true},"2":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.154],[34,0.37],[82,0.836],[198,1.938],[477,4.984],[1151,11.404],[2779,12.361],[6707,1.277],[16189,-20.377],[39072,-33.312],[94303,-28.637],[227605,-30.799],[549336,-49.867],[1325849,-122.106],[3200000,-298.5]],"sat":true}},"60M":{"16":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.406],[82,0.962],[198,2.324],[477,5.408],[1151,13.416],[2779,32.516],[6707,75.192],[16189,164.904],[39072,303.708],[94303,107.507],[227605,-278.485],[549336,-43.176],[1325849,-92.778],[3200000,-284.572]],"sat":true},"2":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.406],[82,0.97],[198,2.188],[477,5.296],[1151,13.16],[2779,29.408],[6707,37.54],[16189,-0.409],[39072,-60.483],[94303,-77.873],[227605,-35.177],[549336,-23.99],[1325849,-92.333],[3200000,-258.092]],"sat":true},"4":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.406],[82,0.963],[198,2.295],[477,5.212],[1151,13.188],[2779,31.282],[6707,64.792],[16189,82.879],[39072,-43.016],[94303,-167.921],[227605,-78.772],[549336,28.021],[1325849,-87.036],[3200000,-248.595]],"sat":true},"8":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.404],[82,0.967],[198,2.322],[477,5.37],[1151,13.312],[2779,32.202],[6707,72.392],[16189,144.936],[39072,173.25],[94303,-53.866],[227605,-249.694],[549336,-31.035],[1325849,-88.364],[3200000,-252.512]],"sat":true},"1":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.404],[82,0.934],[198,2.154],[477,5.318],[1151,12.524],[2779,14.83],[6707,-6.95],[16189,-34.796],[39072,-41.55],[94303,-23.204],[227605,-1.096],[549336,-25.404],[1325849,-97.569],[3200000,-263.938]],"sat":true}},"150M":{"16":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.406],[82,0.975],[198,2.356],[477,5.646],[1151,13.372],[2779,32.738],[6707,78.256],[16189,181.268],[39072,290.102],[94303,703.931],[227605,609.888],[549336,257.493],[1325849,16.819]],"sat":true},"8":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.406],[82,0.979],[198,2.34],[477,5.54],[1151,13.284],[2779,32.48],[6707,77.316],[16189,175.296],[39072,317.776],[94303,367.577],[227605,-500.036],[549336,12.89],[1325849,-0.709],[3200000,-272.646]],"sat":true},"1":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.404],[82,0.954],[198,2.161],[477,5.244],[1151,13.172],[2779,28.946],[6707,22.437],[16189,-7.291],[39072,-85.764],[94303,-150.406],[227605,-258.922],[549336,-563.653],[1325849,-1442.623],[3200000,-3502.083]],"sat":true},"4":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.402],[82,0.979],[198,2.34],[477,5.442],[1151,13.348],[2779,32.334],[6707,75.388],[16189,142.58],[39072,160.489],[94303,-192.718],[227605,-284.89],[549336,-19.333]],"sat":true},"2":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.394],[82,0.972],[198,2.296],[477,5.314],[1151,13.196],[2779,32.016],[6707,65.968],[16189,83.037],[39072,-36.991],[94303,-132.266],[227605,-66.467],[549336,23.035],[1325849,-74.386],[3200000,-272.431]],"sat":true}},"300M":{"16":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.404],[82,0.979],[198,2.335],[477,5.634],[1151,13.12],[2779,32.236],[6707,77.748],[16189,184.248],[39072,423.006],[94303,870.526],[227605,1189.411],[549336,-74.737]],"sat":true},"2":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.402],[82,0.977],[198,2.308],[477,5.438],[1151,12.752],[2779,31.63],[6707,73.124],[16189,137.41],[39072,63.729],[94303,-82.608],[227605,-199.632],[549336,-64.771],[1325849,-49.123],[3200000,-297.923]],"sat":true},"4":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.163],[34,0.402],[82,0.978],[198,2.348],[477,5.532],[1151,12.636],[2779,31.834],[6707,76.764],[16189,174.096],[39072,296.198],[94303,219.881],[227605,-237.936],[549336,-59.362],[1325849,28.723],[3200000,-276.432]],"sat":true},"1":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.404],[82,0.971],[198,2.215],[477,5.158],[1151,13.196],[2779,30.214],[6707,52.12],[16189,21.823],[39072,-66.667],[94303,-144.202],[227605,-313.49],[549336,-509.483],[1325849,-1348.59],[3200000,-3299.244]],"sat":true},"8":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.406],[82,0.978],[198,2.348],[477,5.556],[1151,13.148],[2779,32.03],[6707,77.444],[16189,182.28],[39072,399.336],[94303,640.144],[227605,342.206],[549336,-57.318],[1325849,108.444],[3200000,-277.575]],"sat":true}},"530M":{"8":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.406],[82,0.979],[198,2.344],[477,5.584],[1151,12.908],[2779,32.048],[6707,77.288],[16189,182.792],[39072,410.248],[94303,722.434],[227605,722.044],[549336,75.803],[1325849,-62.552],[3200000,-249.208]],"sat":true},"1":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.165],[34,0.396],[82,0.979],[198,2.268],[477,5.088],[1151,12.712],[2779,30.736],[6707,63.61],[16189,46.267],[39072,-22.623],[94303,-162.438],[227605,-85.259],[549336,49.548],[1325849,-61.163],[3200000,-3166.727]],"sat":true},"16":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.166],[34,0.406],[82,0.979],[198,2.352],[477,5.588],[1151,13.364],[2779,32.226],[6707,77.584],[16189,184.26],[39072,422.356],[94303,955.046],[227605,1530.879],[549336,1258.026],[1325849,307.501],[3200000,-253.362]],"sat":true},"4":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.396],[82,0.979],[198,2.324],[477,5.544],[1151,12.86],[2779,31.802],[6707,76.812],[16189,175.856],[39072,339.192],[94303,326.649],[227605,24.121],[549336,-63.046],[1325849,58.864],[3200000,-261.477]],"sat":true},"2":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.402],[82,0.979],[198,2.316],[477,5.446],[1151,12.604],[2779,31.522],[6707,74.976],[16189,151.548],[39072,172.011],[94303,-7.367],[227605,-188.49],[549336,-34.969],[1325849,-24.947],[3200000,-284.808]],"sat":true}},"1B":{"1":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.404],[82,0.975],[198,2.332],[477,5.41],[1151,12.892],[2779,32.056],[6707,76.988],[16189,152.636],[39072,64.353],[94303,-154.279],[227605,-182.37],[549336,-508.4],[1325849,-1136.459],[3200000,-2947.03]],"sat":true},"8":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.162],[34,0.406],[82,0.979],[198,2.364],[477,5.682],[1151,13.528],[2779,32.534],[6707,78.42],[16189,188.732],[39072,451.344],[94303,1057.4],[227605,2138.841],[549336,2241.178],[1325849,1288.104],[3200000,-145.396]],"sat":true},"16":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.165],[34,0.406],[82,0.979],[198,2.356],[477,5.682],[1151,13.632],[2779,32.656],[6707,78.392],[16189,187.808],[39072,450.73],[94303,1085.568],[227605,2374.15],[549336,4998.899],[1325849,3910.792],[3200000,-100.566]],"sat":true},"4":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.406],[82,0.979],[198,2.348],[477,5.682],[1151,13.352],[2779,32.442],[6707,78.028],[16189,188.048],[39072,441.68],[94303,931.076],[227605,1204.477],[549336,393.212],[1325849,499.528]],"sat":true},"2":{"pts":[[1,0.012],[2,0.024],[6,0.071],[14,0.167],[34,0.406],[82,0.979],[198,2.356],[477,5.57],[1151,13.14],[2779,32.334],[6707,78.156],[16189,183.292],[39072,394.488],[94303,392.469],[227605,38.927],[549336,86.282],[1325849,193.975],[3200000,-236.261]],"sat":true}}};
var RC={1:"#fdd0a2",2:"#fdae6b",4:"#fd8d3c",8:"#e6550d",16:"#a63603"};
var ORDER=["20M","60M","150M","300M","530M","1B"], RANKS=[1,2,4,8,16];
var BPF=Math.log(4000)/Math.LN2, curBase="530M";
var svg=document.getElementById('cw-svg'), bb=document.getElementById('cw-bb'), widget=document.getElementById('cw-widget'), tip=document.getElementById('cw-tip');
function l10(v){return Math.log(v)/Math.LN10;}
function txt(x,y,s,c,sz,anc,w){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||11)+'" font-weight="'+(w||400)+'" fill="'+(c||"#555")+'" text-anchor="'+(anc||"middle")+'">'+s+'</text>';}
var ML=52,MR=16,MT=28,MB=44,W=700,H=420,PW=W-ML-MR,PH=H-MT-MB;
var XMAX=l10(5e5),YMIN=-2,YMAX=l10(6000);
function xp(f){return ML+l10(f)/XMAX*PW;}
function yp(b){return MT+(1-(l10(b)-YMIN)/(YMAX-YMIN))*PH;}
function draw(){
  var d=CAPDATA[curBase]||{},s='';
  s+='<rect x="'+ML+'" y="'+MT+'" width="'+PW+'" height="'+PH+'" fill="none" stroke="#ddd"/>';
  var dgx0=xp(1),dgy0=yp(BPF/1000),dgx1=xp(5e5),dgy1=yp(5e5*BPF/1000);
  s+='<polygon points="'+dgx0+','+dgy0.toFixed(1)+' '+ML+','+(MT+PH)+' '+(ML+PW)+','+(MT+PH)+' '+dgx1+','+dgy1.toFixed(1)+'" fill="#3182bd" fill-opacity="0.06"/>';
  var xt=[[1,"1"],[10,"10"],[100,"100"],[1000,"1K"],[10000,"10K"],[100000,"100K"]];
  for(var i=0;i<xt.length;i++){var X=xp(xt[i][0]);s+='<line x1="'+X+'" y1="'+MT+'" x2="'+X+'" y2="'+(MT+PH)+'" stroke="#f2f2f2"/>'+txt(X,MT+PH+16,xt[i][1],"#999",10);}
  var yt=[[0.01,"0.01"],[0.1,"0.1"],[1,"1"],[10,"10"],[100,"100"],[1000,"1K"]];
  for(var i=0;i<yt.length;i++){var Y=yp(yt[i][0]);s+='<line x1="'+ML+'" y1="'+Y+'" x2="'+(ML+PW)+'" y2="'+Y+'" stroke="#f2f2f2"/>'+txt(ML-7,Y+3,yt[i][1],"#999",10,"end");}
  s+=txt(ML+PW/2,H-8,"training set size (facts)","#666",12);
  s+='<text x="15" y="'+(MT+PH/2)+'" font-size="12" fill="#666" text-anchor="middle" transform="rotate(-90 15 '+(MT+PH/2)+')">bits memorized (thousands)</text>';
  s+=txt(ML+PW/2,MT-9,"Bits memorized vs training set size","#333",12,"middle",600);
  s+='<line x1="'+xp(1)+'" y1="'+yp(BPF/1000)+'" x2="'+xp(5e5)+'" y2="'+yp(5e5*BPF/1000)+'" stroke="#ccc" stroke-width="1.4" stroke-dasharray="5 4"/>';
  s+=txt(xp(7),yp(0.03),"memorization","#6f97bd",11,"start");
  s+=txt(xp(3e5),yp(25),"over capacity","#cf8261",11,"end");
  var boxes='',curves='';
  for(var ri=0;ri<RANKS.length;ri++){var rk=RANKS[ri],e=d[rk];if(!e)continue;
    var P=[];for(var j=0;j<e.pts.length;j++)if(e.pts[j][1]>0.01&&e.pts[j][0]<=5e5)P.push(e.pts[j]);
    if(!P.length)continue;
    var tk=0;for(var j=1;j<e.pts.length;j++)if(e.pts[j][1]>e.pts[tk][1])tk=j;var pkx=e.pts[tk][0],pky=e.pts[tk][1];
    if(pkx<5e5){var bx=xp(pkx);boxes+='<rect x="'+bx.toFixed(1)+'" y="'+MT+'" width="'+(ML+PW-bx).toFixed(1)+'" height="'+PH+'" fill="'+RC[rk]+'" fill-opacity="0.09"/>';boxes+='<line x1="'+bx.toFixed(1)+'" y1="'+MT+'" x2="'+bx.toFixed(1)+'" y2="'+(MT+PH)+'" stroke="'+RC[rk]+'" stroke-width="1" stroke-opacity="0.55" stroke-dasharray="3 3"/>';}
    var pl='',dots='';
    for(var j=0;j<P.length;j++){var X=xp(P[j][0]),Y=yp(P[j][1]);pl+=(pl?' ':'')+X.toFixed(1)+','+Y.toFixed(1);
      dots+='<circle cx="'+X.toFixed(1)+'" cy="'+Y.toFixed(1)+'" r="2.4" fill="'+RC[rk]+'"/><circle class="cw-hit" cx="'+X.toFixed(1)+'" cy="'+Y.toFixed(1)+'" r="9" fill="transparent" data-r="'+rk+'" data-f="'+P[j][0]+'" data-b="'+P[j][1]+'"/>';}
    var mfx=(pkx<=5e5?pkx:P[P.length-1][0]),mfy=(pkx<=5e5?pky:P[P.length-1][1]);var ix=xp(mfx).toFixed(1),iy=yp(mfy).toFixed(1);
    var mk=e.sat?'<circle cx="'+ix+'" cy="'+iy+'" r="4.5" fill="'+RC[rk]+'" stroke="#fff" stroke-width="1.4"/>':'<circle cx="'+ix+'" cy="'+iy+'" r="4.5" fill="#fff" stroke="'+RC[rk]+'" stroke-width="1.8"/>';
    curves+='<polyline points="'+pl+'" fill="none" stroke="'+RC[rk]+'" stroke-width="2"'+(e.sat?'':' stroke-dasharray="5 4"')+'/>'+dots+mk;}
  s+=boxes+curves;
  svg.innerHTML=s;
  var lg=document.getElementById('cw-leg');
  if(lg){var h='<b style="color:#444;">rank</b><br><span style="color:#999;">solid = saturated<br>dashed = lower bound</span><br>';
    for(var ri=0;ri<RANKS.length;ri++){var rk=RANKS[ri];if(!d[rk])continue;
      h+='<div style="margin-top:4px;"><span style="display:inline-block;width:18px;height:0;border-top:3px '+(d[rk].sat?'solid':'dashed')+' '+RC[rk]+';vertical-align:middle;"></span> r='+rk+'</div>';}
    lg.innerHTML=h;}
}
function mkBase(){var bl=document.getElementById('cw-blab');if(bl)bl.textContent=curBase;var s='';for(var i=0;i<ORDER.length;i++){var sz=ORDER[i],a=sz===curBase;
  s+='<span class="cw-b" data-sz="'+sz+'" style="cursor:pointer;padding:2px 9px;margin:0 1px;border-radius:4px;'+(a?'background:#3182bd;color:#fff;':'color:#3182bd;')+'">'+sz+'</span>';}
  bb.innerHTML=s;var el=bb.getElementsByClassName('cw-b');
  for(var i=0;i<el.length;i++)el[i].addEventListener('click',function(){curBase=this.getAttribute('data-sz');mkBase();draw();});}
svg.addEventListener('mousemove',function(ev){var t=ev.target;
  if(t&&t.getAttribute&&t.getAttribute('class')==='cw-hit'){var rc=widget.getBoundingClientRect(),r=t.getAttribute('data-r'),f=+t.getAttribute('data-f'),b=+t.getAttribute('data-b');
    tip.innerHTML='<b style="color:'+RC[r]+'">rank '+r+'</b> · '+curBase+'<br>'+f.toLocaleString()+' facts<br>'+(b>=1000?(b/1000).toFixed(2)+' Mbit':Math.round(b).toLocaleString()+' kbit');
    tip.style.display='block';tip.style.left=(ev.clientX-rc.left+12)+'px';tip.style.top=(ev.clientY-rc.top+12)+'px';
  }else tip.style.display='none';});
svg.addEventListener('mouseleave',function(){tip.style.display='none';});
mkBase();draw();
})();
</script>

<p style="font-size:12px;color:#777;text-align:center;margin:2px auto 0 auto;">Bits memorized against training-set size for the <b id="cw-blab" style="color:#08519c">530M</b> base, one line per LoRA rank, log-log.</p>
<br>
Each curve rides the diagonal while it has room, then peels off at its own ceiling. Intuitively, smaller bases peel sooner, the 20M near ten thousand facts and the 530M past a hundred thousand. At rank 4, the 20M adapter crosses from about 104 kbit at 18k facts to roughly -39 kbit at 400k, having overwritten more than it kept. Pushed far past capacity, the smallest model even does worse than random guessing, where its stored bits turn <b>negative</b>. Schulman et al.'s *LoRA Without Regret* sees the first half of this shape on real tasks: loss curves track full fine-tuning until a rank-dependent threshold, then fall off as the adapter runs out of capacity. The overshoot into negative territory, though, is ours; in their setting a past-capacity LoRA learns less efficiently rather than regressing.

### The capacity law {#size}

How can we predict the capacity of a given base and adapter pair before training? Naturally, we can turn to scaling laws. Some dimensions we discuss thus far provide valuable signals: the base parameter count $N$, number of tokens $T$ saw in pretraining, and the adapter's trainable parameter count $p \approx 2\,r\,d_\text{model} \times (\text{layers} \times \text{matrices})$, with the adapter on every attention and MLP projection. Since DataDecide supplies matched checkpoints across abundant model sizes and pretraining steps, we can fit a scaling law over some (base, rank) cell.

We posit a joint power law, $C \propto N^{a}\, p^{b}\, T^{c}$, and estimate the exponents by least squares in log space,

$$\log C = a\log N + b\log p + c\log T + \text{const}$$

We can visualize the performance of fitting on different variable combinations:

<div id="pva-widget" style="position:relative;width:100%;margin:18px 0 4px 0;font-family:inherit;color:#333;">
  <div id="pva-tabs" style="text-align:center;margin-bottom:8px;font-size:12px;color:#555;"></div>
  <div style="display:flex;flex-wrap:nowrap;justify-content:center;align-items:flex-start;gap:8px;">
    <svg id="pva-svg" viewBox="0 0 600 440" style="flex:1 1 auto;min-width:0;max-width:600px;display:block;"></svg>
    <div id="pva-leg" style="flex:0 0 auto;font-size:11px;color:#555;padding:8px 4px 0 4px;width:120px;"></div>
  </div>
  <div id="pva-tip" style="position:absolute;pointer-events:none;display:none;background:#fff;border:1px solid #ddd;border-radius:5px;padding:5px 8px;font-size:11px;color:#333;box-shadow:0 1px 4px rgba(0,0,0,0.12);z-index:5;white-space:nowrap;"></div>
</div>
<script>
(function(){
var CO={"cells":[{"szM":60,"r":16,"C":303708,"p3":269655,"pnp":283936,"ppt":237935},{"szM":300,"r":16,"C":1189411,"p3":1091225,"pnp":1192688,"ppt":1013050},{"szM":530,"r":8,"C":722434,"p3":861058,"pnp":876497,"ppt":812029},{"szM":530,"r":1,"C":63610,"p3":104134,"pnp":93798,"ppt":102405},{"szM":530,"r":16,"C":1530879,"p3":1741186,"pnp":1846163,"ppt":1619280},{"szM":20,"r":16,"C":79117,"p3":114070,"pnp":122643,"ppt":98756},{"szM":60,"r":2,"C":37540,"p3":32611,"pnp":30385,"ppt":30006},{"szM":300,"r":2,"C":137410,"p3":131970,"pnp":127635,"ppt":127755},{"szM":20,"r":1,"C":5655,"p3":6822,"pnp":6231,"ppt":6245},{"szM":60,"r":4,"C":82879,"p3":65945,"pnp":64000,"ppt":59835},{"szM":20,"r":8,"C":43624,"p3":56410,"pnp":58227,"ppt":49524},{"szM":300,"r":4,"C":296198,"p3":266863,"pnp":268836,"ppt":254759},{"szM":150,"r":16,"C":703931,"p3":523404,"pnp":555858,"ppt":476334},{"szM":150,"r":8,"C":367578,"p3":258836,"pnp":263903,"ppt":238870},{"szM":150,"r":1,"C":28946,"p3":31303,"pnp":28241,"ppt":30124},{"szM":530,"r":4,"C":339192,"p3":425813,"pnp":416132,"ppt":407212},{"szM":530,"r":2,"C":172011,"p3":210575,"pnp":197566,"ppt":204207},{"szM":300,"r":1,"C":52120,"p3":65262,"pnp":60597,"ppt":64066},{"szM":300,"r":8,"C":640144,"p3":539636,"pnp":566249,"ppt":508020},{"szM":60,"r":8,"C":173250,"p3":133351,"pnp":134803,"ppt":119319},{"szM":20,"r":4,"C":24199,"p3":27896,"pnp":27644,"ppt":24835},{"szM":60,"r":1,"C":14830,"p3":16127,"pnp":14426,"ppt":15047},{"szM":20,"r":2,"C":12360,"p3":13795,"pnp":13125,"ppt":12454},{"szM":150,"r":4,"C":160490,"p3":128000,"pnp":125293,"ppt":119787},{"szM":150,"r":2,"C":83037,"p3":63299,"pnp":59485,"ppt":60070},{"szM":300,"r":16,"C":315156,"p3":473192,"pnp":1192688,"ppt":574236,"step":true},{"szM":20,"r":4,"C":19697,"p3":18777,"pnp":27644,"ppt":18978,"step":true},{"szM":300,"r":1,"C":54544,"p3":63518,"pnp":60597,"ppt":62898,"step":true},{"szM":530,"r":4,"C":387964,"p3":390665,"pnp":416132,"ppt":384062,"step":true},{"szM":300,"r":8,"C":672372,"p3":525214,"pnp":566249,"ppt":498755,"step":true},{"szM":60,"r":16,"C":274675,"p3":231646,"pnp":283936,"ppt":214600,"step":true},{"szM":150,"r":16,"C":633021,"p3":453585,"pnp":555858,"ppt":432182,"step":true},{"szM":300,"r":16,"C":503646,"p3":714883,"pnp":1192688,"ppt":760043,"step":true},{"szM":530,"r":4,"C":332160,"p3":354641,"pnp":416132,"ppt":359630,"step":true},{"szM":530,"r":16,"C":1071986,"p3":1264207,"pnp":1846163,"ppt":1302765,"step":true},{"szM":60,"r":8,"C":137496,"p3":114554,"pnp":134803,"ppt":107617,"step":true},{"szM":300,"r":16,"C":785844,"p3":910036,"pnp":1192688,"ppt":895480,"step":true},{"szM":530,"r":8,"C":445245,"p3":404624,"pnp":876497,"ppt":486115,"step":true},{"szM":530,"r":1,"C":51636,"p3":48934,"pnp":93798,"ppt":61304,"step":true},{"szM":60,"r":1,"C":12444,"p3":13854,"pnp":14426,"ppt":13571,"step":true},{"szM":150,"r":16,"C":420293,"p3":382195,"pnp":555858,"ppt":384713,"step":true},{"szM":20,"r":4,"C":23213,"p3":23902,"pnp":27644,"ppt":22360,"step":true},{"szM":150,"r":8,"C":229032,"p3":189005,"pnp":263903,"ppt":192924,"step":true},{"szM":300,"r":4,"C":243862,"p3":222553,"pnp":268836,"ppt":225193,"step":true},{"szM":150,"r":1,"C":26865,"p3":22858,"pnp":28241,"ppt":24330,"step":true},{"szM":150,"r":16,"C":641751,"p3":486529,"pnp":555858,"ppt":453267,"step":true},{"szM":20,"r":8,"C":26144,"p3":25132,"pnp":58227,"ppt":28593,"step":true},{"szM":530,"r":4,"C":306525,"p3":309166,"pnp":416132,"ppt":327616,"step":true},{"szM":60,"r":1,"C":14623,"p3":15185,"pnp":14426,"ppt":14445,"step":true},{"szM":300,"r":1,"C":26542,"p3":28300,"pnp":60597,"ppt":36315,"step":true},{"szM":300,"r":8,"C":227263,"p3":234005,"pnp":566249,"ppt":287965,"step":true},{"szM":60,"r":8,"C":146544,"p3":125564,"pnp":134803,"ppt":114539,"step":true},{"szM":20,"r":1,"C":4408,"p3":3039,"pnp":6231,"ppt":3606,"step":true},{"szM":60,"r":4,"C":49070,"p3":48778,"pnp":64000,"ppt":48751,"step":true},{"szM":20,"r":8,"C":43324,"p3":54467,"pnp":58227,"ppt":48358,"step":true},{"szM":20,"r":1,"C":6587,"p3":6587,"pnp":6231,"ppt":6098,"step":true},{"szM":150,"r":4,"C":69083,"p3":61867,"pnp":125293,"ppt":73095,"step":true},{"szM":150,"r":1,"C":28718,"p3":27127,"pnp":28241,"ppt":27332,"step":true},{"szM":150,"r":8,"C":304650,"p3":224309,"pnp":263903,"ppt":216729,"step":true},{"szM":150,"r":16,"C":228757,"p3":252981,"pnp":555858,"ppt":290663,"step":true},{"szM":150,"r":8,"C":337882,"p3":240600,"pnp":263903,"ppt":227302,"step":true},{"szM":300,"r":4,"C":171020,"p3":174827,"pnp":268836,"ppt":191134,"step":true},{"szM":150,"r":1,"C":28242,"p3":29098,"pnp":28241,"ppt":28665,"step":true},{"szM":20,"r":16,"C":44898,"p3":76779,"pnp":122643,"ppt":75467,"step":true},{"szM":60,"r":16,"C":174617,"p3":199459,"pnp":283936,"ppt":193860,"step":true},{"szM":530,"r":1,"C":64522,"p3":95539,"pnp":93798,"ppt":96583,"step":true},{"szM":530,"r":16,"C":1236344,"p3":1450156,"pnp":1846163,"ppt":1430067,"step":true},{"szM":300,"r":4,"C":309732,"p3":259731,"pnp":268836,"ppt":250113,"step":true},{"szM":530,"r":8,"C":818976,"p3":789982,"pnp":876497,"ppt":765864,"step":true},{"szM":60,"r":16,"C":302336,"p3":253908,"pnp":283936,"ppt":228405,"step":true},{"szM":20,"r":1,"C":4740,"p3":4592,"pnp":6231,"ppt":4773,"step":true},{"szM":20,"r":8,"C":27142,"p3":37969,"pnp":58227,"ppt":37845,"step":true},{"szM":20,"r":16,"C":75791,"p3":97739,"pnp":122643,"ppt":88915,"step":true},{"szM":60,"r":4,"C":74082,"p3":56650,"pnp":64000,"ppt":53967,"step":true},{"szM":530,"r":4,"C":252573,"p3":200096,"pnp":416132,"ppt":243775,"step":true},{"szM":300,"r":16,"C":1197186,"p3":1062060,"pnp":1192688,"ppt":994575,"step":true},{"szM":530,"r":8,"C":666726,"p3":717137,"pnp":876497,"ppt":717143,"step":true},{"szM":530,"r":1,"C":64658,"p3":86729,"pnp":93798,"ppt":90439,"step":true},{"szM":20,"r":16,"C":41802,"p3":50822,"pnp":122643,"ppt":57018,"step":true},{"szM":20,"r":4,"C":17096,"p3":12429,"pnp":27644,"ppt":14339,"step":true},{"szM":530,"r":8,"C":591125,"p3":625180,"pnp":876497,"ppt":653305,"step":true},{"szM":60,"r":4,"C":64692,"p3":62094,"pnp":64000,"ppt":57439,"step":true},{"szM":300,"r":4,"C":130018,"p3":115721,"pnp":268836,"ppt":144407,"step":true},{"szM":20,"r":16,"C":94212,"p3":110141,"pnp":122643,"ppt":96432,"step":true},{"szM":530,"r":1,"C":62670,"p3":75608,"pnp":93798,"ppt":82388,"step":true},{"szM":530,"r":16,"C":721944,"p3":818209,"pnp":1846163,"ppt":969370,"step":true},{"szM":300,"r":1,"C":50352,"p3":54426,"pnp":60597,"ppt":56631,"step":true},{"szM":150,"r":4,"C":119444,"p3":93467,"pnp":125293,"ppt":96747,"step":true},{"szM":300,"r":8,"C":443559,"p3":450034,"pnp":566249,"ppt":449061,"step":true},{"szM":20,"r":1,"C":5962,"p3":5845,"pnp":6231,"ppt":5623,"step":true},{"szM":20,"r":8,"C":39252,"p3":48334,"pnp":58227,"ppt":44589,"step":true},{"szM":300,"r":1,"C":36137,"p3":42755,"pnp":60597,"ppt":48066,"step":true},{"szM":530,"r":16,"C":1617491,"p3":1597461,"pnp":1846163,"ppt":1527221,"step":true},{"szM":150,"r":4,"C":145180,"p3":118982,"pnp":125293,"ppt":113987,"step":true},{"szM":300,"r":8,"C":311344,"p3":353526,"pnp":566249,"ppt":381143,"step":true},{"szM":150,"r":8,"C":127770,"p3":125105,"pnp":263903,"ppt":145760,"step":true},{"szM":150,"r":1,"C":19118,"p3":15130,"pnp":28241,"ppt":18382,"step":true},{"szM":150,"r":4,"C":159152,"p3":110926,"pnp":125293,"ppt":108684,"step":true},{"szM":60,"r":8,"C":100002,"p3":98637,"pnp":134803,"ppt":97216,"step":true},{"szM":20,"r":4,"C":23633,"p3":26935,"pnp":27644,"ppt":24250,"step":true},{"szM":60,"r":1,"C":11952,"p3":11929,"pnp":14426,"ppt":12260,"step":true},{"szM":1000,"r":1,"C":152636,"p3":186752,"pnp":188937,"ppt":194152,"stale":true},{"szM":1000,"r":8,"C":2241178,"p3":1544202,"pnp":1765528,"ppt":1539551,"stale":true},{"szM":1000,"r":16,"C":4998899,"p3":3122604,"pnp":3718727,"ppt":3070043,"stale":true},{"szM":1000,"r":4,"C":1204477,"p3":763644,"pnp":838214,"ppt":772047,"stale":true},{"szM":1000,"r":2,"C":394488,"p3":377640,"pnp":397956,"ppt":387162,"stale":true}],"meta":{"f3":{"a":-0.242,"b":1.016,"c":0.595,"r2":0.981,"r2_1b":0.898},"fnp":{"a":0.377,"b":1.075,"r2":0.978,"r2_1b":0.958},"fpt":{"b":0.996,"c":0.404,"r2":0.977,"r2_1b":0.911},"n":101}};
var m=CO.meta;
var FITS={
  "3":{lab:"N · p · T",key:"p3",eq:"N^"+m.f3.a+" · p^"+m.f3.b+" · T^"+m.f3.c,r2:m.f3.r2,n:101},
  "np":{lab:"N · p",key:"pnp",eq:"N^"+m.fnp.a+" · p^"+m.fnp.b,r2:m.fnp.r2,n:25},
  "pt":{lab:"p · T",key:"ppt",eq:"p^"+m.fpt.b+" · T^"+m.fpt.c,r2:m.fpt.r2,n:101}
};
var cur="3";
var HUE={20:196,60:220,150:250,300:284,530:326,1000:18},SAT={20:58,60:60,150:52,300:50,530:58,1000:70};
var RIDX={1:0,2:1,4:2,8:3,16:4};
function col(sz,r){return "hsl("+HUE[sz]+","+SAT[sz]+"%,"+(70-RIDX[r]*7.5)+"%)";}
var SZS=[20,60,150,300,530,1000],SZL={20:"20M",60:"60M",150:"150M",300:"300M",530:"530M",1000:"1B"};
var svg=document.getElementById("pva-svg"),tip=document.getElementById("pva-tip"),leg=document.getElementById("pva-leg"),widget=document.getElementById("pva-widget"),tabs=document.getElementById("pva-tabs");
function l10(v){return Math.log(v)/Math.LN10;}
function txt(x,y,s,c,sz,anc,w){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||11)+'" font-weight="'+(w||400)+'" fill="'+(c||"#555")+'" text-anchor="'+(anc||"middle")+'">'+s+'</text>';}
var ML=60,MR=16,MT=28,MB=52,W=600,H=440,PW=W-ML-MR,PH=H-MT-MB;
var Xa=l10(4000),Xb=l10(7e6),Ya=Xa,Yb=Xb;
function xp(v){return ML+(l10(v)-Xa)/(Xb-Xa)*PW;}
function yp(v){return MT+(1-(l10(v)-Ya)/(Yb-Ya))*PH;}
function draw(){
  var f=FITS[cur];
  var s='<rect x="'+ML+'" y="'+MT+'" width="'+PW+'" height="'+PH+'" fill="none" stroke="#ddd"/>';
  var t=[[1e4,"10⁴"],[1e5,"10⁵"],[1e6,"10⁶"]];
  for(var i=0;i<t.length;i++){var Y=yp(t[i][0]);s+='<line x1="'+ML+'" y1="'+Y+'" x2="'+(ML+PW)+'" y2="'+Y+'" stroke="#f4f4f4"/>'+txt(ML-7,Y+3,t[i][1],"#999",10,"end");
    var X=xp(t[i][0]);s+='<line x1="'+X+'" y1="'+MT+'" x2="'+X+'" y2="'+(MT+PH)+'" stroke="#f4f4f4"/>'+txt(X,MT+PH+16,t[i][1],"#999",10);}
  s+='<line x1="'+xp(4000)+'" y1="'+yp(4000)+'" x2="'+xp(7e6)+'" y2="'+yp(7e6)+'" stroke="#bbb" stroke-width="1.4" stroke-dasharray="5 4"/>';
  s+=txt(ML+PW/2,MT+PH+34,"predicted capacity   "+f.eq,"#666",12);
  s+='<text x="14" y="'+(MT+PH/2)+'" font-size="12" fill="#666" text-anchor="middle" transform="rotate(-90 14 '+(MT+PH/2)+')">actual capacity C (bits)</text>';
  for(var i=0;i<CO.cells.length;i++){var c=CO.cells[i];if(!c.step||cur==="np")continue;var X=xp(c[f.key]),Y=yp(c.C),cc=col(c.szM,c.r);
    s+='<circle class="pva-hit" cx="'+X+'" cy="'+Y+'" r="2.3" fill="'+cc+'" fill-opacity="0.35" data-t="pretraining-step checkpoint" data-z="'+SZL[c.szM]+'" data-r="'+c.r+'" data-v="'+c.C+'"/>';}
  for(var i=0;i<CO.cells.length;i++){var c=CO.cells[i];if(c.step)continue;var X=xp(c[f.key]),Y=yp(c.C),cc=col(c.szM,c.r);
    if(c.stale){s+='<rect class="pva-hit" x="'+(X-4)+'" y="'+(Y-4)+'" width="8" height="8" fill="none" stroke="'+cc+'" stroke-width="1.6" data-t="1B (held out)" data-z="'+SZL[c.szM]+'" data-r="'+c.r+'" data-v="'+c.C+'"/>';}
    else{s+='<circle class="pva-hit" cx="'+X+'" cy="'+Y+'" r="4.6" fill="'+cc+'" stroke="#fff" stroke-width="1.1" data-t="final checkpoint" data-z="'+SZL[c.szM]+'" data-r="'+c.r+'" data-v="'+c.C+'"/>';}
  }
  s+=txt(ML+PW/2,MT-8,"actual vs predicted capacity   R²="+f.r2+"  (fit on "+f.n+")","#333",12,"middle",600);
  svg.innerHTML=s;
  var h='<div style="margin-bottom:8px;"><b style="color:#444;">markers</b><br><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#999;"></span> ≤530M final (fit)<br>'+(cur!=="np"?'<span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#999;opacity:0.4;"></span> pretraining step<br>':'')+'<span style="display:inline-block;width:8px;height:8px;border:1.6px solid #999;"></span> 1B (held out)</div>';
  h+='<div><b style="color:#444;">base &amp; rank</b><br><span style="color:#999;">light r1 &rarr; deep r16</span><br>';
  for(var i=0;i<SZS.length;i++){var z=SZS[i];h+='<div style="margin-top:3px;">';[1,2,4,8,16].forEach(function(rk){h+='<span style="display:inline-block;width:11px;height:11px;background:'+col(z,rk)+';"></span>';});h+=' '+SZL[z]+'</div>';}
  h+='</div>';leg.innerHTML=h;
}
function drawTabs(){var h='<span style="color:#777;margin-right:8px;">Free variables:</span>';for(var k in FITS){h+='<span class="pva-tab" data-k="'+k+'" style="cursor:pointer;padding:3px 11px;margin:0 2px;border-radius:4px;'+(k===cur?'background:#3182bd;color:#fff;':'color:#3182bd;')+'">'+FITS[k].lab+'</span>';}
  tabs.innerHTML=h;var el=tabs.getElementsByClassName("pva-tab");
  for(var i=0;i<el.length;i++)el[i].addEventListener("click",function(){cur=this.getAttribute("data-k");drawTabs();draw();});}
svg.addEventListener("mousemove",function(ev){var t=ev.target;if(t&&t.getAttribute&&t.getAttribute("class")&&t.getAttribute("class").indexOf("pva-hit")>=0){var rc=widget.getBoundingClientRect();tip.innerHTML="<b>"+t.getAttribute("data-z")+" · r="+t.getAttribute("data-r")+"</b> "+t.getAttribute("data-t")+"<br>"+(+t.getAttribute("data-v")/1e3).toFixed(0)+"k bits";tip.style.display="block";tip.style.left=(ev.clientX-rc.left+12)+"px";tip.style.top=(ev.clientY-rc.top+12)+"px";}else tip.style.display="none";});
svg.addEventListener("mouseleave",function(){tip.style.display="none";});
drawTabs();draw();
})();
</script>
<!-- <p style="font-size:12px;color:#777;text-align:center;margin:2px auto 0 auto;">Actual against predicted capacity for every (base, rank) pair, log–log. The tabs ablate different fitting of free variables.</p> -->
<br>

Our fit is done on models up to 530M, with 1B held out. The $N \cdot p$ fit uses the 25 finals (≤530M, ranks 1–16); the $p \cdot T$ and $N \cdot p \cdot T$ fits add the pretrain step checkpoints (all six sizes, ranks 1/4/8/16). With any fit involving T, we use 101 points.

Two findings stand out. First, the exponent on $p$ lands near one in every view ($0.97$ to $1.08$): capacity is essentially linear in the adapter's trainable parameters. Second, one variable on model quality is adequate. With pretraining tokens in the fit, the exponent on $N$ turns slightly negative, so the base's raw size adds no predictive power on top:

$$C \propto N^{-0.24}\, p^{1.02}\, T^{0.60} \quad (R^2 = 0.98)$$

Dropping $T$ lets size stand in as its proxy at nearly the same quality,

$$C \propto N^{0.38}\, p^{1.08} \quad (R^2 = 0.98)$$

In "Distillation Scaling Laws", Busbridge et al. also treat base size and pretraining tokens as two handles on the base's quality, where teacher size and teacher tokens reach the student only through the teacher's loss. It seems we can use $T$ when the pretraining budget is known, but otherwise $N$ can stand in: in DataDecide, models are trained to roughly 100 tokens per model parameter, about 5x Chinchilla-optimal.

Is the base-quality term causal, or is $N$ merely correlated with it? A bigger base is also a more heavily pretrained one, so the clean test holds each base and adapter fixed and varies only how long the base was pretrained, walking DataDecide's checkpoints from random initialization to the final model (plotted as capacity per adapter parameter, $C/p$, to put all sizes on one scale).

<figure style="margin:26px 0;text-align:center;">
<div id="cpt-widget" style="position:relative;width:100%;margin:26px 0 4px 0;font-family:inherit;color:#333;">
  <div style="display:flex;flex-wrap:nowrap;justify-content:center;align-items:flex-start;gap:8px;">
    <svg id="cpt-svg" viewBox="0 0 620 400" style="flex:1 1 auto;min-width:0;max-width:620px;display:block;"></svg>
    <div id="cpt-leg" style="flex:0 0 auto;font-size:11px;color:#555;padding:8px 4px 0 4px;width:118px;"></div>
  </div>
</div>
<script>
(function(){
var CPT=[{"szM":20,"r":1,"pts":[[0.492,0.0394],[0.983,0.0423],[1.475,0.0533],[1.802,0.0588],[1.912,0.0505]]},{"szM":20,"r":4,"pts":[[0.492,0.0382],[0.983,0.044],[1.475,0.0518],[1.802,0.0528],[1.912,0.054]]},{"szM":20,"r":8,"pts":[[0.492,0.0292],[0.983,0.0303],[1.475,0.0438],[1.802,0.0484],[1.912,0.0487]]},{"szM":20,"r":16,"pts":[[0.492,0.0233],[0.983,0.0251],[1.475,0.0423],[1.802,0.0526],[1.912,0.0442]]},{"szM":60,"r":1,"pts":[[1.72,0.0023],[3.441,0.0689],[4.424,0.0717],[5.161,0.0842],[5.71,0.0854]]},{"szM":60,"r":4,"pts":[[1.72,0.0006],[3.441,0.0707],[4.424,0.1067],[5.161,0.0932],[5.71,0.1194]]},{"szM":60,"r":8,"pts":[[1.72,0.0003],[3.441,0.072],[4.424,0.099],[5.161,0.1055],[5.71,0.1248]]},{"szM":60,"r":16,"pts":[[1.72,0.0001],[3.441,0.0629],[4.424,0.0989],[5.161,0.1089],[5.71,0.1094]]},{"szM":150,"r":1,"pts":[[4.424,0.0812],[8.847,0.1141],[11.796,0.122],[13.271,0.12],[15.004,0.123]]},{"szM":150,"r":4,"pts":[[4.424,0.0734],[8.847,0.1269],[11.796,0.169],[13.271,0.1542],[15.004,0.1704]]},{"szM":150,"r":8,"pts":[[4.424,0.0678],[8.847,0.1216],[11.796,0.1618],[13.271,0.1794],[15.004,0.1952]]},{"szM":150,"r":16,"pts":[[4.424,0.0607],[8.847,0.1116],[11.796,0.1681],[13.271,0.1704],[15.004,0.1869]]},{"szM":300,"r":1,"pts":[[7.373,0.07],[14.746,0.0953],[22.118,0.1329],[28.672,0.1439],[30.007,0.1375]]},{"szM":300,"r":4,"pts":[[7.373,0.0858],[14.746,0.1128],[22.118,0.1609],[28.672,0.2043],[30.007,0.1954]]},{"szM":300,"r":8,"pts":[[7.373,0.075],[14.746,0.1027],[22.118,0.1463],[28.672,0.2218],[30.007,0.2111]]},{"szM":300,"r":16,"pts":[[7.373,0.052],[14.746,0.0831],[22.118,0.1296],[28.672,0.1974],[30.007,0.1961]]},{"szM":530,"r":1,"pts":[[14.909,0.1072],[30.966,0.1301],[38.994,0.1342],[45.875,0.1339],[53.019,0.132]]},{"szM":530,"r":4,"pts":[[14.909,0.1311],[30.966,0.1591],[38.994,0.1724],[45.875,0.2013],[53.019,0.176]]},{"szM":530,"r":8,"pts":[[14.909,0.1155],[30.966,0.1534],[38.994,0.173],[45.875,0.2125],[53.019,0.1875]]},{"szM":530,"r":16,"pts":[[14.909,0.0937],[30.966,0.1391],[38.994,0.1604],[45.875,0.2099],[53.019,0.1986]]},{"szM":1000,"r":1,"pts":[[25.231,0.0001],[50.463,0.1681],[75.694,0.1674],[90.112,0.2139],[100.016,0.2157]]},{"szM":1000,"r":4,"pts":[[25.231,0.0],[50.463,0.2882],[75.694,0.3759],[90.112,0.3407],[100.016,0.4255]]},{"szM":1000,"r":8,"pts":[[25.231,0.0],[50.463,0.3144],[75.694,0.3845],[90.112,0.3789],[100.016,0.3958]]},{"szM":1000,"r":16,"pts":[[25.231,0.0],[50.463,0.3051],[75.694,0.4335],[90.112,0.4411],[100.016,0.4415]]}];
var HUE={20:196,60:220,150:250,300:284,530:326,1000:18},SAT={20:58,60:60,150:52,300:50,530:58,1000:70},RIDX={1:0,2:1,4:2,8:3,16:4};
function col(sz,r){return "hsl("+HUE[sz]+","+SAT[sz]+"%,"+(70-RIDX[r]*7.5)+"%)";}
var SZL={20:"20M",60:"60M",150:"150M",300:"300M",530:"530M",1000:"1B"};
var svg=document.getElementById("cpt-svg"),leg=document.getElementById("cpt-leg");
function l10(v){return Math.log(v)/Math.LN10;}
function txt(x,y,s,c,sz,anc,w){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||11)+'" font-weight="'+(w||400)+'" fill="'+(c||"#555")+'" text-anchor="'+(anc||"middle")+'">'+s+'</text>';}
var ML=58,MR=20,MT=32,MB=52,W=620,H=400,PW=W-ML-MR,PH=H-MT-MB;
var Xa=l10(0.3),Xb=l10(130),Ya=l10(0.01),Yb=l10(0.5);
function xp(t){return ML+(l10(t)-Xa)/(Xb-Xa)*PW;}
function yp(v){var lv=l10(v<0.008?0.008:v);return MT+(1-(lv-Ya)/(Yb-Ya))*PH;}
var s='<rect x="'+ML+'" y="'+MT+'" width="'+PW+'" height="'+PH+'" fill="none" stroke="#ddd"/>';
[[0.01,"0.01"],[0.02,"0.02"],[0.05,"0.05"],[0.1,"0.1"],[0.2,"0.2"],[0.5,"0.5"]].forEach(function(t){var Y=yp(t[0]);s+='<line x1="'+ML+'" y1="'+Y+'" x2="'+(ML+PW)+'" y2="'+Y+'" stroke="#f4f4f4"/>'+txt(ML-8,Y+3,t[1],"#999",10,"end");});
[[0.5,"0.5B"],[1,"1B"],[2,"2B"],[5,"5B"],[10,"10B"],[20,"20B"],[50,"50B"],[100,"100B"]].forEach(function(t){var X=xp(t[0]);s+='<line x1="'+X+'" y1="'+MT+'" x2="'+X+'" y2="'+(MT+PH)+'" stroke="#f4f4f4"/>'+txt(X,MT+PH+16,t[1],"#999",10);});
s+=txt(ML+PW/2,H-8,"pretraining tokens T","#666",12);
s+='<text x="14" y="'+(MT+PH/2)+'" font-size="12" fill="#666" text-anchor="middle" transform="rotate(-90 14 '+(MT+PH/2)+')">bits per parameter  C/p  (log)</text>';
var fy=yp(0.012);s+='<line x1="'+ML+'" y1="'+fy+'" x2="'+(ML+PW)+'" y2="'+fy+'" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="5 4"/>';
s+=txt(ML+6,fy-4,"random-init floor (no pretraining, ~0.01)","#94a3b8",10,"start");
CPT.forEach(function(c){var cc=col(c.szM,c.r),pl='';
  c.pts.forEach(function(p){if(p[1]<0.012)return;pl+=(pl?' ':'')+xp(p[0]).toFixed(1)+','+yp(p[1]).toFixed(1);});
  s+='<polyline points="'+pl+'" fill="none" stroke="'+cc+'" stroke-width="2.2"/>';
  c.pts.forEach(function(p){if(p[1]<0.012)return;s+='<circle cx="'+xp(p[0]).toFixed(1)+'" cy="'+yp(p[1]).toFixed(1)+'" r="3.4" fill="'+cc+'"><title>'+SZL[c.szM]+' r'+c.r+'  T='+p[0]+'B  C/p='+p[1]+'</title></circle>';});
});
var bases=[];CPT.forEach(function(c){if(bases.indexOf(c.szM)<0)bases.push(c.szM);});bases.sort(function(a,b){return b-a;});
s+=txt(ML+PW/2,MT-12,"Bits per parameter rises with pretraining","#333",13,"middle",600);
svg.innerHTML=s;
// legend: per base, a rank ramp (light r1 -> deep r16), matching the actual-vs-predicted figure.
var h='<div><b style="color:#444;">base &amp; rank</b><br><span style="color:#999;">light r1 &rarr; deep r16</span>';
bases.forEach(function(z){h+='<div style="margin-top:4px;line-height:12px;">';[1,4,8,16].forEach(function(rk){h+='<span style="display:inline-block;width:11px;height:11px;vertical-align:middle;background:'+col(z,rk)+';"></span>';});h+='<span style="vertical-align:middle;"> '+SZL[z]+'</span></div>';});
h+='</div>';leg.innerHTML=h;
})();
</script>
  <!-- <figcaption style="font-size:12px;color:#777;margin-top:2px;">Bits per LoRA-param against pretraining tokens $T$ (log–log).</figcaption> -->
</figure>

Capacity rises monotonically with pretraining, at every size and rank. A random base sits at the injection floor, near 0.01 bits per parameter, because it has no representations to remap; pretraining lifts it severalfold, from about 0.05 at 20M to roughly 0.2 at the mid sizes and 0.4 at 1B by the end of the sweep. Holding size and parameter count fixed while adding tokens still raises capacity, so the base-quality term in the law is causal, not just a correlation with size. The lift shows up across every rank measured (1, 4, 8, 16), not just one.

When fit on models up to 530M, the law predicts 1B's capacity within 0.8 to 1.4 times measured (the held-out $R^2 = 0.96$). Notably, a fit on $p$ alone collapses on the held-out 1B ($R^2 = 0.45$). Adding either $N$ or $T$ as an axis restores our predictability. I have found that a fit on <=150M can work almost equally well ($R^2 = 0.92$ for predicting 1B), so a cheap budget might suffice in practice.

With the law, we can quantify precisely the capacity of an adapter rank. Because the exponent on $p$ is essentially one ($p^{1.08}$), dividing the fit through by $p$ gives the *bits per LoRA parameter*,

$$C/p \propto N^{0.38}\, p^{0.08}$$

Since the exponent on $p$ is small, density is mostly a property of the base model instead of the adapter size: at 530M it sits between 0.13 and 0.20 across ranks 1 to 16, while the base moves it from about 0.05 at 20M to 0.4 at 1B. Specifically:

<div style="font-family:'SFMono-Regular',Consolas,monospace;font-size:12.5px;line-height:1.7;background:#fafafa;border:1px solid #ececec;border-radius:4px;padding:12px 14px;margin:18px 0;color:#333;white-space:pre-wrap;">1B, rank 16    C = 5.0 Mbit (peak of its memorization curve)
               p = 11.3M trainable parameters
               C/p = 0.44 bits per parameter

20M, rank 16   C = 79 kbit,  p = 1.8M   →   C/p = 0.04</div>

Thus, in our suite, a LoRA parameter holds about <b>0.05 to 0.4 bits</b> per trainable parameter.

<b>How do these densities compare to full training?</b> Morris et al. train GPT-family models from scratch on random data, the same protocol we borrow here, and read a plateau of about 3.6 bits per parameter (also in bfloat16). Allen-Zhu and Li get at the number differently: they train on synthetic biographies and count how many facts can be probed back out, landing near 2 bits per parameter. Against either reading, a LoRA is roughly an order of magnitude less dense than the weights it patches.

Even at 1B the density sits well below the full training ceiling, but a low density is not a weakness in itself. Thinky's blog post corroborates with the conclusion that a LoRA keeps pace with full finetuning until the finetuning set asks for more bits than the adapter can hold. In the next section, we examine more closely where the capacity boundary falls from the perspective of recall.

## From capacity to recall {#recall}

Our $bits$ and capacity ceiling for LoRA have closely followed Morris's framing. In practice, we might want to also gage model's *recall* ability given that it has been train on a set of $D$ facts. Here, capacity governs both storage (our "bits" defined in previous session) and recall -- simply asking how many facts model can guess correct.

{% comment %} cap-widget (storage saturates + recall bends) — hidden for now
<figure style="margin:22px 0;text-align:center;">
<div id="cap-widget" style="position:relative;width:100%;margin:8px 0 4px 0;font-family:inherit;color:#333;">
  <div style="display:flex;gap:36px;justify-content:center;font-size:13px;margin-bottom:8px;">
    <label>base model &nbsp;<b id="cap-slab" style="color:#08519c;">530M</b><br>
      <input id="cap-sz" type="range" min="0" max="5" step="1" value="4" style="width:150px;"></label>
    <label>LoRA rank &nbsp;<b id="cap-rlab" style="color:#08519c;">16</b><br>
      <input id="cap-rr" type="range" min="0" max="4" step="1" value="4" style="width:150px;"></label>
  </div>
  <svg id="cap-svg" viewBox="0 0 620 390" style="width:100%;max-width:640px;display:block;margin:0 auto;"></svg>
</div>
<script>
(function(){
var CAP={"cells":{"60_16":{"C":303708,"Dc":25381,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.001,1.0],[34,0.001,1.0],[82,0.003,0.976],[198,0.008,0.965],[477,0.018,0.904],[1151,0.044,0.962],[2779,0.107,0.977],[6707,0.248,0.934],[16189,0.543,0.793],[39072,1.0,0.444],[94303,0.354,0.019],[227605,-0.917,0.001],[549336,-0.142,0.002],[1325849,-0.305,0.0],[3200000,-0.937,0.0]]},"300_16":{"C":1189411,"Dc":99401,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.0,1.0],[34,0.0,1.0],[82,0.001,1.0],[198,0.002,0.97],[477,0.005,0.981],[1151,0.011,0.934],[2779,0.027,0.967],[6707,0.065,0.977],[16189,0.155,0.961],[39072,0.356,0.888],[94303,0.732,0.633],[227605,1.0,0.207],[549336,-0.063,0.011]]},"530_8":{"C":722434,"Dc":60375,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.0,1.0],[34,0.001,1.0],[82,0.001,1.0],[198,0.003,0.985],[477,0.008,0.975],[1151,0.018,0.901],[2779,0.044,0.963],[6707,0.107,0.971],[16189,0.253,0.942],[39072,0.568,0.835],[94303,1.0,0.443],[227605,0.999,0.08],[549336,0.105,0.006],[1325849,-0.087,0.002],[3200000,-0.345,0.0]]},"530_1":{"C":63610,"Dc":5316,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.001,1.0],[14,0.003,0.929],[34,0.006,0.971],[82,0.015,1.0],[198,0.036,0.949],[477,0.08,0.874],[1151,0.2,0.912],[2779,0.483,0.929],[6707,1.0,0.75],[16189,0.727,0.096],[39072,-0.356,0.003],[94303,-2.554,0.001],[227605,-1.34,0.001],[549336,0.779,0.002],[1325849,-0.962,0.0],[3200000,-49.783,0.0]]},"530_16":{"C":1530879,"Dc":127938,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.0,1.0],[34,0.0,1.0],[82,0.001,1.0],[198,0.002,0.99],[477,0.004,0.973],[1151,0.009,0.955],[2779,0.021,0.965],[6707,0.051,0.977],[16189,0.12,0.957],[39072,0.276,0.879],[94303,0.624,0.757],[227605,1.0,0.324],[549336,0.822,0.051],[1325849,0.201,0.004],[3200000,-0.166,0.0]]},"20_16":{"C":79117,"Dc":6612,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.001,1.0],[14,0.002,0.929],[34,0.005,1.0],[82,0.012,0.951],[198,0.026,0.732],[477,0.064,0.801],[1151,0.165,0.903],[2779,0.387,0.873],[6707,0.777,0.6],[16189,1.0,0.197],[39072,-0.249,0.001],[94303,-1.264,0.0],[227605,-0.851,0.0],[549336,-0.499,0.0],[1325849,-1.709,0.0],[3200000,-3.935,0.0]]},"60_2":{"C":37540,"Dc":3137,"pts":[[1,0.0,1.0],[2,0.001,1.0],[6,0.002,1.0],[14,0.004,1.0],[34,0.011,1.0],[82,0.026,0.976],[198,0.058,0.859],[477,0.141,0.91],[1151,0.351,0.962],[2779,0.783,0.887],[6707,1.0,0.249],[16189,-0.011,0.021],[39072,-1.611,0.003],[94303,-2.074,0.002],[227605,-0.937,0.001],[549336,-0.639,0.001],[1325849,-2.46,0.0],[3200000,-6.875,0.0]]},"300_2":{"C":137410,"Dc":11484,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.001,1.0],[14,0.001,1.0],[34,0.003,0.971],[82,0.007,1.0],[198,0.017,0.965],[477,0.04,0.927],[1151,0.093,0.918],[2779,0.23,0.959],[6707,0.532,0.915],[16189,1.0,0.587],[39072,0.464,0.046],[94303,-0.601,0.001],[227605,-1.453,0.001],[549336,-0.471,0.001],[1325849,-0.357,0.001],[3200000,-2.168,0.0]]},"20_1":{"C":5655,"Dc":473,"pts":[[1,0.002,1.0],[2,0.004,1.0],[6,0.013,1.0],[14,0.029,1.0],[34,0.054,0.765],[82,0.142,0.768],[198,0.343,0.727],[477,0.84,0.761],[1151,1.0,0.202],[2779,-0.214,0.009],[6707,-0.938,0.005],[16189,-5.514,0.001],[39072,-2.344,0.001],[94303,-2.056,0.001],[227605,-52.042,0.0],[549336,-7.775,0.0],[1325849,-20.062,0.0],[3200000,-56.109,0.0]]},"60_4":{"C":82879,"Dc":6926,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.001,1.0],[14,0.002,1.0],[34,0.005,1.0],[82,0.012,0.976],[198,0.028,0.944],[477,0.063,0.862],[1151,0.159,0.945],[2779,0.377,0.94],[6707,0.782,0.745],[16189,1.0,0.23],[39072,-0.519,0.001],[94303,-2.026,0.001],[227605,-0.95,0.001],[549336,0.338,0.001],[1325849,-1.05,0.0],[3200000,-2.999,0.0]]},"20_8":{"C":43624,"Dc":3646,"pts":[[1,0.0,1.0],[2,0.001,1.0],[6,0.002,1.0],[14,0.003,0.857],[34,0.009,0.941],[82,0.021,0.829],[198,0.044,0.702],[477,0.116,0.786],[1151,0.295,0.896],[2779,0.653,0.766],[6707,1.0,0.309],[16189,0.294,0.035],[39072,-1.107,0.001],[94303,-2.754,0.0],[227605,-1.151,0.0],[549336,-1.118,0.0],[1325849,-3.235,0.0],[3200000,-7.631,0.0]]},"300_4":{"C":296198,"Dc":24754,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.001,0.929],[34,0.001,0.971],[82,0.003,1.0],[198,0.008,0.995],[477,0.019,0.948],[1151,0.043,0.87],[2779,0.107,0.96],[6707,0.259,0.968],[16189,0.588,0.891],[39072,1.0,0.459],[94303,0.742,0.093],[227605,-0.803,0.004],[549336,-0.2,0.002],[1325849,0.097,0.001],[3200000,-0.933,0.0]]},"150_16":{"C":703931,"Dc":58829,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.0,1.0],[34,0.001,1.0],[82,0.001,1.0],[198,0.003,0.99],[477,0.008,0.985],[1151,0.019,0.964],[2779,0.047,0.988],[6707,0.111,0.987],[16189,0.258,0.936],[39072,0.412,0.464],[94303,1.0,0.4],[227605,0.866,0.11],[549336,0.366,0.015],[1325849,0.024,0.001],[3200000,-0.413,0.0]]},"1000_1":{"C":152636,"Dc":12756,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.001,1.0],[34,0.003,1.0],[82,0.006,0.988],[198,0.015,0.985],[477,0.035,0.912],[1151,0.084,0.924],[2779,0.21,0.975],[6707,0.504,0.974],[16189,1.0,0.716],[39072,0.422,0.058],[94303,-1.011,0.008],[227605,-1.195,0.003],[549336,-3.331,0.002],[1325849,-7.446,0.001],[3200000,-19.308,0.0]]},"1000_8":{"C":2241178,"Dc":187299,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.0,0.929],[34,0.0,1.0],[82,0.0,1.0],[198,0.001,1.0],[477,0.003,0.994],[1151,0.006,0.972],[2779,0.015,0.981],[6707,0.035,0.985],[16189,0.084,0.984],[39072,0.201,0.972],[94303,0.472,0.917],[227605,0.954,0.629],[549336,1.0,0.128],[1325849,0.575,0.015],[3200000,-0.065,0.001]]},"150_8":{"C":367578,"Dc":30719,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.0,1.0],[34,0.001,1.0],[82,0.003,1.0],[198,0.006,0.97],[477,0.015,0.943],[1151,0.036,0.947],[2779,0.088,0.983],[6707,0.21,0.98],[16189,0.477,0.897],[39072,0.865,0.515],[94303,1.0,0.166],[227605,-1.36,0.022],[549336,0.035,0.004],[1325849,-0.002,0.001],[3200000,-0.742,0.0]]},"150_1":{"C":28946,"Dc":2419,"pts":[[1,0.0,1.0],[2,0.001,1.0],[6,0.002,1.0],[14,0.006,1.0],[34,0.014,1.0],[82,0.033,0.976],[198,0.075,0.874],[477,0.181,0.895],[1151,0.455,0.977],[2779,1.0,0.875],[6707,0.775,0.106],[16189,-0.252,0.016],[39072,-2.963,0.003],[94303,-5.196,0.001],[227605,-8.945,0.001],[549336,-19.473,0.0],[1325849,-49.838,0.0],[3200000,-120.987,0.0]]},"530_4":{"C":339192,"Dc":28347,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.0,1.0],[34,0.001,0.971],[82,0.003,1.0],[198,0.007,0.965],[477,0.016,0.954],[1151,0.038,0.898],[2779,0.094,0.955],[6707,0.226,0.966],[16189,0.518,0.895],[39072,1.0,0.588],[94303,0.963,0.108],[227605,0.071,0.006],[549336,-0.186,0.003],[1325849,0.174,0.001],[3200000,-0.771,0.0]]},"530_2":{"C":172011,"Dc":14375,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.001,1.0],[34,0.002,0.971],[82,0.006,1.0],[198,0.013,0.975],[477,0.032,0.941],[1151,0.073,0.894],[2779,0.183,0.944],[6707,0.436,0.942],[16189,0.881,0.7],[39072,1.0,0.17],[94303,-0.043,0.005],[227605,-1.096,0.002],[549336,-0.203,0.002],[1325849,-0.145,0.001],[3200000,-1.656,0.0]]},"300_1":{"C":52120,"Dc":4356,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.001,1.0],[14,0.003,1.0],[34,0.008,1.0],[82,0.019,0.988],[198,0.042,0.874],[477,0.099,0.866],[1151,0.253,0.97],[2779,0.58,0.909],[6707,1.0,0.527],[16189,0.419,0.041],[39072,-1.279,0.001],[94303,-2.767,0.001],[227605,-6.015,0.001],[549336,-9.775,0.001],[1325849,-25.875,0.0],[3200000,-63.301,0.0]]},"300_8":{"C":640144,"Dc":53498,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.0,1.0],[34,0.001,1.0],[82,0.002,1.0],[198,0.004,0.99],[477,0.009,0.954],[1151,0.021,0.931],[2779,0.05,0.964],[6707,0.121,0.976],[16189,0.285,0.944],[39072,0.624,0.802],[94303,1.0,0.346],[227605,0.535,0.033],[549336,-0.09,0.004],[1325849,0.169,0.002],[3200000,-0.434,0.0]]},"60_8":{"C":173250,"Dc":14479,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.001,1.0],[34,0.002,1.0],[82,0.006,0.976],[198,0.013,0.955],[477,0.031,0.906],[1151,0.077,0.954],[2779,0.186,0.97],[6707,0.418,0.886],[16189,0.837,0.628],[39072,1.0,0.196],[94303,-0.311,0.003],[227605,-1.441,0.002],[549336,-0.179,0.001],[1325849,-0.51,0.0],[3200000,-1.458,0.0]]},"20_4":{"C":24199,"Dc":2022,"pts":[[1,0.0,1.0],[2,0.001,1.0],[6,0.003,1.0],[14,0.007,1.0],[34,0.016,0.912],[82,0.038,0.89],[198,0.08,0.697],[477,0.204,0.78],[1151,0.517,0.865],[2779,1.0,0.567],[6707,0.851,0.104],[16189,-1.137,0.002],[39072,-2.375,0.0],[94303,-2.051,0.0],[227605,-1.546,0.0],[549336,-1.889,0.0],[1325849,-5.007,0.0],[3200000,-11.837,0.0]]},"60_1":{"C":14830,"Dc":1239,"pts":[[1,0.001,1.0],[2,0.002,1.0],[6,0.005,1.0],[14,0.011,1.0],[34,0.027,1.0],[82,0.063,0.976],[198,0.145,0.874],[477,0.359,0.925],[1151,0.845,0.91],[2779,1.0,0.249],[6707,-0.469,0.007],[16189,-2.346,0.003],[39072,-2.802,0.003],[94303,-1.565,0.001],[227605,-0.074,0.001],[549336,-1.713,0.001],[1325849,-6.579,0.0],[3200000,-17.798,0.0]]},"20_2":{"C":12360,"Dc":1033,"pts":[[1,0.001,1.0],[2,0.002,1.0],[6,0.006,1.0],[14,0.012,0.786],[34,0.03,0.853],[82,0.068,0.805],[198,0.157,0.732],[477,0.403,0.82],[1151,0.923,0.755],[2779,1.0,0.163],[6707,0.103,0.029],[16189,-1.649,0.001],[39072,-2.695,0.001],[94303,-2.317,0.001],[227605,-2.492,0.0],[549336,-4.034,0.0],[1325849,-9.879,0.0],[3200000,-24.149,0.0]]},"150_4":{"C":160490,"Dc":13412,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.001,1.0],[34,0.003,0.971],[82,0.006,1.0],[198,0.015,0.985],[477,0.034,0.929],[1151,0.083,0.967],[2779,0.201,0.979],[6707,0.47,0.955],[16189,0.888,0.636],[39072,1.0,0.175],[94303,-1.201,0.03],[227605,-1.775,0.002],[549336,-0.12,0.002]]},"1000_16":{"C":4998899,"Dc":417766,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.0,1.0],[34,0.0,1.0],[82,0.0,1.0],[198,0.0,0.995],[477,0.001,0.998],[1151,0.003,0.99],[2779,0.007,0.986],[6707,0.016,0.982],[16189,0.038,0.979],[39072,0.09,0.97],[94303,0.217,0.959],[227605,0.475,0.795],[549336,1.0,0.582],[1325849,0.782,0.064],[3200000,-0.02,0.001]]},"1000_4":{"C":1204477,"Dc":100660,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.0,1.0],[34,0.0,1.0],[82,0.001,1.0],[198,0.002,0.99],[477,0.005,0.996],[1151,0.011,0.956],[2779,0.027,0.98],[6707,0.065,0.979],[16189,0.156,0.98],[39072,0.367,0.943],[94303,0.773,0.714],[227605,1.0,0.2],[549336,0.326,0.019],[1325849,0.415,0.005],[3200000,-0.165,0.0]]},"150_2":{"C":83037,"Dc":6940,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.001,1.0],[14,0.002,1.0],[34,0.005,0.941],[82,0.012,1.0],[198,0.028,0.939],[477,0.064,0.918],[1151,0.159,0.962],[2779,0.386,0.982],[6707,0.794,0.796],[16189,1.0,0.21],[39072,-0.445,0.022],[94303,-1.593,0.001],[227605,-0.8,0.002],[549336,0.277,0.001],[1325849,-0.896,0.0],[3200000,-3.281,0.0]]},"1000_2":{"C":394488,"Dc":32968,"pts":[[1,0.0,1.0],[2,0.0,1.0],[6,0.0,1.0],[14,0.0,1.0],[34,0.001,1.0],[82,0.002,1.0],[198,0.006,0.99],[477,0.014,0.958],[1151,0.033,0.939],[2779,0.082,0.976],[6707,0.198,0.982],[16189,0.465,0.947],[39072,1.0,0.767],[94303,0.995,0.136],[227605,0.099,0.014],[549336,0.219,0.006],[1325849,0.492,0.002],[3200000,-0.599,0.0]]}}};
var ORDER=[["20M",20],["60M",60],["150M",150],["300M",300],["530M",530],["1B",1000]],RANKS=[1,2,4,8,16];
var SZL={20:"20M",60:"60M",150:"150M",300:"300M",530:"530M",1000:"1B"};
var BLUE="#3182bd",OR="#e6801f";
var svg=document.getElementById("cap-svg"),szEl=document.getElementById("cap-sz"),rrEl=document.getElementById("cap-rr"),slab=document.getElementById("cap-slab"),rlab=document.getElementById("cap-rlab");
function l10(v){return Math.log(v)/Math.LN10;}
function fmt(n){return n>=1e6?(n/1e6).toFixed(2)+"M":n>=1e3?(n/1e3).toFixed(0)+"k":(""+n);}
function txt(x,y,s,c,sz,anc,w){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||11)+'" font-weight="'+(w||400)+'" fill="'+(c||"#555")+'" text-anchor="'+(anc||"middle")+'">'+s+'</text>';}
var ML=54,MR=18,MT=34,MB=50,W=620,H=390,PW=W-ML-MR,PH=H-MT-MB;
var Xa=0,Xb=l10(3.2e6),Ya=-0.3,Yb=1.1;
function xp(d){return ML+(l10(d)-Xa)/(Xb-Xa)*PW;}
function yp(v){return MT+(1-(v-Ya)/(Yb-Ya))*PH;}
function cl(v){return v<Ya?Ya:(v>Yb?Yb:v);}
function draw(){
  var curSz=ORDER[+szEl.value][1],curR=RANKS[+rrEl.value];
  slab.textContent=SZL[curSz];rlab.textContent=curR;
  var c=CAP.cells[curSz+"_"+curR];
  var s='<rect x="'+ML+'" y="'+MT+'" width="'+PW+'" height="'+PH+'" fill="none" stroke="#ddd"/>';
  [0,0.5,1].forEach(function(v){var Y=yp(v);s+='<line x1="'+ML+'" y1="'+Y+'" x2="'+(ML+PW)+'" y2="'+Y+'" stroke="#f4f4f4"/>'+txt(ML-8,Y+3,v.toFixed(1),"#999",10,"end");});
  var y0=yp(0);s+='<line x1="'+ML+'" y1="'+y0+'" x2="'+(ML+PW)+'" y2="'+y0+'" stroke="#e8e8e8"/>';
  [[1,"1"],[100,"100"],[1e4,"10K"],[1e6,"1M"]].forEach(function(t){var X=xp(t[0]);s+='<line x1="'+X+'" y1="'+MT+'" x2="'+X+'" y2="'+(MT+PH)+'" stroke="#f4f4f4"/>'+txt(X,MT+PH+16,t[1],"#999",10);});
  s+=txt(ML+PW/2,H-8,"dataset size D (facts)","#666",12);
  if(!c){s+=txt(ML+PW/2,MT+PH/2,"no data","#bbb",14,"middle",600);svg.innerHTML=s;return;}
  var pk=0,pkx=c.pts[0][0],pj;for(pj=0;pj<c.pts.length;pj++){if(c.pts[pj][1]>pk){pk=c.pts[pj][1];pkx=c.pts[pj][0];}}
  var xc=xp(pkx);s+='<line x1="'+xc.toFixed(1)+'" y1="'+MT+'" x2="'+xc.toFixed(1)+'" y2="'+(MT+PH)+'" stroke="#c0392b" stroke-width="1.3" stroke-dasharray="4 3"/>'+txt(xc+4,MT+PH-6,"capacity","#c0392b",10,"start");
  function poly(idx,col){var pl='',i;for(i=0;i<c.pts.length;i++){pl+=(pl?' ':'')+xp(c.pts[i][0]).toFixed(1)+','+yp(cl(c.pts[i][idx])).toFixed(1);}return '<polyline points="'+pl+'" fill="none" stroke="'+col+'" stroke-width="2.4"/>';}
  s+=poly(1,BLUE)+poly(2,OR);
  var i;for(i=0;i<c.pts.length;i++){var q=c.pts[i];
    s+='<circle cx="'+xp(q[0]).toFixed(1)+'" cy="'+yp(cl(q[1])).toFixed(1)+'" r="2.8" fill="'+BLUE+'"><title>'+q[0]+' facts &middot; stored '+(q[1]*100).toFixed(0)+'% of C</title></circle>';
    s+='<circle cx="'+xp(q[0]).toFixed(1)+'" cy="'+yp(cl(q[2])).toFixed(1)+'" r="2.8" fill="'+OR+'"><title>'+q[0]+' facts &middot; recall '+q[2].toFixed(2)+'</title></circle>';}
  var lx=ML+12,ly=MT+PH-38;
  s+='<rect x="'+lx+'" y="'+ly+'" width="12" height="12" fill="'+BLUE+'"/>'+txt(lx+17,ly+10,"stored bits / C","#555",11,"start");
  s+='<rect x="'+lx+'" y="'+(ly+18)+'" width="12" height="12" fill="'+OR+'"/>'+txt(lx+17,ly+28,"recall (accuracy)","#555",11,"start");
  s+=txt(ML+PW-8,MT+14,"C = "+fmt(c.C)+" bits","#333",11,"end",600);
  s+=txt(ML+PW-8,MT+29,"reached at "+fmt(pkx)+" facts","#c0392b",11,"end");
  s+=txt(ML+PW/2,MT-14,"Storage saturates and recall bends at capacity   ("+SZL[curSz]+", rank "+curR+")","#333",12,"middle",600);
  svg.innerHTML=s;
}
szEl.addEventListener("input",draw);rrEl.addEventListener("input",draw);draw();
})();
</script>
  <figcaption style="font-size:12px;color:#777;margin-top:2px;">Stored bits (as a fraction of capacity $C$) and recall against dataset size, for a selected base and rank. Slide either up and the capacity point moves right: both curves shift with it, storage topping out at $C$ just as recall bends down. The red line marks where storage reaches $C$; the next plot rescales this point to $D_c = C/\log_2 V$.</figcaption>
</figure>
{% endcomment %}

For a single base and rank, both stored bits and recall move with dataset size in units of capacity $D_c = C/\log_2 V$. Below $D_c$ the adapter recalls almost every fact and stored bits rise; around $D_c$ storage saturates at $C$ and recall bends over together. Storage stops rising at the same dataset size where recall starts falling, sharing capacity as the yardstick for the size of the training set.[^bitsrecall]

In the following plot, we collapse our family onto one curve by dividing each curve's dataset by its capacity in facts, $D_c = C/\log_2 V$.

<div id="rc-widget" style="position:relative;width:100%;margin:18px 0 4px 0;font-family:inherit;color:#333;">
  <svg id="rc-svg" viewBox="0 0 700 470" style="width:100%;max-width:720px;display:block;margin:0 auto;"></svg>
  <div id="rc-tip" style="position:absolute;pointer-events:none;display:none;background:#fff;border:1px solid #ddd;border-radius:5px;padding:5px 8px;font-size:11px;color:#333;box-shadow:0 1px 4px rgba(0,0,0,0.12);z-index:5;white-space:nowrap;"></div>
</div>
<script>
(function(){
var RC={"fit":[{"szM":60,"r":16,"pts":[[-2.873,1.0],[-2.491,0.976],[-2.108,0.965],[-1.726,0.904],[-1.343,0.962],[-0.961,0.977],[-0.578,0.934],[-0.195,0.793],[0.187,0.444],[0.57,0.019],[0.953,0.001],[1.335,0.002],[1.718,0.0],[2.101,0.0]]},{"szM":300,"r":16,"pts":[[-2.701,0.97],[-2.319,0.981],[-1.936,0.934],[-1.554,0.967],[-1.171,0.977],[-0.788,0.961],[-0.406,0.888],[-0.023,0.633],[0.36,0.207],[0.742,0.011]]},{"szM":530,"r":8,"pts":[[-2.867,1.0],[-2.484,0.985],[-2.102,0.975],[-1.72,0.901],[-1.337,0.963],[-0.954,0.971],[-0.572,0.942],[-0.189,0.835],[0.194,0.443],[0.576,0.08],[0.959,0.006],[1.342,0.002],[1.724,0.0]]},{"szM":530,"r":1,"pts":[[-2.947,1.0],[-2.579,0.929],[-2.194,0.971],[-1.812,1.0],[-1.429,0.949],[-1.047,0.874],[-0.665,0.912],[-0.282,0.929],[0.101,0.75],[0.484,0.096],[0.866,0.003],[1.249,0.001],[1.632,0.001],[2.014,0.002],[2.397,0.0],[2.78,0.0]]},{"szM":530,"r":16,"pts":[[-2.81,0.99],[-2.428,0.973],[-2.046,0.955],[-1.663,0.965],[-1.28,0.977],[-0.898,0.957],[-0.515,0.879],[-0.132,0.757],[0.25,0.324],[0.633,0.051],[1.015,0.004],[1.398,0.0]]},{"szM":20,"r":16,"pts":[[-3.042,1.0],[-2.674,0.929],[-2.289,1.0],[-1.907,0.951],[-1.524,0.732],[-1.142,0.801],[-0.759,0.903],[-0.376,0.873],[0.006,0.6],[0.389,0.197],[0.772,0.001],[1.154,0.0],[1.537,0.0],[1.92,0.0],[2.302,0.0],[2.685,0.0]]},{"szM":60,"r":2,"pts":[[-2.718,1.0],[-2.35,1.0],[-1.965,1.0],[-1.583,0.976],[-1.2,0.859],[-0.818,0.91],[-0.435,0.962],[-0.053,0.887],[0.33,0.249],[0.713,0.021],[1.095,0.003],[1.478,0.002],[1.861,0.001],[2.243,0.001],[2.626,0.0],[3.009,0.0]]},{"szM":300,"r":2,"pts":[[-2.914,1.0],[-2.529,0.971],[-2.146,1.0],[-1.763,0.965],[-1.382,0.927],[-0.999,0.918],[-0.616,0.959],[-0.234,0.915],[0.149,0.587],[0.532,0.046],[0.914,0.001],[1.297,0.001],[1.68,0.001],[2.062,0.001],[2.445,0.0]]},{"szM":20,"r":1,"pts":[[-2.674,1.0],[-2.373,1.0],[-1.896,1.0],[-1.528,1.0],[-1.143,0.765],[-0.761,0.768],[-0.378,0.727],[0.004,0.761],[0.387,0.202],[0.769,0.009],[1.152,0.005],[1.535,0.001],[1.917,0.001],[2.3,0.001],[2.683,0.0]]},{"szM":60,"r":4,"pts":[[-2.694,1.0],[-2.309,1.0],[-1.927,0.976],[-1.544,0.944],[-1.162,0.862],[-0.779,0.945],[-0.397,0.94],[-0.014,0.745],[0.369,0.23],[0.751,0.001],[1.134,0.001],[1.517,0.001],[1.899,0.001],[2.282,0.0],[2.665,0.0]]},{"szM":20,"r":8,"pts":[[-2.784,1.0],[-2.416,0.857],[-2.03,0.941],[-1.648,0.829],[-1.265,0.702],[-0.883,0.786],[-0.501,0.896],[-0.118,0.766],[0.265,0.309],[0.647,0.035],[1.03,0.001],[1.413,0.0],[1.795,0.0],[2.178,0.0],[2.561,0.0],[2.943,0.0]]},{"szM":300,"r":4,"pts":[[-2.862,0.971],[-2.48,1.0],[-2.097,0.995],[-1.715,0.948],[-1.333,0.87],[-0.95,0.96],[-0.567,0.968],[-0.184,0.891],[0.198,0.459],[0.581,0.093],[0.964,0.004],[1.346,0.002],[1.729,0.001],[2.112,0.0]]},{"szM":150,"r":16,"pts":[[-2.856,1.0],[-2.473,0.99],[-2.091,0.985],[-1.709,0.964],[-1.326,0.988],[-0.943,0.987],[-0.56,0.936],[-0.178,0.464],[0.205,0.4],[0.588,0.11],[0.97,0.015],[1.353,0.001],[1.736,0.0]]},{"szM":150,"r":8,"pts":[[-2.956,1.0],[-2.574,1.0],[-2.191,0.97],[-1.809,0.943],[-1.426,0.947],[-1.044,0.983],[-0.661,0.98],[-0.278,0.897],[0.104,0.515],[0.487,0.166],[0.87,0.022],[1.252,0.004],[1.635,0.001],[2.018,0.0]]},{"szM":150,"r":1,"pts":[[-2.605,1.0],[-2.238,1.0],[-1.852,1.0],[-1.47,0.976],[-1.087,0.874],[-0.705,0.895],[-0.323,0.977],[0.06,0.875],[0.443,0.106],[0.826,0.016],[1.208,0.003],[1.591,0.001],[1.974,0.001],[2.356,0.0],[2.739,0.0]]},{"szM":530,"r":4,"pts":[[-2.921,0.971],[-2.539,1.0],[-2.156,0.965],[-1.774,0.954],[-1.391,0.898],[-1.009,0.955],[-0.626,0.966],[-0.243,0.895],[0.139,0.588],[0.522,0.108],[0.905,0.006],[1.287,0.003],[1.67,0.001],[2.053,0.0]]},{"szM":530,"r":2,"pts":[[-3.011,1.0],[-2.626,0.971],[-2.244,1.0],[-1.861,0.975],[-1.479,0.941],[-1.097,0.894],[-0.714,0.944],[-0.331,0.942],[0.052,0.7],[0.434,0.17],[0.817,0.005],[1.2,0.002],[1.582,0.002],[1.965,0.001],[2.348,0.0]]},{"szM":300,"r":1,"pts":[[-2.861,1.0],[-2.493,1.0],[-2.108,1.0],[-1.725,0.988],[-1.342,0.874],[-0.961,0.866],[-0.578,0.97],[-0.195,0.909],[0.187,0.527],[0.57,0.041],[0.953,0.001],[1.335,0.001],[1.718,0.001],[2.101,0.001],[2.483,0.0],[2.866,0.0]]},{"szM":300,"r":8,"pts":[[-2.815,1.0],[-2.432,0.99],[-2.05,0.954],[-1.667,0.931],[-1.284,0.964],[-0.902,0.976],[-0.519,0.944],[-0.136,0.802],[0.246,0.346],[0.629,0.033],[1.012,0.004],[1.394,0.002],[1.777,0.0]]},{"szM":60,"r":8,"pts":[[-3.015,1.0],[-2.629,1.0],[-2.247,0.976],[-1.864,0.955],[-1.482,0.906],[-1.1,0.954],[-0.717,0.97],[-0.334,0.886],[0.048,0.628],[0.431,0.196],[0.814,0.003],[1.196,0.002],[1.579,0.001],[1.962,0.0],[2.344,0.0]]},{"szM":20,"r":4,"pts":[[-3.005,1.0],[-2.528,1.0],[-2.16,1.0],[-1.774,0.912],[-1.392,0.89],[-1.009,0.697],[-0.627,0.78],[-0.245,0.865],[0.138,0.567],[0.521,0.104],[0.903,0.002],[1.286,0.0],[1.669,0.0],[2.051,0.0],[2.434,0.0],[2.817,0.0]]},{"szM":60,"r":1,"pts":[[-2.792,1.0],[-2.315,1.0],[-1.947,1.0],[-1.562,1.0],[-1.179,0.976],[-0.797,0.874],[-0.415,0.925],[-0.032,0.91],[0.351,0.249],[0.733,0.007],[1.116,0.003],[1.499,0.003],[1.881,0.001],[2.264,0.001],[2.647,0.001],[3.029,0.0]]},{"szM":20,"r":2,"pts":[[-3.014,1.0],[-2.713,1.0],[-2.236,1.0],[-1.868,0.786],[-1.483,0.853],[-1.1,0.805],[-0.717,0.732],[-0.336,0.82],[0.047,0.755],[0.43,0.163],[0.812,0.029],[1.195,0.001],[1.578,0.001],[1.96,0.001],[2.343,0.0],[2.726,0.0]]},{"szM":150,"r":4,"pts":[[-2.981,1.0],[-2.596,0.971],[-2.214,1.0],[-1.831,0.985],[-1.449,0.929],[-1.066,0.967],[-0.684,0.979],[-0.301,0.955],[0.082,0.636],[0.464,0.175],[0.847,0.03],[1.23,0.002],[1.612,0.002]]},{"szM":150,"r":2,"pts":[[-2.695,1.0],[-2.31,0.941],[-1.928,1.0],[-1.545,0.939],[-1.163,0.918],[-0.78,0.962],[-0.397,0.982],[-0.015,0.796],[0.368,0.21],[0.751,0.022],[1.133,0.001],[1.516,0.002],[1.899,0.001],[2.281,0.0],[2.664,0.0]]}],"b1":[[-2.96,1.0,1],[-2.574,1.0,1],[-2.192,0.988,1],[-1.809,0.985,1],[-1.427,0.912,1],[-1.045,0.924,1],[-0.662,0.975,1],[-0.279,0.974,1],[0.104,0.716,1],[0.486,0.058,1],[0.869,0.008,1],[1.251,0.003,1],[1.634,0.002,1],[2.017,0.001,1],[2.399,0.0,1],[-2.976,1.0,8],[-2.594,0.994,8],[-2.211,0.972,8],[-1.829,0.981,8],[-1.446,0.985,8],[-1.063,0.984,8],[-0.681,0.972,8],[-0.298,0.917,8],[0.085,0.629,8],[0.467,0.128,8],[0.85,0.015,8],[1.233,0.001,8],[-2.942,0.998,16],[-2.56,0.99,16],[-2.177,0.986,16],[-1.794,0.982,16],[-1.412,0.979,16],[-1.029,0.97,16],[-0.646,0.959,16],[-0.264,0.795,16],[0.119,0.582,16],[0.502,0.064,16],[0.884,0.001,16],[-2.706,0.99,4],[-2.324,0.996,4],[-1.942,0.956,4],[-1.559,0.98,4],[-1.176,0.979,4],[-0.794,0.98,4],[-0.411,0.943,4],[-0.028,0.714,4],[0.354,0.2,4],[0.737,0.019,4],[1.12,0.005,4],[1.502,0.0,4],[-2.987,1.0,2],[-2.604,1.0,2],[-2.221,0.99,2],[-1.84,0.958,2],[-1.457,0.939,2],[-1.074,0.976,2],[-0.692,0.982,2],[-0.309,0.947,2],[0.074,0.767,2],[0.456,0.136,2],[0.839,0.014,2],[1.222,0.006,2],[1.604,0.002,2],[1.987,0.0,2],[-2.957,1.0,8],[-2.575,0.987,8],[-2.192,0.983,8],[-1.81,0.984,8],[-1.427,0.986,8],[-1.044,0.979,8],[-0.662,0.969,8],[-0.279,0.926,8],[0.104,0.619,8],[0.486,0.125,8],[0.869,0.019,8],[1.252,0.001,8],[-2.956,1.0,1],[-2.571,1.0,1],[-2.188,1.0,1],[-1.805,0.975,1],[-1.424,0.948,1],[-1.041,0.93,1],[-0.658,0.98,1],[-0.276,0.954,1],[0.107,0.716,1],[0.49,0.077,1],[0.872,0.009,1],[1.255,0.002,1],[1.638,0.002,1],[2.02,0.001,1],[2.403,0.0,1],[-2.782,0.998,16],[-2.399,0.975,16],[-2.017,0.978,16],[-1.634,0.979,16],[-1.251,0.969,16],[-0.869,0.957,16],[-0.486,0.912,16],[-0.103,0.671,16],[0.279,0.265,16],[0.662,0.015,16],[1.045,0.0,16],[-2.934,0.996,16],[-2.552,0.982,16],[-2.169,0.983,16],[-1.786,0.981,16],[-1.404,0.983,16],[-1.021,0.977,16],[-0.638,0.945,16],[-0.256,0.826,16],[0.127,0.558,16],[0.509,0.042,16],[0.892,0.001,16],[-2.851,1.0,1],[-2.466,1.0,1],[-2.084,0.988,1],[-1.701,0.98,1],[-1.319,0.925,1],[-0.936,0.889,1],[-0.553,0.972,1],[-0.171,0.948,1],[0.212,0.465,1],[0.594,0.028,1],[0.977,0.003,1],[1.36,0.002,1],[1.742,0.001,1],[2.125,0.0,1],[2.508,0.0,1],[-2.876,0.99,8],[-2.494,0.983,8],[-2.111,0.951,8],[-1.729,0.976,8],[-1.346,0.985,8],[-0.963,0.952,8],[-0.581,0.956,8],[-0.198,0.812,8],[0.185,0.418,8],[0.567,0.022,8],[0.95,0.005,8],[1.333,0.0,8],[-3.035,1.0,4],[-2.652,0.995,4],[-2.271,0.992,4],[-1.888,0.946,4],[-1.505,0.969,4],[-1.123,0.982,4],[-0.74,0.982,4],[-0.357,0.939,4],[0.025,0.662,4],[0.408,0.168,4],[0.791,0.017,4],[1.173,0.003,4],[1.556,0.0,4],[-2.942,0.998,16],[-2.559,0.99,16],[-2.177,0.985,16],[-1.794,0.986,16],[-1.411,0.979,16],[-1.029,0.976,16],[-0.646,0.962,16],[-0.263,0.79,16],[0.119,0.58,16],[0.502,0.06,16],[0.885,0.001,16],[-2.993,1.0,4],[-2.61,0.995,4],[-2.228,0.99,4],[-1.845,0.951,4],[-1.462,0.969,4],[-1.08,0.981,4],[-0.697,0.982,4],[-0.314,0.951,4],[0.068,0.717,4],[0.451,0.136,4],[0.834,0.018,4],[1.216,0.005,4],[1.599,0.0,4],[-2.85,0.929,1],[-2.464,1.0,1],[-2.082,1.0,1],[-1.699,0.97,1],[-1.317,0.925,1],[-0.935,0.935,1],[-0.552,0.967,1],[-0.169,0.934,1],[0.214,0.46,1],[0.596,0.069,1],[0.979,0.009,1],[1.361,0.004,1],[1.744,0.002,1],[2.127,0.0,1],[2.509,0.0,1],[-2.963,0.985,8],[-2.581,0.99,8],[-2.199,0.961,8],[-1.816,0.976,8],[-1.433,0.984,8],[-1.051,0.984,8],[-0.668,0.978,8],[-0.285,0.926,8],[0.097,0.644,8],[0.48,0.108,8],[0.863,0.011,8],[1.245,0.001,8],[-2.92,1.0,4],[-2.537,0.995,4],[-2.155,0.992,4],[-1.773,0.943,4],[-1.39,0.973,4],[-1.007,0.975,4],[-0.624,0.958,4],[-0.242,0.925,4],[0.141,0.54,4],[0.523,0.053,4],[0.906,0.007,4],[1.289,0.003,4],[1.671,0.0,4]],"b1pred":[[-2.667,1.0,1],[-2.285,0.988,1],[-1.902,0.985,1],[-1.52,0.912,1],[-1.137,0.924,1],[-0.754,0.975,1],[-0.372,0.974,1],[0.011,0.716,1],[0.393,0.058,1],[0.776,0.008,1],[1.159,0.003,1],[1.541,0.002,1],[1.924,0.001,1],[2.307,0.0,1],[-2.872,1.0,8],[-2.49,0.994,8],[-2.108,0.972,8],[-1.725,0.981,8],[-1.342,0.985,8],[-0.96,0.984,8],[-0.577,0.972,8],[-0.194,0.917,8],[0.188,0.629,8],[0.571,0.128,8],[0.954,0.015,8],[1.336,0.001,8],[-2.814,0.998,16],[-2.431,0.99,16],[-2.049,0.986,16],[-1.666,0.982,16],[-1.283,0.979,16],[-0.901,0.97,16],[-0.518,0.959,16],[-0.135,0.795,16],[0.247,0.582,16],[0.63,0.064,16],[1.013,0.001,16],[-2.932,1.0,4],[-2.549,0.99,4],[-2.167,0.996,4],[-1.784,0.956,4],[-1.402,0.98,4],[-1.019,0.979,4],[-0.636,0.98,4],[-0.254,0.943,4],[0.129,0.714,4],[0.512,0.2,4],[0.894,0.019,4],[1.277,0.005,4],[1.66,0.0,4],[-2.99,1.0,2],[-2.608,1.0,2],[-2.225,0.99,2],[-1.843,0.958,2],[-1.461,0.939,2],[-1.078,0.976,2],[-0.695,0.982,2],[-0.313,0.947,2],[0.07,0.767,2],[0.453,0.136,2],[0.835,0.014,2],[1.218,0.006,2],[1.601,0.002,2],[1.983,0.0,2]],"meta":{"k":2.34,"mid":1.44,"r2_fit":0.978,"r2_1b":0.991,"r2_1b_pred":0.984,"dist_fit":0.035,"dist_1b":0.026,"dist_1bpred":0.03,"n_fit":25}};
var HUE={20:196,60:220,150:250,300:284,530:326},SAT={20:58,60:60,150:52,300:50,530:58};
var RIDX={1:0,2:1,4:2,8:3,16:4};
function col(sz,r){return "hsl("+HUE[sz]+","+SAT[sz]+"%,"+(66-RIDX[r]*7)+"%)";}
function ocol(r){return "hsl(30,85%,"+(68-RIDX[r]*7)+"%)";}
var SZS=[20,60,150,300,530],SZL={20:"20M",60:"60M",150:"150M",300:"300M",530:"530M"};
var svg=document.getElementById("rc-svg"),tip=document.getElementById("rc-tip"),widget=document.getElementById("rc-widget");
function txt(x,y,s,c,sz,anc,w){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||11)+'" font-weight="'+(w||400)+'" fill="'+(c||"#555")+'" text-anchor="'+(anc||"middle")+'">'+s+'</text>';}
function xm(X,Y,r,col,extra){return '<path class="rc-hit" d="M'+(X-r).toFixed(1)+' '+(Y-r).toFixed(1)+'L'+(X+r).toFixed(1)+' '+(Y+r).toFixed(1)+'M'+(X-r).toFixed(1)+' '+(Y+r).toFixed(1)+'L'+(X+r).toFixed(1)+' '+(Y-r).toFixed(1)+'" stroke="'+col+'" stroke-width="1.5" fill="none"'+extra+'/>';}
var ML=56,MR=18,MT=28,MB=74,W=700,H=470,PW=W-ML-MR,PH=H-MT-MB,Xa=-3,Xb=3;
function xp(lx){return ML+(lx-Xa)/(Xb-Xa)*PW;}
function yp(v){return MT+(1-v)*PH;}
var m=RC.meta;
var s='<rect x="'+ML+'" y="'+MT+'" width="'+PW+'" height="'+PH+'" fill="none" stroke="#ddd"/>';
// operating regimes (universal in D/D_c): store <1, lose 1–5, collapse >5
var z5=Math.log(5)/Math.LN10;
s+='<rect x="'+xp(Xa).toFixed(1)+'" y="'+MT+'" width="'+(xp(0)-xp(Xa)).toFixed(1)+'" height="'+PH+'" fill="rgba(44,162,95,0.07)"/>';
s+='<rect x="'+xp(0).toFixed(1)+'" y="'+MT+'" width="'+(xp(z5)-xp(0)).toFixed(1)+'" height="'+PH+'" fill="rgba(230,150,40,0.10)"/>';
s+='<rect x="'+xp(z5).toFixed(1)+'" y="'+MT+'" width="'+(xp(Xb)-xp(z5)).toFixed(1)+'" height="'+PH+'" fill="rgba(192,57,43,0.08)"/>';
for(var v=0;v<=1.0001;v+=0.25){var Y=yp(v);s+='<line x1="'+ML+'" y1="'+Y+'" x2="'+(ML+PW)+'" y2="'+Y+'" stroke="#f4f4f4"/>'+txt(ML-7,Y+3,v.toFixed(2),"#999",10,"end");}
for(var e=-3;e<=3;e++){var X=xp(e);s+='<line x1="'+X+'" y1="'+MT+'" x2="'+X+'" y2="'+(MT+PH)+'" stroke="#f4f4f4"/>'+txt(X,MT+PH+16,(e===0?"1":"10^"+e),"#999",10);}
s+=txt(ML+PW/2,MT+PH+32,"dataset size / capacity   D / D_c","#666",12);
s+='<text x="13" y="'+(MT+PH/2)+'" font-size="12" fill="#666" text-anchor="middle" transform="rotate(-90 13 '+(MT+PH/2)+')">fact recall (accuracy)</text>';
var X1=xp(0);s+='<line x1="'+X1+'" y1="'+MT+'" x2="'+X1+'" y2="'+(MT+PH)+'" stroke="#c0392b" stroke-dasharray="3 3"/>'+txt(X1+4,MT+12,"D = D_c","#c0392b",10,"start");
var X5=xp(z5);s+='<line x1="'+X5.toFixed(1)+'" y1="'+MT+'" x2="'+X5.toFixed(1)+'" y2="'+(MT+PH)+'" stroke="#c0392b" stroke-dasharray="2 3" opacity="0.55"/>'+txt(X5+4,MT+12,"5·D_c","#c0392b",10,"start");
for(var i=0;i<RC.fit.length;i++){var c=RC.fit[i],cc=col(c.szM,c.r);
  for(var j=0;j<c.pts.length;j++){var X=xp(c.pts[j][0]),Y=yp(c.pts[j][1]);
    s+='<circle class="rc-hit" cx="'+X.toFixed(1)+'" cy="'+Y.toFixed(1)+'" r="2.1" fill="'+cc+'" opacity="0.55" data-t="fit" data-z="'+SZL[c.szM]+'" data-r="'+c.r+'" data-x="'+c.pts[j][0]+'" data-a="'+c.pts[j][1]+'"/>';}}
var fl="";for(var e=-3;e<=3;e+=0.05){var rr=1/(1+Math.pow(Math.pow(10,e)/m.mid,m.k));fl+=(fl?" ":"")+xp(e).toFixed(1)+","+yp(rr).toFixed(1);}
s+='<polyline points="'+fl+'" fill="none" stroke="#111" stroke-width="2.4"/>';
var xmid=Math.log(m.mid)/Math.LN10;s+='<circle cx="'+xp(xmid).toFixed(1)+'" cy="'+yp(0.5).toFixed(1)+'" r="3.6" fill="#111"/>'+txt(xp(xmid)+9,yp(0.5)-5,"½ recall at D = "+m.mid+"·D_c","#111",10,"start",600);
for(var i=0;i<RC.b1pred.length;i++){var p=RC.b1pred[i],X=xp(p[0]),Y=yp(p[1]);
  s+='<circle class="rc-hit" cx="'+X.toFixed(1)+'" cy="'+Y.toFixed(1)+'" r="3.4" fill="none" stroke="'+ocol(p[2])+'" stroke-width="1.4" data-t="pred" data-r="'+p[2]+'" data-x="'+p[0]+'" data-a="'+p[1]+'"/>';}
for(var i=0;i<RC.b1.length;i++){var p=RC.b1[i];s+=xm(xp(p[0]),yp(p[1]),3,ocol(p[2]),' data-t="meas" data-r="'+p[2]+'" data-x="'+p[0]+'" data-a="'+p[1]+'"');}
s+=txt(ML+PW/2,MT-8,"recall = 1 / (1 + (D / "+m.mid+"·D_c)^"+m.k+")      fit R²="+m.r2_fit+"   ·   1B held-out R²="+m.r2_1b,"#333",12,"middle",600);
s+=txt(ML+PW-8,MT+18,"Mean error: "+m.dist_fit,"#666",11,"end",600);
var lx=ML+10,ly=MT+PH-74;
s+='<circle cx="'+(lx+9)+'" cy="'+ly+'" r="3" fill="#999" opacity="0.7"/>'+txt(lx+24,ly+3,"fit cells (20M–530M, "+m.n_fit+")","#666",10,"start");
s+='<line x1="'+lx+'" y1="'+(ly+16)+'" x2="'+(lx+18)+'" y2="'+(ly+16)+'" stroke="#111" stroke-width="2.4"/>'+txt(lx+24,ly+19,"fitted sigmoid","#666",10,"start");
s+=xm(lx+9,ly+32,3.4,ocol(16),'')+txt(lx+24,ly+35,"1B held out (measured C)","#666",10,"start");
s+='<circle cx="'+(lx+9)+'" cy="'+(ly+48)+'" r="3.4" fill="none" stroke="'+ocol(16)+'" stroke-width="1.4"/>'+txt(lx+24,ly+51,"1B (predicted C, final)","#666",10,"start");
var by=MT+PH+52,bx=ML;s+=txt(bx,by-11,"base & rank (r1 → r16):","#888",10,"start");
for(var i=0;i<SZS.length;i++){var z=SZS[i],xx=bx+i*74;[1,2,4,8,16].forEach(function(rk,j){s+='<rect x="'+(xx+j*9)+'" y="'+by+'" width="9" height="10" fill="'+col(z,rk)+'"/>';});s+=txt(xx+22,by+22,SZL[z],"#666",10);}
var xx6=bx+5*74;[1,2,4,8,16].forEach(function(rk,j){s+='<rect x="'+(xx6+j*9)+'" y="'+by+'" width="9" height="10" fill="'+ocol(rk)+'"/>';});s+=txt(xx6+22,by+22,"1B","#666",10);
svg.innerHTML=s;
svg.addEventListener("mousemove",function(ev){var t=ev.target;if(t&&t.getAttribute&&t.getAttribute("class")==="rc-hit"){var rc=widget.getBoundingClientRect();
  var tt=t.getAttribute("data-t"),x=(+t.getAttribute("data-x")).toFixed(2),a=(+t.getAttribute("data-a")).toFixed(2),h;
  if(tt=="fit")h="<b>"+t.getAttribute("data-z")+" · r"+t.getAttribute("data-r")+" (fit)</b>";
  else if(tt=="meas")h="<b>1B held out · r"+t.getAttribute("data-r")+" (measured C)</b>";
  else h="<b>1B · r"+t.getAttribute("data-r")+" (predicted C)</b>";
  tip.innerHTML=h+"<br>D/D_c=10^"+x+" · recall "+a;tip.style.display="block";tip.style.left=(ev.clientX-rc.left+12)+"px";tip.style.top=(ev.clientY-rc.top+12)+"px";}else tip.style.display="none";});
svg.addEventListener("mouseleave",function(){tip.style.display="none";});
})();
</script>

<p style="font-size:12px;color:#777;text-align:center;margin:2px auto 0 auto;">Every memorization curve, rescaled by its own capacity in facts $D_c = C/\log_2 V$. The sigmoid is fit on the ≤530M cells alone. Shading marks the regimes: <span style="color:#2ca25f;">store</span> ($D < D_c$), <span style="color:#c9822b;">lose</span> ($D_c$ to $5D_c$), <span style="color:#c0392b;">collapse</span> ($> 5D_c$).</p>

Recall stays near one before crossing $D_c$, halves when passing it, and is gone by about $5\,D_c$. The same behavior applies at almost every size and rank, with a single shared steepness $k \approx 2.3$.

By substituting in capacity, we can fit Recall as function of the base, the adapter, and the dataset directly,

$$\text{recall}(N, p, D) \approx \frac{1}{1 + \big(D\,\log_2 V \,/\, 1.44\,C(N,p)\big)^{k}}, \qquad k \approx 2.3$$

Before training, we can -- similar to scaling law -- apply the fact set size to this equation. Below that $C(N,p)/\log_2 V$, the adapter keeps everything. Above it, it will start trading old facts for new.

## Real-world post-training

The synthetic law makes rank *buy* capacity: $D$ structureless facts force an update spanning on the order of $D$ directions, so a bigger adapter stores more. Does that carry over to real fine-tuning? Does a task with more data, or more information-dense data, need more rank? To find out we ran a spread of real tasks and, for each, measured two things: the reduction in test loss each LoRA rank buys, and the effective rank of the update $\Delta W = BA$ the adapter actually learns. The synthetic random facts are the memorization control at one extreme.

We evaluate on the following tasks:
- **random facts**: the synthetic $(\text{entity}, \text{relation}, \text{value})$ triplets defined earlier, with each fact worth 12 bits and no shared structure to exploit.
- **Aya**: multilingual instruction following, teaching an English base model a language it has barely seen. Knowledge injection, the high-information extreme.
- **PopQA**: real Wikidata facts phrased as questions, split by entity popularity. Very close to our synthetic set and is fully grounded.
- **Super-NaturalInstructions**: a pool of genuinely different instruction task types (translation, arithmetic, NER, summarization, sentiment, and more).

**Setup.** Every adapter uses LoRA on all linear layers (attention and MLP), trained with each task's own objective. In metrics, we consider:
1. Test loss, the held-out negative log-likelihood in bits/token, over the frozen base at each rank
2. **Effective rank** of the update $\Delta W$, which counts how many singular directions carry "real" weight. We compute it as $\exp\big(-\sum_i p_i \log p_i\big)$, the perplexity of the normalized singular values $p_i = s_i/\sum_j s_j$, and take the median over the 65 adapted matrices. A low effective rank means one shared transformation; a high one means many independent directions.

### Aya: Injecting multilinguality

The first task is multilingual instruction tuning on the Aya Collection, thanks to the great effort from Cohere Labs community. Given DataDecide pretraining is primarily English we deem this to be an information-heavy task where model might memorize language the way it memorizes random facts.

*Hypothesis.* The capacity law helps us understand the budget before any training. Plugging the 1B base and each adapter into $C \propto N^{0.38}\,p^{1.08}$ outputs about 190 kbit at rank 1 and 3.7 Mbit at rank 16. We use test loss on the language as the cost of storing the per-language token, which lands in the range of 2 to 6 bits per token (third column of the table below). We can call this 4, which gives us a quick napkin math for predicted capacity:

<div style="font-family:'SFMono-Regular',Consolas,monospace;font-size:12.5px;line-height:1.7;background:#fafafa;border:1px solid #ececec;border-radius:4px;padding:12px 14px;margin:18px 0;color:#333;white-space:pre-wrap;">rank 1     law: ~190 kbit  →  budget ~47k tokens     we train on ~8M (170×)  →  <span style="color:#c0392b;">should collapse</span>
rank 16    law: ~3.7 Mbit  →  budget ~0.9M tokens    we train on ~8M (9×)    →  might hold</div>

If the adapter stored language the way it stores random facts, rank 1 would collapse and rank 16 would pull far ahead.

<table style="margin:16px auto;border-collapse:collapse;font-size:13.5px;color:#333;">
<tr style="border-bottom:1px solid #ccc;color:#666;text-align:left;"><th style="padding:4px 18px;">base</th><th style="padding:4px 18px;">language</th><th style="padding:4px 18px;">base test loss</th><th style="padding:4px 18px;">gain @ rank 1</th><th style="padding:4px 18px;">gain @ rank 16</th></tr>
<tr><td style="padding:3px 18px;vertical-align:middle;" rowspan="4">1B</td><td style="padding:3px 18px;">Hindi</td><td style="padding:3px 18px;text-align:center;">2.08</td><td style="padding:3px 18px;text-align:center;">+0.19</td><td style="padding:3px 18px;text-align:center;">+0.17</td></tr>
<tr><td style="padding:3px 18px;">German</td><td style="padding:3px 18px;text-align:center;">3.68</td><td style="padding:3px 18px;text-align:center;">+0.09</td><td style="padding:3px 18px;text-align:center;">+0.09</td></tr>
<tr><td style="padding:3px 18px;">Turkish</td><td style="padding:3px 18px;text-align:center;">4.04</td><td style="padding:3px 18px;text-align:center;">+0.26</td><td style="padding:3px 18px;text-align:center;">+0.25</td></tr>
<tr style="border-bottom:1px solid #ddd;"><td style="padding:3px 18px;">Swahili</td><td style="padding:3px 18px;text-align:center;">5.34</td><td style="padding:3px 18px;text-align:center;"><b>+1.02</b></td><td style="padding:3px 18px;text-align:center;"><b>+0.98</b></td></tr>
<tr><td style="padding:3px 18px;vertical-align:middle;" rowspan="4">530M</td><td style="padding:3px 18px;">Hindi</td><td style="padding:3px 18px;text-align:center;">2.29</td><td style="padding:3px 18px;text-align:center;">+0.23</td><td style="padding:3px 18px;text-align:center;">+0.23</td></tr>
<tr><td style="padding:3px 18px;">German</td><td style="padding:3px 18px;text-align:center;">4.14</td><td style="padding:3px 18px;text-align:center;">+0.09</td><td style="padding:3px 18px;text-align:center;">+0.07</td></tr>
<tr><td style="padding:3px 18px;">Turkish</td><td style="padding:3px 18px;text-align:center;">4.49</td><td style="padding:3px 18px;text-align:center;">+0.30</td><td style="padding:3px 18px;text-align:center;">+0.29</td></tr>
<tr><td style="padding:3px 18px;">Swahili</td><td style="padding:3px 18px;text-align:center;">6.06</td><td style="padding:3px 18px;text-align:center;"><b>+1.36</b></td><td style="padding:3px 18px;text-align:center;"><b>+1.33</b></td></tr>
</table>

*Result.* Our hypothesis on the effectiveness of rank on Aya is negative. What moves the gain is how unfamiliar the language is to the base (reduction in test loss, bits/token, at 25.6k examples). Rank 1 matches rank 16 within $0.05$ bits in all eight cells, even at 170× its supposed storage budget. The gain tracks the base instead: Swahili, the language the base knows least, gains $10\times$ more than German.[^hindi] Our capacity math says a rank-1 adapter would be hopeless here, but it is not.

**Why rank 1 might suffice.** Our hypothesis treated language as storage with one independent fact per direction. However, our result suggests a language update is high-information yet low-rank. Linguistically, a language has properties a transformer can learn well (eg. word order, morphology), so one small correction could cover more sentences, which carries over to sentences the adapter never saw.

**Aya (8 languages).** We can stress test the hypothesis further by training a single adapter on eight typologically diverse languages across five scripts. If each language were its own direction, the combined update should need eight. In reality, our effective rank stays near $1.5$. The languages appear to share a cross-lingual subspace, and the combined update remains one correction. The spectrum graph will elaborate on this result.

### Super-NaturalInstructions: Massive multitask

In the same spirit, Super-NLI can help us test our capacity laws in the massive multi-task setting. Languages might share a subspace, but genuinely different tasks might not. We combine eight task types from Super-NLI (arithmetic, NER, summarization, sentiment, POS tagging, QA) into one adapter. Here, the effective rank was $\approx 2$ and did not climb with the number of tasks. In fact, the effective rank fell, potentially as a shared <b>instruction following direction</b> absorbed much of the update. Our NLI unrelated tasks are not especially orthogonal in weight space.

### PopQA: Long tail facts

The real-world task closest to our synthetic set is [PopQA](https://arxiv.org/abs/2212.10511), real Wikidata facts phrased as single-answer questions ("What is the capital of ___?"). Some of its entities are famous and some are obscure, measured by the Wikipedia pageviews of the subject entity. We split the dataset into three popularity bins — **head** (popular entities), **torso** (middle), and **tail** (obscure) — and finetune on each. Our bet is on the tail. An obscure entity barely appears in pretraining, so facts about it should be the closest thing to our random triples, and the adapter should have to store them.

Once again, our hypothesis is negative. Gains are rank-flat and the update stays near one direction in every bin as measured by bits/token (in contrast, random fact memorization sits at effective rank $\approx 14$):


<table style="margin:16px auto;border-collapse:collapse;font-size:13.5px;color:#333;">
<tr style="border-bottom:1px solid #ccc;color:#666;text-align:left;"><th style="padding:4px 18px;">PopQA bin (1B)</th><th style="padding:4px 18px;">median pageviews</th><th style="padding:4px 18px;">gain @ rank 1</th><th style="padding:4px 18px;">gain @ rank 16</th><th style="padding:4px 18px;">effective rank</th></tr>
<tr><td style="padding:3px 18px;">tail (obscure)</td><td style="padding:3px 18px;text-align:center;">208</td><td style="padding:3px 18px;text-align:center;">+2.19</td><td style="padding:3px 18px;text-align:center;">+2.21</td><td style="padding:3px 18px;text-align:center;">1.7</td></tr>
<tr><td style="padding:3px 18px;">torso</td><td style="padding:3px 18px;text-align:center;">572</td><td style="padding:3px 18px;text-align:center;">+1.14</td><td style="padding:3px 18px;text-align:center;">+0.98</td><td style="padding:3px 18px;text-align:center;">2.0</td></tr>
<tr><td style="padding:3px 18px;">head (popular)</td><td style="padding:3px 18px;text-align:center;">3592</td><td style="padding:3px 18px;text-align:center;">+1.79</td><td style="padding:3px 18px;text-align:center;">+1.60</td><td style="padding:3px 18px;text-align:center;">1.9</td></tr>
</table>

Long-tail factual QA still *generalizes*. The answers are real entities the base already represents, so the adapter build on existing pathways rather than storing new bits, and one or two directions are enough.

### Rank tracks structure, not data

Across single languages, multilingual injection, massive multi-task, and long-tail facts: **every real task we measured used an effective rank of 1 to 3, while random facts spread across ~14.** Even when we stress test the adapter, such as growing the fact set from 500 to 549k, every real task stays at 1 to 3 regardless of how much data it sees.

Visually, we can observe the spectrum of the updates themselves. Each adapted layer's $\Delta W$ decomposes by SVD into independent directions, $\Delta W = \sum_i \sigma_i\, u_i v_i^\top$, with $\sigma_i$ the strength of each. We normalize the 16 singular values of each layer to sum to one (only the *shape* matters), and average over the 65 layers. A spectrum piled onto $\sigma_1$ is a single-direction update; a flat spectrum fills all sixteen:

<figure style="margin:26px 0;text-align:center;">
<div id="hm-widget" style="position:relative;width:100%;margin:8px 0 4px 0;font-family:inherit;color:#333;">
  <svg id="hm-svg" viewBox="0 0 720 312" style="width:100%;max-width:720px;display:block;margin:0 auto;"></svg>
  <div id="hm-tip" style="position:absolute;pointer-events:none;display:none;background:#fff;border:1px solid #ddd;border-radius:5px;padding:5px 8px;font-size:11px;color:#333;box-shadow:0 1px 4px rgba(0,0,0,0.12);z-index:5;white-space:nowrap;"></div>
</div>
<script>
(function(){
var ROWS=[
 {lab:"random facts (14.1)",s:[0.1745,0.1147,0.0898,0.0752,0.0677,0.0614,0.0568,0.0526,0.0491,0.0457,0.0429,0.04,0.0373,0.0344,0.0311,0.0267]},
 {lab:"PopQA · tail (1.7)",s:[0.8642,0.0615,0.0196,0.0106,0.0078,0.0059,0.0049,0.0041,0.0037,0.0033,0.003,0.0027,0.0025,0.0023,0.0021,0.0019]},
 {lab:"SuperNI · 8 tasks (2.1)",s:[0.8484,0.0548,0.0221,0.0132,0.0098,0.0079,0.0068,0.006,0.0053,0.0048,0.0043,0.004,0.0037,0.0033,0.0031,0.0027]},
 {lab:"Swahili (1.5)",s:[0.8872,0.0765,0.011,0.0054,0.0036,0.0029,0.0023,0.0019,0.0016,0.0015,0.0013,0.0012,0.0011,0.001,0.0009,0.0008]},
 {lab:"German (1.5)",s:[0.8824,0.0714,0.0137,0.0072,0.0047,0.0035,0.0028,0.0024,0.0021,0.0018,0.0017,0.0015,0.0014,0.0012,0.0011,0.001]},
 {lab:"8 languages combined (1.5)",s:[0.891,0.0667,0.0151,0.0071,0.004,0.0029,0.0023,0.0019,0.0016,0.0014,0.0012,0.0011,0.001,0.0009,0.0008,0.0007]}];
var svg=document.getElementById("hm-svg"),tip=document.getElementById("hm-tip"),widget=document.getElementById("hm-widget");
function txt(x,y,str,c,sz,anc,w){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||11)+'" font-weight="'+(w||400)+'" fill="'+(c||"#555")+'" text-anchor="'+(anc||"middle")+'">'+str+'</text>';}
var STOPS=[[0,0,4],[40,11,84],[120,28,109],[197,58,88],[242,124,74],[252,220,140]];
function col(v){var t=(Math.log(Math.max(v,1e-3))/Math.LN10+3)/3;t=Math.max(0,Math.min(1,t));
  var fdx=t*(STOPS.length-1),i=Math.floor(fdx);if(i>STOPS.length-2)i=STOPS.length-2;var u=fdx-i,a=STOPS[i],b=STOPS[i+1];
  return 'rgb('+Math.round(a[0]+(b[0]-a[0])*u)+','+Math.round(a[1]+(b[1]-a[1])*u)+','+Math.round(a[2]+(b[2]-a[2])*u)+')';}
var ML=168,MR=70,MT=38,MB=42,W=720,H=312,PW=W-ML-MR,PH=H-MT-MB,NC=16;
var cw=PW/NC,rh=PH/ROWS.length;
var s=txt(ML+PW/2,20,"Memorization fills the spectrum; other tasks collapse to one direction","#333",13,"middle",600);
ROWS.forEach(function(d,i){var y=MT+i*rh;
  s+=txt(ML-9,y+rh/2+3.5,d.lab,"#333",10.5,"end");
  for(var j=0;j<NC;j++){var x=ML+j*cw;
    s+='<rect class="hm-hit" data-lab="'+d.lab+'" data-i="'+(j+1)+'" data-v="'+d.s[j].toFixed(4)+'" x="'+x.toFixed(1)+'" y="'+y.toFixed(1)+'" width="'+(cw+0.5).toFixed(1)+'" height="'+(rh+0.5).toFixed(1)+'" fill="'+col(d.s[j])+'"/>';}
});
[1,4,8,12,16].forEach(function(t){var X=ML+(t-0.5)*cw;s+=txt(X,MT+PH+16,""+t,"#999",10);});
s+=txt(ML+PW/2,H-6,"singular value of ΔW  (largest → smallest)","#666",12);
var cbx=ML+PW+16,cbw=12,cbh=PH;
for(var q=0;q<=40;q++){var t=q/40,v=Math.pow(10,-3+3*t),yy=MT+cbh*(1-t);
  s+='<rect x="'+cbx+'" y="'+yy.toFixed(1)+'" width="'+cbw+'" height="'+(cbh/40+0.6).toFixed(1)+'" fill="'+col(v)+'"/>';}
[[1,"1"],[0.1,"0.1"],[0.01,"0.01"],[0.001,"0.001"]].forEach(function(t){var yy=MT+cbh*(1-(Math.log(t[0])/Math.LN10+3)/3);s+=txt(cbx+cbw+3,yy+3,t[1],"#999",8.5,"start");});
svg.innerHTML=s;
svg.addEventListener("mousemove",function(ev){var t=ev.target;if(t&&t.getAttribute&&t.getAttribute("class")==="hm-hit"){var rc=widget.getBoundingClientRect();
  tip.innerHTML="<b>"+t.getAttribute("data-lab")+"</b><br>σ"+t.getAttribute("data-i")+" = "+t.getAttribute("data-v");
  tip.style.display="block";tip.style.left=(ev.clientX-rc.left+12)+"px";tip.style.top=(ev.clientY-rc.top+12)+"px";}else tip.style.display="none";});
svg.addEventListener("mouseleave",function(){tip.style.display="none";});
})();
</script>
<figcaption style="font-size:12px;color:#777;margin-top:6px;">Mean singular-value spectrum of ΔW across the 65 adapted matrices (rank-16 adapters, 1B base); each row normalized to sum 1, log scale. The number after each label is the effective rank. Hover any cell for its value.</figcaption>
</figure>

The spectrum makes the two regimes visible at a glance. Random facts spread their weight across all sixteen directions, while every real task piles onto the first one or two, including the combined 8-language and 8-task adapters.

**The practical reading.** Once again, we can see that binding constraint is data and base quality, rather than adapter rank, where capability can be bought with more examples or a stronger base.

**Previous works.** Morris et al. emphasize the difficulty of separating memorization from the rest of a model's behavior: *"a language model prompted to add two numbers can output the answer without having seen the equation before."* PopQA is a case in point here. Obscure facts show a large loss drop that looks like memorization, yet the update is low-rank and generalizing. Their remedy separates the two in the loss, decomposing against a reference model pretrained from scratch; our adapter's rank separates them in the weights, since memorization fills up the ranks and generalization does not. The rank-flatness itself is an older thread. Aghajanyan et al. found that fine-tuning has a tiny intrinsic dimension, and Schulman et al.'s *LoRA Without Regret* argues from capacity that rank 1 already exceeds what most post-training needs. The one place the literature sees rank clearly bite, Biderman et al.'s *LoRA Learns Less*, finds LoRA trailing full fine-tuning on code and math, with the gap closing only as rank grows — exactly the information-dense, memorization-like regime our spectrum picks out.

## How to know the smallest rank?

In production, we are always interested in shipping the smallest rank. It is the cheapest to store, to serve, to merge back into the base, and leaves no capacity idle. The usual way to find out is to sweep several ranks for a comparison, which multiplies the number of training runs. The results above point to a cheaper route, given that we have observed adapter rank is fixed by task structure rather than by data volume.

Unfortunately, task structure is hard to gauge. Two things set the rank a task needs: enough directions to express the transformation itself, which is 1 to 3 for every real task we measured, and enough capacity to cover whatever must be stored outright. Real tasks are bound by the first; pure fact memorization by the second. Neither can be read off the frozen base. In our experiments, the rank of the loss gradient at initialization does not predict the rank of the converged update — the update's rank is a property of the optimization trajectory, not of the initial geometry. Instead of predicting the rank, we can measure it via a quick adapter training run.

### A simple method

Training cost is nearly independent of rank: for the 1B base a rank-32 adapter is only about 0.2% of the trainable parameters, so rank 32 and rank 4 are both reasonably cheap. The usual reason to search over rank therefore disappears. We train once at a generous fixed rank $R$ (32 or 64), and recover the rank the task needs from that single run.

Make the target precise. Fix a tolerance $\epsilon$, write $\Delta W^\star$ for a converged rank-$R$ update and $\Delta W^\star_k$ for its rank-$k$ SVD truncation, and let $\mathcal{L}$ be the held-out loss. The rank worth deploying is the smallest that surrenders no more than $\epsilon$ of the achievable loss reduction,

$$r^\star \;=\; \min\big\{\, k \;:\; \mathcal{L}(\Delta W^\star_k) \;\le\; \mathcal{L}(\Delta W^\star) + \epsilon \,\big\}.$$

The point is that $r^\star$ is a property of the *already-trained* $\Delta W^\star$, so no retraining is needed to find it. Per the [Eckart–Young–Mirsky theorem](https://en.wikipedia.org/wiki/Low-rank_approximation), the top-$k$ SVD truncation is the best rank-$k$ approximation of a matrix you can make, and its error is exactly the singular values you threw away, $\lVert \Delta W^\star - \Delta W^\star_k\rVert_F^2 = \sum_{i>k}\sigma_i^2$. As long as the loss varies smoothly with the weights, a small tail means a small change in loss, and the spectrum of one generous run pins down $r^\star$.

**Algorithm.** Given a base, task data, a "generous" rank $R$, and tolerance $\epsilon$:

<div style="margin:18px 0;padding:16px 20px;background:#f6f8fa;border:1px solid #e4e9ee;border-radius:8px;display:flex;flex-direction:column;gap:10px;">
  <div style="display:flex;gap:14px;align-items:baseline;">
    <span style="flex:0 0 auto;font-size:15px;font-weight:700;color:#3182bd;">1</span>
    <div style="line-height:1.6;font-size:13.5px;"><b>Train once, generously.</b> Fit the adapter at rank $R$ to convergence.</div>
  </div>
  <div style="display:flex;gap:14px;align-items:baseline;">
    <span style="flex:0 0 auto;font-size:15px;font-weight:700;color:#3182bd;">2</span>
    <div style="line-height:1.6;font-size:13.5px;"><b>Decompose.</b> Take the SVD of each layer's converged update, $\Delta W^\star_\ell = U_\ell \Sigma_\ell V_\ell^\top$.</div>
  </div>
  <div style="display:flex;gap:14px;align-items:baseline;">
    <span style="flex:0 0 auto;font-size:15px;font-weight:700;color:#3182bd;">3</span>
    <div style="line-height:1.6;font-size:13.5px;"><b>Read off the rank.</b> $r^\star$ is the smallest $k$ that keeps $1-\epsilon'$ of the spectral energy, $\sum_{i\le k}\sigma_i^2 / \sum_i \sigma_i^2$ — a weight-space proxy for the loss tolerance, justified by Eckart–Young–Mirsky above.</div>
  </div>
  <div style="display:flex;gap:14px;align-items:baseline;">
    <span style="flex:0 0 auto;font-size:15px;font-weight:700;color:#3182bd;">4</span>
    <div style="line-height:1.6;font-size:13.5px;"><b>Truncate and ship.</b> Keep the top $r^\star$ directions and refold them into the LoRA factors $B, A$.</div>
  </div>
</div>

**Results.** To confirm the measured rank is the rank the task needs, we truncate each trained rank-32 adapter to rank $k$ by its rank-$k$ SVD and remeasure the test loss, with no retraining. The regimes separate cleanly. For the three real tasks the loss is already at its floor by rank 1 and stays flat out to rank 32. Interestingly, the <b>truncated adapter for PopQA</b> is even slightly better at rank 1, where discarding directions acts as helpful _regularization_. A skill's $\Delta W$ already lives in one or two directions, so its rank-1 approximation reconstructs almost the entire update. Random-fact recall is the opposite: the loss keeps falling all the way to rank 32, since each direction stores more facts. The measured effective rank therefore coincides with the rank the task uses, and whether truncation is free or costly identifies the regime after the fact. The rank trajectory shows the same signal even earlier: a skill's update collapses toward one direction over training while memorization climbs, so the direction of the trend separates the two well before convergence.

<figure style="margin:26px 0;text-align:center;">
<div id="rt-widget" style="position:relative;width:100%;margin:8px 0 4px 0;font-family:inherit;color:#333;">
  <svg id="rt-svg" viewBox="0 0 640 400" style="width:100%;max-width:660px;display:block;margin:0 auto;"></svg>
</div>
<script>
(function(){
var SERIES=[
 {lab:"random facts (200k)",col:"#c0392b",w:3,end:"3 epochs",p:[[50,5.89],[100,2.02],[200,1.43],[500,1.27],[1000,1.47],[2000,1.98],[5000,2.91],[10000,4.83],[20000,9.93]]},
 {lab:"random facts (8k)",col:"#e8836e",w:2,end:"80 epochs",p:[[20,16.86],[50,6.29],[100,1.94],[200,1.44],[400,1.33],[800,1.4],[1600,1.79],[3200,3.32],[5000,5.85],[10000,12.18],[20000,19.27]]},
 {lab:"Swahili",col:"#3182bd",w:2,p:[[20,23.97],[50,22.86],[100,21.01],[200,14.9],[400,5.05],[800,2.49],[1600,1.69],[3200,1.47],[5000,1.43],[10000,1.43],[20000,1.61]]},
 {lab:"SuperNI (8 tasks)",col:"#e6801f",w:2,p:[[20,24.42],[50,22.29],[100,19.78],[200,12.07],[400,6.67],[800,3.39],[1600,2.23],[3200,2.01],[5000,1.86],[10000,1.81],[20000,1.91]]},
 {lab:"PopQA",col:"#2ca25f",w:2,p:[[20,21.83],[50,18.24],[100,10.18],[200,2.87],[400,1.94],[800,1.58],[1600,1.53],[3200,1.54],[5000,1.63],[10000,1.92],[20000,2.63]]}];
var svg=document.getElementById("rt-svg");
function l10(v){return Math.log(v)/Math.LN10;}
function txt(x,y,s,c,sz,anc,w){return '<text x="'+x+'" y="'+y+'" font-size="'+(sz||11)+'" font-weight="'+(w||400)+'" fill="'+(c||"#555")+'" text-anchor="'+(anc||"middle")+'">'+s+'</text>';}
var ML=52,MR=16,MT=42,MB=46,W=640,H=400,PW=W-ML-MR,PH=H-MT-MB;
var Xa=l10(18),Xb=l10(24000),Ya=0,Yb=l10(26);
function xp(v){return ML+(l10(v)-Xa)/(Xb-Xa)*PW;}
function yp(v){return MT+(1-(l10(v)-Ya)/(Yb-Ya))*PH;}
var s='<rect x="'+ML+'" y="'+MT+'" width="'+PW+'" height="'+PH+'" fill="none" stroke="#ddd"/>';
s+=txt(ML+PW/2,24,"Trajectory of effective ranks across tasks","#333",13,"middle",600);
[1,2,5,10,20].forEach(function(t){var Y=yp(t);s+='<line x1="'+ML+'" y1="'+Y+'" x2="'+(ML+PW)+'" y2="'+Y+'" stroke="#f4f4f4"/>'+txt(ML-8,Y+3,""+t,"#999",10,"end");});
[20,100,1000,10000].forEach(function(t){var X=xp(t);s+='<line x1="'+X+'" y1="'+MT+'" x2="'+X+'" y2="'+(MT+PH)+'" stroke="#f6f6f6"/>'+txt(X,MT+PH+16,t>=1000?(t/1000)+"k":""+t,"#999",10);});
s+=txt(ML+PW/2,H-8,"training step (log)","#666",12);
s+='<text x="14" y="'+(MT+PH/2)+'" font-size="12" fill="#666" text-anchor="middle" transform="rotate(-90 14 '+(MT+PH/2)+')">effective rank of ΔW</text>';
SERIES.forEach(function(d){var pl='',i;
  for(i=0;i<d.p.length;i++){pl+=(pl?' ':'')+xp(d.p[i][0]).toFixed(1)+','+yp(d.p[i][1]).toFixed(1);}
  s+='<polyline points="'+pl+'" fill="none" stroke="'+d.col+'" stroke-width="'+d.w+'"/>';
  for(i=0;i<d.p.length;i++){s+='<circle cx="'+xp(d.p[i][0]).toFixed(1)+'" cy="'+yp(d.p[i][1]).toFixed(1)+'" r="3" fill="'+d.col+'"><title>'+d.lab+' — step '+d.p[i][0]+': eff rank '+d.p[i][1]+'</title></circle>';}
  if(d.end){var lp=d.p[d.p.length-1];s+=txt(xp(lp[0])-6,yp(lp[1])+3,d.end,d.col,9.5,"end",600);}
});
var lx=ML+PW-196,ly=MT+8;
SERIES.forEach(function(d,j){var yy=ly+j*17;
  s+='<line x1="'+lx+'" y1="'+yy+'" x2="'+(lx+18)+'" y2="'+yy+'" stroke="'+d.col+'" stroke-width="'+d.w+'"/>'+txt(lx+24,yy+4,d.lab,d.col,10.5,"start");});
svg.innerHTML=s;
})();
</script>
<figcaption style="font-size:12px;color:#777;margin-top:2px;">Effective rank of ΔW over a single rank-32 training run (1B base, log–log). Each dot is a checkpoint. Both random-fact sizes climb as facts accumulate — the 8k set faster, since at a fixed step count it has made more passes over its data — while the real tasks fall to rank 1–2 within a few hundred steps and stay there.</figcaption>
</figure>

The truncated adapter also holds for decoding, not just test loss. From greedy generation, the rank-1 version reproduces the full-rank outputs almost token for token, and PopQA exact-match is flat from rank 1 (about 0.15, against 0 for the frozen base) while random fact recall again climbs with rank. So the truncation is capable of preserving generation quality, not only its likelihood.

## Limitations

- Our experiments are limited to a dense transformer architecture and the layers where LoRA is applied.
- Every adapter keeps the token embeddings and LM head frozen, with LoRA on the attention and MLP linear layers only. For tasks that shift the token distribution, such as a new language, training the embeddings may add capability that our adapter architecture cannot reach. We saw hints of this on the multilingual task, but did not pursue it further.
- We lightly tune learning rates across all experiments, but cannot guarantee that they are optimal.
- Task gains are measured mainly as test loss. We spotchecked greedy generation under truncation and found the rank-1 adapter reproduces the full adapter's outputs (PopQA exact-match flat from rank 1; fact recall climbing with rank), buthave not run a broad downstream evaluation across all tasks.

## Notation

<details>
<summary style="cursor:pointer;color:#3182bd;font-size:13.5px;">Show the symbol table</summary>

<table style="width:100%;border-collapse:collapse;font-size:13.5px;margin:8px 0 0 0;">
<tr style="border-top:1px solid #eaecef;color:#888;text-align:left;"><th style="padding:6px 16px;width:110px;">symbol</th><th style="padding:6px 16px;">meaning</th></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;"><i>N</i></td><td style="padding:6px 16px;">base-model parameter count (model size)</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;"><i>T</i></td><td style="padding:6px 16px;">number of tokens the base was pretrained on</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;"><i>r</i></td><td style="padding:6px 16px;">LoRA rank</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;"><i>p</i></td><td style="padding:6px 16px;">adapter trainable parameters, &asymp; 2&thinsp;<i>r</i>&thinsp;<i>d</i><sub>model</sub> per matrix, summed over layers</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;"><i>D</i></td><td style="padding:6px 16px;">fine-tuning dataset size (number of facts)</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;"><i>V</i></td><td style="padding:6px 16px;">candidate values per fact (here <i>V</i> = 4000)</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;">NLL<sub>2</sub></td><td style="padding:6px 16px;">negative log-likelihood of the answer, in bits</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;">bits</td><td style="padding:6px 16px;">memorized bits for a fact = log<sub>2</sub><i>V</i> &minus; NLL<sub>2</sub>(answer | query)</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;"><i>C</i></td><td style="padding:6px 16px;">capacity: the maximum memorized bits over dataset size <i>D</i></td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;"><i>b</i> = <i>C</i>/<i>p</i></td><td style="padding:6px 16px;">bits stored per adapter parameter (density), &asymp; 0.05–0.4 for the 20M–1B bases tested</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;"><i>D</i><sub>c</sub></td><td style="padding:6px 16px;">capacity in facts, <i>D</i><sub>c</sub> = <i>C</i> / log<sub>2</sub><i>V</i></td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;"><i>k</i></td><td style="padding:6px 16px;">steepness of the recall curve (sigmoid exponent)</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;">&Delta;<i>W</i></td><td style="padding:6px 16px;">the adapter's update, &Delta;<i>W</i> = (&alpha;/<i>r</i>)&thinsp;<i>BA</i>, per adapted layer</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;">&sigma;<sub>i</sub></td><td style="padding:6px 16px;">singular values of &Delta;<i>W</i>; their normalized entropy exponentiated is the effective rank</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;"><i>R</i>, <i>r</i><sup>&#9733;</sup></td><td style="padding:6px 16px;">generous training rank and deployed (truncation) rank</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;">&epsilon;, &epsilon;&prime;</td><td style="padding:6px 16px;">loss tolerance for truncation, and its spectral-energy proxy</td></tr>
<tr style="border-top:1px solid #f0f0f0;"><td style="padding:6px 16px;"><i>W</i><sub>0</sub>, <i>A</i>, <i>B</i>, &alpha;</td><td style="padding:6px 16px;">LoRA update <i>W</i> = <i>W</i><sub>0</sub> + (&alpha;/<i>r</i>)&thinsp;<i>BA</i>; only <i>A</i>, <i>B</i> are trained</td></tr>
</table>
</details>


## References

Thanks to Claude Opus 4.8 for assistance in making these figures and writing code for the experiments.

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
- Aghajanyan et al. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.*
- Biderman et al. (2024). *LoRA Learns Less and Forgets Less.*
- Morris et al. (2025). *How Much Do Language Models Memorize?*
- Allen-Zhu & Li (2024). *Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws.*
- Busbridge et al. (2025). *Distillation Scaling Laws.*
- Schulman et al. (2025). *LoRA Without Regret.* Thinking Machines Lab. [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora)
- Magnusson et al. (2025). *DataDecide: How to Predict Best Pretraining Data with Small Experiments.* Ai2.
- Singh et al. (2024). *Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning.*
- Mallen et al. (2023). *When Not to Trust Language Models (PopQA).*

## Citation

If you find this useful, please cite it as:

```bibtex
@misc{nguyen2026loracapacity,
  author       = {Nguyen, Tai},
  title        = {Capacity of a LoRA Adapter (Part 1)},
  year         = {2026},
  howpublished = {\url{https://taidnguyen.github.io/blog/lora-capacity/}},
  note         = {Blog post}
}
```

Tai Nguyen. "Capacity of a LoRA Adapter (Part 1)." 2026. [taidnguyen.github.io/blog/lora-capacity](https://taidnguyen.github.io/blog/lora-capacity/)

[^hindi]: Hindi looks like the most familiar language in the table, which is suspicious for an English-pretrained base. I suspect that the low loss is a tokenizer artifact. The OLMo tokenizer contains little Devanagari, so it shatters Hindi text into many tiny tokens, and each tiny token is easy to predict on its own. A Hindi sentence might simply spend many more tokens than a German one. Regardless, Hindi's gain is still flat in rank so our claim still stands.

[^bitsrecall]: Stored bits and recall are two readouts of the same per-fact distribution $P(a \mid q)$. Bits integrate the model's confidence ($\log_2 V - \text{NLL}$), so they can go negative once overwriting makes the model worse than a uniform guess on old facts. Recall thresholds the same distribution (is the top answer right?), so it is bounded between chance and one. Below $D_c$ the two move together; past it, overwriting is visible only in bits, while recall floors at chance.

[^morrisbits]: Morris et al. define the information content of a uniform-random dataset as $H(x) = N\,S\,\log_2 V$ and estimate the leftover code length under the trained model by its negative log likelihood, then take memorization to be the difference. With single-token answers ($S=1$) that reduces to the per-fact expression above. Because the data is uniform random, there is no generalizable structure to subtract, so I use the uniform prior directly rather than the reference-model term they need for real text.
