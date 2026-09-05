---
layout: blog
title: "Capacity of a LoRA Adapter (Part 2): Quantization"
date: 2026-08-03
categories: blog
topic: capacity, LoRA, quantization, scaling
published: false
permalink: /blog/lora-capacity-part2/
description: "Push weight compression to the limit"
read_time: 2
comments: false
sidenotes: true
---
## Introduction

In [Part 1](/blog/lora-capacity/), we measured that a LoRA adapter holds about **0.05 to 0.4 bits per trainable parameter**, set by the quality of its base. Meanwhile, each of those parameters sits in a **16-bit** weight. The adapter carries 40× to 300× less information than the precision it is stored in. It is mostly empty.

This part asks what happens when we stop paying for the empty space. We quantize the adapter, then the base it borrows from, and push both toward the floor. First, what does quantization actually do to a weight? It snaps each weight to the nearest value on a coarse shared grid. The *scheme* decides where the grid lines fall, and the *bit-width* decides how many lines there are:

<div id="qs-wrap" style="width:100%;margin:16px 0 4px 0;font-family:inherit;">
  <div id="qs-ctrl" style="text-align:center;margin-bottom:4px;"></div>
  <svg id="qs-svg" viewBox="0 0 680 306" style="width:100%;display:block;margin:0 auto;"></svg>
</div>
<script>
(function(){
  var NF4=[-1,-0.6962,-0.5251,-0.3949,-0.2844,-0.1848,-0.0910,0,0.0796,0.1609,0.2461,0.3379,0.4407,0.5626,0.7230,1];
  var FP4=[-6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6].map(function(v){return v/6;});
  var svg=document.getElementById("qs-svg"),ctrl=document.getElementById("qs-ctrl");
  var X0=118,X1=560, RY={dist:44,bf16:92,sym:136,asym:180,nf4:222,fp4:264};
  var state={b:4};
  function Xw(w){return X0+(w+1)/2*(X1-X0);}
  function symLev(b){if(b===1)return[-0.5,0.5];var q=Math.pow(2,b-1)-1,a=[];for(var k=-q;k<=q;k++)a.push(k/q);return a;}
  function asymLev(b){var n=Math.pow(2,b),lo=-0.3,hi=1.0,a=[];for(var k=0;k<n;k++)a.push(lo+k*(hi-lo)/(n-1));return a;}
  function lab(y,t,col){return '<text x="108" y="'+(y+3)+'" font-size="10" fill="'+(col||"#555")+'" text-anchor="end">'+t+'</text>';}
  function cnt(y,t){return '<text x="'+(X1+10)+'" y="'+(y+3)+'" font-size="9" fill="#999">'+t+'</text>';}
  function row(y,lev,c){var s='<line x1="'+X0+'" y1="'+y+'" x2="'+X1+'" y2="'+y+'" stroke="#ddd"/>';
    var draw=lev,tw=1.4;
    if(lev.length>48){draw=[];for(var k=0;k<48;k++)draw.push(lev[Math.round(k*(lev.length-1)/47)]);tw=0.6;}   // dense comb: too many to count
    draw.forEach(function(w){var x=Xw(w);s+='<line x1="'+x+'" y1="'+(y-7)+'" x2="'+x+'" y2="'+(y+7)+'" stroke="'+c+'" stroke-width="'+tw+'"/>';});return s;}
  function render(){
    var b=state.b, dy=RY.dist, path="M"+X0+","+(dy+16);
    for(var i=0;i<=60;i++){var w=-1+2*i/60,f=Math.exp(-Math.pow((w-0.12)/0.3,2));path+=" L"+Xw(w).toFixed(1)+","+(dy+16-f*30).toFixed(1);}
    path+=" L"+X1+","+(dy+16)+" Z";
    var s='<line x1="'+Xw(0)+'" y1="30" x2="'+Xw(0)+'" y2="278" stroke="#eee"/>';
    s+='<path d="'+path+'" fill="#e6eef5" stroke="#9ecae1" stroke-width="1"/>';
    s+=lab(dy,"weights")+cnt(dy,"a real adapter");
    s+='<rect x="'+X0+'" y="'+(RY.bf16-6)+'" width="'+(X1-X0)+'" height="12" fill="#c6dbef"/>'+lab(RY.bf16,"bf16")+cnt(RY.bf16,"continuous");
    var sl=symLev(b); s+=row(RY.sym,sl,"#c0392b")+lab(RY.sym,"INT symmetric")+cnt(RY.sym,b===1?"2 &middot; sign&times;mean":(sl.length>40?sl.length+" levels":(Math.pow(2,b)-1)+" levels"));
    var al=asymLev(b); s+=row(RY.asym,al,"#2c7fb8")+lab(RY.asym,"INT asymmetric")+cnt(RY.asym,al.length>40?al.length+" levels":Math.pow(2,b)+" levels");
    if(al.length<=40){var zp=al.reduce(function(p,c){return Math.abs(c)<Math.abs(p)?c:p;});
      s+='<circle cx="'+Xw(zp)+'" cy="'+RY.asym+'" r="3.4" fill="none" stroke="#2c7fb8" stroke-width="1.3"/><text x="'+Xw(zp)+'" y="'+(RY.asym-11)+'" font-size="7.5" fill="#2c7fb8" text-anchor="middle">zero-pt</text>';}
    var on=(b===4), gc="#d3d3d3";   // NF4/FP4 always shown; greyed unless at 4-bit (they are 4-bit-only)
    s+=row(RY.nf4,NF4,on?"#756bb1":gc)+lab(RY.nf4,"NF4",on?null:"#bbb")+cnt(RY.nf4,on?"16 levels":"16 &middot; 4-bit only");
    s+=row(RY.fp4,FP4,on?"#31a354":gc)+lab(RY.fp4,"FP4",on?null:"#bbb")+cnt(RY.fp4,on?"16 levels":"16 &middot; 4-bit only");
    ["&minus;max","0","+max"].forEach(function(t,i){s+='<text x="'+Xw(i-1)+'" y="293" font-size="8.5" fill="#aaa" text-anchor="middle">'+t+'</text>';});
    svg.innerHTML=s;}
  function renderCtrl(){var h='<span style="color:#999;font-size:11px;margin-right:6px;">bit-width:</span>';
    [8,4,3,2,1].forEach(function(b){var on=state.b===b;
      h+='<button data-b="'+b+'" style="border:1px solid '+(on?"#c0392b":"#d0d0d0")+";background:"+(on?"#c0392b":"#fff")+";color:"+(on?"#fff":"#666")+';border-radius:4px;padding:2px 9px;margin:0 2px;cursor:pointer;font-size:11px;">'+b+"-bit</button>";});
    ctrl.innerHTML=h;}
  ctrl.addEventListener("click",function(e){var b=e.target;if(b.tagName!=="BUTTON")return;state.b=+b.getAttribute("data-b");renderCtrl();render();});
  renderCtrl();render();
})();
</script>
<p style="font-size:12px;color:#777;text-align:center;margin:6px auto 16px auto;max-width:680px;">How each scheme lays its grid over a real, positively skewed adapter weight distribution (top). Use the bit-width buttons to change the number of levels. NF4 and FP4 are fixed 4-bit codebooks, so they light up only at 4-bit.</p>

## Outline

The full post lands next week. Planned sections:

- **Topology of LoRA adapter weights.** What the trained weights look like and where each grid lands on them.
- **How small can the adapter get?** A sweep from 16 bits down to 1, and why asymmetric integer grids hold up best.
- **Quantizing base and adapter together.** QLoRA-style bases and training under quantization noise.
- **Capacity per byte.** The cheapest way to store and serve an adapter.
