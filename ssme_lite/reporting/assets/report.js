'use strict';
(() => {
const D = JSON.parse(document.getElementById('report-data').textContent);
const $ = id => document.getElementById(id);
const M = D.metadata, names = D.models, metrics = Object.keys(D.metrics);
const earlyStopping = M.early_stopping ?? true;
const labels = {accuracy:'Accuracy', auc:'AUC', auprc:'AUPRC', ece:'ECE'};
const colors = ['#6a69b6','#629b99','#c19270','#9e86a3','#869ab5','#a8a473','#b77d88'];
const ink = '#292a36', accent = '#6a69b6', muted = '#a6a8b6';
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite = v => typeof v === 'number' && Number.isFinite(v);
const fmt = (v, n=4) => finite(v) ? v.toFixed(n) : '—';
const pct = v => finite(v) ? (100*v).toFixed(1)+'%' : '—';
const short = i => String(names[i]).replace(/^alg_/, '');
const code = i => 'M'+String(i+1).padStart(2,'0');
const col = i => colors[i % colors.length];
const metricName = k => labels[k] || k;
let metric = M.primary_metric || metrics[0], pinned = null, hovering = null, spaceColor = 'class';
let pairA = 0, pairB = Math.min(1,names.length-1);
const mobile = () => window.matchMedia('(max-width: 760px)').matches;
const tip = text => `data-tip="${esc(text)}"`;
const text = (x,y,s,attrs='') => `<text x="${x}" y="${y}" ${attrs}>${esc(s)}</text>`;
const line = (x1,y1,x2,y2,attrs='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
const circle = (x,y,r,attrs='') => `<circle cx="${x}" cy="${y}" r="${r}" ${attrs}/>`;
const rect = (x,y,w,h,attrs='') => `<rect x="${x}" y="${y}" width="${Math.max(0,w)}" height="${Math.max(0,h)}" ${attrs}/>`;
const svg = (w,h,body,title) => `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title>${body}</svg>`;
const empty = message => `<div class="empty">${esc(message)}</div>`;
const extent = (values, clamp=false) => {
 const valid=values.filter(finite); if(!valid.length)return [0,1];
 let lo=Math.min(...valid), hi=Math.max(...valid), pad=Math.max((hi-lo)*.15,.001);
 return clamp ? [Math.max(0,lo-pad),Math.min(1,hi+pad)] : [lo-pad,hi+pad];
};
const scale = (domain,a,b) => v => a+(v-domain[0])/(domain[1]-domain[0] || 1)*(b-a);
const tick = v => Math.abs(v)>0 && Math.abs(v)<.001 ? v.toExponential(1) : Math.abs(v)>=100 ? v.toFixed(0) : v.toFixed(3).replace(/0+$/,'').replace(/\.$/,'');
const axisTick = (v,domain) => {
 const digits=Math.min(8,Math.max(0,Math.ceil(-Math.log10((domain[1]-domain[0])/4))+1));
 return Math.abs(v)>0&&Math.abs(v)<.0001?v.toExponential(1):String(Number(v.toFixed(digits)));
};
function axisX(domain,left,right,top,bottom,label='') {
 const sx=scale(domain,left,right); let s='';
 for(let i=0;i<=4;i++){const v=domain[0]+(domain[1]-domain[0])*i/4,x=sx(v);s+=line(x,top,x,bottom,'class="grid"')+text(x,bottom+22,axisTick(v,domain),'text-anchor="middle" class="axis-label"');}
 if(label)s+=text((left+right)/2,bottom+43,label,'text-anchor="middle" class="axis-label"');
 return s;
}
function dotPlot(key,small=false){
 const d=D.metrics[key], w=small?(mobile()?220:290):(mobile()?420:690), left=small?40:(mobile()?115:155), right=w-(small?20:mobile()?58:69), row=small?32:38;
 const order=small?names.map((_,i)=>i):d.order;
 const h=order.length*row+54, domain=extent(d.rows.flatMap(r=>[r.estimate,r.interval_low,r.interval_high]),true), sx=scale(domain,left,right);
 let s=axisX(domain,left,right,10,h-44);
 order.forEach((j,r)=>{const y=25+r*row,a=d.rows[j],lead=j===d.order[0], c=lead?accent:muted;
 s+=`<g data-model="${j}" ${tip(`${names[j]}\n${metricName(key)}: ${fmt(a.estimate,6)}\nSampling interval: ${fmt(a.interval_low,6)} – ${fmt(a.interval_high,6)}`)}>`;
 s+=text(left-13,y+4,small?code(j):short(j),'text-anchor="end"');
 if(finite(a.estimate)){
 if(finite(a.interval_low)&&finite(a.interval_high)) s+=line(sx(a.interval_low),y,sx(a.interval_high),y,`stroke="${c}" stroke-width="3" stroke-linecap="round"`);
 s+=circle(sx(a.estimate),y,lead?5.5:4.2,`fill="${c}" stroke="white" stroke-width="1.5"`);
 if(!small)s+=text(w-5,y+4,fmt(a.estimate),'text-anchor="end" class="value-text"');
 } else s+=text(right,y+4,'unavailable','text-anchor="end"');
 s+='</g>';
 });return svg(w,h,s,`${metricName(key)} estimates and sampling intervals`);
}
function heatmap(matrix,rowLabels,columnLabels,{domain=[0,1],percent=true,modelRows=true,diagonal=false}={}){
 const n=rowLabels.length,m=columnLabels.length,cell=43,left=75,top=37,w=left+m*cell+10,h=top+n*cell+15;
 let s='';columnLabels.forEach((l,j)=>s+=text(left+j*cell+cell/2,20,l,'text-anchor="middle"'));
 rowLabels.forEach((l,i)=>{
 s+=`<g ${modelRows?`data-model="${i}"`:''}>`+text(left-12,top+i*cell+cell*.62,l,'text-anchor="end"');
 for(let j=0;j<m;j++){
 const v=matrix[i][j],t=finite(v)?Math.max(0,Math.min(1,(v-domain[0])/(domain[1]-domain[0]))):0;
 const rgb=[244,244,249].map((c,k)=>Math.round(c+([106,105,182][k]-c)*t));
 const caption=modelRows?`${names[i]} / ${columnLabels[j]}: ${percent?pct(v):fmt(v,3)}`:`${l} / ${columnLabels[j]}: ${fmt(v)}`;
 s+=`<g ${tip(caption)}>`+rect(left+j*cell+1,top+i*cell+1,cell-3,cell-3,`fill="${diagonal&&i===j?'#f4f4f8':finite(v)?`rgb(${rgb})`:'#fafafa'}" rx="3"`);
 s+=text(left+j*cell+cell/2,top+i*cell+cell*.62,diagonal&&i===j?'—':finite(v)?(percent?(v*100).toFixed(0):v.toFixed(2)):'—',`text-anchor="middle" style="font-size:11px;fill:${t>.65&&!(diagonal&&i===j)?'white':ink}"`);
 s+='</g>';
 }s+='</g>';
 });return svg(w,h,s,'Matrix: '+columnLabels.join(', '))+`<div class="heat-legend"><span>${percent?'Frequency / share (%)':'Pearson r'}</span><span>${percent?'0':domain[0]}</span><i></i><span>${percent?'100':domain[1]}</span></div>`;
}
function histogramChart(hist,{w=520,h=235,xlabel='',zero=false}={}){
 if(!hist.n)return empty('No sampling data available');
 if(mobile())w=400;
 if(zero&&finite(hist.point_mass)){
  const left=45,right=w-20,top=28,bottom=h-49,domain=extent([0,hist.point_mass]),sx=scale(domain,left,right);
  let s=axisX(domain,left,right,top,bottom,xlabel);
  s+=line(sx(0),top,sx(0),bottom,`stroke="${ink}" stroke-dasharray="4 4"`);
  s+=line(sx(hist.point_mass),top+15,sx(hist.point_mass),bottom,`stroke="${accent}" stroke-width="3"`)+circle(sx(hist.point_mass),top+15,5,`fill="${accent}"`);
  s+=text(sx(hist.point_mass),top-2,`Point: ${fmt(hist.point_mass,6)} · n = ${hist.n}`,`text-anchor="${hist.point_mass>0?'end':'start'}" class="value-text"`);
  return svg(w,h,s,'Degenerate paired delta distribution');
 }
 const left=45,right=w-20,top=20,bottom=h-49,domain=[hist.edges[0],hist.edges.at(-1)],sx=scale(domain,left,right),max=Math.max(...hist.counts,1),sy=scale([0,max],bottom,top);
 let s=axisX(domain,left,right,top,bottom,xlabel);
 [0,max].forEach(v=>s+=text(left-8,sy(v)+4,String(v),'text-anchor="end" class="axis-label"'));
 hist.counts.forEach((n,i)=>{const a=hist.edges[i],b=hist.edges[i+1];s+=rect(sx(a)+1,sy(n),sx(b)-sx(a)-2,bottom-sy(n),`fill="${accent}" opacity=".68" ${tip(`${tick(a)} – ${tick(b)}\nCount: ${n}`)}`);});
 if(zero&&domain[0]<=0&&domain[1]>=0)s+=line(sx(0),top,sx(0),bottom,`stroke="${ink}" stroke-dasharray="4 4"`);
 return svg(w,h,s,'Histogram: '+xlabel);
}
function violinPlot(){
 const d=D.metrics[metric],w=mobile()?420:1060,left=mobile()?115:165,right=mobile()?370:1000,row=48,h=d.order.length*row+64;
 const domain=extent(d.distributions.flatMap(v=>v.quantiles),true),sx=scale(domain,left,right);let s=axisX(domain,left,right,8,h-50,metricName(metric));
 d.order.forEach((j,i)=>{
 const v=d.distributions[j],y=28+i*row;s+=`<g data-model="${j}" ${tip(`${names[j]}\nValid samples: ${v.n}\nMedian: ${fmt(v.quantiles[2],6)}`)}>`+text(left-18,y+4,short(j),'text-anchor="end"');
 if(v.n){
 if(v.density.length){const max=Math.max(...v.density),points=v.x.map((x,k)=>`${sx(x)},${y-17*v.density[k]/max}`).concat(v.x.map((x,k)=>`${sx(x)},${y+17*v.density[k]/max}`).reverse());s+=`<polygon points="${points.join(' ')}" fill="${i===0?accent:'#a5a9bb'}" fill-opacity=".25" stroke="${i===0?accent:'#a5a9bb'}" stroke-width="1"/>`;}
 s+=line(sx(v.quantiles[1]),y,sx(v.quantiles[3]),y,`stroke="${accent}" stroke-width="4" stroke-linecap="round"`)+circle(sx(v.quantiles[2]),y,4,`fill="${accent}" stroke="white" stroke-width="1"`);
 if(!v.density.length)s+=text(right+10,y+4,'point','class="axis-label"');
 }s+='</g>';
 });return svg(w,h,s,'Metric sampling violin distributions');
}
function seriesChart(series,{label='',threshold=null,unitRange=false,nonnegative=false}={}){
 const history=D.fit_history,w=mobile()?420:530,h=225,left=62,right=w-20,top=20,bottom=175;
 if(!history.length||!series.some(s=>s.values.some(finite)))return empty('This diagnostic was not saved with the current fit. Refit to show it.');
 const xs=history.map(r=>r.iteration),xd=xs.length===1?[xs[0]-.5,xs[0]+.5]:[xs[0],xs.at(-1)];
 const values=series.flatMap(s=>s.values).filter(finite);if(finite(threshold))values.push(threshold);
 let yd=unitRange?[0,1]:extent(values);
 if(!unitRange&&finite(threshold))yd=[0,Math.max(...values,1e-12)*1.15];
 else if(!unitRange&&Math.max(...values)===Math.min(...values)){const v=values[0],pad=Math.max(Math.abs(v)*.01,.01);yd=[v-pad,v+pad];}
 if(nonnegative&&!finite(threshold))yd=[Math.max(0,Math.min(...values)*.95),Math.max(1e-10,Math.max(...values)*1.05)];
 const sx=scale(xd,left,right),sy=scale(yd,bottom,top);let s='';
 for(let i=0;i<=3;i++){const v=yd[0]+(yd[1]-yd[0])*i/3;s+=line(left,sy(v),right,sy(v),'class="grid"')+text(left-9,sy(v)+4,axisTick(v,yd),'text-anchor="end" class="axis-label"');}
 const xt=xs.length<=5?xs:[xs[0],xs[Math.floor(xs.length/2)],xs.at(-1)];xt.forEach(x=>s+=text(sx(x),bottom+23,x,'text-anchor="middle" class="axis-label"'));
 s+=text((left+right)/2,bottom+43,'EM iteration','text-anchor="middle" class="axis-label"');
 if(finite(threshold))s+=line(left,sy(threshold),right,sy(threshold),'stroke="#bd9778" stroke-dasharray="4 4"')+text(right,sy(threshold)-6,'tol = '+tick(threshold),'text-anchor="end" class="axis-label"');
 series.forEach((a,j)=>{const points=a.values.map((v,i)=>finite(v)?`${sx(xs[i])},${sy(v)}`:null).filter(Boolean);s+=`<polyline points="${points.join(' ')}" fill="none" stroke="${col(j)}" stroke-width="2"/>`;a.values.forEach((v,i)=>{if(finite(v))s+=circle(sx(xs[i]),sy(v),4,`fill="${col(j)}" ${tip(`${a.name}\nIteration ${xs[i]}: ${fmt(v,8)}`)}`);});});
 return svg(w,h,s,label)+(series.length>1?series.map((a,j)=>`<span class="legend-item"><i class="swatch" style="background:${col(j)}"></i>${esc(a.name)}</span>`).join(''):'');
}
function scatterPlot(points,{pca=true}={}){
 if(!points.length)return empty('No unlabeled samples');
 const w=mobile()?420:(pca?780:530),h=pca?(mobile()?350:440):240,left=56,right=w-25,top=20,bottom=h-48;
 const xd=extent(points.map(p=>pca?p.x:p.confidence),!pca),yd=extent(points.map(p=>pca?p.y:p.entropy));
 if(!pca)yd[0]=Math.max(0,yd[0]);
 const sx=scale(xd,left,right),sy=scale(yd,bottom,top);let s=axisX(xd,left,right,top,bottom,pca?'PC1':'Posterior confidence');
 for(let i=0;i<=4;i++){const v=yd[0]+(yd[1]-yd[0])*i/4;s+=line(left,sy(v),right,sy(v),'class="grid"')+text(left-9,sy(v)+4,axisTick(v,yd),'text-anchor="end" class="axis-label"');}
 s+=text(13,(top+bottom)/2,pca?'PC2':'Entropy (nats)',`text-anchor="middle" transform="rotate(-90 13 ${(top+bottom)/2})" class="axis-label"`);
 const anchors=$('anchors').checked;
 for(const p of points){
 let c=accent;
 if(pca){if(spaceColor==='class')c=col(p.class);if(spaceColor==='status')c=p.labeled?accent:'#c0c3d1';if(spaceColor==='confidence')c=continuousColor(p.confidence);if(spaceColor==='entropy')c=continuousColor(p.entropy/Math.log(D.fit_pool.n_classes));}
 const opacity=pca&&anchors&&!p.labeled?.06:p.labeled?.96:.40;
 const desc=`Sample ${p.id}\n${p.labeled?'Labeled anchor':'Unlabeled'} · posterior class ${p.class}\nConfidence: ${fmt(p.confidence,8)}\nEntropy: ${fmt(p.entropy,8)}`;
 s+=circle(sx(pca?p.x:p.confidence),sy(pca?p.y:p.entropy),pca?(p.labeled?5:2.8):3,`fill="${c}" fill-opacity="${opacity}" stroke="${pca&&p.labeled?ink:'none'}" stroke-width="1.2" ${tip(desc)}`);
 }return svg(w,h,s,pca?'PCA of ALR prediction vectors':'Posterior confidence and entropy');
}
function densityChart(d, xlabel){
 if(!d?.n)return empty('No density data available. Regenerate the report.');
 if(finite(d.point_mass))return empty(`Point mass: ${fmt(d.point_mass,8)} · n = ${d.n}. KDE smoothing is not applied.`);
 const w=520,h=270,left=65,right=495,top=28,bottom=215;
 const domain=[d.x[0],d.x.at(-1)],sx=scale(domain,left,right),peak=Math.max(...d.density),sy=scale([0,peak],bottom,top);
 let s=line(left,bottom,right,bottom,`stroke="${ink}"`);
 for(let i=0;i<=4;i++){const v=domain[0]+(domain[1]-domain[0])*i/4;s+=text(sx(v),bottom+20,v.toFixed(6),'text-anchor="middle" class="axis-label"');}
 s+=text((left+right)/2,h-8,xlabel,'text-anchor="middle" class="axis-label"');
 [0,peak/2,peak].forEach(v=>s+=text(left-8,sy(v)+4,v.toPrecision(3),'text-anchor="end" class="axis-label"'));
 s+=text(left,15,'Density · zoomed axis','class="axis-label"');
 const path=d.x.map((v,i)=>`${i?'L':'M'}${sx(v)},${sy(d.density[i])}`).join(' ');
 s+=`<path d="${path} L${right},${bottom} L${left},${bottom} Z" fill="${accent}" fill-opacity=".18"/><path d="${path}" fill="none" stroke="${accent}" stroke-width="2.5"/>`;
 return svg(w,h,s,xlabel+' boundary-corrected density');
}
function render3D(){
 const host=$('pca-3d');if(!host)return;
 const state=host.viewState||(host.viewState={yaw:.55,pitch:-.35,zoom:1});
 const points=D.prediction_space.points;
 if(!points.every(p=>finite(p.z))){host.innerHTML=empty('Three-dimensional coordinates are unavailable. Regenerate the report.');return;}
 const radius=Math.max(1e-12,...points.flatMap(p=>[Math.abs(p.x),Math.abs(p.y),Math.abs(p.z)]));
 const project=(x,y,z)=>{const a=x*Math.cos(state.yaw)+z*Math.sin(state.yaw),b=-x*Math.sin(state.yaw)+z*Math.cos(state.yaw);return [390+a/radius*155*state.zoom,240-(y*Math.cos(state.pitch)-b*Math.sin(state.pitch))/radius*155*state.zoom,y*Math.sin(state.pitch)+b*Math.cos(state.pitch)];};
 let s='';
 [[radius,0,0],[0,radius,0],[0,0,radius]].forEach((v,i)=>{const a=project(...v.map(x=>-x)),b=project(...v);s+=line(a[0],a[1],b[0],b[1],`stroke="#a5a9bb" stroke-dasharray="4 4"`)+text(b[0]+8,b[1],`PC${i+1} · ${pct(D.prediction_space.variance[i])}`,'class="axis-label"');});
 const anchors=$('anchors').checked;
 points.map(p=>({p,q:project(p.x,p.y,p.z)})).sort((a,b)=>a.q[2]-b.q[2]).forEach(({p,q})=>{s+=circle(q[0],q[1],p.labeled?5:3,`fill="${col(p.class)}" fill-opacity="${anchors&&!p.labeled?.06:p.labeled?.95:.5}" stroke="${p.labeled?ink:'none'}" ${tip(`Sample ${p.id}\nClass ${p.class}\nPC1 ${fmt(p.x,3)} · PC2 ${fmt(p.y,3)} · PC3 ${fmt(p.z,3)}`)}`);});
 host.innerHTML=svg(780,480,s,'Three-dimensional PCA, colored by inferred class');
}
function continuousColor(t){const a=[205,210,218],b=[79,77,157];return `rgb(${a.map((v,j)=>Math.round(v+(b[j]-v)*Math.max(0,Math.min(1,t)))).join(',')})`;}
function renderSpace(){
 $('pca').innerHTML=scatterPlot(D.prediction_space.points);
 render3D();
 let items=[];
 if(spaceColor==='class')items=Array.from({length:D.fit_pool.n_classes},(_,k)=>[col(k),'Class '+k]);
 else if(spaceColor==='status')items=[[accent,'Labeled'],['#c0c3d1','Unlabeled']];
 else items=[[continuousColor(0),spaceColor==='confidence'?'0':'0 nats'],[continuousColor(1),spaceColor==='confidence'?'1':fmt(Math.log(D.fit_pool.n_classes),3)+' nats']];
 $('space-legend').innerHTML=items.map(([c,l])=>`<span class="legend-item"><i class="swatch" style="background:${c}"></i>${esc(l)}</span>`).join('');
 const p=D.posterior,k=D.fit_pool.n_classes;
 const explanations={
 class:'Inferred class: color is the class with the largest SSME posterior probability. It answers which class a sample at this location is assigned to. Color on unlabeled points is an inference.',
 confidence:`Posterior confidence: the maximum class probability, max P(y | s). Darker color means SSME puts more mass on a single class. This is not a candidate model's own confidence.${k===2?' For example, [0.5, 0.5] has confidence 0.5 and [0.99, 0.01] has confidence 0.99.':''} Unlabeled samples in this report span ${fmt(p.min_confidence,6)}–${fmt(p.max_confidence,6)}. Colors look similar when that range sits near 1.`,
 entropy:`Posterior entropy: how spread the class probabilities are, H = −Σ p log p. Darker color means the probabilities are more spread out and the latent class is less certain. Values near 0 are concentrated on one class.${k===2?' For example, [0.5, 0.5] has entropy 0.693 nats and [0.99, 0.01] has about 0.056 nats. In the binary case, entropy and confidence rank the same uncertainty in opposite directions.':''} Unlabeled entropy in this report spans ${fmt(p.min_entropy,6)}–${fmt(p.max_entropy,6)} nats.`,
 status:`Label status: purple marks the ${D.fit_pool.n_labeled} truly labeled samples and gray marks the ${D.fit_pool.n_unlabeled} unlabeled samples. The plot shows where the few labels sit in prediction space. Color is not accuracy or confidence.`,
 };
 $('space-explanation').textContent=explanations[spaceColor]+(spaceColor==='confidence'||spaceColor==='entropy'?' Labeled anchors have a one-hot posterior, so their confidence is 1 and their entropy is 0.':'');
 document.querySelectorAll('[data-color]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.color===spaceColor)));
}
function renderContribution(){
 const c=D.contribution,root=$('contribution-content');
 if(!c){root.innerHTML='<p class="performance-reading">Leave-one-model-out diagnostics were not computed. Call report(contribution=True) to show posterior impact and performance changes from actual refits.</p>';return;}
 const w=mobile()?420:650,left=mobile()?115:160,right=w-65,row=38,h=names.length*row+60;
 const max=Math.max(...c.rows.map(r=>r.posterior_total_variation),1e-8),domain=[0,max*1.1],sx=scale(domain,left,right);
 let plot=axisX(domain,left,right,10,h-45,'Mean posterior total variation');
 c.order.forEach((i,r)=>{const a=c.rows[i],y=25+r*row;plot+=`<g data-model="${i}" ${tip(`${a.removed_model}\nMean TV: ${fmt(a.posterior_total_variation,8)}\nClass flips: ${pct(a.posterior_class_flip_rate)}`)}>`+text(left-12,y+4,short(i),'text-anchor="end"')+line(left,y,sx(a.posterior_total_variation),y,'stroke="#d8d7e9" stroke-width="2"')+circle(sx(a.posterior_total_variation),y,5,`fill="${r===0?accent:muted}"`)+text(w-4,y+4,fmt(a.posterior_total_variation,6),'text-anchor="end" class="value-text"')+'</g>';});
 const best=c.rows[c.order[0]];
 root.innerHTML=`<div class="contribution-intro"><p>The original visible labels, full-model initialization, scalar bandwidth, and ${c.protocol.fixed_epochs} iterations are held fixed across ${c.n_refits} extra fits. Every effect is computed on ${c.n_samples.toLocaleString()} unlabeled samples.</p><p><strong>Largest effect: ${esc(best.removed_model)}</strong> · mean posterior TV = ${fmt(best.posterior_total_variation,6)}. A larger value means removing that input changes the result more. It is not a score for making SSME more accurate.</p></div><div class="two-col"><figure><h3>Effect on the posterior</h3>${svg(w,h,plot,'Leave-one-model-out posterior total variation')}<figcaption>For each sample, compute ½Σ|full posterior − posterior after removal|, then average. The range is 0–1. In the binary case this equals the mean absolute change in the positive-class posterior.</figcaption></figure><figure><h3>Effect on the other models' Accuracy estimates</h3><div id="contribution-matrix"></div><figcaption>Rows are removed models. Columns are models still evaluated. Values are Accuracy after removal minus Accuracy before removal, in percentage points. Purple is an increase and teal is a decrease. The diagonal is excluded. An increase does not mean the estimate is closer to the truth.</figcaption></figure></div><details class="data-details" open><summary>Per-model effects</summary><div class="table-wrap"><table><thead><tr><th>Removed model</th><th>Mean TV</th><th>Class flip rate</th><th>Mean |Δ Accuracy| (pp)</th><th>Mean |Δ rank|</th><th>Remaining Top-1 changed</th><th>JS divergence (nats)</th></tr></thead><tbody>${c.order.map(i=>{const a=c.rows[i];return `<tr data-model="${i}"><td>${esc(a.removed_model)}</td><td>${fmt(a.posterior_total_variation,6)}</td><td>${pct(a.posterior_class_flip_rate)}</td><td>${fmt(100*a.remaining_accuracy_mean_abs_change,4)}</td><td>${fmt(a.remaining_rank_mean_abs_change,3)}</td><td>${a.remaining_top1_changed?'Yes':'No'}</td><td>${fmt(a.posterior_js_divergence_nats,6)}</td></tr>`;}).join('')}</tbody></table></div></details><p class="caption">Ranks compare only models that remain after the removal, so deleting the leader is not itself counted as a ranking effect. Accuracy uses the exact posterior expectation and adds no extra label-sampling noise.</p><details class="data-details"><summary>What contribution means, and what is held fixed</summary><p class="performance-reading">This is a conditional ablation under a fixed initialization. The shared initialization comes from the full set of model predictions, so the diagnostic does not cover how removal would change that initialization, and the scalar bandwidth is not reselected. A small effect can come from overlapping information or a concentrated posterior. That value alone does not show that a model is useless. Without a separate set of true labels, a change in the estimate is not an accuracy gain, and these numbers are not normalized into shares that add to 100%.</p></details>`;
 const cell=43,ml=65,mt=30,mw=ml+names.length*cell+8,mh=mt+names.length*cell+15;
 const limit=Math.max(...c.accuracy_delta_matrix.flat().filter(finite).map(Math.abs),1e-12);let matrix='';
 names.forEach((_,j)=>matrix+=text(ml+j*cell+cell/2,17,code(j),'text-anchor="middle"'));
 names.forEach((_,i)=>{matrix+=`<g data-model="${i}">`+text(ml-10,mt+i*cell+26,code(i),'text-anchor="end"');names.forEach((_,j)=>{const v=c.accuracy_delta_matrix[i][j],nearZero=finite(v)&&Math.abs(v)<=1e-12,t=finite(v)&&!nearZero?Math.abs(v)/limit:0;const end=v<0?[98,155,153]:[106,105,182],rgb=[247,247,250].map((a,k)=>Math.round(a+(end[k]-a)*t));matrix+=`<g ${tip(`Remove ${names[i]} → ${names[j]}\nΔ Accuracy: ${finite(v)?(100*v).toExponential(4):'—'} percentage points`)}>`+rect(ml+j*cell+1,mt+i*cell+1,cell-3,cell-3,`fill="rgb(${rgb})" rx="3"`)+text(ml+j*cell+cell/2,mt+i*cell+26,finite(v)?nearZero?'≈0':(v>0?'+':'')+(100*v).toFixed(2):'—',`text-anchor="middle" style="font-size:10px;fill:${t>.65?'white':ink}"`)+'</g>';});matrix+='</g>';});
 $('contribution-matrix').innerHTML=svg(mw,mh,matrix,'Signed Accuracy changes in percentage points')+'<p class="caption">Display tolerance: |Δ Accuracy| ≤ 10⁻¹² is shown as ≈0 in a neutral color. The raw value stays in the hover tip and in the exported data.</p>';
}
function renderPair(){
 const d=D.metrics[metric],p=d.pairs[`${pairA}:${pairB}`];
 if(!p){$('pair-stats').innerHTML='';$('delta-chart').innerHTML=empty('At least two distinct candidate models are required.');return;}
 $('pair-stats').innerHTML=[[pct(p.a_better),'A beats B'],[pct(p.b_better),'B beats A'],[pct(p.tie),'Tie'],[String(p.n),'Valid paired draws']].map(([v,l])=>`<div><div class="stat-number">${v}</div><div class="stat-label">${l}</div></div>`).join('');
 $('delta-chart').innerHTML=histogramChart(p.delta,{w:1000,h:205,xlabel:`Δ ${metricName(metric)} · A − B`,zero:true});
 $('pair-a').value=String(pairA);$('pair-b').value=String(pairB);
 for(const option of $('pair-a').options)option.disabled=Number(option.value)===pairB;
 for(const option of $('pair-b').options)option.disabled=Number(option.value)===pairA;
}
function updateHighlight(){
 const selected=pinned ?? hovering;document.body.classList.toggle('has-highlight',selected!==null);
 document.querySelectorAll('[data-model]').forEach(el=>el.classList.toggle('highlighted',Number(el.dataset.model)===selected));
 document.querySelectorAll('.model-chip').forEach(el=>el.setAttribute('aria-pressed',String(Number(el.dataset.model)===pinned)));
}
function renderMetric(){
 const d=D.metrics[metric],j=d.order[0],second=d.order[1],r=d.rows[j],validLeader=finite(r.estimate),direction=metric==='ece'?'lowest':'highest';
 document.querySelectorAll('.metric-tabs').forEach(el=>el.innerHTML=metrics.map(k=>`<button data-metric="${k}" aria-pressed="${k===metric}">${metricName(k)}</button>`).join(''));
 $('kpi-metric').textContent=metricName(metric);$('hero-dot').innerHTML=dotPlot(metric);
 $('hero-summary').innerHTML=`<div><span class="eyebrow">LEADING ESTIMATE</span><div class="leader-name">${validLeader?esc(short(j)):'No estimate available'}</div><span class="subtle">${metricName(metric)} · ${metric==='ece'?'LOWER':'HIGHER'} IS BETTER</span><div class="score">${fmt(r.estimate)}</div></div><div><div class="mini-row">Rank-1 stability<strong>${validLeader?pct(d.rank1[j]):'—'}</strong></div><div class="mini-row">Closest competitor<strong>${second!==undefined&&finite(d.rows[second].estimate)?esc(short(second)):'—'}</strong></div><div class="mini-row">Valid ranked evaluations<strong>${d.rank_valid_draws} / ${M.n_draws}</strong></div></div>`;
 $('ranking-valid').textContent=`${metricName(metric)} · ${d.rank_valid_draws} / ${M.n_draws} complete valid draws`;
 $('rank-heatmap').innerHTML=heatmap(d.rank_frequencies,names.map((_,i)=>code(i)),names.map((_,i)=>'#'+(i+1)));
 $('pair-heatmap').innerHTML=heatmap(d.pairwise_wins,names.map((_,i)=>code(i)),names.map((_,i)=>code(i)),{diagonal:true});
 $('rank-bars').innerHTML=`<div class="table-wrap"><table><thead><tr><th>Model</th><th>Rank-1 stability</th><th>Top-2 frequency</th><th>Median rank</th></tr></thead><tbody>${d.order.map(i=>`<tr data-model="${i}"><td>${esc(names[i])}</td><td><span style="display:inline-block;vertical-align:middle;width:${finite(d.rank1[i])?d.rank1[i]*80:0}px;height:4px;background:${accent};margin-right:12px"></span>${pct(d.rank1[i])}</td><td>${pct(d.top2[i])}</td><td>${d.median_rank[i]??'—'}</td></tr>`).join('')}</tbody></table></div>`;
 $('violin').innerHTML=violinPlot();pairA=j;pairB=second??j;renderPair();
 updateHighlight();
}
function renderStatic(){
 document.title=(M.title||'Model Evaluation')+' · SSME-Lite';$('report-title').textContent=M.title;
 $('dataset-label').textContent=M.dataset||'REAL-WORLD INPUT';
 document.querySelectorAll('.coverage').forEach(e=>e.textContent=pct(M.confidence));
 const f=D.fit_pool;
 $('kpis').innerHTML=[[names.length,'CANDIDATE MODELS'],[f.n_labeled.toLocaleString(),'LABELED ANCHORS'],[f.n_unlabeled.toLocaleString(),'UNLABELED SAMPLES'],[metricName(metric),'PRIMARY METRIC']].map(([v,l],i)=>`<div class="kpi"><div class="number" ${i===3?'id="kpi-metric"':''}>${v}</div><div class="label">${l}</div></div>`).join('');
 $('composition').innerHTML=`<div class="composition-bar"><span style="width:${100*f.n_labeled/f.n_samples}%"></span><span style="flex:1"></span></div><div class="composition-legend"><div><strong>${f.n_labeled.toLocaleString()}</strong><span>Labeled · ${pct(f.n_labeled/f.n_samples)}</span></div><div><strong>${f.n_unlabeled.toLocaleString()}</strong><span>Unlabeled · ${pct(f.n_unlabeled/f.n_samples)}</span></div></div>`;
 $('scope-caption').textContent=`Fit pool: ${f.n_samples.toLocaleString()} samples. Performance is estimated on ${M.target==='all'?'the full fit pool':'unlabeled samples only'} (n = ${M.n_samples}).${M.split_seed!==undefined?' This report is a single fit from seed '+M.split_seed+'.':''}`;
 $('performance-scope').textContent=`This section evaluates ${names.length} candidate models on ${M.target==='all'?'the full fit pool':'unlabeled samples'} (${M.n_samples.toLocaleString()} samples). ${M.n_labeled} known labels stay fixed in every draw. The remaining labels are drawn from the fitted posterior, for ${M.n_draws} draws. Each table column is colored on its own. Shade shows relative standing within that metric and is not a score that can be averaged across metrics.`;
 const config=[['Task',f.n_classes===2?'Binary classification':'Multiclass'],['Classes',f.n_classes],['Kernel','Gaussian KDE'],['Bandwidth rule',M.bandwidth_rule],['Bandwidth',fmt(M.bandwidth,6)],['Labeled weight',M.labeled_weight],['EM epochs',M.n_iter+(earlyStopping?' (early stop)':' (fixed)')],['Metric samples',M.n_draws],['ALR dimensions',D.prediction_space.alr_dimensions]];
 $('config-grid').innerHTML=config.map(([k,v])=>`<dl><dt>${k}</dt><dd>${esc(v)}</dd></dl>`).join('');
 $('model-key').innerHTML=names.map((n,i)=>`<button class="model-chip" data-model="${i}" aria-pressed="false"><span class="code">${code(i)}</span>${esc(n)}</button>`).join('');
 $('performance-matrix').innerHTML=`<div class="table-wrap"><table><thead><tr><th>Candidate model</th>${metrics.map(k=>`<th>${metricName(k)} ${k==='ece'?'↓':'↑'}</th>`).join('')}</tr></thead><tbody>${names.map((name,i)=>`<tr data-model="${i}"><td><span class="subtle">${code(i)}</span> &nbsp; ${esc(name)}</td>${metrics.map(k=>{const d=D.metrics[k],r=d.rows[i],values=d.rows.map(r=>r.estimate).filter(finite),domain=values.length?[Math.min(...values),Math.max(...values)]:[0,1];let t=domain[1]!==domain[0]?(r.estimate-domain[0])/(domain[1]-domain[0]):.5;if(k==='ece')t=1-t;return `<td style="background:rgba(106,105,182,${finite(r.estimate)?.04+t*.19:0});color:${d.order[0]===i?accent:ink};font-family:var(--mono)">${fmt(r.estimate)}</td>`;}).join('')}</tr>`).join('')}</tbody></table></div>`;
 $('small-multiples').innerHTML=metrics.map(k=>`<figure><h3>${metricName(k)} ${k==='ece'?'↓':'↑'}</h3>${dotPlot(k,true)}</figure>`).join('');
 $('metric-table').innerHTML=`<table><thead><tr><th>Model</th><th>Metric</th><th>Estimate</th><th>Sampling low</th><th>Sampling high</th><th>MC standard error</th><th>Valid draws</th></tr></thead><tbody>${metrics.flatMap(k=>D.metrics[k].rows.map((r,i)=>`<tr data-model="${i}"><td>${esc(r.model)}</td><td>${metricName(k)}</td><td>${fmt(r.estimate,6)}</td><td>${fmt(r.interval_low,6)}</td><td>${fmt(r.interval_high,6)}</td><td>${fmt(r.mc_standard_error,8)}</td><td>${r.valid_draws}</td></tr>`)).join('')}</tbody></table>`;
 for(const id of ['pair-a','pair-b'])$(id).innerHTML=names.map((n,i)=>`<option value="${i}">${esc(n)}</option>`).join('');
 const l=D.landscape;
 $('correlation').innerHTML=heatmap(l.correlation,names.map((_,i)=>code(i)),names.map((_,i)=>code(i)),{domain:[-1,1],percent:false});
 $('agreement').innerHTML=heatmap(l.agreement,names.map((_,i)=>code(i)),names.map((_,i)=>code(i)));
 $('correlation-caption').textContent='Pearson correlation of '+l.correlation_definition+'. Correlation is undefined for a constant prediction and is shown as —.';
 const pairLabel=p=>p?p.map(i=>short(i)).join(' / '):'—';
 $('landscape-summary').innerHTML=[['MOST SIMILAR',pairLabel(l.most_similar),true],['MOST DIFFERENT',pairLabel(l.most_different),true],['MEAN CORRELATION',fmt(l.mean_correlation,3),false],['MEAN AGREEMENT',pct(l.mean_agreement),false]].map(([title,value,small])=>`<div><span class="eyebrow">${title}</span><div class="value ${small?'small':''}">${esc(value)}</div></div>`).join('');
 const sp=D.prediction_space;
 $('variance').innerHTML=sp.variance.map((v,i)=>`<div class="variance-row"><div class="top"><span>PC${i+1}</span><span>${pct(v)}</span></div><div class="variance-track"><span style="width:${100*v}%"></span></div></div>`).join('');
 $('space-caption').textContent=`The first two components explain ${pct(sp.variance[0]+sp.variance[1])} of the variance, and the first three explain ${pct(sp.variance.reduce((a,b)=>a+b,0))}. Showing ${sp.points.length.toLocaleString()} / ${sp.total.toLocaleString()} points. On large data, unlabeled points use a deterministic display sample and every anchor is kept.`;
 renderSpace();
 renderContribution();
 const p=D.posterior;
 $('sampling-diagnostic').textContent=finite(p.mean_confidence)?`The mean maximum posterior probability on unlabeled samples is ${(100*p.mean_confidence).toFixed(6)}%. Each draw is expected to move ${fmt(p.expected_nonmodal_labels_per_draw,6)} samples off their most likely class (Σ[1 − max P]). When that count is small, a finite number of draws can still collapse the interval to a point, even after many EM iterations.`:'';
 $('posterior-note').innerHTML=`<span class="big">${pct(p.concentrated_fraction)}</span><p>of unlabeled samples satisfy max P(y | s) ≥ ${p.threshold.toFixed(2)}. That is how concentrated the current posterior is. A sampling distribution that collapses to a point does not rule out uncertainty from the fit or from the bandwidth.</p>`;
 $('confidence-hist').innerHTML=densityChart(p.confidence_density, 'Maximum posterior probability');
 $('entropy-hist').innerHTML=densityChart(p.entropy_density, 'Entropy (nats)');

 let cs='';const cw=520,ch=f.n_classes*70+65;cs+=axisX([0,1],98,470,15,ch-45,'Proportion');
 for(let k=0;k<f.n_classes;k++){
 const yy=35+k*70,prior=p.class_priors[k],share=p.n_unlabeled?p.class_counts[k]/p.n_unlabeled:null;
 cs+=text(80,yy+8,'Class '+k,'text-anchor="end"')+rect(98,yy-6,372*prior,10,`fill="${col(k)}" ${tip(`Class ${k} fitted prior: ${pct(prior)}`)}`);
 if(finite(share))cs+=rect(98,yy+11,372*share,10,`fill="${col(k)}" fill-opacity=".35" ${tip(`Class ${k} unlabeled composition: ${pct(share)}`)}`);
 }$('class-structure').innerHTML=svg(cw,ch,cs,'Fitted priors and posterior composition')+'<span class="legend-item">Solid: fitted prior</span><span class="legend-item">Light: unlabeled composition</span>';
 $('ambiguous-table').innerHTML=p.ambiguous.length?`<table><thead><tr><th>Sample id</th>${Array.from({length:f.n_classes},(_,k)=>`<th>P(class ${k})</th>`).join('')}<th>Confidence</th><th>Entropy (nats)</th></tr></thead><tbody>${p.ambiguous.map(a=>`<tr><td>${esc(a.id)}</td>${a.posterior.map(v=>`<td>${fmt(v,6)}</td>`).join('')}<td>${fmt(a.confidence,6)}</td><td>${a.entropy>0&&a.entropy<.000001?a.entropy.toExponential(3):fmt(a.entropy,6)}</td></tr>`).join('')}</tbody></table>`:empty('The fit pool has no unlabeled samples.');

}
renderStatic();renderMetric();
const threeHost=$('pca-3d');let drag3D=null;
threeHost.style.touchAction='none';
threeHost.addEventListener('pointerdown',e=>{drag3D=[e.clientX,e.clientY];threeHost.setPointerCapture(e.pointerId);});
threeHost.addEventListener('pointermove',e=>{if(!drag3D)return;threeHost.viewState.yaw+=(e.clientX-drag3D[0])*.008;threeHost.viewState.pitch+=(e.clientY-drag3D[1])*.008;drag3D=[e.clientX,e.clientY];render3D();});
['pointerup','pointercancel'].forEach(name=>threeHost.addEventListener(name,()=>drag3D=null));
threeHost.addEventListener('wheel',e=>{e.preventDefault();threeHost.viewState.zoom=Math.max(.4,Math.min(2,threeHost.viewState.zoom*Math.exp(-e.deltaY*.001)));render3D();},{passive:false});
threeHost.addEventListener('keydown',e=>{const v=threeHost.viewState;if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','='].includes(e.key))return;e.preventDefault();if(e.key==='ArrowLeft')v.yaw-=.1;if(e.key==='ArrowRight')v.yaw+=.1;if(e.key==='ArrowUp')v.pitch-=.1;if(e.key==='ArrowDown')v.pitch+=.1;if(e.key==='+'||e.key==='=')v.zoom=Math.min(2,v.zoom+.1);if(e.key==='-')v.zoom=Math.max(.4,v.zoom-.1);render3D();});
$('reset-3d').addEventListener('click',()=>{threeHost.viewState=null;render3D();});
document.addEventListener('click',event=>{
 const b=event.target.closest('[data-metric]');if(b){metric=b.dataset.metric;renderMetric();}
 const c=event.target.closest('[data-color]');if(c){spaceColor=c.dataset.color;renderSpace();}
 const m=event.target.closest('.model-chip');if(m){const i=Number(m.dataset.model);pinned=pinned===i?null:i;updateHighlight();}
});
$('anchors').addEventListener('change',renderSpace);
$('pair-a').addEventListener('change',e=>{pairA=Number(e.target.value);renderPair();});
$('pair-b').addEventListener('change',e=>{pairB=Number(e.target.value);renderPair();});
document.addEventListener('pointerover',event=>{const m=event.target.closest('[data-model]');if(m){hovering=Number(m.dataset.model);updateHighlight();}const t=event.target.closest('[data-tip]');if(t){$('tooltip').textContent=t.dataset.tip;$('tooltip').hidden=false;}});
document.addEventListener('pointermove',event=>{const t=$('tooltip');if(t.hidden)return;t.style.left=Math.max(8,Math.min(event.clientX+14,innerWidth-t.offsetWidth-12))+'px';t.style.top=Math.max(8,Math.min(event.clientY+14,innerHeight-t.offsetHeight-12))+'px';});
document.addEventListener('pointerout',event=>{if(event.target.closest('[data-model]')){hovering=null;updateHighlight();}if(event.target.closest('[data-tip]'))$('tooltip').hidden=true;});
window.addEventListener('scroll',()=>{$('tooltip').hidden=true;},{passive:true});
document.addEventListener('keydown',e=>{if(e.key==='Escape'){pinned=null;hovering=null;updateHighlight();$('tooltip').hidden=true;}});
window.matchMedia('(max-width: 760px)').addEventListener('change',()=>{renderStatic();renderMetric();});
$('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(D,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='ssme-report-data.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
const navLinks=[...document.querySelectorAll('nav a')];
const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){const id=entry.target.id;navLinks.forEach(a=>{const active=a.hash==='#'+id;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});}},{rootMargin:'-10% 0px -60% 0px'});
document.querySelectorAll('main section').forEach(section=>observer.observe(section));
// Restore deep links after the embedded data has populated chart dimensions.
requestAnimationFrame(()=>{const section=document.getElementById(location.hash.slice(1));if(section)section.scrollIntoView({behavior:'instant'});});
})();
