'use strict';
const assert=require('node:assert/strict'), path=require('node:path'), fs=require('node:fs'), vm=require('node:vm');
const body=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const programs=[...body.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]);
assert.equal(programs.length,3);
assert(!/<iframe\b/.test(body));
assert(!/(?:src|href)=["'](?:https?:|\/\/)/i.test(body));
const ids=[...body.matchAll(/\bid="([a-z-]+)"/g)].map(m=>m[1]);
class Node{
 constructor(tag){this.tagName=tag.toUpperCase();this.children=[];this.handlers={};this.dataset={};this.attrs={};this.textContent='';this.classList={toggle:(n,v)=>this.classList[n]=v};this.style={setProperty:(k,v)=>this.style[k]=v}}
 append(...c){for(const n of c){n.parent=this;this.children.push(n)}}
 replaceChildren(...c){this.children=[];this.append(...c)}
 remove(){this.parent.children=this.parent.children.filter(n=>n!==this)}
 addEventListener(k,f){this.handlers[k]=f}
 event(k,e={}){return this.handlers[k]?.(e)}
 click(){this.event('click')}
 focus(){this.focused=true}
 setAttribute(k,v){this.attrs[k]=v}
 getAttribute(k){return this.attrs[k]}
 removeAttribute(k){delete this.attrs[k]}
}
const empty=[0,0,0,2,0,0,0,2,0,0,0,0,0,0,0,0];
const iso=(past=0)=>new Date(Date.now()-past).toISOString();
const rows={channels:[
 {name:'训练Alpha',mode:'progress_log',kind:'training',status:'running',updated_at:iso(),phase:'train_cache',latest:{step:64,total:512,loss:1.843,epoch:1,epochs:2},timings:{train_encoding_seconds:32.5},history:[{timestamp:iso(),phase:'cache',step:64},{timestamp:iso(5000),phase:'starting',step:0}]},
 {name:'真推理B',mode:'in_session',kind:'inference',status:'running',updated_at:iso(),phase:'native_inference',latest:{turn:52,cap:120,decision_ms:2008,score:176,tile:32,board:empty}},
 {name:'静态C',mode:'static_result',kind:'benchmark',status:'completed',updated_at:iso(200000),timings:{cpu_median_wall_seconds:4.5}},
 {name:'待命D',mode:'progress_log',kind:'training',status:'waiting'},
 {name:'过期E',mode:'in_session',kind:'inference',status:'stale',updated_at:iso(450000)},
]};
function harness(location){
 const refs={};for(const id of ids)refs[id]=new Node('div');
 refs.speed.value='1';refs.replay.hidden=true;
 const panels=['left','right'].map(slot=>{
  const p=new Node('article');p.dataset.slot=slot;p.fields={};
  for(const n of ['.board','h2','.panel-source','.status','.score-label','.score','.move-label','.move-name','.arrow','.best','.decision-note','.final'])p.fields[n]=new Node('div');
  p.querySelector=k=>p.fields[k];return p;
 });
 const types=['all','training','inference','benchmark'];
 const filters=Object.fromEntries(types.map(t=>{
  const p=new Node('button');p.dataset.monitorFilter=t;return [t,p]
 }));
 const doc=new Node('document'),win=new Node('window');win.location=location;
 doc.querySelectorAll=k=>k==='.panel'?panels:Object.values(filters);doc.getElementById=id=>refs[id];doc.createElement=k=>new Node(k);
 let counter=0,offline=false;
 const requests=[],timers=new Map();
 const context={document:doc,window:win,HTMLElement:Node,Element:Node,Intl,TextEncoder,
  matchMedia:()=>({matches:true,addEventListener(){}}),
  setTimeout:(cb,ms)=>{const id=++counter;timers.set(id,{cb,ms});return id},clearTimeout:id=>timers.delete(id),
  fetch:async(url,config)=>{requests.push({url,method:config.method});if(offline)throw Error('down');return {ok:true,json:async()=>rows}}};
 for(const p of programs)vm.runInNewContext(p,context);
 return {refs,panels,filters,requests,goOffline(){offline=true},tickPoll(){const pair=[...timers].find(([,t])=>t.ms===2400);assert(pair);timers.delete(pair[0]);pair[1].cb()}};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const collect=e=>[String(e.textContent),...e.children.map(collect)].join(' ');
(async()=>{
 const h=harness({protocol:'http:',hostname:'127.0.0.1'});
 await flush();
 assert.deepEqual(h.requests,[{url:'/channels',method:'GET'}]);
 const r=h.refs;
 assert.equal(r['arcade-monitor-pane'].hidden,false);
 assert.equal(r['arcade-review-pane'].hidden,true);
 assert.equal(r['monitor-sources'].children.length,5);
 assert.match(collect(r['monitor-sources']),/训练Alpha.*已记录 64 \/ 512.*训练轮次.*1 \/ 2.*损失.*1.843/);
 assert.match(collect(r['monitor-sources']),/真推理B.*该步推理耗时.*2,008 毫秒.*最新记录中的棋盘/);
 assert.match(collect(r['monitor-sources']),/静态C.*已完成.*静态结果文件.*文件时间/);
 assert.match(collect(r['monitor-sources']),/待命D.*等数据/);
 assert.match(collect(r['monitor-sources']),/过期E.*更新可能过期/);
 h.filters.training.click();assert.equal(r['monitor-sources'].children.length,2);
 assert(!collect(r['monitor-sources']).includes('真推理B'));
 h.filters.inference.click();assert.equal(r['monitor-sources'].children.length,2);
 h.filters.benchmark.click();assert.equal(r['monitor-sources'].children.length,1);
 h.goOffline();h.tickPoll();await flush();
 assert.equal(r['monitor-connection'].textContent,'连接中断');
 assert.match(collect(r['monitor-sources']),/静态C.*已完成/);
 assert.equal(h.requests.every(x=>x.method==='GET'&&x.url==='/channels'),true);
 const f=harness({protocol:'file:',hostname:''});
 await flush();const a=f.refs;
 assert.equal(f.requests.length,0);assert.match(a['monitor-description'].textContent,/没有已联接/);
 a['arcade-review-tab'].click();assert.equal(a['arcade-review-pane'].hidden,false);
 const sample=fs.readFileSync(path.join(__dirname,'samples','sample-events.jsonl'),'utf8');
 a['file-input'].files=[{name:'sample-events.jsonl',size:Buffer.byteLength(sample),text:async()=>sample}];
 a['file-input'].event('change');await flush();
 assert.equal(a['turn-output'].textContent,'第 0 / 4 步');
 a['jump-end'].click();assert.equal(a['turn-output'].textContent,'第 4 / 4 步');
 assert.equal(f.panels[0].fields['.score'].textContent,'8');
 assert.equal(f.panels[0].fields['.status'].textContent,'步数到限');
 a['arcade-monitor-tab'].click();a['arcade-review-tab'].click();assert.equal(a['turn-output'].textContent,'第 4 / 4 步');
 assert.equal(f.requests.length,0);
 const outside=harness({protocol:'https:',hostname:'example.invalid'});await flush();assert.equal(outside.requests.length,0);
 console.log('PASS combined dashboard: GET-only loopback, real signal and missing/stale/static classification, filters, disconnect retention, file replay and tab-state persistence.');
})().catch(e=>{console.error(e);process.exitCode=1});
