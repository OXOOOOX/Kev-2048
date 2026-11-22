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
 click(){if(!this.disabled)return this.event('click')}
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
function harness(location,folderSupported=false){
 const refs={};for(const id of ids)refs[id]=new Node('div');
 refs.speed.value='1';refs.replay.hidden=true;
 if(folderSupported)refs['folder-input'].webkitdirectory=true;
 const panels=['left','right'].map(slot=>{
  const p=new Node('article');p.dataset.slot=slot;p.fields={};
  for(const n of ['.board','h2','.panel-source','.run-meta','.status','.score-label','.score','.move-label','.move-name','.arrow','.best','.decision-note','.final'])p.fields[n]=new Node('div');
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
 assert.equal(a['import-tools'].hidden,true);
 assert.equal(a['toggle-import'].getAttribute('aria-expanded'),'false');
 assert.equal(a['toggle-import'].focused,true);
 a['toggle-import'].click();assert.equal(a['import-tools'].hidden,false);
 a['toggle-import'].click();assert.equal(a['import-tools'].hidden,true);
 a['arcade-monitor-tab'].click();a['arcade-review-tab'].click();assert.equal(a['turn-output'].textContent,'第 4 / 4 步');
 assert.equal(f.requests.length,0);
 const folder=harness({protocol:'file:',hostname:''},true);
 const b=folder.refs;b['arcade-review-tab'].click();
 assert.equal(b['pick-folder'].hidden,false);
 assert.equal(a['pick-folder'].hidden,true);
 let readCount=0;
 const file=(p,mtime,contents=sample,size=Buffer.byteLength(contents))=>({
  name:p.split('/').at(-1),webkitRelativePath:p,lastModified:mtime,size,
  text:async()=>{readCount++;return contents}
 });
 b['folder-input'].files=[
  file('runs/a/moves.jsonl',20),
  file('runs/b/benchmark.json',200,'{}'),
  file('runs/b/worker.progress.jsonl',190,'{}'),
  file('project/experiments/traces/stock.jsonl',40),
  file('runs/live/8e10ad515d634e63a231d66d74bdbab1.jsonl',35),
  file('runs/b/big-moves.jsonl',60,sample,17*1048576),
  file('project/elsewhere/stock.jsonl',50),
 ];
 b['folder-input'].event('change');
 assert.equal(readCount,0); // enumerating a folder never reads any file body
 const found=b['folder-list'].children;
 assert.equal(found.length,4);
 assert.deepEqual(found.map(x=>x.children[1].children[0].textContent),[
  'project/experiments/traces/stock.jsonl','runs/live/8e10ad515d634e63a231d66d74bdbab1.jsonl','runs/a/moves.jsonl','runs/b/big-moves.jsonl'
 ]);
 assert.equal(found[3].children[0].disabled,true);
 assert.equal(b['open-folder-files'].disabled,true);
 for(const i of [0,2]){found[i].children[0].checked=true;found[i].children[0].event('change')}
 assert.match(b['folder-selection'].textContent,/已选 2 次运行/);
 b['open-folder-files'].click();await flush();
 assert.equal(readCount,2);assert.equal(b['turn-output'].textContent,'第 0 / 4 步');
 assert.match(folder.panels[0].fields['.panel-source'].textContent,/project\/experiments\/traces\/stock.jsonl/);
 assert.match(folder.panels[1].fields['.panel-source'].textContent,/runs\/a\/moves.jsonl/);
 assert.equal(b['import-tools'].hidden,true);
 b['toggle-import'].click();assert.equal(b['folder-list'].children[0].children[0].checked,true);
 assert.equal(folder.requests.length,0);

 // Load additional candidates without losing choices or reading bodies.
 const paged=harness({protocol:'file:',hostname:''},true),p=paged.refs;
 const beforeListing=readCount;
 p['folder-input'].files=Array.from({length:121},(_,i)=>file('runs/run-'+i+'/moves.jsonl',200-i));
 p['folder-input'].files.push(file('runs/new-empty/moves.jsonl',1000,'',0));
 p['folder-input'].event('change');
 assert.equal(p['folder-list'].children.length,120);
 const first=p['folder-list'].children[0].children[0];first.checked=true;first.event('change');
 p['more-folder-files'].click();
 assert.equal(p['folder-list'].children.length,122);
 assert.equal(p['folder-list'].children[0].children[0].checked,true);
 const later=p['folder-list'].children[120].children[0];later.checked=true;later.event('change');
 assert.equal(p['folder-list'].children[121].children[0].disabled,true);
 assert.match(p['folder-selection'].textContent,/已选 2 次运行/);
 assert.equal(readCount,beforeListing);
 p['open-folder-files'].click();await flush();
 assert.match(p['import-status'].textContent,/已载入 2 次运行/);
 assert.equal(readCount,beforeListing+2);

 // A selected run includes sidecars, counts once and shows reported identity.
 const trace=require('./launcher/parser.js').parseTrace(sample)[0],last=trace.frames.at(-1);
 const summary={checkpoint_head_sha256:'a'.repeat(64),device:'cuda',cap:last.p,games:[{seed:trace.seed,turn:last.p,score:last.s,tile:last.m,stop:'cap'}]};
 const manifest={schema_version:1,kind:'play',variant:'测试模型 A',files:{moves:'moves.jsonl',result:'result.json'},model:{head_sha256:'a'.repeat(64)}};
 const withMeta=harness({protocol:'file:',hostname:''},true),m=withMeta.refs;
 m['file-input'].files=[file('run-a/moves.jsonl',1),file('run-a/result.json',1,JSON.stringify(summary)),file('run-a/run.json',1,JSON.stringify(manifest))];
 m['file-input'].event('change');await flush();
 assert.match(m['import-status'].textContent,/已载入 1 次运行/);
 assert.match(withMeta.panels[0].fields.h2.textContent,/测试模型 A/);
 assert.match(withMeta.panels[0].fields['.run-meta'].textContent,/cuda.*aaaaaaaaaaaa/);
 m['jump-end'].click();assert.equal(withMeta.panels[0].fields['.status'].textContent,'步数到限');
 // A failing batch keeps the previous board, position and selection, and names the path.
 const oldChoice=m['left-select'].value;
 m['file-input'].files=[file('good/moves.jsonl',1),file('broken/moves.jsonl',1,'broken')];
 m['file-input'].event('change');await flush();
 assert.match(m['import-status'].textContent,/broken\/moves.jsonl.*保留/);
 assert.equal(m['turn-output'].textContent,'第 4 / 4 步');
 assert.equal(m['left-select'].value,oldChoice);
 assert.equal(m['import-tools'].hidden,false);
 assert.equal(m.pick.disabled,false);
 // Nine runs remain prohibited even when listing spans more than one page.
 m['file-input'].files=Array.from({length:9},(_,i)=>file('runs/'+i+'/moves.jsonl',1));
 m['file-input'].event('change');await flush();assert.match(m['import-status'].textContent,/最多 8 次运行/);

 // Unsupported JSON summaries do not replace an existing successful import.
 m['file-input'].files=[file('summary-only/result.json',1,JSON.stringify(summary))];
 m['file-input'].event('change');await flush();assert.match(m['import-status'].textContent,/摘要不能单独回放/);
 assert.equal(m['turn-output'].textContent,'第 4 / 4 步');

 // A stale asynchronous import must not overwrite a newer import or its collapsed state.
 let releaseSlow;
 const slow=file('slow/moves.jsonl',1);slow.text=()=>new Promise(resolve=>{releaseSlow=resolve});
 m['file-input'].files=[slow];m['file-input'].event('change');await flush();
 assert.equal(m.pick.disabled,true);
 m['import-panel'].event('drop',{preventDefault(){},dataTransfer:{files:[file('new/moves.jsonl',1)]}});
 await flush();const newSource=withMeta.panels[0].fields['.panel-source'].textContent;
 releaseSlow(sample);await flush();
 assert.equal(withMeta.panels[0].fields['.panel-source'].textContent,newSource);
 assert.match(newSource,/new\/moves.jsonl/);
 assert.equal(m['import-tools'].hidden,true);assert.equal(m.pick.disabled,false);
 const outside=harness({protocol:'https:',hostname:'example.invalid'});await flush();assert.equal(outside.requests.length,0);
 console.log('PASS combined dashboard: monitor isolation, replay, run sidecars, collapse/focus, paged selection, usable-first ordering, path errors, failed import retention and asynchronous replacement.');
})().catch(e=>{console.error(e);process.exitCode=1});
