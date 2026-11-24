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
 dispatchEvent(e){this.event(e.type,e);return true}
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
 {name:'真推理B',mode:'in_session',kind:'inference',status:'running',updated_at:iso(),phase:'native_inference',latest:{turn:52,cap:120,decision_ms:2008,score:176,tile:32,board:empty,games:9,active_games:2,total:20}},
 {name:'静态C',mode:'static_result',kind:'benchmark',status:'completed',updated_at:iso(200000),timings:{cpu_median_wall_seconds:4.5}},
 {name:'待命D',mode:'progress_log',kind:'training',status:'waiting'},
 {name:'过期E',mode:'in_session',kind:'inference',status:'stale',updated_at:iso(450000)},
]};
function harness(location,folderSupported=false,streaming=false,route,loadReplay,previewReplay){
 const refs={};for(const id of ids)refs[id]=new Node('div');
 refs.speed.value='1';refs.replay.hidden=true;refs['play-seed'].value='';
 for(const name of ['archive','folder'])refs[name+'-sort'].value='time';
 if(folderSupported)refs['folder-input'].webkitdirectory=true;
 const panels=['left','right'].map(slot=>{
  const p=new Node('article');p.dataset.slot=slot;p.fields={};
  for(const n of ['.board','h2','.panel-source','.run-meta','.status','.score-label','.score','.move-label','.move-name','.arrow','.best','.decision-note','.final'])p.fields[n]=new Node('div');
  p.querySelector=k=>p.fields[k];return p;
 });
 const types=['all','training','evaluation','inference','benchmark'];
 const filters=Object.fromEntries(types.map(t=>{
  const p=new Node('button');p.dataset.monitorFilter=t;return [t,p]
 }));
 const doc=new Node('document'),win=new Node('window');win.location=location;
 doc.querySelectorAll=k=>k==='.panel'?panels:Object.values(filters);doc.getElementById=id=>refs[id];doc.createElement=k=>new Node(k);
 let counter=0,offline=false,monotonic=0;
 const requests=[],streams=[],timers=new Map();
 const context={document:doc,window:win,HTMLElement:Node,Element:Node,Intl,TextEncoder,performance:{now:()=>monotonic},
  Event:class {constructor(type){this.type=type}},
  matchMedia:()=>({matches:true,addEventListener(){}}),
  setTimeout:(cb,ms)=>{const id=++counter;timers.set(id,{cb,ms});return id},clearTimeout:id=>timers.delete(id),
  fetch:async(url,config)=>{
   requests.push({url,method:config.method,...(config.body?{body:JSON.parse(config.body)}:{})});
   if(offline)throw Error('down');
   return route?.(url,config)||{ok:true,json:async()=>rows};
  }};
 if(streaming)context.EventSource=class extends Node{
  static CLOSED=2;
  constructor(url){super('event-source');this.url=url;this.readyState=0;streams.push(this)}
  close(){this.readyState=2}
 };
 for(const [index,program] of programs.entries()){
  vm.runInNewContext(program,context);
  if(index!==0)continue;
  if(loadReplay)context.KevRuns.loadRun=loadReplay;
  if(previewReplay)context.KevRuns.describeReplay=previewReplay;
 }
 function tick(ms,elapsed=ms){const pair=[...timers].find(([,t])=>t.ms===ms);assert(pair);timers.delete(pair[0]);monotonic+=elapsed;pair[1].cb()}
 const hasTimer=ms=>[...timers.values()].some(timer=>timer.ms===ms);
 return {refs,panels,filters,requests,streams,goOffline(){offline=true},tickPoll(){tick(2400)},tickReview(){tick(400)},
  tickReplay(){tick(650)},replayScheduled(){return hasTimer(650)},tickTraining:tick,hasTimer};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const collect=e=>[String(e.textContent),...e.children.map(collect)].join(' ');
function changeSort(refs,name,value){const control=refs[name+'-sort'];control.value=value;control.event('change')}
function filterTarget(refs,name,checked){const control=refs[name+'-target'];control.checked=checked;control.event('change')}
const replayRow=card=>card.children[0];
const tileProgress=card=>card.children[1];
const archiveNames=refs=>refs['archive-list'].children.map(card=>replayRow(card).children[0].textContent.replace(/^★ /,''));
const folderCards=refs=>refs['folder-list'].children.map(item=>item.children[1].children[0]);
const folderGames=refs=>folderCards(refs).map(replayRow);
const folderNames=refs=>folderGames(refs).map(item=>item.children[0].textContent);
const directoryMarkers=card=>tileProgress(card).children[1].children.filter(item=>item.tagName==='BUTTON');
function checkTileProgress(card,fills,reached=false){
 const progress=tileProgress(card),[bar,axis]=progress.children;
 assert.equal(progress.dataset.reached,String(reached));
 assert.deepEqual(bar.children.map(part=>part.style['--segment-width']),['512','512','1024']);
 assert.deepEqual(bar.children.map(part=>part.style['--segment-fill']),fills);
 assert.deepEqual(axis.children.map(tick=>[tick.textContent,tick.style.left]),[['0','0%'],['512','25%'],['1024','50%'],['2048','100%']]);
}
function replayFixture(seed,model,tiles){
 const frames=tiles.map((tile,turn)=>({p:turn,m:tile,b:[tile,...Array(15).fill(0)],
  a:turn?'left':null,n:null,s:turn*4,g:turn?4:0}));
 return {seed,runTitle:model,policy:'native',frames,status:'cap'};
}
async function checkDirectoryClip(h,marker,start,end,turns){
 const refs=h.refs;
 marker.click();await flush();
 assert.equal(refs['turn-output'].textContent,`第 ${start} / ${turns} 步`);
 assert.equal(refs.play.getAttribute('aria-label'),'暂停');
 for(let turn=start+1;turn<=end;turn++){
  h.tickReplay();assert.equal(refs['turn-output'].textContent,`第 ${turn} / ${turns} 步`);
 }
 assert.equal(refs.play.getAttribute('aria-label'),'播放');
 assert.equal(h.replayScheduled(),false);
}
(async()=>{
 const h=harness({protocol:'http:',hostname:'127.0.0.1'});
 await flush();
 assert.deepEqual(h.requests,[{url:'/channels',method:'GET'}]);
 const r=h.refs;
 assert.equal(r['arcade-monitor-pane'].hidden,false);
 assert.equal(r['arcade-review-pane'].hidden,true);
 assert.equal(r['arcade-play-pane'].hidden,true);
 assert.equal(r['monitor-sources'].children.length,5);
 assert.match(collect(r['monitor-sources']),/训练Alpha.*已记录 64 \/ 512.*训练轮次.*1 \/ 2.*损失.*1.843/);
 assert.match(collect(r['monitor-sources']),/真推理B.*该步推理耗时.*2,008 毫秒.*最新记录中的棋盘/);
 assert.match(collect(r['monitor-sources']),/已完成对局 9 \/ 20.*进行中对局 2/);
 assert.match(collect(r['monitor-sources']),/静态C.*已完成.*静态结果文件.*文件时间/);
 assert.match(collect(r['monitor-sources']),/待命D.*等数据/);
 assert.match(collect(r['monitor-sources']),/过期E.*更新可能过期/);
 const finished=iso(120000),trial='lr8e-5-kl015';
 const results={channels:[rows.channels[2],{
   name:'8000 盘面训练',mode:'progress_log',kind:'training',status:'completed',updated_at:iso(60000),
   latest:{phase:'completed',step:1,total:1},history:[
    {phase:'head_fit',trial,epoch:15,dev_best_rate:0.5375},
    {phase:'head_fit',trial,epoch:16,dev_best_rate:0.5235},
    {phase:'trial_completed',trial,epoch:15,loss:0.9424},
   ],
  },{
   name:'8000 候选筛选',mode:'progress_log',kind:'evaluation',status:'completed',updated_at:iso(30000),
   latest:{phase:'completed',stage:'screened_out_30',games:30,reference_games_reused:90,elapsed_seconds:17.531},
   history:[{phase:'screening_completed',timestamp:finished,games:30,promote_to_100:false,candidate_vector:[0,0,0,1042],threshold:[0,0,2,1548]}],
  }]};
 const completed=harness({protocol:'http:',hostname:'127.0.0.1'},false,false,()=>({ok:true,json:async()=>results}));
 await flush();
 const [screening,training]=completed.refs['monitor-sources'].children.map(collect);
 assert.match(screening,/8000 候选筛选.*测评 · 联接测评日志.*未达扩展门槛.*已完成 30 \/ 30 局.*初筛得分中位数 1,042.*最强对照得分中位数 1,548.*复用对照 90 局/);
 assert.doesNotMatch(screening,/17.531/);
 assert.match(training,/8000 盘面训练.*训练轮次 16.*保留轮次 15.*验证损失 0.942.*老师动作命中率 53.75%/);
 completed.filters.evaluation.click();
 assert.deepEqual(completed.refs['monitor-sources'].children.map(collect),[screening]);
 const before=[[2,0,0,2],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
 const after=[[4,0,0,0],[0,0,0,0],[0,0,0,2],[0,0,0,0]];
 const official={arm:'official',seed:17,turn:9,score:8,max_tile:4,before,after,action:'left',probabilities:{left:0.75,right:0.25},stop:null};
 const candidate={...official,arm:'candidate',action:'right',probabilities:{left:0.12,right:0.88},stop:'terminal'};
 const other={...official,arm:'other',score:12};
 const nextSeed={...official,seed:18,score:20};
 const evaluation={id:'watch-6',name:'整局开发评测',mode:'gameplay',kind:'evaluation',status:'running',
  phase:'gameplay',updated_at:iso(),latest:{timestamp:iso()},games:72,total_games:300,
  arms:['official','candidate','other'],batch:{number:5,total:13,seeds:[17,18]},
  decisions_per_second:5.134,activity:{seeds:2,models:2,games:3},episodes:[official,candidate,other,nextSeed]};
 rows.channels.push(evaluation);
 h.tickPoll();await flush();
 const gameBox=r['monitor-gameplay'];
 assert.match(collect(gameBox),/已完成 72 \/ 300 局.*官方基线.*第 9 步.*AI 选择 ← 左 · 75%.*不可走/);
 assert.match(collect(gameBox),/平均落子速度 5.13 步\/秒.*评测批次 5 \/ 13.*进行中种子 2 \/ 2.*进行中模型 2 \/ 3.*进行中对局 3 \/ 6/);
 assert.match(collect(gameBox),/candidate.*自然结束.*AI 选择 → 右 · 88%.*other.*得分 12/);
 const [gameSummary,compareHead,,seedTabs,gameDetail]=gameBox.children[0].children;
 const progress=()=>gameSummary.children.find(node=>node.tagName==='PROGRESS');
 assert.equal(progress().value,72);assert.equal(progress().max,300);
 const [models]=gameDetail.children;
 const officialModel=models.children[0];
 const decision=officialModel.children.at(-1);
 assert.match(collect(decision.children[1]),/0%.*100%/);
 const tracks=decision.children[2].children.map(row=>row.children[1].children[0]);
 assert.equal(tracks.length,4);
 assert.equal(tracks[0].getAttribute('aria-hidden'),'true');
 assert.equal(tracks[2].getAttribute('aria-valuemax'),'100');
 assert.equal(tracks[2].style['--probability-fill'],'75%');
 assert.equal(tracks[3].style['--probability-fill'],'25%');
 const [firstTab,secondTab]=seedTabs.children;
 assert.equal(firstTab.getAttribute('aria-selected'),'true');
 compareHead.children[1].click();
 assert.match(models.children[0].children[3].getAttribute('aria-label'),/^落子前盘面：2、0、0、2/);
 secondTab.click();
 assert.equal(secondTab.getAttribute('aria-selected'),'true');
 assert.match(collect(models),/官方基线.*得分 20.*candidate.*等待首步.*other.*等待首步/);
 evaluation.games=73;nextSeed.turn=10;
 h.tickPoll();await flush();
 assert.equal(seedTabs.children[1],secondTab);
 assert.equal(secondTab.getAttribute('aria-selected'),'true');
 assert.equal(progress().value,73);
 assert.match(collect(gameBox),/第 10 步.*AI 选择 ← 左 · 75%/);
 h.filters.training.click();assert.equal(gameBox.children.length,0);
 h.filters.inference.click();assert.equal(gameBox.children.length,0);
 h.filters.evaluation.click();assert.match(collect(gameBox),/得分 20/);
 evaluation.batch={number:6,total:13,seeds:[19]};evaluation.episodes=[];
 evaluation.reused_games=100;evaluation.games=160;
 evaluation.activity={seeds:0,models:0,games:0};
 h.tickPoll();await flush();
 assert.equal(seedTabs.children[0].textContent,'19');
 assert.equal(seedTabs.children[0].getAttribute('aria-selected'),'true');
 assert.match(collect(gameBox),/评测批次 6 \/ 13.*进行中种子 0 \/ 1.*进行中对局 0 \/ 3/);
 assert.match(collect(gameBox),/已复用官方基线 100 局/);
 const seedRequests=new Map();
 const archiveWatch={...evaluation,seed_catalog:Array.from({length:100},(_,i)=>i),arms:['official','candidate','other']};
 const pickerHarness=harness({protocol:'http:',hostname:'127.0.0.1'},false,true,url=>
  new Promise(resolve=>seedRequests.set(url,resolve)));
 const sendCatalog=()=>pickerHarness.streams[0].event('channels',{data:JSON.stringify({channels:[archiveWatch]})});
 sendCatalog();
 const [,,seedNav,quickTabs,seedPanel]=pickerHarness.refs['monitor-gameplay'].children[0].children;
 const [pickerGroup,backToBatch,retrySeed,seedMessage]=seedNav.children;
 const seedSelect=pickerGroup.children[1];
 assert.equal(pickerGroup.children[0].textContent,'全部 100 个种子');
 assert.equal(seedSelect.children.length,100);
 const chooseSeed=seed=>{seedSelect.value=String(seed);seedSelect.event('change')};
 const answerSeed=(seed,episodes,ok=true)=>{
  const resolve=seedRequests.get('/gameplay/seed?channel=watch-6&seed='+seed);
  assert(resolve);resolve({ok,json:async()=>({channel:'watch-6',seed,episodes})});
 };
 chooseSeed(0);chooseSeed(52);
 assert.equal(quickTabs.hidden,true);assert.match(seedMessage.textContent,/正在读取种子 52/);
 answerSeed(52,[{...official,seed:52,score:321},{...candidate,seed:52,score:432},{...other,seed:52,score:543}]);
 await flush();
 answerSeed(0,[{...official,seed:0,score:9876}]);await flush();
 assert.match(collect(seedPanel),/官方基线.*得分 321.*candidate.*得分 432.*other.*得分 543/);
 assert.doesNotMatch(collect(seedPanel),/9,876/);
 sendCatalog();assert.equal(seedSelect.value,'52');assert.equal(backToBatch.hidden,false);
 chooseSeed(99);answerSeed(99,[],false);await flush();assert.equal(retrySeed.hidden,false);
 retrySeed.click();answerSeed(99,[]);await flush();assert.match(seedMessage.textContent,/0 \/ 3 个模型已有落子记录/);
 backToBatch.click();assert.equal(quickTabs.hidden,false);assert.equal(seedSelect.value,'19');assert.equal(backToBatch.hidden,true);
 rows.channels.pop();
 h.filters.all.click();
 h.filters.training.click();assert.equal(r['monitor-sources'].children.length,2);
 assert(!collect(r['monitor-sources']).includes('真推理B'));
 h.filters.inference.click();assert.equal(r['monitor-sources'].children.length,2);
 h.filters.benchmark.click();assert.equal(r['monitor-sources'].children.length,1);
 h.goOffline();h.tickPoll();await flush();
 assert.equal(r['monitor-connection'].textContent,'连接中断');
 assert.match(collect(r['monitor-sources']),/静态C.*已完成/);
 assert.equal(h.requests.every(x=>x.method==='GET'&&x.url==='/channels'),true);
 const push=harness({protocol:'http:',hostname:'127.0.0.1'},false,true);
 const [stream]=push.streams;
 assert.equal(stream.url,'/channels/events');
 stream.event('channels',{data:JSON.stringify(rows)});
 assert.equal(push.refs['monitor-connection'].textContent,'实时推送已联接');
 stream.event('error');
 assert.equal(push.refs['monitor-connection'].textContent,'连接中断');
 stream.event('channels',{data:JSON.stringify(rows)});
 assert.equal(push.refs['monitor-connection'].textContent,'实时推送已联接');
 assert.deepEqual(push.requests,[]);
 const loraTime=iso(150000);
 const loraStep={timestamp:loraTime,phase:'lora_fit',status:'running',trial:'lora',
  epoch:1,optimizer_step:70,steps:125,mean_training_loss:1.409999};
 const lora={id:'training-lora',name:'第三轮训练',kind:'training',mode:'progress_log',status:'running',phase:'lora_fit',
  updated_at:loraTime,stale_after_seconds:360,latest:loraStep,history:[
   {...loraStep,timestamp:iso(800000),trial:'lora-pilot',optimizer_step:32,steps:32},
   {...loraStep,timestamp:iso(750000),trial:'lora-control-timing',optimizer_step:4,steps:4},
   {...loraStep,timestamp:iso(700000),trial:'lora-head-control',optimizer_step:10},loraStep,
  ]};
 const pushLora=()=>stream.event('channels',{data:JSON.stringify({channels:[lora]})});
 const loraCard=()=>push.refs['monitor-sources'].children[0];
 const loraBar=()=>loraCard().children.find(node=>node.tagName==='PROGRESS');
 const loraPill=()=>loraCard().children[0].children[1];
 pushLora();
 assert.equal(loraPill().dataset.tone,'active');
 assert.equal(loraBar().value,70);assert.equal(loraBar().max,125);
 assert.equal(loraBar().getAttribute('aria-label'),'优化器步数 70 / 125 步');
 assert.match(collect(loraCard()),/1 步 = 用一批训练盘面更新一次模型参数/);
 assert.match(collect(loraCard()),/LoRA 正式训练.*优化器步数 70 \/ 125 步.*平均训练损失 1\.4100/);
 assert.match(collect(loraCard()),/只训头对照训练.*只训头对照测速.*LoRA 试跑/);
 const boardView=()=>loraCard().children.find(node=>node.className==='training-batch');
 assert.equal(boardView(),undefined);
 function loraBatch(step,tile){
  const samples=Array.from({length:16},(_,i)=>({row:step+i+1,
   board:[[0,2,0,i===1?tile:4],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
   stage:i===1?'middle':'early',source_policy:i===1?'official':'teacher',best_actions:i===1?['left','down']:['right'],
   ...(i===1?{candidate:{choice:'right',probabilities:{up:null,down:.22,left:.16,right:.62}}}:{})}));
  return {trial:'lora',epoch:1,optimizer_step:step,steps:125,samples,boards_per_second:4,pace_source:'measured'};
 }
 lora.training_batch=loraBatch(70,512);pushLora();
 const [boardToggle,batchPanel]=boardView().children;
 assert.equal(boardToggle.textContent,'查看训练盘面');
 assert.equal(boardToggle.getAttribute('aria-controls'),batchPanel.id);
 assert.equal(boardToggle.getAttribute('aria-expanded'),'false');
 assert.equal(batchPanel.hidden,true);
 boardToggle.click();
 const [batchHeading,playback,pace,batchSample,batchChoices]=batchPanel.children;
 const [position,back,play,forward,speedLabel,latestBatch]=playback.children;
 const speed=speedLabel.children[0];
 assert.equal(boardToggle.getAttribute('aria-expanded'),'true');
 assert.equal(batchPanel.hidden,false);
 assert.match(batchHeading.textContent,/已完成训练批次.*70 \/ 125/);
 assert.match(loraCard().children.find(node=>node.tagName==='SMALL').textContent,/本步共 16 张盘面/);
 assert.equal(batchChoices.children.length,16);
 assert.equal(batchChoices.children[0].getAttribute('aria-pressed'),'true');
 assert.match(batchSample.children[0].getAttribute('aria-label'),/^训练样本 1 的盘面：0、2、0、4/);
 assert.match(collect(batchSample),/训练数据行 71.*对局阶段 前期.*来源策略 教师模型.*老师推荐方向 → 右/);
 assert.match(collect(batchSample),/本次日志未记录候选决策/);
 assert.match(pace.textContent,/实测吞吐 4 盘\/秒.*等效轮播 4 盘\/秒/);
 push.tickTraining(250);
 assert.equal(position.textContent,'本批第 2 / 16 张');
 assert.match(collect(batchSample),/候选首选 → 右 · 62%.*不可走/);
 speed.value='2';speed.event('change');
 assert.match(pace.textContent,/等效轮播 8 盘\/秒/);
 push.tickTraining(125,375);
 assert.equal(position.textContent,'本批第 5 / 16 张');
 batchChoices.children[1].click();
 assert.equal(batchChoices.children[1].getAttribute('aria-pressed'),'true');
 assert.match(batchSample.children[0].getAttribute('aria-label'),/^训练样本 2 的盘面：0、2、0、512/);
 assert.match(collect(batchSample),/训练数据行 72.*对局阶段 中期.*来源策略 官方基线.*老师推荐方向 ← 左、↓ 下/);
 assert.equal(play.textContent,'继续轮播');assert(!push.hasTimer(125));
 lora.updated_at=iso();lora.latest={...loraStep,timestamp:lora.updated_at,optimizer_step:80,mean_training_loss:1.359521};
 lora.training_batch=loraBatch(80,1024);
 lora.history.push(lora.latest);pushLora();
 assert.equal(loraBar().value,80);
 assert.match(collect(loraCard()),/优化器步数 80 \/ 125 步.*平均训练损失 1\.3595/);
 assert.equal(boardView().children[0],boardToggle);
 assert.equal(batchPanel.hidden,false);
 assert.match(batchHeading.textContent,/已完成训练批次.*70 \/ 125/);
 assert.equal(latestBatch.hidden,false);
 assert.match(collect(batchSample),/训练数据行 72/);
 latestBatch.click();
 assert.match(batchHeading.textContent,/已完成训练批次.*80 \/ 125/);
 assert.equal(position.textContent,'本批第 1 / 16 张');assert.equal(latestBatch.hidden,true);
 forward.click();
 assert.match(batchSample.children[0].getAttribute('aria-label'),/^训练样本 2 的盘面：0、2、0、1024/);
 assert.match(collect(batchSample),/训练数据行 82/);
 back.click();speed.value='slow';speed.event('change');play.click();
 assert.match(pace.textContent,/逐盘慢看 1 盘\/秒/);assert(push.hasTimer(1000));
 boardToggle.click();
 assert(!push.hasTimer(1000));
 assert.equal(boardToggle.getAttribute('aria-expanded'),'false');
 assert.equal(batchPanel.hidden,true);
 lora.updated_at=iso(361000);pushLora();
 assert.equal(loraPill().dataset.tone,'old');
 assert.match(loraPill().textContent,/更新可能过期/);
 const trainingStep={...loraStep,phase:'optimizer',trial:'stock',epoch:2,optimizer_step:32,steps:32,mean_training_loss:1.25};
 const native={...lora,id:'native-stock',name:'Stock 训练',status:'completed',phase:'completed',
  latest:{phase:'completed',status:'completed',timestamp:iso()},training_step:trainingStep,
  training_batch:{...trainingStep,batch_size:128,samples:Array.from({length:64},(_,i)=>({row:i+1,board:before,best_actions:['left']}))}};
 stream.event('channels',{data:JSON.stringify({channels:[native]})});
 assert.match(collect(loraCard()),/Stock 训练.*已完成.*优化器步数 32 \/ 32 步.*平均训练损失 1\.2500/);
 boardView().children[0].click();
 const [nativeHeading,,nativePace,nativeSample,nativeSamples]=boardView().children[1].children;
 assert.match(loraCard().children.find(node=>node.tagName==='SMALL').textContent,/本步共 128 张盘面/);
 assert.match(nativePace.textContent,/训练速度待测.*1 盘\/秒/);
 assert.match(nativeHeading.textContent,/显示 64 \/ 128 张/);
 assert.equal(nativeSamples.children.length,64);
 nativeSamples.children[63].click();
 assert.match(collect(nativeSample),/训练数据行 64.*老师推荐方向 ← 左/);
 const f=harness({protocol:'file:',hostname:''});
 await flush();const a=f.refs;
 assert.equal(f.requests.length,0);assert.match(a['monitor-description'].textContent,/文件模式/);
 a['arcade-review-tab'].click();assert.equal(a['arcade-review-pane'].hidden,false);
 const sample=fs.readFileSync(path.join(__dirname,'samples','sample-events.jsonl'),'utf8');
 const trace=require('./launcher/parser.js').parseTrace(sample)[0],last=trace.frames.at(-1);
 const savedId='a'.repeat(32)+'/candidate',legacyId='b'.repeat(32)+'/official';
 const savedReplay={id:savedId,model:'星标候选',seed:trace.seed,turns:last.p,score:last.s,max_tile:last.m,time:'2026-11-23T02:15:00Z',starred:true};
 const legacyReplay={...savedReplay,id:legacyId,model:'旧官方模型',starred:false};
 let exposed=[legacyReplay];
 const archive=harness({protocol:'http:',hostname:'127.0.0.1'},true,true,url=>{
  if(url==='/replay/catalog')return {ok:true,json:async()=>({games:exposed})};
  if(url==='/replay/record?id='+encodeURIComponent(savedId))return {ok:true,json:async()=>({id:savedId,moves:sample,
   manifest:{schema_version:1,kind:'play',variant:'星标候选',files:{moves:'moves.jsonl'},seeds:[trace.seed]}})};
  return null;
 });
 assert.deepEqual(archive.requests,[]);
 const saved=archive.refs;
 saved['arcade-review-tab'].click();await flush();
 assert.equal(archive.requests.at(-1).url,'/replay/catalog');
 assert.match(collect(saved['archive-list']),/旧官方模型.*文件时间/);
 assert.doesNotMatch(collect(saved['archive-list']),/★/);
 exposed=[savedReplay,legacyReplay];
 saved['arcade-monitor-tab'].click();saved['arcade-review-tab'].click();await flush();
 assert.match(collect(saved['archive-list']),/★ 星标候选.*种子 96652001.*步数 4.*分数 8.*最大方块 4.*保存时间/);
 saved['archive-filter'].value='候选';saved['archive-filter'].event('input');
 assert.equal(saved['archive-list'].children.length,1);
 assert.equal(archive.requests.filter(request=>request.url.startsWith('/replay/record')).length,0);
 replayRow(saved['archive-list'].children[0]).click();await flush();
 assert.equal(archive.requests.at(-1).url,'/replay/record?id='+encodeURIComponent(savedId));
 assert.equal(saved.replay.hidden,false);assert.equal(saved.replay.focused,true);
 assert.match(archive.panels[0].fields['h2'].textContent,/星标候选/);
 assert.equal(saved['turn-output'].textContent,'第 0 / 4 步');
 saved['jump-end'].click();assert.equal(archive.panels[0].fields['.score'].textContent,'8');

 // Sorting and the target filter apply to catalog rows before rendering or fetching a trace.
 const catalog=[
  {model:'最高分（旧）',score:80000,max_tile:2048,turns:900,time:'2026-11-20T00:00:00Z'},
  {model:'最新且步数最多',score:7000,max_tile:1024,turns:1100,time:'2026-11-23T00:00:00Z'},
  {model:'最大方块',score:70000,max_tile:4096,turns:800,time:'2026-11-22T00:00:00Z'},
  {model:'最高分（新）',score:80000,max_tile:2048,turns:1001,time:'2026-11-21T00:00:00Z'},
  {model:'未知数据',score:null,max_tile:null,turns:null,time:null},
 ].map((game,i)=>({...game,id:(i+1).toString().repeat(32)+'/candidate',seed:i+20}));
 saved['archive-filter'].value='';
 exposed=catalog;saved['refresh-archive'].click();await flush();
 assert.deepEqual(archiveNames(saved),['最新且步数最多','最大方块','最高分（新）','最高分（旧）','未知数据']);
 changeSort(saved,'archive','score');
 assert.deepEqual(archiveNames(saved),['最高分（新）','最高分（旧）','最大方块','最新且步数最多','未知数据']);
 changeSort(saved,'archive','max_tile');
 assert.deepEqual(archiveNames(saved),['最大方块','最高分（新）','最高分（旧）','最新且步数最多','未知数据']);
 filterTarget(saved,'archive',true);
 assert.deepEqual(archiveNames(saved),['最大方块','最高分（新）','最高分（旧）']);
 saved['archive-filter'].value='最高分（旧）';saved['archive-filter'].event('input');
 assert.deepEqual(archiveNames(saved),['最高分（旧）']);
 saved['archive-filter'].value='';saved['archive-filter'].event('input');
 filterTarget(saved,'archive',false);
 changeSort(saved,'archive','turns');
 assert.deepEqual(archiveNames(saved),['最新且步数最多','最高分（新）','最高分（旧）','最大方块','未知数据']);
 changeSort(saved,'archive','time');
 assert.deepEqual(archiveNames(saved),['最新且步数最多','最大方块','最高分（新）','最高分（旧）','未知数据']);
 assert.equal(archive.requests.filter(request=>request.url.startsWith('/replay/record')).length,1);
 assert.equal(saved['turn-output'].textContent,'第 4 / 4 步');
 let storedRows=saved['archive-list'].children;
 checkTileProgress(storedRows[0],['100%','100%','0%']);
 checkTileProgress(storedRows[1],['100%','100%','100%'],true);
 assert.match(collect(storedRows[1]),/最大方块 4,096/);
 checkTileProgress(storedRows[2],['100%','100%','100%'],true);
 checkTileProgress(storedRows[4],['0%','0%','0%']);
 assert.match(tileProgress(storedRows[4]).children[0].children[0].title,/进度未知/);
 exposed=[{...catalog[0],max_tile:512},{...catalog[1],max_tile:256}];
 saved['refresh-archive'].click();await flush();
 storedRows=saved['archive-list'].children;
 checkTileProgress(storedRows[0],['50%','0%','0%']);
 checkTileProgress(storedRows[1],['100%','0%','0%']);

 a['file-input'].files=[{name:'sample-events.jsonl',size:Buffer.byteLength(sample),text:async()=>sample}];
 a['file-input'].event('change');await flush();
 assert.equal(a['turn-output'].textContent,'第 0 / 4 步');
 a['jump-end'].click();assert.equal(a['turn-output'].textContent,'第 4 / 4 步');
 assert.equal(f.panels[0].fields['.score'].textContent,'8');
 assert.equal(f.panels[0].fields['.status'].textContent,'步数到限');
 assert.equal(a['milestones'].hidden,true);
 assert.equal(a['milestone-list'].children.length,0);
 assert.equal(a['milestone-ticks'].children.length,0);
 assert.equal(a['import-tools'].hidden,true);
 assert.equal(a['toggle-import'].getAttribute('aria-expanded'),'false');
 assert.equal(a['toggle-import'].focused,true);
 a['toggle-import'].click();assert.equal(a['import-tools'].hidden,false);
 a['toggle-import'].click();assert.equal(a['import-tools'].hidden,true);
 a['arcade-review-tab'].event('keydown',{key:'ArrowRight',preventDefault(){},stopPropagation(){}});
 assert.equal(a['arcade-play-pane'].hidden,false);
 assert.equal(a['arcade-play-tab'].focused,true);
 assert.match(a['play-feedback'].textContent,/连接本机工作台/);
 a['arcade-monitor-tab'].click();a['arcade-review-tab'].click();assert.equal(a['turn-output'].textContent,'第 4 / 4 步');
 assert.equal(f.requests.length,0);

 // Exercise the milestone view with short, already-parsed frame histories.
 const milestoneGames=[
  replayFixture(12,'模型一',[4,16,32,64,128,256,512,512,1024,2048,2048,2048,4096]),
  replayFixture(12,'模型二',[4,16,32,64,128,256,512,512,512,1024,1024,2048]),
  replayFixture(13,'低方块棋局',[4,8,16,32,64]),
 ];
 const clip=harness({protocol:'file:',hostname:''},false,false,undefined,async()=>({games:milestoneGames}));
 const marks=clip.refs;
 marks['arcade-review-tab'].click();
 marks['file-input'].files=[{name:'moves.jsonl',size:Buffer.byteLength(sample),text:async()=>sample}];
 marks['file-input'].event('change');await flush();
 assert.equal(marks['seed-select'].value,'12');
 assert.equal(marks['milestones'].hidden,false);
 const milestones=()=>marks['milestone-list'].children;
 const findMilestone=(side,tile)=>milestones().find(({dataset})=>dataset.slot===side&&Number(dataset.tile)===tile);
 const pressed=()=>milestones().filter(item=>item.getAttribute('aria-pressed')==='true');
 assert.equal(milestones().length,6);
 for(const [side,expected] of [['left',[6,8,9]],['right',[6,9,11]]]){
  assert.deepEqual([512,1024,2048].map(tile=>Number(findMilestone(side,tile).dataset.turn)),expected);
 }
 assert.equal(pressed().length,0);
 assert.equal(marks['milestone-play'].hidden,true);
 assert.match(body.match(/<[^>]+id="milestone-ticks"[^>]*>/)[0],/aria-hidden="true"/);
 const renderedTicks=marks['milestone-ticks'].children.map(marker=>marker.style.left);
 assert(renderedTicks.includes('50%')&&renderedTicks.includes('75%'));

 findMilestone('right',512).click();
 assert.equal(marks['turn-output'].textContent,'第 6 / 12 步');
 assert.deepEqual(pressed(),[findMilestone('right',512)]);
 findMilestone('left',2048).click();
 assert.equal(marks['turn-output'].textContent,'第 9 / 12 步');
 assert.deepEqual(pressed(),[findMilestone('left',2048)]);
 assert.equal(marks['milestone-play'].hidden,false);
 assert(!marks['milestone-play'].disabled);
 assert.match(marks['milestone-play'].textContent,/第 5 步/);
 marks['milestone-play'].click();
 assert.equal(marks['turn-output'].textContent,'第 5 / 12 步');
 assert.equal(marks.play.getAttribute('aria-label'),'暂停');
 for(let turn=6;turn<=9;turn++){
  clip.tickReplay();assert.equal(marks['turn-output'].textContent,`第 ${turn} / 12 步`);
 }
 assert.equal(marks.play.getAttribute('aria-label'),'播放');
 assert.equal(clip.replayScheduled(),false);
 marks.play.click();
 for(let turn=10;turn<=12;turn++){
  clip.tickReplay();assert.equal(marks['turn-output'].textContent,`第 ${turn} / 12 步`);
 }
 assert.equal(clip.replayScheduled(),false);

 findMilestone('right',1024).click();marks['milestone-play'].click();
 assert.equal(clip.replayScheduled(),true);
 marks['right-select'].value='';marks['right-select'].event('change');
 assert.equal(clip.replayScheduled(),false);
 assert.equal(milestones().length,3);
 assert(milestones().every(item=>item.dataset.slot==='left'));
 assert.equal(pressed().length,0);
 assert.equal(marks['milestone-play'].hidden,true);
 findMilestone('left',512).click();marks['milestone-play'].click();
 assert.equal(clip.replayScheduled(),true);
 marks['seed-select'].value='13';marks['seed-select'].event('change');
 assert.equal(clip.replayScheduled(),false);
 assert.equal(marks['milestones'].hidden,true);
 assert.equal(milestones().length,0);
 assert.equal(marks['turn-output'].textContent,'第 0 / 4 步');

 // Directory actions open the chosen archive, then time the clip from its verified trace.
 const archiveClips=[
  {id:'c'.repeat(32)+'/official',model:'模型一',seed:12,turns:12,max_tile:4096,milestones:{512:6}},
  {id:'d'.repeat(32)+'/candidate',model:'模型二',seed:12,turns:11,max_tile:2048,milestones:{1024:7,2048:11}},
 ];
 const archiveClip=harness({protocol:'http:',hostname:'127.0.0.1'},false,true,url=>{
  if(url==='/replay/catalog')return {ok:true,json:async()=>({games:archiveClips})};
  const game=archiveClips.find(entry=>url==='/replay/record?id='+encodeURIComponent(entry.id));
  if(game)return {ok:true,json:async()=>({id:game.id,moves:sample,manifest:{variant:game.model}})};
  return null;
 },async entry=>({games:[milestoneGames[archiveClips.findIndex(game=>entry.path.includes(game.id))]]}));
 const clipArchive=archiveClip.refs;
 clipArchive['arcade-review-tab'].click();await flush();
 const [firstArchive,secondArchive]=clipArchive['archive-list'].children;
 assert.deepEqual(directoryMarkers(firstArchive).map(marker=>Number(marker.dataset.tile)),[512]);
 assert.deepEqual(directoryMarkers(secondArchive).map(marker=>Number(marker.dataset.tile)),[1024,2048]);
 const archiveJump=directoryMarkers(secondArchive)[0];
 assert.match(collect(archiveJump),/1,?024.*第\s*7\s*步/);
 assert.match(archiveJump.getAttribute('aria-label'),/从合成前回看/);
 await checkDirectoryClip(archiveClip,archiveJump,5,9,11);
 assert.equal(archiveClip.requests.at(-1).url,'/replay/record?id='+encodeURIComponent(archiveClips[1].id));
 assert.equal(archiveClip.panels[0].fields.h2.textContent,'模型二');
 replayRow(firstArchive).click();await flush();
 assert.equal(clipArchive['turn-output'].textContent,'第 0 / 12 步');
 assert.equal(archiveClip.panels[0].fields.h2.textContent,'模型一');
 assert.equal(archiveClip.replayScheduled(),false);

 const playableModels=[{id:'official',name:'官方基线',kind:'model'},
  ...['候选一','候选二','候选三'].map((name,index)=>({id:'candidate-'+index,name,kind:'model',
   training_summary:'冻结编码器，用教师标注训练策略头。',
   evaluation:index===0?{latest:{games:30,mean_score:1521.733333,label:'第 13 轮 · 新开发开局'},
    overall:{games:60,mean_score:1763.53333},batches:[
     {games:30,mean_score:1521.733333,label:'第 13 轮 · 新开发开局'},
     {games:30,mean_score:2005.333333,label:'第 12 轮 · 固定开发开局'}]}:null})),
  {id:'teacher',name:'教师模型',kind:'teacher'}];
 const openingChoice={revision:1,models:playableModels,session:null};
 const liveSession={id:'live-test',seed:146,cap:5000,status:'running',active:true,saved:false,games:playableModels.map(model=>({
  model:model.id,name:model.name,kind:model.kind,status:'running',turn:1,score:4,max_tile:4,board:after,action:'left',
  decision_ms:3.25,...(model.kind==='teacher'?{action_values:{left:9.5,right:-2.25}}:{probabilities:{left:0.8,right:0.2}})
 }))};
 let revision=1, gameSession=null, delayFrame=false, releaseFrame;
 const playState=()=>({...openingChoice,revision,session:gameSession});
 const instant=harness({protocol:'http:',hostname:'127.0.0.1'},false,true,(url,settings)=>{
  const payload=settings.body?JSON.parse(settings.body):{};
  if(url.startsWith('/play/frame?')){
   const {searchParams}=new URL(url,'http://127.0.0.1');
   const turn=Number(searchParams.get('turn')), id=searchParams.get('id');
   const games=turn===0?gameSession.games.map(({model,name,kind})=>({model,name,kind,turn:0,score:0,board:before,max_tile:2,status:'ready'})):
    liveSession.games.filter(game=>gameSession.games.some(current=>current.model===game.model));
   const frame={id,turn,games};
   if(delayFrame&&turn===1)return {ok:true,json:()=>new Promise(resolve=>{releaseFrame=()=>resolve(frame)})};
   return {ok:true,json:async()=>frame};
  }
  if(!url.startsWith('/play/'))return null;
  if(url==='/play/start')gameSession={...liveSession,id:payload.seed===42?'new-game':liveSession.id,
   seed:payload.seed??146,games:liveSession.games.filter(game=>payload.models.includes(game.model))};
  if(url==='/play/pause')gameSession={...gameSession,status:'paused'};
  if(url==='/play/resume')gameSession={...gameSession,status:'running'};
  if(url==='/play/stop')gameSession={...gameSession,status:'stopped',active:false};
  if(url==='/play/save')gameSession={...gameSession,saved:true};
  if(url!=='/play/state')revision++;
  const response=playState();return {ok:true,json:async()=>response};
 });
 const u=instant.refs;
 assert.deepEqual(instant.requests,[]);
 u['arcade-monitor-tab'].event('keydown',{key:'End',preventDefault(){},stopPropagation(){}});
 await flush();
 assert.equal(u['arcade-play-pane'].hidden,false);
 assert.equal(u['play-model-options'].children.length,5);
 const modelCheckbox=option=>option.children[0].children[0];
 assert.equal(modelCheckbox(u['play-model-options'].children[0]).checked,true);
 assert.match(collect(u['play-model-options'].children[0]),/暂无整局评测/);
 const firstCandidate=u['play-model-options'].children[1];
 assert.equal(modelCheckbox(firstCandidate).getAttribute('aria-label'),'候选一');
 assert.match(collect(firstCandidate),/整局平均分.*1,521\.7.*30 局.*第 13 轮.*60 个开局均分 1,763\.5.*教师标注/);
 assert.match(collect(firstCandidate.children[1]),/各轮均分.*第 12 轮.*30 局.*2,005\.3 分/);
 for(const option of u['play-model-options'].children.slice(1)){
  modelCheckbox(option).checked=true;
  modelCheckbox(option).event('change');
 }
 u['play-form'].event('submit',{preventDefault(){}});await flush();
 assert.deepEqual(instant.requests[1],{url:'/play/start',method:'POST',body:{models:playableModels.map(model=>model.id)}});
 assert.equal(u['play-boards'].children.length,5);
 assert.match(collect(u['play-boards']),/官方基线.*3.25 毫秒.*80%/);
 const teacherCard=u['play-boards'].children[4];
 assert.match(collect(teacherCard),/教师模型.*评分 9.5.*-2.25/);
 assert.doesNotMatch(collect(teacherCard),/%/);
 const liveFeed=instant.streams.find(source=>source.url==='/play/events');
 function teacherMoves(turn,score){
  gameSession={...gameSession,games:gameSession.games.map(game=>game.kind==='teacher'?{...game,turn,score}:game)};
  revision++;liveFeed.event('play',{data:JSON.stringify(playState())});
 }
 teacherMoves(2,16);
 assert.match(collect(teacherCard),/得分 16.*第 2 步/);
 assert.equal(u['play-save'].disabled,true);
 u['play-pause'].click();await flush();
 assert.deepEqual(instant.requests.at(-1),{url:'/play/pause',method:'POST',body:{id:'live-test'}});
 assert.equal(u['play-session-state'].textContent,'已暂停推理');
 u['play-prev'].click();await flush();
 assert.deepEqual(instant.requests.at(-1),{url:'/play/frame?id=live-test&turn=1',method:'GET'});
 assert.match(collect(teacherCard),/得分 4.*第 1 步.*评分 9.5/);
 assert.equal(u['play-cursor'].textContent,'回看 · 第 1 / 2 步');
 u['play-review'].click();assert.equal(u['play-review'].textContent,'暂停回看');
 instant.tickReview();await flush();
 assert.match(collect(teacherCard),/得分 16.*第 2 步/);
 assert.equal(u['play-review'].textContent,'播放回看');
 assert.equal(u['play-session-state'].textContent,'已暂停推理');
 u['play-seek'].value='0';u['play-seek'].event('input');await flush();
 assert.match(collect(teacherCard),/开局.*得分 0.*第 0 步/);
 assert.doesNotMatch(collect(teacherCard),/评分/);
 delayFrame=true;
 u['play-next'].click();await flush();
 u['play-prev'].click();await flush();
 releaseFrame();await flush();delayFrame=false;
 assert.equal(u['play-cursor'].textContent,'回看 · 第 0 / 2 步');
 u['play-pause'].click();await flush();
 assert.deepEqual(instant.requests.at(-1),{url:'/play/resume',method:'POST',body:{id:'live-test'}});
 assert.equal(u['play-cursor'].textContent,'实时 · 第 2 / 2 步');
 u['play-prev'].click();await flush();
 teacherMoves(3,32);
 assert.equal(u['play-cursor'].textContent,'回看 · 第 1 / 3 步');
 assert.match(collect(teacherCard),/得分 4.*第 1 步/);
 u['play-latest'].click();
 assert.match(collect(teacherCard),/得分 32.*第 3 步/);
 u['play-stop'].click();await flush();
 assert.deepEqual(instant.requests.at(-1),{url:'/play/stop',method:'POST',body:{id:'live-test'}});
 assert.match(u['play-session-state'].textContent,/已停止/);
 assert.match(u['play-save-note'].textContent,/开新局或重启服务/);
 u['play-save'].click();await flush();
 assert.deepEqual(instant.requests.at(-1),{url:'/play/save',method:'POST',body:{id:'live-test'}});
 assert.equal(u['play-save'].getAttribute('aria-pressed'),'true');
 assert.equal(u['play-save'].disabled,true);
 assert.match(u['play-save-note'].textContent,/已保存，可在「回看」中载入/);
 for(const option of u['play-model-options'].children.slice(1)){
  modelCheckbox(option).checked=false;
  modelCheckbox(option).event('change');
 }
 u['play-seed'].value='42';u['play-form'].event('submit',{preventDefault(){}});await flush();
 assert.deepEqual(instant.requests.at(-1),{url:'/play/start',method:'POST',body:{models:['official'],seed:42}});
 assert.equal(u['play-session-title'].textContent,'种子 42');
 assert.equal(u['play-boards'].children.length,1);
 assert.equal(u['play-save'].getAttribute('aria-pressed'),'false');
 u['arcade-play-tab'].event('keydown',{key:'Home',preventDefault(){},stopPropagation(){}});
 assert.equal(liveFeed.readyState,2);
 assert.equal(u['arcade-monitor-pane'].hidden,false);
 const monitorOnly=harness({protocol:'http:',hostname:'localhost'},false,true,
  url=>url==='/play/state'?{ok:false,status:404}:null);
 monitorOnly.refs['arcade-play-tab'].click();await flush();
 assert.match(monitorOnly.refs['play-feedback'].textContent,/未启用即时游玩/);
 assert.equal(monitorOnly.refs['play-start'].disabled,true);
 assert.deepEqual(monitorOnly.streams.map(source=>source.url),['/channels/events']);

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
 await flush();
 const found=b['folder-list'].children;
 assert.equal(found.length,4);
 assert.deepEqual(folderGames(b).map(game=>game.title.split(' · ')[0]),[
  'project/experiments/traces/stock.jsonl','runs/live/8e10ad515d634e63a231d66d74bdbab1.jsonl','runs/a/moves.jsonl','runs/b/big-moves.jsonl'
 ]);
 assert.match(collect(found[0]),/种子 96652001.*步数 4.*分数 8.*最大方块 4.*文件时间/);
 assert.equal(found[3].children[0].disabled,true);
 assert.equal(b['open-folder-files'].disabled,true);
 const listed=readCount;
 b['folder-filter'].value='runs/a/';b['folder-filter'].event('input');
 assert.deepEqual(folderGames(b).map(game=>game.title),['runs/a/moves.jsonl']);
 folderGames(b)[0].click();await flush();
 assert.equal(readCount,listed+1);assert.equal(b['turn-output'].textContent,'第 0 / 4 步');
 assert.match(folder.panels[0].fields['.panel-source'].textContent,/runs\/a\/moves.jsonl/);
 assert.equal(folder.panels[1].hidden,true);
 b['folder-filter'].value='';b['folder-filter'].event('input');
 assert.equal(b['folder-results'].hidden,false);
 for(const i of [0,2]){const box=b['folder-list'].children[i].children[0];box.checked=true;box.event('change')}
 assert.match(b['folder-selection'].textContent,/已选 2 次运行/);
 b['open-folder-files'].click();await flush();
 assert.equal(readCount,listed+3);assert.equal(b['turn-output'].textContent,'第 0 / 4 步');
 assert.match(folder.panels[0].fields['.panel-source'].textContent,/project\/experiments\/traces\/stock.jsonl/);
 assert.match(folder.panels[1].fields['.panel-source'].textContent,/runs\/a\/moves.jsonl/);
 assert.equal(b['import-tools'].hidden,true);
 b['toggle-import'].click();assert.equal(b['folder-list'].children[0].children[0].checked,true);
 assert.equal(folder.requests.length,0);

 const mixed=harness({protocol:'file:',hostname:''},true),mix=mixed.refs;
 const secondPolicy=sample.trim().split('\n').map(line=>JSON.stringify({...JSON.parse(line),profile:'candidate',run_id:'other-run'})).join('\n');
 mix['folder-input'].files=[file('runs/mixed/moves.jsonl',20,sample.trimEnd()+'\n'+secondPolicy)];
 mix['folder-input'].event('change');await flush();
 const games=folderGames(mix),candidateGame=games.find(game=>/candidate/.test(collect(game)));
 assert.equal(games.length,2);assert.match(collect(candidateGame),/candidate.*种子 96652001/);
 const secondPolicyRow=mix['folder-list'].children[1].children[0];
 secondPolicyRow.checked=true;secondPolicyRow.event('change');
 assert(mix['folder-list'].children.every(row=>row.children[0].checked));
 assert.match(mix['folder-selection'].textContent,/已选 1 次运行/);
 folderGames(mix).find(game=>/candidate/.test(collect(game))).click();await flush();
 assert.match(mixed.panels[0].fields['h2'].textContent,/候选版本/);
 assert.equal(mixed.panels[1].hidden,true);

 // Two same-seed policies share one folder file. The marker still opens the chosen policy.
 const comparedPolicies=[
  {...milestoneGames[0],policy:'native',runId:'policy-one'},
  {...milestoneGames[1],policy:'candidate',runId:'policy-two'},
 ];
 const folderClip=harness({protocol:'file:',hostname:''},true,false,undefined,async entry=>{
  assert.equal(entry.path,'runs/policies/moves.jsonl');
  return {games:comparedPolicies};
 },async()=>comparedPolicies.map(({seed,runTitle,policy,runId},index)=>({
  seed,model:runTitle,policy,runId,max_tile:2048,turns:index===0?12:11,milestones:{2048:index===0?9:11},
 })));
 const clipFolder=folderClip.refs;
 clipFolder['arcade-review-tab'].click();
 clipFolder['folder-input'].files=[file('runs/policies/moves.jsonl',20)];
 clipFolder['folder-input'].event('change');await flush();
 const secondFolder=folderCards(clipFolder).find(card=>collect(replayRow(card)).includes('模型二'));
 const [folderJump]=directoryMarkers(secondFolder);
 assert.equal(Number(folderJump.dataset.tile),2048);
 assert.match(collect(folderJump),/2,?048.*第\s*11\s*步/);
 assert.match(folderJump.getAttribute('aria-label'),/从合成前回看/);
 await checkDirectoryClip(folderClip,folderJump,7,11,11);
 assert.match(folderClip.panels[0].fields.h2.textContent,/模型二/);
 assert.equal(folderClip.panels[1].hidden,true);
 replayRow(secondFolder).click();await flush();
 assert.equal(clipFolder['turn-output'].textContent,'第 0 / 11 步');
 assert.equal(folderClip.replayScheduled(),false);

 // Load additional candidates without losing choices.
 const paged=harness({protocol:'file:',hostname:''},true),p=paged.refs;
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
 await flush();const beforeOpen=readCount;
 p['open-folder-files'].click();await flush();
 assert.match(p['import-status'].textContent,/已载入 2 次运行/);
 assert.equal(readCount,beforeOpen+2);

 // A result beyond the first page can lead any metric and match the target filter.
 const sorted=harness({protocol:'file:',hostname:''},true),v=sorted.refs;
 const preview=(model,mtime,stats)=>[
  file('runs/'+model+'/moves.jsonl',mtime),
  file('runs/'+model+'/result.json',mtime,JSON.stringify({games:[{seed:trace.seed,...stats}]})),
 ];
 v['folder-input'].files=[...Array.from({length:121},(_,i)=>file('runs/filler-'+i+'/moves.jsonl',200-i)),
  ...preview('较新文件',300,{turn:20,score:50,tile:1024,ended_at:'2026-11-21T00:00:00Z'}),
  ...preview('较新对局',1,{turn:30,score:100,tile:2048,ended_at:'2026-11-23T00:00:00Z'}),
  ...preview('高分旧局',2,{turn:150,score:9000,tile:4096}),
  ...preview('更多步数',3,{turn:200,score:4000,tile:2048}),
  ...preview('缺少统计',4,{}),
 ];
 v['folder-input'].event('change');await flush();
 assert.equal(v['folder-list'].children.length,120);
 assert.deepEqual(folderNames(v).slice(0,2),['较新对局','较新文件']);
 checkTileProgress(folderCards(v)[0],['100%','100%','100%'],true);
 checkTileProgress(folderCards(v)[1],['100%','100%','0%']);
 assert(!folderNames(v).includes('高分旧局'));
 const chosen=v['folder-list'].children[0].children[0];chosen.checked=true;chosen.event('change');
 changeSort(v,'folder','score');
 assert.deepEqual(folderNames(v).slice(0,4),['高分旧局','更多步数','较新对局','较新文件']);
 changeSort(v,'folder','max_tile');
 assert.deepEqual(folderNames(v).slice(0,4),['高分旧局','较新对局','更多步数','较新文件']);
 filterTarget(v,'folder',true);
 assert.deepEqual(folderNames(v),['高分旧局','较新对局','更多步数']);
 assert.equal(v['folder-list'].children[1].children[0].checked,true);
 assert.match(v['folder-selection'].textContent,/已选 1 次运行/);
 changeSort(v,'folder','turns');
 assert.deepEqual(folderNames(v),['更多步数','高分旧局','较新对局']);
 v['folder-filter'].value='旧局';v['folder-filter'].event('input');
 assert.deepEqual(folderNames(v),['高分旧局']);
 v['folder-filter'].value='';v['folder-filter'].event('input');
 changeSort(v,'folder','time');
 assert.deepEqual(folderNames(v),['较新对局','更多步数','高分旧局']);
 filterTarget(v,'folder',false);
 assert.equal(v['folder-list'].children.length,120);
 assert.equal(v['folder-list'].children[0].children[0].checked,true);

 // Finishing a superseded preview must not publish rows from the previous folder.
 let finishPreview;
 const pendingPreview=file('old/stale/moves.jsonl',400);
 pendingPreview.text=()=>new Promise(resolve=>{finishPreview=resolve});
 v['folder-input'].files=[pendingPreview];v['folder-input'].event('change');await flush();
 v['folder-input'].files=preview('新文件夹',3,{turn:40,score:2048,tile:2048});
 v['folder-input'].event('change');await flush();
 filterTarget(v,'folder',true);
 assert.deepEqual(folderNames(v),['新文件夹']);
 finishPreview(sample);await flush();
 assert.deepEqual(folderNames(v),['新文件夹']);

 // A selected run includes sidecars, counts once and shows reported identity.
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
 console.log('PASS combined dashboard: monitor isolation, live play controls, recording, timeline, star save, replay import, directory clips, milestones, sorting, target filter and retention.');
})().catch(e=>{console.error(e);process.exitCode=1});
