'use strict';
(()=>{
 const $=id=>document.getElementById(id);
 const loc=window.location;
 const loopback=loc.protocol==='http:'&&['127.0.0.1','localhost'].includes(loc.hostname)&&!loc.username&&!loc.password;
 const tabs={monitor:$('arcade-monitor-tab'),review:$('arcade-review-tab')};
 const panes={monitor:$('arcade-monitor-pane'),review:$('arcade-review-pane')};
 const status=$('monitor-connection'), sourceBox=$('monitor-sources'), desc=$('monitor-description');
 const filterNodes=[...document.querySelectorAll('[data-monitor-filter]')];
 const typeNames={training:'训练',inference:'推理',benchmark:'测速'};
 const stateNames={ready:'就绪',loading:'加载中',running:'进行中',move:'进行中',paused:'已暂停',terminal:'自然结束',
   cap:'达到步数上限',capped:'达到步数上限',stopped:'已停止',error:'失败',completed:'已完成',complete:'已完成',
   waiting:'等数据',missing:'没找到源',stale:'更新可能过期',unknown:'状态未确认',incomplete:'记录不完整',unavailable:'暂不可读'};
 const phaseNames={loading:'加载模型',loading_model:'加载模型',encoding:'编码',cache:'缓存',train_encoding:'编码训练数据',fit:'优化决策头',head_fit:'优化决策头',train:'训练',training:'训练',test:'测试',evaluate:'验证',dev:'验证',development_evaluation:'验证',reload:'重载核验',official_reload:'重载核验',save:'保存',checkpoint_save:'保存检查点',done:'完成',completed:'完成',native_inference:'原生 Kev 推理',train_cache:'缓存训练特征',dev_cache:'缓存验证特征',starting:'启动训练进程',inference:'推理',running:'运行中',waiting:'等待数据'};
 const modeNames={progress_log:'联接进度日志',static_result:'静态结果文件',in_session:'当前会话',live:'实时源',stream:'实时源',file_tail:'联接日志',watch:'联接日志',linked:'联接源',static:'静态文件',imported:'已导入',result:'结果文件',completed:'完成记录',recorded:'离线记录',archive:'离线记录',archived:'离线记录',waiting:'等待来源'};
 const knownActive=new Set(['running','move','loading']);
 const staticModes=new Set(['static_result','static','imported','result','completed','recorded','archive','archived']);
 let filter='all',snapshot=null,connected=false,received=false,inflight=false,nextPoll=null,suspended=false;
 const own=(o,k)=>k!==undefined&&Object.prototype.hasOwnProperty.call(o,k);
 const finite=x=>typeof x==='number'&&Number.isFinite(x);
 const format=new Intl.NumberFormat('zh-CN',{maximumFractionDigits:3});
 const dated=new Intl.DateTimeFormat('zh-CN',{dateStyle:'short',timeStyle:'medium'});
 const text=(s,max=136)=>typeof s==='string'&&s.trim()?s.trim().slice(0,max):'';
 const fmtDate=s=>{
  const ms=typeof s==='number'?(s<1e11?s*1000:s):Date.parse(s||'');
  return finite(ms)?dated.format(ms):'未记录';
 };
 const at=s=>typeof s==='number'?(s<1e11?s*1000:s):Date.parse(s||'');
 const n=x=>finite(x)?format.format(x):'—';
 const label=(map,k,fallback)=>own(map,k)?map[k]:(text(k,60)||fallback);
 function make(tag,klass='',txt=''){
  const e=document.createElement(tag);if(klass)e.className=klass;if(txt)e.textContent=txt;return e;
 }
 function advisory(message){
  const e=make('div','card monitor-empty',message);sourceBox.replaceChildren(e);
 }
 function capsule(c,when){
  const state=typeof c.status==='string'?c.status:'unknown';
  const age=Date.now()-at(when);
  const stale=!staticModes.has(c.mode)&&(!connected||state==='stale'||(knownActive.has(state)&&finite(age)&&age>90000));
  const z=make('span','pill',stale&&state!=='stale'?'更新可能过期':label(stateNames,state,'状态未确认'));
  z.dataset.tone=stale?'old':knownActive.has(state)?'active':state==='error'?'error':'normal';
  return z;
 }
 function boardGrid(board){
  if(!Array.isArray(board))return null;
  const b=board.length===4&&board.every(r=>Array.isArray(r)&&r.length===4)?board.flat():board;
  if(b.length!==16||!b.every(x=>Number.isInteger(x)&&x>=0&&x<=1048576&&(x===0||(x&(x-1))===0)))return null;
  const e=make('div','monitor-board');e.setAttribute('role','img');
  e.setAttribute('aria-label','记录中的最新棋盘：'+b.join('、'));
  for(const x of b){const tile=make('span','tile',x?String(x):'');tile.dataset.value=x;tile.setAttribute('aria-hidden','true');e.append(tile)}
  return e;
 }
 function metric(container,key,value){
  const e=make('div','monitor-metric');
  e.append(make('dt','',key),make('dd','',value));container.append(e);
 }
 function history(ch,card){
  if(!Array.isArray(ch.history))return;
  const log=make('ol','monitor-history');
  const events=ch.history.filter(e=>e&&typeof e==='object').slice(-12).sort((a,b)=>(at(b.timestamp||b.updated_at||b.at)||0)-(at(a.timestamp||a.updated_at||a.at)||0)).slice(0,5);
  for(const e of events){
   const p=finite(e.step)?'第 '+n(e.step)+' 步':finite(e.turn)?'第 '+n(e.turn)+' 步':'';
   const what=[label(phaseNames,e.phase,''),p,own(stateNames,e.status)?stateNames[e.status]:''].filter(Boolean).join(' · ');
   if(!what)continue;
   log.append(make('li','',fmtDate(e.timestamp||e.updated_at||e.at)+' · '+what));
  }
  if(log.children.length)card.append(make('h3','','最近更新'),log);
 }
 function channelCard(c){
  const card=make('article','card monitor-card'), header=make('div','monitor-top');
  const latest=c.latest&&typeof c.latest==='object'?c.latest:{};
  const phase=c.phase||latest.phase;
  const last=c.updated_at||latest.updated_at||latest.timestamp;
  const statusLabel=label(stateNames,c.status,'状态未确认');
  const kind=label(typeNames,c.kind,'其他');
  const modeLabel=label(modeNames,c.mode,'未确认来源类型');
  header.append(make('h2','',text(c.name,100)||'未命名来源'),capsule(c,last));card.append(header);
  card.append(make('p','monitor-meta',kind+' · '+modeLabel+' · '+(staticModes.has(c.mode)&&!latest.timestamp?'文件时间：':'最后更新：')+fmtDate(last)));
  const stage=label(phaseNames,phase,'未报告阶段');
  card.append(make('p','monitor-stage','阶段：'+stage+(statusLabel==='更新可能过期'?'':' · '+statusLabel)));
  const count=finite(latest.step)?latest.step:finite(latest.turn)?latest.turn:undefined;
  const total=finite(latest.total)?latest.total:finite(latest.cap)?latest.cap:undefined;
  if(finite(count)){
   if(finite(total)&&total>0&&count>=0&&count<=total){
    const progress=make('progress','monitor-progress');progress.max=total;progress.value=count;
    progress.setAttribute('aria-label','已记录进度：'+n(count)+' / '+n(total));card.append(progress);
    card.append(make('p','monitor-meta','已记录 '+n(count)+' / '+n(total)));
   }else card.append(make('p','monitor-meta','已记录 '+n(count)+' 步'));
  }
  const metrics=make('dl','monitor-metrics');
  if(finite(latest.epoch))metric(metrics,'训练轮次',n(latest.epoch)+(finite(latest.epochs)?' / '+n(latest.epochs):''));
  if(finite(latest.loss))metric(metrics,'损失',n(latest.loss));
  if(finite(latest.score))metric(metrics,'已记录得分',n(latest.score));
  if(finite(latest.tile||latest.max_tile))metric(metrics,'最大方块',n(latest.tile||latest.max_tile));
  if(finite(latest.decision_ms))metric(metrics,'该步推理耗时',n(latest.decision_ms)+' 毫秒');
  if(finite(latest.elapsed_seconds))metric(metrics,'已记录的耗时',n(latest.elapsed_seconds)+' 秒');
  if(finite(latest.train)&&finite(latest.dev))metric(metrics,'训练 / 验证盘面',n(latest.train)+' / '+n(latest.dev));
  const timings=c.timings&&typeof c.timings==='object'&&!Array.isArray(c.timings)?c.timings:{};
  for(const [k,v] of Object.entries(timings).slice(0,12)){
   if(!finite(v)||v<0)continue;
   const [suffix,unit]=k.endsWith('_seconds')?['_seconds','秒']:k.endsWith('_ms')?['_ms','毫秒']:['',''];
   if(!unit)continue;
   const phase=suffix?k.slice(0,-suffix.length):k;
   metric(metrics,label(phaseNames,phase.replace(/_/g,' '),phase),n(v)+' '+unit);
  }
  if(metrics.children.length)card.append(metrics);
  const board=boardGrid(latest.board);
  if(board){card.append(make('h3','','最新记录中的棋盘'),board)}
  const reported=text(latest.checkpoint||c.checkpoint,64);
  if(reported&&/^[A-Za-z0-9_. -]{1,64}$/.test(reported))card.append(make('p','monitor-meta','检查点：'+reported));
  if(!board&&!metrics.children.length&&!last&&!phase) {
   card.append(make('p','monitor-meta','这个来源还未报告阶段、进度或耗时。'));
  }
  history(c,card);
  return card;
 }
 function render(){
  const chans=snapshot?.channels;
  status.dataset.tone=connected?'active':received?'old':'normal';
  status.textContent=!loopback?'未联接本机服务':connected?'本机服务已联接':received?'连接中断':'正在查询本机服务';
  if(!loopback){
   desc.textContent='当前在文件模式中，没有已联接的实时来源。切到「即时回看」仍可读本地逐步记录。';
   advisory('未联接训练或推理来源。此页不会启动任务、改参数或补写数据。');
   return;
  }
  if(connected)desc.textContent='根据本机服务实际报告显示。只读观察，不会开始、暂停或配置任何任务。';
  else if(received)desc.textContent='连接中断，保留最后一次收到的记录；以下不是当前实时状态。';
  else desc.textContent='正在等待本机服务提供观察来源。';
  if(!Array.isArray(chans)){advisory(connected?'服务未报告观察来源。':received?'暂未收到来源记录，等待重连。':'尚未收到观察来源。');return}
  const good=chans.filter(c=>c&&typeof c==='object'&&(filter==='all'||c.kind===filter));
  if(!good.length){advisory(filter==='all'?'还没有已联接的观察来源。':'当前没有此类来源。');return}
  sourceBox.replaceChildren(...good.slice(0,24).map(channelCard));
 }
 function show(tab,focus=false){
  if(tab==='monitor'){
   const b=$('play');if(b?.getAttribute('aria-label')==='暂停')b.click();
  }
  for(const name of Object.keys(tabs)){
   const selected=name===tab;
   tabs[name].setAttribute('aria-selected',String(selected));tabs[name].tabIndex=selected?0:-1;
   panes[name].hidden=!selected;
  }
  if(focus)tabs[tab].focus();
 }
 for(const [name,tab] of Object.entries(tabs)){
  tab.addEventListener('click',()=>show(name));
  tab.addEventListener('keydown',event=>{
   if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
   event.preventDefault();event.stopPropagation();
   show(event.key==='Home'?'monitor':event.key==='End'?'review':name==='review'?'monitor':'review',true);
  });
 }
 for(const tab of filterNodes){
  tab.addEventListener('click',()=>{
   filter=tab.dataset.monitorFilter;
   for(const item of filterNodes)item.setAttribute('aria-pressed',String(item===tab));
   render();
  });
 }
 async function poll(){
  if(!loopback||inflight||suspended)return;
  inflight=true;
  try{
   const request={method:'GET',cache:'no-store',credentials:'same-origin'};
   if(typeof AbortSignal!=='undefined'&&typeof AbortSignal.timeout==='function')request.signal=AbortSignal.timeout(6000);
   const response=await fetch('/channels',request);
   if(!response.ok)throw Error('source offline');
   const data=await response.json();
   if(!data||!Array.isArray(data.channels))throw Error('invalid source status');
   snapshot=data;connected=true;
  }catch{connected=false}
  finally{
   received=true;inflight=false;render();
   if(!suspended)nextPoll=setTimeout(poll,2400);
  }
 }
 window.addEventListener('pagehide',()=>{suspended=true;clearTimeout(nextPoll)});
 window.addEventListener('pageshow',()=>{if(suspended){suspended=false;poll()}});
 show('monitor');render();poll();
})();
