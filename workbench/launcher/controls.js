'use strict';
(()=>{
 /* SHARED_MOTION */
 const $=id=>document.getElementById(id);
 const loc=window.location;
 const loopback=loc.protocol==='http:'&&['127.0.0.1','localhost'].includes(loc.hostname)&&!loc.username&&!loc.password;
 const tabs={monitor:$('arcade-monitor-tab'),review:$('arcade-review-tab'),play:$('arcade-play-tab')};
 const panes={monitor:$('arcade-monitor-pane'),review:$('arcade-review-pane'),play:$('arcade-play-pane')};
 const tabOrder=Object.keys(tabs);
 const status=$('monitor-connection'), sourceBox=$('monitor-sources'), gameBox=$('monitor-gameplay'), desc=$('monitor-description');
 const filterNodes=[...document.querySelectorAll('[data-monitor-filter]')];
 const typeNames={training:'训练',evaluation:'测评',inference:'推理',benchmark:'测速'};
 const stateNames={ready:'就绪',loading:'加载中',running:'进行中',move:'进行中',paused:'已暂停',terminal:'自然结束',
   cap:'达到步数上限',capped:'达到步数上限',stopped:'已停止',error:'失败',completed:'已完成',complete:'已完成',
   waiting:'等数据',missing:'没找到源',stale:'更新可能过期',unknown:'状态未确认',incomplete:'记录不完整',unavailable:'暂不可读'};
 const phaseNames={loading:'加载模型',loading_model:'加载模型',encoding:'编码',cache:'缓存',train_encoding:'编码训练数据',fit:'优化决策头',head_fit:'优化决策头',train:'训练',training:'训练',test:'测试',evaluate:'验证',dev:'验证',development_evaluation:'验证',reload:'重载核验',official_reload:'重载核验',save:'保存',checkpoint_save:'保存检查点',done:'完成',completed:'完成',native_inference:'原生 Kev 推理',train_cache:'缓存训练特征',dev_cache:'缓存验证特征',starting:'启动训练进程',inference:'推理',running:'运行中',waiting:'等待数据'};
 const improvementPhases={inputs_verified:'训练输入核验完成',loading_official_encoder:'加载官方编码器',
  feature_cache_reused:'复用特征缓存',fitting:'拟合决策头',fit_epoch:'优化决策头',trial_completed:'试验完成',
  head_screening:'决策头初筛',lora_pilot:'LoRA 试跑',lora_budget_decision:'确认 LoRA 训练轮数',lora_failed:'LoRA 训练失败',
  gameplay_start:'开始整局评测',gameplay_incomplete:'整局评测未完成',budget_stop:'达到运行时限',error:'执行出错'};
 const loraTrials={lora:'LoRA 正式训练','lora-pilot':'LoRA 试跑','lora-control-timing':'只训头对照测速',
  'lora-head-control':'只训头对照训练'};
 const modeNames={progress_log:'联接进度日志',static_result:'静态结果文件',in_session:'当前会话',live:'实时源',stream:'实时源',file_tail:'联接日志',watch:'联接日志',linked:'联接源',static:'静态文件',imported:'已导入',result:'结果文件',completed:'完成记录',recorded:'离线记录',archive:'离线记录',archived:'离线记录',waiting:'等待来源'};
 const knownActive=new Set(['running','move','loading']);
 const gameplayPhases={gameplay:'整局对照评测',episode_completed:'完成一局',screening_completed:'首轮筛选完成',inference_gate:'核对推理一致性',budget_stop:'达到运行时限',baseline_reused:'已复用官方基线'};
 const screeningStages={screened_out_30:'初筛完成 · 未达扩展门槛',full_100:'100 局评测完成'};
 const directions={up:'↑ 上',down:'↓ 下',left:'← 左',right:'→ 右'};
 const trainingStages={early:'前期',middle:'中期',late:'后期'};
 const trainingPolicies={teacher:'教师模型',official:'官方基线',stock:'原版策略',perturbed:'扰动策略'};
 const trainingBoardViews=new Map();
 let trainingBoardCount=0;
 const gameViews=new Map();
 let shownGames='';
 const staticModes=new Set(['static_result','static','imported','result','completed','recorded','archive','archived']);
 let filter='all',snapshot=null,connected=false,received=false,inflight=false,nextPoll=null,suspended=false,stream=null;
 const own=(o,k)=>k!==undefined&&Object.prototype.hasOwnProperty.call(o,k);
 const finite=x=>typeof x==='number'&&Number.isFinite(x);
 const format=new Intl.NumberFormat('zh-CN',{maximumFractionDigits:3});
 const lossFormat=new Intl.NumberFormat('zh-CN',{minimumFractionDigits:4,maximumFractionDigits:4});
 const percent=new Intl.NumberFormat('zh-CN',{style:'percent',maximumFractionDigits:2});
 const paceNumber=new Intl.NumberFormat('zh-CN',{maximumSignificantDigits:4});
 const dated=new Intl.DateTimeFormat('zh-CN',{dateStyle:'short',timeStyle:'medium'});
 const clock=new Intl.DateTimeFormat('zh-CN',{timeStyle:'medium'});
 const text=(s,max=136)=>typeof s==='string'&&s.trim()?s.trim().slice(0,max):'';
 const fmtDate=s=>{
  const ms=typeof s==='number'?(s<1e11?s*1000:s):Date.parse(s||'');
  return finite(ms)?dated.format(ms):'未记录';
 };
 const at=s=>typeof s==='number'?(s<1e11?s*1000:s):Date.parse(s||'');
 const n=x=>finite(x)?format.format(x):'—';
 const label=(map,k,fallback)=>own(map,k)?map[k]:(text(k,60)||fallback);
 function stageLabel(c,phase,fallback,trial){
  if(c.kind==='training'){
   if(phase==='lora_fit'||phase==='optimizer')return own(loraTrials,trial)?loraTrials[trial]:
    phase==='lora_fit'?'LoRA 优化':text(trial,80)||'优化模型';
   if(own(improvementPhases,phase))return improvementPhases[phase];
  }
  return c.kind==='evaluation'&&own(gameplayPhases,phase)?gameplayPhases[phase]:label(phaseNames,phase,fallback);
 }
 const optimizerProgress=(c,e)=>c.kind==='training'&&['lora_fit','optimizer'].includes(e?.phase)&&Number.isSafeInteger(e.optimizer_step)&&
  Number.isSafeInteger(e.steps)&&e.steps>0&&e.optimizer_step>=0&&e.optimizer_step<=e.steps;
 const optimizerText=e=>'优化器步数 '+n(e.optimizer_step)+' / '+n(e.steps)+' 步';
 function trainingStepNote(step,batch){
  const same=batch&&batch.trial===step.trial&&batch.epoch===step.epoch&&batch.optimizer_step===step.optimizer_step;
  const size=same?(batch.batch_size??(step.phase==='lora_fit'?batch.samples?.length:null)):null;
  return '1 步 = 用一批训练盘面更新一次模型参数。'+(Number.isSafeInteger(size)&&size>0?'本步共 '+n(size)+' 张盘面。':'');
 }
 const trainingLoss=(c,e)=>optimizerProgress(c,e)&&finite(e.mean_training_loss)
  ?lossFormat.format(e.mean_training_loss):null;
 function make(tag,klass='',txt=''){
  const e=document.createElement(tag);if(klass)e.className=klass;if(txt)e.textContent=txt;return e;
 }
 function advisory(message){
  syncTrainingVisibility();
  renderGameplay([]);
  const e=make('div','card monitor-empty',message);sourceBox.replaceChildren(e);
 }
 function capsule(c,when){
  const state=typeof c.status==='string'?c.status:'unknown';
  const age=Date.now()-at(when);
  const staleAfter=finite(c.stale_after_seconds)&&c.stale_after_seconds>0?1000*c.stale_after_seconds:90000;
  const stale=!staticModes.has(c.mode)&&(!connected||state==='stale'||(knownActive.has(state)&&finite(age)&&age>staleAfter));
  const z=make('span','pill',stale&&state!=='stale'?'更新可能过期':label(stateNames,state,'状态未确认'));
  z.dataset.tone=stale?'old':knownActive.has(state)?'active':state==='error'?'error':'normal';
  return z;
 }
 function boardGrid(board,title='记录中的最新棋盘'){
  if(!Array.isArray(board))return null;
  const b=board.length===4&&board.every(r=>Array.isArray(r)&&r.length===4)?board.flat():board;
  if(b.length!==16||!b.every(x=>Number.isInteger(x)&&x>=0&&x<=1048576&&(x===0||(x&(x-1))===0)))return null;
  const e=make('div','monitor-board');e.setAttribute('role','img');
  e.setAttribute('aria-label',title+'：'+b.join('、'));
  for(const x of b){const tile=make('span','tile',x?String(x):'');tile.dataset.value=x;tile.setAttribute('aria-hidden','true');e.append(tile)}
  return e;
 }
 function trainingBoardView(){
  const element=make('div','training-batch');
  const toggle=make('button','training-batch-toggle','查看训练盘面');
  const panel=make('div','training-batch-panel');
  panel.id='training-batch-panel-'+(++trainingBoardCount);panel.hidden=true;
  panel.setAttribute('role','region');panel.setAttribute('aria-label','训练盘面');
  toggle.type='button';toggle.setAttribute('aria-expanded','false');toggle.setAttribute('aria-controls',panel.id);
  toggle.dataset.trainingControl='true';
  const heading=make('p','monitor-meta');
  const sampleList=make('div','training-batch-samples');
  sampleList.setAttribute('role','group');sampleList.setAttribute('aria-label','批次样本');
  const display=make('div','training-batch-display');
  const toolbar=make('div','training-playback');
  function control(label){
   const button=make('button','',label);button.type='button';button.dataset.trainingControl='true';return button;
  }
  const previous=control('上一张'), play=control('暂停轮播'), next=control('下一张'), follow=control('查看最新批次');
  const count=make('span','training-sample-count');
  const speedLabel=make('label','','速度 '), speed=make('select');
  speed.setAttribute('aria-label','盘面轮播速度');speed.dataset.trainingControl='true';
  for(const [value,label] of [['0.01','0.01×'],['0.1','0.1×'],['0.5','0.5×'],['1','1× 训练节奏'],['2','2×'],['4','4×'],['slow','逐盘慢看']]){
   const option=make('option','',label);option.value=value;speed.append(option);
  }
  speed.value='1';speedLabel.append(speed);
  const pace=make('p','monitor-meta training-pace');
  follow.hidden=true;
  toolbar.append(count,previous,play,next,speedLabel,follow);
  panel.append(heading,toolbar,pace,display,sampleList);element.append(toggle,panel);
  let selected=0,buttons=[],batch=null,latest=null,channel=null,playing=true,visible=false,timer=null;
  let startTime=0,startIndex=0;
  const batchKey=b=>[b.trial,b.epoch,b.optimizer_step,b.timestamp,b.samples[0].row].join('|');
  const now=()=>performance.now();
  function trainingRate(){return finite(batch?.boards_per_second)&&batch.boards_per_second>0?batch.boards_per_second:null}
  function playbackRate(){return speed.value==='slow'?1:(trainingRate()??1)*Number(speed.value)}
  function clearClock(){if(timer!==null){clearTimeout(timer);timer=null}}
  function schedule(delay){timer=setTimeout(tick,Math.min(60000,Math.max(60,Math.ceil(delay))))}
  function restartClock(){
   clearClock();
   if(!visible||panel.hidden||!playing||batch.samples.length<2)return;
   startTime=now();startIndex=selected;schedule(1000/playbackRate());
  }
  function tick(){
   timer=null;
   const rate=playbackRate(), elapsed=Math.max(0,now()-startTime)*rate/1000;
   const advances=Math.floor(elapsed), index=(startIndex+advances)%batch.samples.length;
   if(index!==selected){selected=index;renderSample()}
   schedule((advances+1-elapsed)*1000/rate);
  }
  function playbackStatus(){
   const rate=trainingRate();
   const measure=rate===null?'训练速度待测':(batch.pace_source==='estimated'?'估算吞吐 ':'实测吞吐 ')+paceNumber.format(rate)+' 盘/秒';
   const effective=playbackRate();
   pace.textContent=measure+' · '+(speed.value==='slow'?'逐盘慢看 ':rate===null?'暂按 ':'等效轮播 ')+paceNumber.format(effective)+' 盘/秒'+
    (effective>1000/60?' · 跳帧展示':'');
   count.textContent='本批第 '+n(selected+1)+' / '+n(batch.samples.length)+' 张';
   play.textContent=playing?'暂停轮播':'继续轮播';
   play.setAttribute('aria-label',play.textContent);
   follow.hidden=!latest||batchKey(latest)===batchKey(batch);
  }
  function renderSample(){
   const sample=batch.samples[selected];
   for(const [i,button] of buttons.entries())button.setAttribute('aria-pressed',String(i===selected));
   const grid=boardGrid(sample?.board,'训练样本 '+(selected+1)+' 的盘面');
   const details=make('dl','monitor-metrics');
   if(Number.isSafeInteger(sample?.row)&&sample.row>0)metric(details,'训练数据行',n(sample.row));
   if(own(trainingStages,sample?.stage))metric(details,'对局阶段',trainingStages[sample.stage]);
   if(own(trainingPolicies,sample?.source_policy))metric(details,'来源策略',trainingPolicies[sample.source_policy]);
   const actions=Array.isArray(sample?.best_actions)?sample.best_actions.filter(move=>typeof move==='string'&&own(directions,move)):[];
   const compare=make('div','training-decisions');
   const teacher=make('div','training-teacher'),candidate=make('div','training-candidate');
   teacher.append(make('h3','','老师推荐方向'),make('p','training-teacher-choice',actions.map(move=>directions[move]).join('、')||'未记录'));
   candidate.append(make('h3','','候选模型 · 本步更新前'));
   const decision=sample?.candidate;
   if(decision?.probabilities&&own(directions,decision.choice)&&finite(decision.probabilities[decision.choice])){
    renderGameplayDecision(candidate,{action:decision.choice,probabilities:decision.probabilities},null,'候选首选');
   }else candidate.append(make('p','monitor-meta','本次日志未记录候选决策'));
   compare.append(teacher,candidate);
   display.replaceChildren(grid||make('p','monitor-meta','此样本的盘面暂不可读。'),details,compare);
   playbackStatus();
  }
  function selectSample(i){playing=false;clearClock();selected=i;renderSample()}
  function renderBatch(){
   const count=batch.samples.length;
   selected=Math.min(selected,count-1);
   if(buttons.length!==count){
    buttons=Array.from({length:count},(_,i)=>{
     const button=make('button','',String(i+1));button.type='button';button.dataset.trainingControl='true';
     button.setAttribute('aria-label','样本 '+(i+1));
     button.addEventListener('click',()=>selectSample(i));
     return button;
    });
    sampleList.replaceChildren(...buttons);
   }
   const total=batch.batch_size>count?' · 显示 '+n(count)+' / '+n(batch.batch_size)+' 张':' · '+n(count)+' 张';
   heading.textContent='已完成训练批次 · '+stageLabel(channel,batch.phase||'lora_fit','',batch.trial)+
    ' · 第 '+n(batch.epoch)+' 轮 · '+optimizerText(batch)+total;
   renderSample();
  }
  function useLatest(){
   if(batchKey(latest)!==batchKey(batch))selected=0;
   batch=latest;renderBatch();restartClock();
  }
  previous.addEventListener('click',()=>selectSample((selected-1+batch.samples.length)%batch.samples.length));
  next.addEventListener('click',()=>selectSample((selected+1)%batch.samples.length));
  play.addEventListener('click',()=>{
   playing=!playing;
   if(playing){useLatest()}else{clearClock();playbackStatus()}
  });
  follow.addEventListener('click',()=>{playing=true;useLatest()});
  speed.addEventListener('change',()=>{playbackStatus();restartClock()});
  toggle.addEventListener('click',()=>{
   panel.hidden=!panel.hidden;
   toggle.setAttribute('aria-expanded',String(!panel.hidden));
   toggle.textContent=panel.hidden?'查看训练盘面':'收起训练盘面';
   if(panel.hidden){clearClock()}else{playing=true;useLatest()}
  });
  return {element,setVisible(value){
   if(value!==visible){visible=value;restartClock()}
  },update(c,newBatch){
   channel=c;latest=newBatch;
   if(!batch||panel.hidden){batch=latest;selected=0;return}
   if(batchKey(batch)!==batchKey(latest)){
    if(playing)useLatest();else playbackStatus();
   }else{
    const rate=trainingRate();batch=latest;
    playbackStatus();if(rate!==trainingRate())restartClock();
   }
  }};
 }
 let shownTrainingIds=new Set();
 function syncTrainingVisibility(){
  const viewing=!suspended&&!panes.monitor.hidden&&!document.hidden;
  for(const [id,view] of trainingBoardViews)view.setVisible(viewing&&shownTrainingIds.has(id));
 }
 function trainingBoards(c){
  const batch=c.training_batch;
  if(!batch||!optimizerProgress(c,{...batch,phase:batch.phase||'lora_fit'})||batch.optimizer_step===0||
     !Number.isSafeInteger(batch.epoch)||batch.epoch<1||!Array.isArray(batch.samples)||!batch.samples.length||batch.samples.length>64)return null;
  const id=c.id||c.name;
  if(!trainingBoardViews.has(id))trainingBoardViews.set(id,trainingBoardView());
  const view=trainingBoardViews.get(id);view.update(c,batch);return view.element;
 }
 function metric(container,key,value){
  const e=make('div','monitor-metric');
  e.append(make('dt','',key),make('dd','',value));container.append(e);
 }
 function gameplayActivity(c){
  const group=make('dl','monitor-metrics gameplay-activity');
  const active=connected?c.activity:null, batch=c.batch;
  const pair=(value,total)=>n(value)+' / '+n(total);
  const rate=c.decisions_per_second;
  metric(group,'平均落子速度',finite(rate)?n(Math.round(rate*100)/100)+' 步/秒':'—');
  metric(group,'评测批次',pair(batch.number,batch.total));
  metric(group,'进行中种子',pair(active?.seeds,batch.seeds.length));
  metric(group,'进行中模型',pair(active?.models,c.arms.length));
  metric(group,'进行中对局',pair(active?.games,batch.seeds.length*c.arms.length));
  return group;
 }
 function history(ch,card){
  if(!Array.isArray(ch.history))return;
  const log=make('ol','monitor-history');
  const events=ch.history.filter(e=>e&&typeof e==='object').slice(-12).sort((a,b)=>(at(b.timestamp||b.updated_at||b.at)||0)-(at(a.timestamp||a.updated_at||a.at)||0)).slice(0,5);
  for(const [index,e] of events.entries()){
   const newer=events[index-1];
   if(e.phase==='completed'&&e.stage&&e.phase===newer?.phase&&e.stage===newer.stage&&e.games===newer.games)continue;
   const p=optimizerProgress(ch,e)?optimizerText(e):finite(e.games)?'已完成 '+n(e.games)+' 局':
    finite(e.step)?'第 '+n(e.step)+' 步':finite(e.turn)?'第 '+n(e.turn)+' 步':'';
   const loss=trainingLoss(ch,e);
   const what=[stageLabel(ch,e.phase,'',e.trial),p,loss===null?'':'平均训练损失 '+loss,
    own(stateNames,e.status)?stateNames[e.status]:''].filter(Boolean).join(' · ');
   if(!what)continue;
   log.append(make('li','',fmtDate(e.timestamp||e.updated_at||e.at)+' · '+what));
  }
  if(log.children.length)card.append(make('h3','','最近更新'),log);
 }
 function trainingResult(c,latest,metrics){
  if(c.kind!=='training'||c.status!=='completed'||latest.total!==1)return;
  const events=c.history||[];
  const saved=events.findLast(e=>e.phase==='trial_completed'&&finite(e.epoch));
  if(!saved)return;
  const fitted=events.filter(e=>e.phase==='head_fit'&&e.trial===saved.trial&&finite(e.epoch));
  const best=fitted.findLast(e=>e.epoch===saved.epoch);
  const last=fitted.at(-1);
  if(last)metric(metrics,'训练轮次',n(last.epoch));
  metric(metrics,'保留轮次',n(saved.epoch));
  if(finite(saved.loss))metric(metrics,'验证损失',n(saved.loss));
  if(finite(best?.dev_best_rate))metric(metrics,'老师动作命中率',percent.format(best.dev_best_rate));
 }
 function channelCard(c,card=make('article','card monitor-card')){
  const header=make('div','monitor-top'), isGameplay=c.mode==='gameplay';
  card.replaceChildren();
  const latest=c.latest&&typeof c.latest==='object'?c.latest:{};
  const phase=c.phase||latest.phase;
  const last=c.updated_at||latest.updated_at||latest.timestamp;
  const statusLabel=label(stateNames,c.status,'状态未确认');
  const kind=label(typeNames,c.kind,'其他');
  const modeLabel=c.kind==='evaluation'&&(isGameplay||c.mode==='progress_log')?'联接测评日志':label(modeNames,c.mode,'未确认来源类型');
  header.append(make('h2','',text(c.name,100)||'未命名来源'),capsule(c,last));card.append(header);
  card.append(make('p','monitor-meta',kind+' · '+modeLabel+' · '+(staticModes.has(c.mode)&&!latest.timestamp?'文件时间：':'最后更新：')+fmtDate(last)));
  const evaluationLog=c.kind==='evaluation'&&!isGameplay;
  const gate=evaluationLog?c.history?.findLast(e=>e.phase==='screening_completed'):null;
  const episode=evaluationLog?c.history?.findLast(e=>e.phase==='episode_completed'&&finite(e.total)):null;
  const terminalStage=evaluationLog&&own(screeningStages,latest.stage)?latest.stage:null;
  const isScreening=Boolean(terminalStage||gate||episode);
  const isGameCount=isGameplay||isScreening;
  const stage=terminalStage?screeningStages[terminalStage]:stageLabel(c,phase,'未报告阶段',latest.trial);
  card.append(make('p','monitor-stage','阶段：'+stage+(terminalStage||statusLabel==='更新可能过期'?'':' · '+statusLabel)));
  const trialsDone=c.kind==='training'&&c.status==='completed';
  const optimizer=optimizerProgress(c,c.training_step)?c.training_step:optimizerProgress(c,latest)?latest:null;
  const count=isGameplay?c.games:isScreening?latest.games:optimizer?optimizer.optimizer_step:latest.step??latest.turn;
  let total=optimizer?optimizer.steps:latest.total??latest.cap;
  if(isGameplay)total=c.total_games;
  if(isScreening)total=latest.total??episode?.total??gate?.games;
  if(gate?.promote_to_100===true)total=100;
  if(terminalStage)total=latest.games;
  if(finite(count)){
   if(finite(total)&&total>0&&count>=0&&count<=total){
    const progress=make('progress','monitor-progress');progress.max=total;progress.value=count;
    const progressText=optimizer?optimizerText(optimizer):(isGameCount||trialsDone?'已完成 ':'已记录 ')+
     n(count)+' / '+n(total)+(isGameCount?' 局':trialsDone?' 组':'');
    progress.setAttribute('aria-label',progressText);
    card.append(progress,make('p',isGameCount||optimizer?'gameplay-count':'monitor-meta',progressText));
    if(optimizer)card.append(make('small','training-step-note',trainingStepNote(optimizer,c.training_batch)));
   }else card.append(make('p','monitor-meta',isGameCount?'已完成 '+n(count)+' 局':'已记录 '+n(count)+' 步'));
  }
  const metrics=make('dl','monitor-metrics');
  trainingResult(c,latest,metrics);
  const epoch=optimizer?.epoch??latest.epoch;
  if(finite(epoch))metric(metrics,'训练轮次',n(epoch)+(finite(latest.epochs)?' / '+n(latest.epochs):''));
  if(finite(latest.loss))metric(metrics,'损失',n(latest.loss));
  const loss=trainingLoss(c,optimizer);
  if(loss!==null)metric(metrics,'平均训练损失',loss);
  if(!isGameplay&&finite(latest.score))metric(metrics,'已记录得分',n(latest.score));
  if(!isGameCount&&finite(latest.games))
   metric(metrics,'已完成对局',n(latest.games)+(finite(latest.total)?' / '+n(latest.total):''));
  if(!isGameplay&&finite(latest.active_games))metric(metrics,'进行中对局',n(latest.active_games));
  if(gate?.candidate_vector&&gate?.threshold&&finite(gate.games)){
   metric(metrics,'初筛达到 512',n(gate.candidate_vector[2])+' / '+n(gate.games));
   metric(metrics,'初筛得分中位数',n(gate.candidate_vector[3]));
   metric(metrics,'最强对照达到 512',n(gate.threshold[2])+' / '+n(gate.games));
   metric(metrics,'最强对照得分中位数',n(gate.threshold[3]));
  }
  if(finite(latest.reference_games_reused))metric(metrics,'复用对照',n(latest.reference_games_reused)+' 局');
  if(finite(latest.tile||latest.max_tile))metric(metrics,'最大方块',n(latest.tile||latest.max_tile));
  if(finite(latest.decision_ms))metric(metrics,'该步推理耗时',n(latest.decision_ms)+' 毫秒');
  if(!terminalStage&&finite(latest.elapsed_seconds)){
   const duration=n(latest.elapsed_seconds)+' 秒';
   if(isGameplay)card.append(make('p','monitor-meta','已记录的耗时 '+duration));
   else metric(metrics,'已记录的耗时',duration);
  }
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
  if(isGameplay&&c.batch)card.append(gameplayActivity(c));
  if(isGameplay&&finite(c.reused_games)&&c.reused_games>0){
   card.append(make('p','monitor-meta','已复用官方基线 '+n(c.reused_games)+' 局'));
  }
  const board=boardGrid(latest.board);
  if(board){card.append(make('h3','','最新记录中的棋盘'),board)}
  const trainingBatch=c.kind==='training'?trainingBoards(c):null;
  if(trainingBatch)card.append(trainingBatch);
  if(c.kind==='training'&&c.batch_replay_url==='/training/round4-batches'){
   const archive=make('a','training-archive-link','回看全部 20 个训练批次');
   archive.href=c.batch_replay_url;
   card.append(archive);
  }
  const reported=text(latest.checkpoint||c.checkpoint,64);
  if(reported&&/^[A-Za-z0-9_. -]{1,64}$/.test(reported))card.append(make('p','monitor-meta','检查点：'+reported));
  if(!board&&!metrics.children.length&&!last&&!phase) {
   card.append(make('p','monitor-meta','这个来源还未报告阶段、进度或耗时。'));
  }
  if(!isGameplay)history(c,card);
  return card;
 }
 function renderGameplayDecision(model,episode,previous,choiceLabel='AI 选择'){
   const fields=make('div','gameplay-decision');
   model.append(fields);
   const chosen=episode.probabilities[episode.action];
   fields.append(make('p','gameplay-choice',choiceLabel+' '+directions[episode.action]+' · '+percent.format(chosen)));
   const scale=make('div','gameplay-scale');
   scale.setAttribute('aria-hidden','true');
   scale.append(make('span','','0%'),make('span','','100%'));
   fields.append(scale);
   const list=make('dl','gameplay-probabilities');
   fields.append(list);
   for(const [action,name] of Object.entries(directions)){
    const value=episode.probabilities[action];
    const line=make('div','gameplay-probability');
    line.dataset.chosen=String(action===episode.action);
    const direction=make('dt','',name[0]);
    direction.append(make('span','gameplay-direction-name',name.slice(1)));
    const reading=make('dd','gameplay-reading');
    const track=make('span','gameplay-probability-track');
    const fill=make('span','gameplay-probability-fill');
    const probability=finite(value)?percent.format(value):'不可走';
    if(finite(value)){
     const amount=100*value;
     track.style.setProperty('--probability-fill',amount+'%');
     track.setAttribute('role','meter');
     track.setAttribute('aria-label',name+'的模型概率');
     track.setAttribute('aria-valuemin','0');
     track.setAttribute('aria-valuemax','100');
     track.setAttribute('aria-valuenow',String(amount));
     track.setAttribute('aria-valuetext',probability);
    }else track.setAttribute('aria-hidden','true');
    track.append(fill);
    reading.append(track,make('span','gameplay-probability-value',probability));
    line.append(direction,reading);
    list.append(line);
    const earlier=previous?.probabilities?.[action]||0, current=value||0;
    if(previous&&earlier!==current&&typeof fill.animate==='function'){
     fill.animate([{width:100*earlier+'%'},{width:100*current+'%'}],{duration:245,easing:'ease-out'});
    }
   }
 }
 function renderGameplayModel(model,arm,episode,showBefore,live,emptyLabel='等待首步'){
  const head=make('div','gameplay-model-head');
  const finished={target:'达到 2048',terminal:'自然结束',capped:'达到步数上限'}[episode?.stop];
  const badge=make('span','pill',finished||(episode?(live?'进行中':'未结束'):emptyLabel));
  if(episode&&!finished&&live)badge.dataset.tone='active';
  head.append(make('h4','',arm==='official'?'官方基线':arm),badge);
  model.replaceChildren(head);
  if(!episode)return null;
  const updated=at(episode.updated_at);
  const stamp=finite(updated)?' · '+clock.format(updated):'';
  const score=make('p','gameplay-score','得分 '+n(episode.score));
  model.append(score,
      make('p','monitor-meta','第 '+n(episode.turn)+' 步 · 最大 '+n(episode.max_tile)+stamp));
  const grid=boardGrid(showBefore?episode.before:episode.after,showBefore?'落子前盘面':'当前盘面');
  if(grid)model.append(grid);
  renderGameplayDecision(model,episode);
 }
 function gameplayView(id){
  const element=make('article','card monitor-card gameplay-card');
  const summary=make('div'), header=make('div','gameplay-comparison-head'), seedNav=make('div','gameplay-seed-nav');
  const pickerLabel=make('label'), pickerTitle=make('span'), picker=make('select');
  const follow=make('button','gameplay-view-toggle'), retry=make('button','gameplay-view-toggle','重新读取');
  const batchLabel=make('p','monitor-meta');
  picker.setAttribute('aria-label','选择全部测评种子');
  follow.type=retry.type='button';follow.hidden=retry.hidden=true;
  pickerLabel.append(pickerTitle,picker);seedNav.append(pickerLabel,follow,retry,batchLabel);
  batchLabel.setAttribute('aria-live','polite');
  const tabs=make('div','gameplay-seeds'), detail=make('div'), models=make('div','gameplay-models');
  const beforeToggle=make('button','gameplay-view-toggle','落子前盘面'), recent=make('div');
  detail.id=id+'-comparison';
  detail.setAttribute('role','tabpanel');
  models.tabIndex=0;
  models.setAttribute('role','region');
  models.setAttribute('aria-label','模型对照');
  detail.append(models);
  tabs.setAttribute('role','tablist');
  tabs.setAttribute('aria-label','本批种子');
  header.append(make('h3','','同种子对照'),beforeToggle);
  element.append(summary,header,seedNav,tabs,detail,recent);
  let channel, seeds=[], catalog=[], buttons=[], selected, pinned=null, showBefore=false;
  let historySeed=null, historyFrames=[], historyState='', pending=null;
  const slots=new Map();
  beforeToggle.type='button';
  beforeToggle.setAttribute('aria-pressed','false');
  beforeToggle.addEventListener('click',()=>{
   showBefore=!showBefore;
   beforeToggle.setAttribute('aria-pressed',String(showBefore));
   renderModels();
  });
  function renderModels(){
   const inBatch=seeds.includes(selected);
   const episodes=inBatch?channel.episodes:historyFrames;
   const frames=new Map(episodes.filter(item=>item.seed===selected).map(item=>[item.arm,item]));
   const live=inBatch&&connected&&channel.status==='running'&&Date.now()-at(channel.updated_at)<90000;
   const emptyLabel=inBatch?'等待首步':historyState==='loading'?'读取中':historyState==='error'?'暂不可读':'暂无落子记录';
   if(channel.arms.join(' ')!==[...slots.keys()].join(' ')){
    slots.clear();
    for(const arm of channel.arms)slots.set(arm,{node:make('section','gameplay-model')});
    models.replaceChildren(...Array.from(slots.values(),slot=>slot.node));
   }
   for(const [arm,slot] of slots){
    const episode=frames.get(arm);
    const key=[selected,showBefore,live,emptyLabel,episode?.turn,episode?.stop,episode?.updated_at].join('|');
    if(key===slot.key)continue;
    renderGameplayModel(slot.node,arm,episode,showBefore,live,emptyLabel);
    slot.key=key;
   }
  }
  function clearHistory(){
   pending?.controller?.abort();pending=null;historySeed=null;historyFrames=[];historyState='';
  }
  function describeSelection(){
   const inBatch=seeds.includes(selected);
   picker.value=String(selected);follow.hidden=pinned===null;
   follow.textContent=channel.status==='completed'?'返回末批':'跟随当前批次';
   retry.hidden=inBatch||historyState!=='error';
   if(!catalog.length)batchLabel.textContent=channel.batch?'':'等待评测批次数据。';
   else if(inBatch)batchLabel.textContent='第 '+n(channel.batch.number)+' / '+n(channel.batch.total)+' 批 · 本批 '+n(seeds.length)+' 个种子';
   else if(historyState==='loading')batchLabel.textContent='正在读取种子 '+selected+' 的对局记录…';
   else if(historyState==='error')batchLabel.textContent='种子 '+selected+' 的对局暂时无法读取。';
   else batchLabel.textContent='种子 '+selected+' · '+n(historyFrames.length)+' / '+n(channel.arms.length)+' 个模型已有落子记录';
  }
  async function readHistory(seed){
   clearHistory();historySeed=seed;historyState='loading';
   const request={controller:typeof AbortController==='function'?new AbortController():null};
   pending=request;describeSelection();renderModels();
   try{
    const response=await fetch('/gameplay/seed?channel='+encodeURIComponent(id)+'&seed='+encodeURIComponent(seed),
      {method:'GET',cache:'no-store',credentials:'same-origin',signal:request.controller?.signal});
    if(!response.ok)throw Error('seed unavailable');
    const data=await response.json();
    if(pending!==request)return;
    if(data.channel!==id||data.seed!==seed||!Array.isArray(data.episodes))throw Error('invalid seed record');
    historyFrames=data.episodes;historyState='ready';
   }catch{
    if(pending!==request)return;
    historyState='error';
   }
   pending=null;describeSelection();renderModels();
  }
  function activate(seed,focus=false){
   selected=seed;
   const index=seeds.indexOf(seed);
   tabs.hidden=index===-1;
   for(const [position,button] of buttons.entries()){
    button.setAttribute('aria-selected',String(position===index));
    button.tabIndex=position===index?0:-1;
   }
   if(index>=0){
    clearHistory();
    detail.setAttribute('role','tabpanel');detail.removeAttribute('aria-label');detail.setAttribute('aria-labelledby',buttons[index].id);
    if(focus)buttons[index].focus();
   }else{
    detail.setAttribute('role','region');detail.removeAttribute('aria-labelledby');detail.setAttribute('aria-label','种子 '+seed+' 的模型对照');
    if(historySeed!==seed){readHistory(seed);return}
   }
   describeSelection();renderModels();
  }
  function followBatch(){
   pinned=null;
   const running=channel.episodes.find(item=>!item.stop&&seeds.includes(item.seed));
   activate(running?.seed??seeds[0]??catalog[0]);
  }
  picker.addEventListener('change',()=>{
   const value=Number(picker.value);
   if(catalog.includes(value)){pinned=value;activate(value)}
  });
  follow.addEventListener('click',followBatch);
  retry.addEventListener('click',()=>{if(historySeed===selected)readHistory(selected)});
  function seedButton(seed,index){
   const button=make('button','',String(seed));
   button.id=id+'-seed-'+seed;
   button.type='button';
   button.setAttribute('role','tab');
   button.setAttribute('aria-controls',detail.id);
   button.addEventListener('click',()=>{pinned=null;activate(seed)});
   button.addEventListener('keydown',event=>{
    const adjacent={ArrowLeft:(index+seeds.length-1)%seeds.length,
        ArrowRight:(index+1)%seeds.length,Home:0,End:seeds.length-1};
    if(!own(adjacent,event.key))return;
    event.preventDefault();
    pinned=null;activate(seeds[adjacent[event.key]],true);
   });
   return button;
  }
  return {element,update(next){
   channel=next;
   channelCard(channel,summary);
   recent.replaceChildren();
   history(channel,recent);
   const batch=channel.batch;
   const nextCatalog=channel.seed_catalog||[];
   if(nextCatalog.length!==catalog.length||nextCatalog.some((seed,i)=>seed!==catalog[i])){
    catalog=nextCatalog;
    picker.replaceChildren(...catalog.map(seed=>{const option=make('option','',String(seed));option.value=String(seed);return option}));
    pickerTitle.textContent='全部 '+n(catalog.length)+' 个种子';
   }
   pickerLabel.hidden=!catalog.length;
   if(pinned!==null&&!catalog.includes(pinned))pinned=null;
   detail.hidden=beforeToggle.hidden=!batch&&!catalog.length;
   const nextSeeds=batch?.seeds||[];
   if(nextSeeds.length!==seeds.length||nextSeeds.some((seed,index)=>seed!==seeds[index])){
    seeds=nextSeeds;
    buttons=seeds.map(seedButton);
    tabs.replaceChildren(...buttons);
   }
   if(pinned!==null){activate(pinned)}
   else if(seeds.includes(selected)){activate(selected)}
   else if(seeds.length||catalog.length){followBatch()}
   else{clearHistory();tabs.hidden=true;describeSelection()}
  }};
 }
 function renderGameplay(channels){
  const keys=channels.map(c=>c.id);
  const cards=channels.map(c=>{
   if(!gameViews.has(c.id))gameViews.set(c.id,gameplayView(c.id));
   const view=gameViews.get(c.id);view.update(c);return view.element;
  });
  const next=keys.join(' ');
  if(shownGames!==next){
   shownGames=next;gameBox.replaceChildren(...cards);
  }
 }
 function render(){
  shownTrainingIds=new Set();
  const chans=snapshot?.channels;
  status.dataset.tone=connected?'active':received?'old':'normal';
  status.textContent=!loopback?'未联接本机服务':connected?(stream?'实时推送已联接':'本机服务已联接'):received?'连接中断':'正在查询本机服务';
  if(!loopback){
   desc.textContent='当前在文件模式中。切到「回看」可以载入本地逐步记录。';
   advisory('未联接训练或推理来源。此页不会启动任务、改参数或补写数据。');
   return;
  }
  if(connected)desc.textContent='查看训练计划中的训练、测评、推理和测速进展。';
  else if(received)desc.textContent='连接中断，保留最后一次收到的记录；以下不是当前实时状态。';
  else desc.textContent='正在等待本机服务提供观察来源。';
  if(!Array.isArray(chans)){advisory(connected?'服务未报告观察来源。':received?'暂未收到来源记录，等待重连。':'尚未收到观察来源。');return}
  const good=chans.filter(c=>c&&typeof c==='object'&&(filter==='all'||c.kind===filter));
  if(!good.length){advisory(filter==='all'?'还没有已联接的观察来源。':'当前没有此类来源。');return}
  const active=c=>knownActive.has(c.status)||c.status==='paused';
  const age=c=>at(c.updated_at||c.latest?.timestamp)||0;
  const channels=good.sort((a,b)=>Number(active(b))-Number(active(a))||age(b)-age(a)).slice(0,24);
  const games=channels.filter(c=>c.mode==='gameplay');
  const gamesFirst=games.some(active);
  if(gamesFirst&&gameBox.previousElementSibling===sourceBox)gameBox.after(sourceBox);
  if(!gamesFirst&&sourceBox.previousElementSibling===gameBox)gameBox.before(sourceBox);
  renderGameplay(games);
  const activeControl=document.activeElement;
  sourceBox.replaceChildren(...channels.filter(c=>c.mode!=='gameplay').map(c=>channelCard(c)));
  for(const c of channels)if(c.kind==='training'&&c.training_batch)shownTrainingIds.add(c.id||c.name);
  syncTrainingVisibility();
  if(activeControl?.dataset.trainingControl&&sourceBox.contains(activeControl))activeControl.focus({preventScroll:true});
 }
 /* PLAY_VIEW */
 const playView=createPlayView();
 function show(tab,focus=false){
  if(tab!=='review'){
   const b=$('play');if(b?.getAttribute('aria-label')==='暂停')b.click();
  }
  for(const name of tabOrder){
   const selected=name===tab;
   tabs[name].setAttribute('aria-selected',String(selected));tabs[name].tabIndex=selected?0:-1;
   panes[name].hidden=!selected;
  }
  playView.show(tab==='play');
  syncTrainingVisibility();
  if(tab==='review')window.dispatchEvent(new Event('kev-replay-visible'));
  if(focus)tabs[tab].focus();
 }
 for(const [name,tab] of Object.entries(tabs)){
  tab.addEventListener('click',()=>show(name));
  tab.addEventListener('keydown',event=>{
   if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
   event.preventDefault();event.stopPropagation();
   const index=tabOrder.indexOf(name), length=tabOrder.length;
   const adjacent={Home:0,End:length-1,ArrowLeft:(index+length-1)%length,ArrowRight:(index+1)%length};
   show(tabOrder[adjacent[event.key]],true);
  });
 }
 for(const tab of filterNodes){
  tab.addEventListener('click',()=>{
   filter=tab.dataset.monitorFilter;
   for(const item of filterNodes)item.setAttribute('aria-pressed',String(item===tab));
   render();
  });
 }
 function receive(data){
  if(!data||!Array.isArray(data.channels))throw Error('invalid source status');
  snapshot=data;connected=true;received=true;
 }
 function connect(){
  if(!loopback||suspended||stream)return;
  if(typeof EventSource!=='function'){poll();return;}
  const source=new EventSource('/channels/events');
  stream=source;
  source.addEventListener('channels',event=>{
   if(stream!==source)return;
   try{receive(JSON.parse(event.data))}catch{connected=false;received=true}
   render();
  });
  source.addEventListener('error',()=>{
   if(stream!==source)return;
   connected=false;received=true;
   render();
   if(source.readyState===EventSource.CLOSED){
    source.close();stream=null;poll();
   }
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
   receive(await response.json());
  }catch{connected=false}
  finally{
   received=true;inflight=false;render();
   if(!suspended)nextPoll=setTimeout(poll,2400);
  }
 }
 window.addEventListener('pagehide',()=>{
  suspended=true;clearTimeout(nextPoll);stream?.close();stream=null;playView.show(false);syncTrainingVisibility();
 });
 window.addEventListener('pageshow',()=>{if(suspended){suspended=false;connect();playView.show(!panes.play.hidden);syncTrainingVisibility()}});
 function finishMotion(){playView.finish()}
 window.addEventListener('resize',finishMotion);
 document.addEventListener('visibilitychange',()=>{if(document.hidden)finishMotion();syncTrainingVisibility()});
 if(reducedMotion?.addEventListener)reducedMotion.addEventListener('change',()=>{if(reducedMotion.matches)finishMotion()});
 show('monitor');render();connect();
})();
