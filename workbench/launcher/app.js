'use strict';
const $=id=>document.getElementById(id);
const seedSelect=$('seed-select'),leftSelect=$('left-select'),rightSelect=$('right-select');
const seek=$('seek'),play=$('play'),prev=$('prev'),next=$('next'),speed=$('speed'),marker=$('marker');
const turnText=$('turn-output'),ticks=$('ticks'),divergence=$('jump-divergence');
const milestoneList=$('milestone-list'),milestoneTicks=$('milestone-ticks'),milestonePlay=$('milestone-play');
const names={up:['↑','向上'],down:['↓','向下'],left:['←','向左'],right:['→','向右']};
const fmt=new Intl.NumberFormat('zh-CN');
const words={terminal:'自然终局',cap:'步数到限',target:'达成 2048',stopped:'已手动停止',incomplete:'结束原因未知'};
const policyNames={canonical:'标准选项',fixed:'固定槽位',identity:'原始棋盘',d4:'八方向投票',v2:'V2',hard:'V3-hard',greedy:'贪心参考',candidate:'候选版本',native:'原生模型',frozen_v2:'冻结 V2',official_kev_0_8b:'官方 Kev 0.8B'};
const MAX_FILES=8, MAX_TOTAL_BYTES=48*1024*1024, MAX_TOTAL_GAMES=256, MAX_TOTAL_FRAMES=100000;
let groups=new Map(),match={},turn=0,maxTurn=0,timer=null,playing=false,playbackEnd=0,cut=null,loadId=0;
let milestones=[],selectedMilestone=null;
const MILESTONE_TILES=[512,1024,2048], MILESTONE_LEAD=4;
const panels={};
for(const p of document.querySelectorAll('.panel')){
 const get=q=>p.querySelector(q), b=get('.board');
 const nodes=Array.from({length:16},()=>{
  const t=document.createElement('div');t.className='tile';t.setAttribute('aria-hidden','true');b.append(t);return t;
 });
 panels[p.dataset.slot]={element:p,nodes,board:b,title:get('h2'),source:get('.panel-source'),meta:get('.run-meta'),status:get('.status'),scoreLabel:get('.score-label'),score:get('.score'),moveLabel:get('.move-label'),move:get('.move-name'),arrow:get('.arrow'),best:get('.best'),note:get('.decision-note'),final:get('.final')};
}
function options(select,values,chosen){
 select.replaceChildren(...values.map(([key,text])=>{
  const o=document.createElement('option');o.value=key;o.textContent=text;return o;
 }));
 select.value=chosen;
}
const eq=(a,b)=>a.length===b.length&&a.every((x,i)=>x===b[i]);
const maxP=g=>g.frames.at(-1).p;
const frame=(g,p)=>g.frames[Math.min(p,maxP(g))];
const revealFile=g=>'运行 '+g.sourceOrder+' · '+(g.displayFile||g.file);
const policyTitle=g=>(Object.hasOwn(policyNames,g.policy)?policyNames[g.policy]:null)||g.label||g.policy||'原生模型';
const titleOf=g=>g.runTitle?(g.policy==='native'||g.runTitle===policyTitle(g)?g.runTitle:g.runTitle+' · '+policyTitle(g)):policyTitle(g);
const optionName=g=>titleOf(g)+' · '+revealFile(g);
function putPanel(key){
 const g=match[key], refs=panels[key];
 refs.element.hidden=!g;
 if(!g)return;
 const end=maxP(g),shown=Math.min(turn,end),f=frame(g,turn),frozen=turn>end;
 const action=f.a,spawn=f.n,board=f.b;
 refs.title.textContent=titleOf(g);refs.source.textContent=revealFile(g);
 const identity=[g.device?'设备 '+g.device:'',g.headSha?'头指纹 '+g.headSha.slice(0,12):''].filter(Boolean);
 refs.meta.hidden=!identity.length;refs.meta.textContent=identity.join(' · ');
 refs.meta.title=g.headSha?'日志报告的检查点 SHA-256：'+g.headSha:'';
 for(let i=0;i<16;i++){
  const tile=refs.nodes[i];tile.dataset.value=board[i];tile.textContent=board[i]||'';
  const fresh=i===spawn&&!frozen;tile.classList.toggle('spawn',fresh);
  if(fresh)tile.title='本步新方块：第 '+(Math.floor(i/4)+1)+' 行，第 '+(i%4+1)+' 列';
  else tile.removeAttribute('title');
 }
 const rows=[];for(let i=0;i<16;i+=4)rows.push(board.slice(i,i+4).map(x=>x||'空').join('、'));
 refs.board.setAttribute('aria-label',refs.title.textContent+'棋盘：'+rows.join('；'));
 refs.status.textContent=shown===end?words[g.status]:shown?'记录可继续':'开局';
 refs.scoreLabel.textContent=shown?'第 '+shown+' 步后得分':'开局得分';refs.score.textContent=fmt.format(f.s);
 refs.moveLabel.textContent=action?'本步动作':'开局状态';
 refs.arrow.textContent=!frozen&&action?names[action][0]:'';
 refs.move.textContent=frozen?'无新动作':action?names[action][1]:'还未移动';
 refs.best.textContent=f.m;
 refs.final.textContent='第 '+end+' 步 · '+fmt.format(g.frames.at(-1).s)+' 分 · '+words[g.status];
 refs.note.textContent=frozen?'此侧记录止于第 '+end+' 步，棋盘保持不变。':
 g.status==='incomplete'&&shown===end?'仍有合法动作；没有匹配的结束摘要或标记，结束原因未知。':
 action?(f.g?'本步合并 +'+fmt.format(f.g)+' 分':'本步无合并得分'):'等待选择下一步。';
}
function draw(){
 putPanel('left');putPanel('right');
 seek.value=turn;seek.disabled=!maxTurn;seek.setAttribute('aria-valuetext','第 '+turn+' / '+maxTurn+' 步');
 seek.style.setProperty('--position',(maxTurn?100*turn/maxTurn:0)+'%');
 turnText.textContent='第 '+turn+' / '+maxTurn+' 步';prev.disabled=turn<=0;next.disabled=turn>=maxTurn;play.disabled=!maxTurn;
 const pair=match.right;
 const openingMismatch=pair&&!eq(match.left.frames[0].b,pair.frames[0].b);
 const mismatch=pair&&!eq(frame(match.left,turn).b,frame(pair,turn).b);
 $('comparison').dataset.mismatch=String(Boolean(mismatch||openingMismatch));
 $('comparison').textContent=!pair?'当前为单局回放。':openingMismatch?'开局棋盘不同，仅并列观看；同步步数不表示对照了同一开局。':
 '同一步不一定是同一盘面；这里只对齐两局各自的步数。'+(mismatch?' 当前两盘不同。':' 当前两盘相同。');
}
function refreshPlay(){
 play.setAttribute('aria-label',playing?'暂停':'播放');play.innerHTML=playing?
 '<svg aria-hidden="true" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zm8 0h4v14h-4z"/></svg>':
 '<svg aria-hidden="true" viewBox="0 0 24 24" fill="currentColor"><path d="M7 4l14 8-14 8z"/></svg>';
}
/* SHARED_MOTION */
const replayMotion=createBoardMotion(), finishEffects=replayMotion.finish;
if(reducedMotion?.addEventListener)reducedMotion.addEventListener('change',()=>{if(reducedMotion.matches)finishEffects()});
window.addEventListener('resize',finishEffects);
document.addEventListener('visibilitychange',()=>{if(document.hidden)finishEffects()});
function animateForward(t){
 if(reducedMotion?.matches || typeof Element==='undefined' || typeof Element.prototype.animate!=='function')return;
 const inputs=[];
 for(const [key,refs] of Object.entries(panels)){
  const rec=match[key];
  if(!rec||t>maxP(rec))continue;
  const {b:after,a:action,n:spawn}=rec.frames[t];
  const input=boardTransition(refs.board,refs.nodes,rec.frames[t-1].b,after,action,spawn);
  if(input)inputs.push(input);
 }
 replayMotion.animate(inputs,Math.max(.5,Math.min(4,Number(speed.value)||1)));
}
function haltPlayback(){clearTimeout(timer);timer=null;playing=false;playbackEnd=0;refreshPlay()}
function stop(){haltPlayback();finishEffects()}
function setTurn(t,cancel=true,animated=false){
 if(!match.left)return;
 const previous=turn;if(cancel)stop();else finishEffects();
 turn=Math.min(maxTurn,Math.max(0,Number(t)));draw();
 if(animated&&turn===previous+1)animateForward(turn);
}
function stepPlayback(){
 clearTimeout(timer);if(!playing)return;
 if(turn>=playbackEnd){haltPlayback();return}
 timer=setTimeout(()=>{setTurn(turn+1,false,true);stepPlayback()},650/Number(speed.value));
}
function startPlayback(end){
 playbackEnd=end;playing=true;refreshPlay();stepPlayback();
}
const firstMilestone=(game,tile)=>game.frames.find(step=>step.p>0&&step.m>=tile);
function playMilestone(milestone){
 selectMilestone(milestone);
 setTurn(Math.max(0,milestone.turn-MILESTONE_LEAD));
 startPlayback(milestone.turn);
}
function toggle(){
 if(!maxTurn)return;
 if(playing){stop();return}
 if(turn>=maxTurn)setTurn(0,false);
 startPlayback(maxTurn);
}
function selectMilestone(milestone){
 selectedMilestone=milestone;
 for(const item of milestones){
  const selected=item===milestone;
  item.button.setAttribute('aria-pressed',String(selected));
  item.mark.dataset.selected=String(selected);
 }
 milestonePlay.hidden=!milestone;
 if(milestone){
  const start=Math.max(0,milestone.turn-MILESTONE_LEAD);
  milestonePlay.textContent='从第 '+start+' 步播放';
  milestonePlay.setAttribute('aria-label','从第 '+start+' 步播放到首次合出 '+milestone.tile);
 }
}
function renderMilestones(){
 milestones=[];
 for(const [slot,game] of Object.entries(match)){
  if(!game)continue;
  const side=match.right?(slot==='left'?'左侧':'右侧'):'';
  for(const tile of MILESTONE_TILES){
   const first=firstMilestone(game,tile);
   if(!first)continue;
   const button=document.createElement('button'),mark=document.createElement('span');
   button.type='button';button.className='milestone-button';
   button.dataset.slot=slot;button.dataset.tile=tile;button.dataset.turn=first.p;
   button.textContent=(side?side+' ':'')+tile+' · 第 '+first.p+' 步';
   button.setAttribute('aria-label',side+'首次合出 '+tile+' · 第 '+first.p+' 步');
   mark.className='milestone-tick';mark.dataset.slot=slot;
   mark.style.left=100*first.p/maxTurn+'%';
   const item={button,mark,slot,turn:first.p,tile};
   button.addEventListener('click',()=>{setTurn(item.turn);selectMilestone(item)});
   milestones.push(item);
  }
 }
 $('milestones').hidden=!milestones.length;
 milestoneList.replaceChildren(...milestones.map(item=>item.button));
 milestoneTicks.replaceChildren(...milestones.map(item=>item.mark));
 selectMilestone(null);
}
function resolveDivergence(a,b){
 if(!b)return null;
 if(!eq(a.frames[0].b,b.frames[0].b))return {turn:0,text:'从开局盘面就不同'};
 for(let i=1;i<=Math.min(maxP(a),maxP(b));i++){
  if(a.frames[i].a!==b.frames[i].a)return {turn:i,text:'首次走法分岔 · 第 '+i+' 步'};
  if(!eq(a.frames[i].b,b.frames[i].b))return {turn:i,text:'首次盘面分岔 · 第 '+i+' 步'};
 }
 return null;
}
function updateMatch(goTo=0,changed=null){
 const all=groups.get(seedSelect.value);
 if(!all?.length)return;
 let l=all.find(g=>g.uid===leftSelect.value)||all[0],r=all.find(g=>g.uid===rightSelect.value)||null;
 if(r===l){
  if(changed==='right')l=all.find(g=>g!==r);else r=all.find(g=>g!==l&&eq(g.frames[0].b,l.frames[0].b))||all.find(g=>g!==l)||null;
 }
 options(leftSelect,all.map(g=>[g.uid,optionName(g)]),l.uid);
 options(rightSelect,[['','不比较'],...all.filter(g=>g!==l).map(g=>[g.uid,optionName(g)])],r?.uid||'');
 $('right-control').hidden=all.length<2;
 const pair=Boolean(r);$('boards').classList.toggle('single',!pair);
 $('track-hint').textContent=pair?'比较时对齐两边的步数':'查看这局记录的每一步';
 match={left:l,right:r};
 maxTurn=Math.max(maxP(l),r?maxP(r):0);seek.max=maxTurn;
 renderMilestones();
 const marks=[...new Set([0,Math.round(maxTurn/4),Math.round(maxTurn/2),Math.round(maxTurn*3/4),maxTurn])];
 ticks.replaceChildren(...marks.map(t=>{
  const x=document.createElement('span');x.className='tick';x.style.left=(maxTurn?100*t/maxTurn:0)+'%';x.textContent=t;return x;
 }));
 cut=resolveDivergence(l,r);divergence.hidden=!cut;divergence.textContent=cut?.text||'';
 marker.hidden=!pair||maxP(l)===maxP(r);
 if(!marker.hidden){
  const shorter=maxP(l)<maxP(r)?l:r;
  marker.style.left=100*maxP(shorter)/maxTurn+'%';
  marker.textContent=(shorter===l?'左侧':'右侧')+'停止记录 · '+maxP(shorter)+' 步';
  marker.onclick=()=>setTurn(maxP(shorter));
 }
 setTurn(goTo);
}
const openingKey=g=>g.frames[0].b.join(',');
function sameOpeningPair(all){
 const seen=new Map();
 for(const game of all){const key=openingKey(game);if(seen.has(key))return [seen.get(key),game];seen.set(key,game)}
 return null;
}
function chooseSeed(){
 const all=groups.get(seedSelect.value);
 if(!all?.length)return;
 const [left,right]=sameOpeningPair(all)||all;
 options(leftSelect,all.map(g=>[g.uid,optionName(g)]),left.uid);
 options(rightSelect,all.filter(g=>g!==left).map(g=>[g.uid,optionName(g)]),right?.uid||'');
 updateMatch();
}
function importMessage(text,error=false){
 const el=$('import-status');el.dataset.error=String(error);el.textContent=text;
}
function setImportExpanded(expanded){
 $('import-tools').hidden=!expanded;
 $('import-panel').dataset.collapsed=String(!expanded);
 $('toggle-import').hidden=!groups.size;
 $('toggle-import').setAttribute('aria-expanded',String(expanded));
 $('toggle-import').textContent=expanded?'收起导入选项':'导入其他记录';
}
function importError(error){
 setImportExpanded(true);
 importMessage(String(error.message||error)+(groups.size?' 已保留上一次成功导入的回放。':''),true);
}
async function importFiles(files){
 if(!files.length)return;
 try{await importRuns(KevRuns.describeFiles(files,{direct:true}))}
 catch(error){++loadId;$('pick').disabled=false;$('pick-folder').disabled=false;importError(error);updateFolderSelection()}
}
async function importRuns(source,{selection=null,archive=false,focusReplay=false,milestoneTile=null}={}){
 const my=++loadId;
 stop();
 $('pick').disabled=true; $('pick-folder').disabled=true; $('open-folder-files').disabled=true;
 const message=archive?archiveMessage:importMessage;
 message('正在读取对局…');
 try{
  const entries=typeof source==='function'?await source():source;
  if(my!==loadId)return;
  if(!entries.length)throw Error('没有找到逐步日志；摘要不能单独回放。请选择日志或对应运行文件夹。');
  if(entries.length>MAX_FILES||entries.reduce((sum,e)=>sum+e.bytes,0)>MAX_TOTAL_BYTES)
   throw Error('可一次导入最多 8 次运行，含摘要总大小不超过 48 MiB。');
  const unavailable=entries.find(e=>e.disabled);
  if(unavailable)throw Error('「'+unavailable.path+'」：'+unavailable.disabledReason);
  const staged=new Map();let count=0,totalFrames=0;
  for(const [i,entry] of entries.entries()){
   if(my!==loadId)return;
   const {games}=await KevRuns.loadRun(entry);
   if(my!==loadId)return;
   for(const [j,g] of games.entries()){
    if(++count>MAX_TOTAL_GAMES||(totalFrames+=g.frames.length)>MAX_TOTAL_FRAMES)
     throw Error('本次对局数或帧数超过页面上限；请选择更少的结果文件。');
    const group=String(g.seed);
    if(!staged.has(group))staged.set(group,[]);
    staged.get(group).push({...g,displayFile:entry.path,uid:'file-'+i+'-game-'+j,sourceOrder:i+1});
   }
  }
  if(my!==loadId)return;
  if(!count)throw Error('没有找到可回放的逐步记录。');
  const seedKeys=[...staged.keys()];
  const preferred=selection?.seed!=null?String(selection.seed):
   seedKeys.find(k=>sameOpeningPair(staged.get(k)))||seedKeys.find(k=>staged.get(k).length>1)||seedKeys[0];
  const selected=selection&&staged.get(preferred)?.find(game=>
   (selection.policy==null||game.policy===selection.policy)&&(selection.runId==null||game.runId===selection.runId));
  if(selection&&!selected)throw Error('列表中的种子或策略与逐步记录不一致。');
  if(milestoneTile!==null&&(!selected||!firstMilestone(selected,milestoneTile)))
   throw Error('逐步记录里没有合出 '+milestoneTile+' 的那一步。');
  groups=staged;
  options(seedSelect,seedKeys.map(k=>[k,k+' · '+groups.get(k).length+' 局']),preferred);
  $('empty').hidden=true;$('replay').hidden=false;
  if(selected){
   options(leftSelect,groups.get(preferred).map(game=>[game.uid,optionName(game)]),selected.uid);
   options(rightSelect,[['','不比较']],'');updateMatch();
  }
  else chooseSeed();
  if(archive)archiveMessage('已打开 '+titleOf(selected)+' · 种子 '+String(selected.seed)+'。');
  else importMessage('已载入 '+entries.length+' 次运行 · '+count+' 局 · '+groups.size+' 个种子。文件未上传。');
  setImportExpanded(false);
  (focusReplay?$('replay'):$('toggle-import')).focus();
  if(milestoneTile!==null){
   const milestone=milestones.find(item=>item.slot==='left'&&item.tile===milestoneTile);
   if($('arcade-review-pane')?.hidden){selectMilestone(milestone);setTurn(milestone.turn)}
   else playMilestone(milestone);
  }
 }catch(e){
  if(my===loadId){
   if(archive)archiveMessage(String(e.message||e)+(groups.size?' 已保留当前回放。':''),true);
   else importError(e);
  }
 }finally{
  if(my===loadId){$('pick').disabled=false;$('pick-folder').disabled=false;updateFolderSelection()}
 }
}
const drop=$('import-panel'),fileInput=$('file-input');
setImportExpanded(true);
$('toggle-import').addEventListener('click',()=>setImportExpanded($('import-tools').hidden));
$('pick').addEventListener('click',()=>fileInput.click());
fileInput.addEventListener('change',()=>{
 const files=Array.from(fileInput.files||[]);fileInput.value='';importFiles(files);
});
const folderInput=$('folder-input'),pickFolder=$('pick-folder'),openFolder=$('open-folder-files');
const folderSupported='webkitdirectory' in folderInput;
const fDate=new Intl.DateTimeFormat('zh-CN',{dateStyle:'short',timeStyle:'short'});
const PAGE_SIZE=120, PREVIEW_CONCURRENCY=4;
let folderRows=[],visibleCandidates=PAGE_SIZE;
let waitingPreviews=[],nextPreview=0,readingPreviews=0,folderGeneration=0,previewsRemaining=0;
const folderFilter=$('folder-filter'),archiveFilter=$('archive-filter');
const folderSort=$('folder-sort'),archiveSort=$('archive-sort');
const folderTarget=$('folder-target'),archiveTarget=$('archive-target');
const query=input=>String(input.value||'').trim().toLocaleLowerCase('zh-CN');
const contains=(value,q)=>String(value??'').toLocaleLowerCase('zh-CN').includes(q);
const gameMatches=(game,q)=>!q||contains(game.model,q)||contains(game.seed,q);
const hasMetric=n=>Number.isSafeInteger(n)&&n>=0;
const valueOf=n=>hasMetric(n)?fmt.format(n):'—';
const REPLAY_TILE_TICKS=[0,...MILESTONE_TILES], REPLAY_TILE_LIMIT=REPLAY_TILE_TICKS.at(-1);
const reachesTarget=game=>hasMetric(game.max_tile)&&game.max_tile>=REPLAY_TILE_LIMIT;
function replayStamp(value){
 const stamp=typeof value==='string'?Date.parse(value):value;
 return Number.isFinite(stamp)&&stamp>0&&stamp<=8640000000000000?stamp:null;
}
function compareGames(a,b,key){
 const newest=(replayStamp(b.time)??0)-(replayStamp(a.time)??0);
 if(!['score','max_tile','turns'].includes(key))return newest;
 const value=game=>hasMetric(game[key])?game[key]:-1;
 return value(b)-value(a)||newest;
}
function replayTileProgress(game,open,disabled){
 const maxTile=game.max_tile;
 const progress=document.createElement('span');progress.className='replay-tile-progress';
 const known=hasMetric(maxTile);
 progress.dataset.reached=String(known&&maxTile>=REPLAY_TILE_LIMIT);
 const bar=document.createElement('span'),labels=document.createElement('span');
 bar.className='replay-progress-bar';labels.className='replay-progress-labels';
 bar.setAttribute('aria-hidden','true');
 REPLAY_TILE_TICKS.forEach((tick,index)=>{
  const turn=game.milestones?.[tick];
  const canPlay=index>0&&!disabled&&known&&maxTile>=tick&&Number.isSafeInteger(turn)&&turn>0&&turn<=game.turns;
  const label=document.createElement(canPlay?'button':'span');
  if(canPlay){
   labels.dataset.hasSteps='true';
   label.type='button';label.className='replay-progress-marker';label.dataset.tile=tick;
   label.setAttribute('aria-label','从合成前回看 '+tick+' · 第 '+turn+' 步');
   label.title='从第 '+Math.max(0,turn-MILESTONE_LEAD)+' 步播放到第 '+turn+' 步';
   const target=document.createElement('span'),step=document.createElement('span');
   target.textContent=String(tick);step.textContent='第 '+turn+' 步';
   label.append(target,step);label.addEventListener('click',()=>open(tick));
  }else{
   label.textContent=String(tick);label.setAttribute('aria-hidden','true');
  }
  label.style.left=100*tick/REPLAY_TILE_LIMIT+'%';labels.append(label);
  if(!index)return;
  const start=REPLAY_TILE_TICKS[index-1],width=tick-start;
  const fill=known?Math.max(0,Math.min(1,(maxTile-start)/width))*100:0;
  const segment=document.createElement('span');segment.className='replay-progress-segment';
  segment.style.setProperty('--segment-width',String(width));
  segment.style.setProperty('--segment-fill',fill+'%');
  segment.title=start+'–'+tick+' · '+(known?Math.round(fill)+'%':'进度未知');
  bar.append(segment);
 });
 progress.append(bar,labels);return progress;
}
function replayCard(game,title,open,disabled=false){
 const card=document.createElement('div');card.className='replay-card';card.dataset.disabled=String(disabled);
 const button=document.createElement('button');button.type='button';button.className='replay-row';
 button.title=title;button.disabled=disabled;
 const model=document.createElement('span');model.className='replay-row-name';
 model.textContent=(game.starred?'★ ':'')+(game.model||'未命名模型');
 const metrics=document.createElement('span');metrics.className='replay-row-stats';
 const timeLabel=game.timeSource==='modified'?'文件时间':game.starred?'保存时间':'时间';
 const stamp=replayStamp(game.time);
 for(const part of ['种子 '+(game.seed??'—'),'步数 '+valueOf(game.turns),'分数 '+valueOf(game.score),
  '最大方块 '+valueOf(game.max_tile),timeLabel+' '+(stamp===null?'—':fDate.format(stamp))]){
  const field=document.createElement('span');field.textContent=part;metrics.append(field);
 }
 button.append(model,metrics);button.addEventListener('click',()=>open(null));
 card.append(button,replayTileProgress(game,open,disabled));return card;
}
function archiveMessage(message,error=false){
 const status=$('archive-status');status.dataset.error=String(error);status.textContent=message;
}
const archiveId=/^[0-9a-f]{32}\/[a-z0-9][a-z0-9-]{0,63}$/;
let archiveRows=[],archiveLimit=PAGE_SIZE,archiveRequest=0;
const address=window.location;
const archivesAvailable=address.protocol==='http:'&&['localhost','127.0.0.1'].includes(address.hostname)&&
 !address.username&&!address.password;
const archiveRequestOptions={method:'GET',cache:'no-store',credentials:'same-origin'};
function openArchive(game,milestoneTile){
 importRuns(async()=>{
  const response=await fetch('/replay/record?id='+encodeURIComponent(game.id),archiveRequestOptions);
  if(!response.ok)throw Error('无法读取这局存档，请刷新存档列表。');
  const record=await response.json();
  if(record?.id!==game.id||typeof record.moves!=='string'||!record.manifest||typeof record.manifest!=='object'||Array.isArray(record.manifest))
   throw Error('本机对局存档格式无效。');
  const file=(name,body)=>({name,webkitRelativePath:'本机存档/'+game.id+'/'+name,
   text:async()=>body,size:new TextEncoder().encode(body).byteLength});
  return KevRuns.describeFiles([file('moves.jsonl',record.moves),file('run.json',JSON.stringify(record.manifest))]);
 },{selection:game,archive:true,focusReplay:true,milestoneTile});
}
function renderArchives(){
 const q=query(archiveFilter), target=archiveTarget.checked;
 const matches=archiveRows.filter(game=>gameMatches(game,q)&&(!target||reachesTarget(game)))
  .sort((a,b)=>compareGames(a,b,archiveSort.value));
 const shown=matches.slice(0,archiveLimit);
 $('archive-list').replaceChildren(...shown.map(game=>replayCard(game,game.model,tile=>openArchive(game,tile))));
 $('archive-filter-wrap').hidden=!archiveRows.length;
 $('more-archives').hidden=shown.length>=matches.length;
 archiveMessage(archiveRows.length?(!matches.length?'没有匹配的存档。':
  '显示 '+shown.length+' / '+matches.length+' 局 · 点击回看。'):'还没有本机存档。在即时游玩结束后点星标保存。');
}
async function refreshSavedReplays(){
 if(!archivesAvailable)return;
 $('archive-panel').hidden=false;
 const ticket=++archiveRequest;
 $('refresh-archive').disabled=true;
 archiveMessage('正在读取本机存档…');
 try{
  const response=await fetch('/replay/catalog',archiveRequestOptions);
  if(ticket!==archiveRequest)return;
  if(!response.ok)throw Error('无法读取本机存档，仍可手动导入。');
  const catalog=await response.json();
  if(ticket!==archiveRequest)return;
  if(!Array.isArray(catalog?.games))throw Error('本机存档列表格式无效。');
  archiveRows=catalog.games.filter(game=>game&&archiveId.test(game.id)&&typeof game.model==='string'&&Number.isSafeInteger(game.seed))
   .map(game=>({...game,timeSource:game.starred?'reported':'modified'}));
  renderArchives();
 }catch(error){if(ticket===archiveRequest)archiveMessage(String(error.message||error),true)}
 finally{if(ticket===archiveRequest)$('refresh-archive').disabled=false}
}
$('refresh-archive').addEventListener('click',refreshSavedReplays);
function applyArchiveControls(){archiveLimit=PAGE_SIZE;$('archive-list').scrollTop=0;renderArchives()}
archiveFilter.addEventListener('input',applyArchiveControls);
archiveSort.addEventListener('change',applyArchiveControls);
archiveTarget.addEventListener('change',applyArchiveControls);
$('more-archives').addEventListener('click',()=>{archiveLimit+=PAGE_SIZE;renderArchives()});
window.addEventListener('kev-replay-visible',refreshSavedReplays);

function updateFolderSelection(){
 const selected=folderRows.filter(x=>x.checked);
 const bytes=selected.reduce((sum,x)=>sum+x.entry.bytes,0);
 const valid=selected.length<=MAX_FILES&&bytes<=MAX_TOTAL_BYTES;
 openFolder.disabled=!selected.length||!valid||$('pick').disabled;
 $('folder-selection').textContent=selected.length
  ?'已选 '+selected.length+' 次运行 · '+(bytes/1048576).toFixed(2)+' MiB'+(valid?'':'，最多 '+MAX_FILES+' 次运行、共 48 MiB')
  :'未选择';
}
function showFolder(files){
 try{
  const entries=KevRuns.describeFiles(files).sort((a,b)=>a.path.localeCompare(b.path,'zh-CN',{numeric:true}));
  folderGeneration++;folderFilter.value='';folderSort.value='time';folderTarget.checked=false;
  folderRows=entries.map(entry=>({entry,checked:false,previews:[]}));
  waitingPreviews=folderRows.filter(row=>!row.entry.disabled);
  nextPreview=0;previewsRemaining=waitingPreviews.length;visibleCandidates=PAGE_SIZE;
  setImportExpanded(true);renderFolder();readPreviews();
 }catch(error){importError(error)}
}
function folderGames(row){
 const {entry}=row;
 const fallback={model:entry.path,time:entry.file.lastModified,timeSource:'modified'};
 return (row.previews.length?row.previews:[fallback]).map(game=>({row,game}));
}
function readPreviews(){
 while(readingPreviews<PREVIEW_CONCURRENCY&&nextPreview<waitingPreviews.length){
  const row=waitingPreviews[nextPreview++], ticket=folderGeneration;
  readingPreviews++;
  (async()=>{
   try{row.previews=await KevRuns.describeReplay(row.entry)}
   catch{row.previews=[]}
   finally{
    readingPreviews--;
    if(ticket===folderGeneration&&--previewsRemaining===0)renderFolder();
    readPreviews();
   }
  })();
 }
}
function renderFolder(){
 const q=query(folderFilter),target=folderTarget.checked;
 const matches=folderRows.flatMap(folderGames)
  .filter(({row,game})=>(contains(row.entry.path,q)||gameMatches(game,q))&&(!target||reachesTarget(game)))
  .sort((a,b)=>Number(a.row.entry.disabled)-Number(b.row.entry.disabled)||compareGames(a.game,b.game,folderSort.value));
 const shown=matches.slice(0,visibleCandidates),checkBoxes=new Map();
 const rows=shown.map(({row,game})=>{
  const {entry}=row;
  const wrap=document.createElement('div');wrap.className='folder-choice';
  const box=document.createElement('input');box.type='checkbox';box.disabled=entry.disabled;box.checked=row.checked;
  box.setAttribute('aria-label','选择用于比较：'+entry.path);
  const details=document.createElement('div');details.className='folder-run';
  const title=entry.path+(entry.disabled?' · '+entry.disabledReason:'');
  details.append(replayCard(game,title,tile=>importRuns([entry],{selection:game,focusReplay:true,milestoneTile:tile}),entry.disabled));
  if(entry.disabled){
   const reason=document.createElement('small');reason.className='folder-help';reason.textContent=entry.disabledReason;details.append(reason);
  }
  wrap.append(box,details);
  if(!checkBoxes.has(row))checkBoxes.set(row,[]);
  const peers=checkBoxes.get(row);
  peers.push(box);
  box.addEventListener('change',()=>{row.checked=box.checked;for(const peer of peers)peer.checked=row.checked;updateFolderSelection()});
  return wrap;
 });
 $('folder-list').replaceChildren(...rows);
 $('folder-results').hidden=false;
 $('folder-filter-wrap').hidden=!folderRows.length;
 $('more-folder-files').hidden=shown.length>=matches.length;
 const summary=$('folder-summary');
 if(previewsRemaining)summary.textContent='正在读取文件夹的对局统计…';
 else if(!folderRows.length)summary.textContent='该文件夹没有找到逐步记录。';
 else if(!matches.length)summary.textContent='没有匹配的棋局。';
 else summary.textContent='已显示 '+shown.length+' / '+matches.length+' 局。';
 updateFolderSelection();
}
pickFolder.hidden=!folderSupported;$('folder-help').hidden=!folderSupported;
pickFolder.addEventListener('click',()=>folderInput.click());
folderInput.addEventListener('change',()=>{
 const files=Array.from(folderInput.files||[]);
 folderInput.value='';
 if(files.length)showFolder(files);
});
openFolder.addEventListener('click',()=>{
 if(!openFolder.disabled)importRuns(folderRows.filter(x=>x.checked).map(x=>x.entry));
});
$('more-folder-files').addEventListener('click',()=>{visibleCandidates+=PAGE_SIZE;renderFolder()});
function applyFolderControls(){visibleCandidates=PAGE_SIZE;$('folder-list').scrollTop=0;renderFolder()}
folderFilter.addEventListener('input',applyFolderControls);
folderSort.addEventListener('change',applyFolderControls);
folderTarget.addEventListener('change',applyFolderControls);
drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.toggle('dragover',true)});
drop.addEventListener('dragleave',()=>drop.classList.toggle('dragover',false));
drop.addEventListener('drop',e=>{
 e.preventDefault();drop.classList.toggle('dragover',false);importFiles(Array.from(e.dataTransfer?.files||[]));
});
seedSelect.addEventListener('change',chooseSeed);
leftSelect.addEventListener('change',()=>updateMatch(turn,'left'));rightSelect.addEventListener('change',()=>updateMatch(turn,'right'));
seek.addEventListener('input',()=>setTurn(seek.value));
prev.addEventListener('click',()=>setTurn(turn-1));next.addEventListener('click',()=>setTurn(turn+1,true,true));
play.addEventListener('click',toggle);speed.addEventListener('change',()=>{finishEffects();stepPlayback()});
$('jump-start').addEventListener('click',()=>setTurn(0));$('jump-end').addEventListener('click',()=>setTurn(maxTurn));
divergence.addEventListener('click',()=>{if(cut)setTurn(cut.turn)});
milestonePlay.addEventListener('click',()=>{
 if(selectedMilestone)playMilestone(selectedMilestone);
});
document.addEventListener('keydown',e=>{
 if(!match.left||e.altKey||e.ctrlKey||e.metaKey||e.shiftKey||e.target instanceof HTMLElement&&/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(e.target.tagName))return;
 if(e.key==='ArrowLeft'){e.preventDefault();setTurn(turn-1)}
 else if(e.key==='ArrowRight'){e.preventDefault();setTurn(turn+1,true,true)}
 else if(e.key===' '){e.preventDefault();toggle()}
});
