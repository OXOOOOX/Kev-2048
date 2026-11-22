'use strict';
const $=id=>document.getElementById(id);
const seedSelect=$('seed-select'),leftSelect=$('left-select'),rightSelect=$('right-select');
const seek=$('seek'),play=$('play'),prev=$('prev'),next=$('next'),speed=$('speed'),marker=$('marker');
const turnText=$('turn-output'),ticks=$('ticks'),divergence=$('jump-divergence');
const names={up:['↑','向上'],down:['↓','向下'],left:['←','向左'],right:['→','向右']};
const fmt=new Intl.NumberFormat('zh-CN');
const words={terminal:'自然终局',cap:'步数到限',stopped:'已手动停止',incomplete:'结束原因未知'};
const policyNames={canonical:'标准选项',fixed:'固定槽位',identity:'原始棋盘',d4:'八方向投票',v2:'V2',hard:'V3-hard',greedy:'贪心参考',candidate:'候选版本',native:'原生模型',frozen_v2:'冻结 V2',official_kev_0_8b:'官方 Kev 0.8B'};
const MAX_FILES=8, MAX_TOTAL_BYTES=48*1024*1024, MAX_TOTAL_GAMES=256, MAX_TOTAL_FRAMES=100000;
let groups=new Map(),match={},turn=0,maxTurn=0,timer=null,playing=false,cut=null,loadId=0;
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
function haltPlayback(){clearTimeout(timer);timer=null;playing=false;refreshPlay()}
function stop(){haltPlayback();finishEffects()}
function setTurn(t,cancel=true,animated=false){
 if(!match.left)return;
 const previous=turn;if(cancel)stop();else finishEffects();
 turn=Math.min(maxTurn,Math.max(0,Number(t)));draw();
 if(animated&&turn===previous+1)animateForward(turn);
}
function stepPlayback(){
 clearTimeout(timer);if(!playing)return;
 if(turn>=maxTurn){haltPlayback();return}
 timer=setTimeout(()=>{setTurn(turn+1,false,true);stepPlayback()},650/Number(speed.value));
}
function toggle(){
 if(!maxTurn)return;
 if(playing){stop();return}
 if(turn>=maxTurn)setTurn(0,false);
 playing=true;refreshPlay();stepPlayback();
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
 $('toggle-import').textContent=expanded?'收起文件列表':'更换记录';
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
async function importRuns(entries){
 const my=++loadId;
 stop();
 $('pick').disabled=true; $('pick-folder').disabled=true; $('open-folder-files').disabled=true;
 importMessage('正在本地读取 '+entries.length+' 次运行…');
 try{
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
  groups=staged;
  const seedKeys=[...groups.keys()];
  const preferred=seedKeys.find(k=>sameOpeningPair(groups.get(k)))||seedKeys.find(k=>groups.get(k).length>1)||seedKeys[0];
  options(seedSelect,seedKeys.map(k=>[k,k+' · '+groups.get(k).length+' 局']),preferred);
  $('empty').hidden=true;$('replay').hidden=false;
  chooseSeed();
  importMessage('已载入 '+entries.length+' 次运行 · '+count+' 局 · '+groups.size+' 个种子。文件未上传。');
  setImportExpanded(false);$('toggle-import').focus();
 }catch(e){
  if(my===loadId)importError(e);
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
const PAGE_SIZE=120;
let folderRows=[],visibleCandidates=PAGE_SIZE;
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
  const entries=KevRuns.describeFiles(files).sort((a,b)=>Number(a.disabled)-Number(b.disabled)||(b.file.lastModified||0)-(a.file.lastModified||0)||a.path.localeCompare(b.path,'zh-CN',{numeric:true}));
  folderRows=entries.map(entry=>({entry,checked:false}));visibleCandidates=PAGE_SIZE;
  setImportExpanded(true);renderFolder();
 }catch(error){importError(error)}
}
function renderFolder(){
 const shown=folderRows.slice(0,visibleCandidates);
 const rows=shown.map(row=>{
  const {entry}=row,f=entry.file;
  const wrap=document.createElement('label');wrap.className='folder-choice';
  const box=document.createElement('input');box.type='checkbox';box.disabled=entry.disabled;box.checked=row.checked;
  const span=document.createElement('span'),path=document.createElement('span'),meta=document.createElement('small');
  path.className='folder-path';path.textContent=entry.path;
  const time=Number.isFinite(f.lastModified)&&f.lastModified>0&&f.lastModified<=8640000000000000?fDate.format(f.lastModified):'修改时间未记录';
  const parts=['日志',entry.resultFile?'结果摘要':'',entry.runFile?'运行信息':''].filter(Boolean).join(' + ');
  meta.className='folder-meta';meta.textContent=(entry.bytes/1048576).toFixed(2)+' MiB · '+time+' · '+parts+(box.disabled?' · '+entry.disabledReason:'');
  span.append(path,meta);wrap.append(box,span);
  box.addEventListener('change',()=>{row.checked=box.checked;updateFolderSelection()});
  return wrap;
 });
 $('folder-list').replaceChildren(...rows);
 $('folder-results').hidden=false;
 $('more-folder-files').hidden=shown.length>=folderRows.length;
 $('folder-summary').textContent=folderRows.length?'已显示 '+shown.length+' / '+folderRows.length+' 次运行。可导入优先，同类按修改时间从新到旧；勾选后才读取内容。':'这个文件夹没有找到常见的对局文件名。可以改用「直接选记录文件」。';
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
document.addEventListener('keydown',e=>{
 if(!match.left||e.altKey||e.ctrlKey||e.metaKey||e.shiftKey||e.target instanceof HTMLElement&&/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(e.target.tagName))return;
 if(e.key==='ArrowLeft'){e.preventDefault();setTurn(turn-1)}
 else if(e.key==='ArrowRight'){e.preventDefault();setTurn(turn+1,true,true)}
 else if(e.key===' '){e.preventDefault();toggle()}
});
