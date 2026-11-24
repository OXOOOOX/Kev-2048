function createPlayView(){
 const form=$('play-form'), options=$('play-model-options'), selection=$('play-selection');
 const seed=$('play-seed'), start=$('play-start'), stop=$('play-stop'), feedback=$('play-feedback');
 const sessionView=$('play-session-view'), title=$('play-session-title'), state=$('play-session-state'), boards=$('play-boards');
 const save=$('play-save'), saveNote=$('play-save-note'), pause=$('play-pause'), latest=$('play-latest');
 const cursorText=$('play-cursor'), seek=$('play-seek'), previous=$('play-prev'), next=$('play-next'), review=$('play-review');
 const names={loading:'加载中',running:'正在推理',paused:'已暂停推理',stopping:'正在停止',completed:'已结束',
  opening:'开局',review:'回看',
  stopped:'已停止',error:'推理失败',terminal:'自然结束',target:'达到 2048',cap:'达到步数上限'};
 const messages={
  idle:'连接本机工作台后可即时游玩。',
  connecting:'正在连接本机推理服务。',
  disabled:'当前本机服务未启用即时游玩。',
  reconnecting:'连接中断，正在重连。',
 };
 const maxModels=8, maxSeed=2147483647, choices=new Map(), slots=new Map();
 const chosen=new Set();
 let visible=false, generation=0, snapshot=null, connection='idle', busy=false, error='';
 let events=null, timer=null, fallback=false, catalog='', sessionId=null, initialized=false;
 let cursor=null, frame=null, frameRequest=0, reviewTimer=null, reviewEnd=null;

 const length=session=>Math.max(0,...session.games.map(game=>game.turn));
 const cursorAt=session=>cursor??length(session);

 function finish(){
  for(const slot of slots.values())slot.motion.finish();
 }
 function stopReview(){
  clearTimeout(reviewTimer);reviewTimer=null;reviewEnd=null;
 }
 function followLatest(){
  stopReview();frameRequest++;cursor=null;frame=null;
 }
 function renderTeacherDecision(card,game){
  const values=game.action_values;
  card.append(make('p','gameplay-choice','搜索选择 '+directions[game.action]+' · 评分 '+n(values[game.action])));
  const list=make('dl','play-teacher-values');
  list.setAttribute('aria-label','各方向搜索评分');
  for(const [action,name] of Object.entries(directions)){
   const row=make('div','play-teacher-value');
   row.dataset.chosen=String(action===game.action);
   row.append(make('dt','',name),make('dd','',finite(values[action])?n(values[action]):'不可走'));
   list.append(row);
  }
  card.append(list);
 }
 function renderGame(slot,game){
  if(slot.previous?.turn===game.turn&&slot.previous.status===game.status)return;
  slot.motion.finish();
  const previous=slot.previous;
  const move=previous&&game.turn===previous.turn+1;
  const animate=move&&visible&&!document.hidden&&!reducedMotion?.matches;
  const header=make('div','gameplay-model-head');
  const status=make('span','pill',label(names,game.status,'状态未确认'));
  status.dataset.tone=game.status==='running'?'active':game.status==='error'?'error':'normal';
  header.append(make('h3','',game.name),status);
  const score=make('p','gameplay-score','得分 '+n(game.score));
  if(animate&&previous.score!==game.score)score.dataset.changed='true';
  const details='第 '+n(game.turn)+' 步 · 最大方块 '+n(game.max_tile)+
   (finite(game.decision_ms)?' · '+n(game.decision_ms)+' 毫秒':'');
  slot.card.replaceChildren(header,score,make('p','monitor-meta',details));
  const grid=boardGrid(game.board,game.name+'当前棋盘');
  if(grid)slot.card.append(grid);
  if(own(directions,game.action)){
   if(game.kind==='teacher'&&game.action_values)renderTeacherDecision(slot.card,game);
   if(game.kind==='model'&&game.probabilities)renderGameplayDecision(slot.card,game,animate?previous:null);
  }
  if(animate&&grid&&Array.isArray(game.before)&&typeof grid.getBoundingClientRect==='function'){
   const before=game.before.flat(), prior=previous.board.flat();
   if(before.every((value,index)=>value===prior[index])){
    const transition=boardTransition(grid,Array.from(grid.children),before,game.board.flat(),game.action);
    if(transition)slot.motion.animate([transition],1.65);
   }
  }
  slot.previous=game;
 }
 function renderSession(session){
  sessionView.hidden=!session;
  if((session?.id??null)!==sessionId){followLatest();finish();slots.clear();sessionId=session?.id??null;}
  if(!session)return;
  title.textContent='种子 '+String(session.seed);
  state.textContent=label(names,session.status,'状态未确认');
  state.dataset.tone=session.status==='paused'?'old':session.active?'active':session.status==='error'?'error':'normal';
  renderTimeline(session);
  if(session.games.length!==slots.size||session.games.some(game=>!slots.has(game.model))){
   finish();slots.clear();
   for(const game of session.games)slots.set(game.model,{card:make('article','gameplay-model'),motion:createBoardMotion()});
   boards.replaceChildren(...Array.from(slots.values(),slot=>slot.card));
  }
  const reviewing=cursor!==null;
  const games=reviewing?frame?.games:session.games;
  for(const game of games||[]){
   let playStatus=game.status;
   if(reviewing&&game.turn===0)playStatus='opening';
   else if(reviewing&&game.status==='running')playStatus='review';
   else if(!reviewing&&session.status==='paused'&&game.status==='running')playStatus='paused';
   renderGame(slots.get(game.model),{...game,status:playStatus});
  }
 }
 function renderTimeline(session){
  const ready=connection==='ready'&&!busy;
  const recorded=length(session), shown=cursorAt(session), reading=cursor!==null&&cursor!==frame?.turn;
  const saved=session.saved===true, paused=session.status==='paused';
  save.disabled=!ready||session.active||saved||!recorded;
  save.setAttribute('aria-pressed',String(saved));
  save.textContent=saved?'★ 已星标保存':'☆ 星标保存';
  const note=saved?'已保存，可在「回看」中载入。':session.active?'对局暂存，结束后可星标保存。':
   recorded?'未保存。开新局或重启服务会丢弃这局。':'这局没有可保存的走法。';
  if(saveNote.textContent!==note)saveNote.textContent=note;
  pause.disabled=!ready||!session.active||!(paused||session.status==='running');
  pause.textContent=paused?'继续推理':'暂停推理';
  latest.disabled=!ready||cursor===null;
  seek.disabled=!ready||!recorded;
  previous.disabled=seek.disabled||shown===0;
  next.disabled=seek.disabled||shown>=recorded;
  review.disabled=reviewEnd===null&&(seek.disabled||reading||shown>=recorded);
  review.textContent=reviewEnd===null?'播放回看':'暂停回看';
  seek.max=String(recorded);seek.value=String(shown);
  seek.style.setProperty('--position',(recorded?100*shown/recorded:0)+'%');
  seek.setAttribute('aria-valuetext','第 '+n(shown)+' / '+n(recorded)+' 步');
  const mode=reading?'载入中':reviewEnd!==null?'正在回放':cursor!==null?'回看':session.active&&!paused?'实时':'最新';
  cursorText.textContent=mode+' · 第 '+n(shown)+' / '+n(recorded)+' 步';
 }
 async function seekFrame(turn){
  const session=snapshot?.session;
  if(!session||!Number.isInteger(turn)||turn<0||turn>length(session))return false;
  cursor=turn;error='';
  const ticket=++frameRequest;
  if(turn===length(session)){
   frame={id:session.id,turn,games:session.games};render();return true;
  }
  render();
  try{
   const response=await request('/play/frame?id='+encodeURIComponent(session.id)+'&turn='+turn);
   const nextFrame=await response.json();
   if(ticket!==frameRequest)return false;
   if(!response.ok)throw Error(text(nextFrame.error,160)||'无法读取这一步。');
   if(nextFrame?.id!==session.id||nextFrame.turn!==turn||!Array.isArray(nextFrame.games)||
       nextFrame.games.length!==slots.size||nextFrame.games.some(game=>!slots.has(game.model)))throw Error('回看记录格式错误。');
   frame=nextFrame;render();return true;
  }catch(err){
   if(ticket===frameRequest){cursor=frame?.turn??null;error=text(err.message,160)||'无法读取这一步。';render();}
   return false;
  }
 }
 function advanceReview(){
  if(reviewEnd===null)return;
  if(cursor>=reviewEnd){stopReview();render();return;}
  reviewTimer=setTimeout(async()=>{
   const pending=seekFrame(cursor+1), ticket=frameRequest;
   const moved=await pending;
   if(ticket!==frameRequest||reviewEnd===null)return;
   if(moved)advanceReview();
   else{stopReview();render();}
  },400);
 }
 function renderChoices(models,session){
  const signature=JSON.stringify(models);
  if(signature!==catalog){
   catalog=signature;
   choices.clear();
   for(const model of models){
    const checkbox=make('input');
    checkbox.type='checkbox';checkbox.value=model.id;
    const option=playModelOption(model,checkbox,make);
    checkbox.addEventListener('change',()=>{
     if(checkbox.checked&&chosen.size<maxModels)chosen.add(model.id);
     if(!checkbox.checked)chosen.delete(model.id);
     error='';render();
    });
    choices.set(model.id,{checkbox,option});
   }
   for(const id of chosen)if(!choices.has(id))chosen.delete(id);
   if(!initialized&&models.length){
    const activeModels=session?.games.map(game=>game.model)||[];
    for(const id of activeModels)if(choices.has(id))chosen.add(id);
    if(!chosen.size)chosen.add(choices.has('official')?'official':models[0].id);
    initialized=true;
   }
   showPlayModelOptions(options,choices,chosen,make);
  }
  const checked=session?.active?new Set(session.games.map(game=>game.model)):chosen;
  for(const [id,choice] of choices){
   choice.checkbox.checked=checked.has(id);
   choice.checkbox.disabled=!checked.has(id)&&checked.size>=maxModels;
  }
 }
 function render(){
  const models=snapshot?.models||[], session=snapshot?.session;
  renderChoices(models,session);
  const ready=connection==='ready';
  selection.disabled=!ready||busy||Boolean(session?.active)||!models.length;
  seed.disabled=selection.disabled;
  start.disabled=selection.disabled||!chosen.size;
  stop.disabled=!ready||busy||!session?.active||session.status==='stopping';
  const message=error||(busy?'正在提交操作。':messages[connection]||
   (!models.length?'没有可用的本机模型。':
    models.length>maxModels&&chosen.size===maxModels&&!session?.active?'每局最多可选 8 个模型。':''));
  if(feedback.textContent!==message)feedback.textContent=message;
  feedback.hidden=!message;
  renderSession(session);
 }
 function receive(next,reset=false){
  if(!Number.isSafeInteger(next?.revision)||!Array.isArray(next.models)||
      (next.session!==null&&!Array.isArray(next.session?.games)))throw Error('本机推理状态格式错误。');
  if(reset||!snapshot||next.revision>=snapshot.revision)snapshot=next;
  connection='ready';
  render();
 }
 function request(url,body){
  const settings={method:body?'POST':'GET',cache:'no-store',credentials:'same-origin'};
  if(body){
   settings.headers={'Content-Type':'application/json'};
   settings.body=JSON.stringify(body);
  }
  if(typeof AbortSignal!=='undefined'&&typeof AbortSignal.timeout==='function')settings.signal=AbortSignal.timeout(6000);
  return fetch(url,settings);
 }
 function schedule(ticket){
  if(visible&&ticket===generation)timer=setTimeout(()=>refresh(ticket),1000);
 }
 function connectEvents(ticket){
  if(typeof EventSource!=='function'||fallback){schedule(ticket);return}
  try{events=new EventSource('/play/events')}
  catch{fallback=true;schedule(ticket);return}
  const source=events;
  source.addEventListener('play',event=>{
   if(ticket!==generation||source!==events)return;
   try{receive(JSON.parse(event.data),connection!=='ready')}
   catch{connection='reconnecting';render()}
  });
  source.addEventListener('error',()=>{
   if(ticket!==generation||source!==events)return;
   connection='reconnecting';render();
   if(source.readyState===EventSource.CLOSED){
    source.close();events=null;fallback=true;schedule(ticket);
   }
  });
 }
 async function refresh(ticket){
  try{
   const response=await request('/play/state');
   if(ticket!==generation)return;
   if(response.status===403||response.status===404){
    snapshot=null;connection='disabled';render();return;
   }
   if(!response.ok)throw Error('state');
   const next=await response.json();
   if(ticket!==generation)return;
   receive(next,connection!=='ready');
   connectEvents(ticket);
  }catch{
   if(ticket!==generation)return;
   connection='reconnecting';render();schedule(ticket);
  }
 }
 async function command(url,payload,after){
  const ticket=generation;
  busy=true;error='';render();
  try{
   const response=await request(url,payload);
   const next=await response.json();
   if(ticket!==generation)return;
   if(!response.ok)throw Error(text(next.error,160)||'操作失败，请重试。');
   after?.();
   receive(next);
  }catch(err){if(ticket===generation)error=text(err.message,160)||'操作失败，请重试。'}
  finally{if(ticket===generation){busy=false;render()}}
 }
 form.addEventListener('submit',event=>{
  event.preventDefault();
  if(start.disabled)return;
  const payload={models:[...chosen]};
  if(seed.value!==''){
   const value=Number(seed.value);
   if(!Number.isInteger(value)||value<0||value>maxSeed){
    error='请输入 0 到 2147483647 之间的整数种子。';render();return;
   }
   payload.seed=value;
  }
  command('/play/start',payload);
 });
 stop.addEventListener('click',()=>{
  if(!stop.disabled)command('/play/stop',{id:snapshot.session.id});
 });
 pause.addEventListener('click',()=>{
  if(pause.disabled)return;
  const session=snapshot.session;
  if(session.status==='paused')command('/play/resume',{id:session.id},followLatest);
  else command('/play/pause',{id:session.id});
 });
 save.addEventListener('click',()=>{
  if(!save.disabled)command('/play/save',{id:snapshot.session.id});
 });
 latest.addEventListener('click',()=>{
  if(latest.disabled)return;
  followLatest();render();
 });
 function browse(turn){stopReview();seekFrame(turn);}
 seek.addEventListener('input',()=>{if(!seek.disabled)browse(Number(seek.value));});
 previous.addEventListener('click',()=>{if(!previous.disabled)browse(cursorAt(snapshot.session)-1);});
 next.addEventListener('click',()=>{if(!next.disabled)browse(cursorAt(snapshot.session)+1);});
 review.addEventListener('click',()=>{
  if(review.disabled)return;
  if(reviewEnd!==null){
   stopReview();frameRequest++;cursor=frame?.turn??null;
  }
  else{reviewEnd=length(snapshot.session);advanceReview();}
  render();
 });
 return {finish,show(next){
  if(next===visible)return;
  visible=next;generation++;busy=false;
  if(!visible){
   events?.close();events=null;clearTimeout(timer);stopReview();frameRequest++;
   cursor=frame?.turn??null;finish();return;
  }
  error='';fallback=false;
  connection=loopback?'connecting':'idle';
  render();
  if(loopback)refresh(generation);
 }};
}
