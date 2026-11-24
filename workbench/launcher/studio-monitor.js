/* Live monitor inside the training, replay and play workspace. */
function createStudioMonitor({root, rail, onOpenTraining, onNavigate, onTitleChange, drawBoard}) {
  const kinds = {all:'全部', training:'训练', evaluation:'测评', inference:'推理 / 采集', benchmark:'测速'};
  const statuses = {ready:'就绪', loading:'加载中', running:'进行中', move:'进行中', paused:'已暂停',
    completed:'已完成', complete:'已完成', terminal:'自然结束', cap:'达到步数上限', capped:'达到步数上限',
    stopped:'已停止', error:'失败', failed:'失败', waiting:'等待记录', stale:'更新延迟',
    unknown:'状态未确认', incomplete:'记录不完整', unavailable:'暂不可读'};
  const phases = {optimizer:'参数更新', lora_fit:'LoRA 参数更新', loading:'加载模型', loading_model:'加载模型',
    starting:'启动', data_collection:'采集盘面', teacher_labeling:'老师标注',
    train_cache:'缓存训练特征', dev_cache:'缓存开发特征', evaluate:'开发集评测',
    train_encoding:'编码训练盘面', dev_encoding:'编码验证盘面', encoding:'编码盘面',
    head_fit:'拟合决策头', fit_epoch:'拟合决策头', head_fit_epoch:'拟合决策头', train:'训练', training:'训练',
    verification:'核验', verify_data:'核验数据', inputs_verified:'已核验输入', trial_completed:'候选训练完成',
    development_evaluation:'开发集验证', checkpoint_save:'保存检查点', official_reload:'重载官方模型',
    initial_model_load:'首次加载模型', episode_completed:'完成一局', training_episode_completed:'完成一局采集',
    training_collection:'采集训练盘面', training_loading_native:'加载采集模型', training_collection_checkpoint:'保存采集进度',
    training_board_quota_check:'核对盘面配额', baseline_reused:'复用官方基线', gameplay:'整局测评', gameplay_start:'开始整局测评',
    inference_gate:'核对推理一致性', screening_completed:'初筛完成', budget_stop:'达到运行时限',
    native_inference:'本机推理', done:'完成', completed:'完成', error:'执行出错', prepare:'准备', wall:'总耗时'};
  const stageNames = {screened_out_30:'初筛完成 · 未达扩展门槛', full_100:'扩展评测完成'};
  const directions = {up:'↑ 上', down:'↓ 下', left:'← 左', right:'→ 右'};
  const active = c => ['running','loading','move'].includes(c.status);
  const finished = c => ['completed','complete','terminal','capped','cap'].includes(c.status);
  const finite = Number.isFinite;
  const numbers = new Intl.NumberFormat('zh-CN',{maximumFractionDigits:3});
  const measuredValue = new Intl.NumberFormat('zh-CN',{maximumSignificantDigits:6});
  const dates = new Intl.DateTimeFormat('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
  const stamp = c => c.updated_at || c.latest?.timestamp;
  const formatTime = s => { const ms=Date.parse(s); return finite(ms)?dates.format(ms):'时间未记录'; };
  const number = n => finite(n)?numbers.format(n):'—';
  const percent = n => (100*n).toFixed(2)+'%';
  const known = (map,key) => Object.hasOwn(map,key)?map[key]:null;
  const kindName = c => known(kinds,c.kind)||'其他';
  const statusName = c => known(statuses,c.status)||'状态未确认';
  const phaseName = value => known(phases,value)||(typeof value==='string'?value.replaceAll('_',' '):'');
  const phaseText = c => known(stageNames,c.latest?.stage)||phaseName(c.phase||c.latest?.phase);
  const events = c => Array.isArray(c.history)?c.history:[];
  const el = (tag,cls='',text='')=>{
    const node=document.createElement(tag);
    if(cls)node.className=cls;
    if(text!=='')node.textContent=text;
    return node;
  };
  const btn = (label,cls,onClick)=>{
    const node=el('button',cls,label);node.type='button';node.addEventListener('click',onClick);return node;
  };
  function badge(node, channel) {
    node.textContent=statusName(channel);
    node.dataset.state=active(channel)?'active':finished(channel)?'done':channel.status;
  }
  function validBoard(value) {
    const flat=Array.isArray(value)?value.flat():[];
    return flat.length===16 && flat.every(n=>Number.isSafeInteger(n)&&n>=0&&n<=1048576) ? flat:null;
  }
  function stepRecord(c) {
    if(c.kind!=='training')return null;
    const hasStep=e=>Number.isSafeInteger(e?.optimizer_step)&&Number.isSafeInteger(e.steps)&&e.steps>0;
    return hasStep(c.training_step)?c.training_step:events(c).findLast(hasStep)||null;
  }
  function resultProgress(c) {
    const latest=c.latest||{}, step=stepRecord(c);
    if(finite(c.collection_progress?.completed))return {value:c.collection_progress.completed,total:c.collection_progress.total,unit:'局',label:'已采集'};
    if(step)return {value:step.optimizer_step,total:step.steps,unit:'步',label:'参数更新'};
    if(c.kind==='evaluation' && finite(c.games??latest.games)){
      const gate=events(c).findLast(e=>e.phase==='screening_completed');
      const episode=events(c).findLast(e=>e.phase==='episode_completed'&&finite(e.total));
      const total=c.total_games??latest.total_games??latest.total??(latest.stage==='screened_out_30'?latest.games:
        gate?.promote_to_100===true?100:episode?.total??gate?.games);
      return {value:c.games??latest.games,total,unit:'局',label:'已完成'};
    }
    if(finite(latest.step))return {value:latest.step,total:latest.total,unit:c.kind==='training'&&finished(c)?'组':'',label:'进度'};
    if(finite(latest.completed_episodes))return {value:latest.completed_episodes,total:latest.total_episodes,unit:'局',label:'已采集'};
    if(finite(latest.turn))return {value:latest.turn,total:latest.cap,unit:'步',label:'已记录'};
    return null;
  }
  function progressValue(p) {
    return number(p.value)+(finite(p.total)&&p.total>0?' / '+number(p.total):'')+(p.unit?' '+p.unit:'');
  }
  function fillProgress(mount,p){
    mount.replaceChildren();
    if(!p)return;
    const label=p.label+' '+progressValue(p);
    mount.append(el('span','sm-progress-label',label));
    if(finite(p.total)&&p.total>0&&p.value>=0&&p.value<=p.total){
      const track=el('progress','sm-progress');track.max=p.total;track.value=p.value;
      track.setAttribute('aria-label',label);mount.append(track);
    }
  }
  function leadingBoard(c) {
    const episode=c.episodes?.find(e=>!e.stop)||c.episodes?.[0];
    return validBoard(c.training_batch?.samples?.[0]?.board||episode?.after||c.latest?.board);
  }
  function glimpse(c) {
    const box=el('span','sm-glimpse'), cells=leadingBoard(c);
    box.setAttribute('aria-hidden','true');
    if(cells){
      box.classList.add('sm-board-mini');
      for(const n of cells){const tile=el('i','tile');tile.dataset.value=n;box.append(tile);}
    }else{
      box.classList.add('sm-kind-symbol');box.dataset.kind=c.kind;
      box.textContent=({training:'训',evaluation:'测',inference:'推',benchmark:'速'})[c.kind]||'录';
    }
    return box;
  }
  function generalMetrics(c) {
    const metrics=new Map(), latest=c.latest||{}, history=events(c), step=stepRecord(c);
    const metric=(label,value)=>{if(value!==null&&value!==undefined&&value!==''&&!metrics.has(label))metrics.set(label,String(value));};
    const amount=(label,value,unit='')=>{if(finite(value))metric(label,number(value)+unit);};
    if(c.kind==='training'){
      amount('轮次',step?.epoch??latest.epoch);
      amount('本批盘面',step?.batch_size??c.training_batch?.batch_size,' 张');
      const loss=step?.mean_training_loss??latest.mean_training_loss;
      if(finite(loss))metric('平均训练损失',loss.toFixed(4));
      if(finite(c.training_batch?.boards_per_second))metric(c.training_batch.pace_source==='estimated'?'估算训练速度':'训练速度',number(c.training_batch.boards_per_second)+' 盘/秒');
      const saved=history.findLast(e=>e.phase==='trial_completed'&&finite(e.epoch));
      if(saved){amount('保留轮次',saved.epoch);if(finite(saved.loss))metric('验证损失',saved.loss.toFixed(4));}
      const fit=saved && history.findLast(e=>e.trial===saved.trial&&e.epoch===saved.epoch&&finite(e.dev_best_rate));
      if(finite(fit?.dev_best_rate))metric('老师走法命中率',percent(fit.dev_best_rate));
    }
    if(finite(latest.loss))metric('记录损失',latest.loss.toFixed(4));
    if(c.kind==='evaluation'){
      const gate=history.findLast(e=>e.phase==='screening_completed');
      if(gate && Array.isArray(gate.candidate_vector)&&Array.isArray(gate.threshold)){
        metric('候选达到 512',number(gate.candidate_vector[2])+' / '+number(gate.games));
        amount('候选得分中位数',gate.candidate_vector[3]);
        metric('对照达到 512',number(gate.threshold[2])+' / '+number(gate.games));
        amount('对照得分中位数',gate.threshold[3]);
      }
      amount('复用对局',c.reused_games??latest.reused_games??latest.reference_games_reused,' 局');
      amount('新跑对局',c.new_games,' 局');
      if(c.batch&&finite(c.batch.number)&&finite(c.batch.total))metric('评测批次',number(c.batch.number)+' / '+number(c.batch.total));
      if(c.activity){
        amount('进行中种子',c.activity.seeds);
        amount('进行中模型',c.activity.models);
        amount('并行对局',c.activity.games);
      }
      amount('落子速度',c.decisions_per_second??latest.decisions_per_second,' 步/秒');
    }
    if(c.kind==='benchmark'){
      const times=c.timings||{};
      amount('CPU 总耗时中位数',times.cpu_median_wall_seconds,' 秒');
      amount('CUDA 总耗时中位数',times.cuda_median_wall_seconds,' 秒');
      if(finite(times.cpu_median_wall_seconds)&&times.cuda_median_wall_seconds>0)metric('CUDA 加速比',number(times.cpu_median_wall_seconds/times.cuda_median_wall_seconds)+'×');
    }
    if(finite(latest.seed))metric('最近种子',String(latest.seed));
    for(const [key,label,unit] of [
      ['turns','最近一局步数',' 步'],['encoded_boards','已编码盘面',' 张'],['cache_hits','缓存命中',' 次'],
      ['score','得分',''],['tile','最大方块',''],['max_tile','最大方块',''],['decision_ms','决策耗时',' ms'],
      ['train','训练盘面',' 张'],['dev','验证盘面',' 张'],['elapsed_seconds','记录耗时',' 秒']
    ])amount(label,latest[key],unit);
    for(const e of [latest,...history.slice().reverse()]){
      if(!Array.isArray(e?.summary_metrics))continue;
      for(const item of e.summary_metrics){
        if(typeof item?.label!=='string'||!item.label.trim())continue;
        const value=item.value;
        if(!finite(value)&&(typeof value!=='string'||!value.trim()))continue;
        const unit=typeof item.unit==='string'?item.unit.trim():'';
        const reading=typeof value==='string'?value.trim():Number.isInteger(value)?number(value):measuredValue.format(value);
        metric(item.label.trim(),reading+(unit?(unit==='%'?'':' ')+unit:''));
      }
    }
    return [...metrics];
  }
  function timingLabel(key) {
    const name=key.replace(/_(seconds|ms)$/,'');
    const match=/^(cpu|cuda|mps)_median_(.+)$/.exec(name);
    return match?match[1].toUpperCase()+' · '+phaseName(match[2])+'中位数':phaseName(name);
  }
  function metricList(rows) {
    const list=el('dl','sm-metric-list');
    for(const [label,value] of rows){
      const item=el('div');item.append(el('dt','',label),el('dd','',value));list.append(item);
    }
    return list;
  }
  function historyItem(e) {
    const fields=[phaseName(e.phase)||known(statuses,e.status)||'更新'];
    if(e.trial)fields.push(e.trial);
    if(finite(e.epoch))fields.push('第 '+number(e.epoch)+' 轮');
    if(finite(e.optimizer_step))fields.push('更新 '+number(e.optimizer_step)+(finite(e.steps)?' / '+number(e.steps):'')+' 步');
    else if(finite(e.games))fields.push(number(e.games)+' 局');
    else if(finite(e.step))fields.push(number(e.step)+(finite(e.total)?' / '+number(e.total):''));
    if(finite(e.mean_training_loss??e.loss))fields.push('损失 '+(e.mean_training_loss??e.loss).toFixed(4));
    return fields.join(' · ');
  }

  root.classList.add('sm-monitor');
  rail.classList.add('sm-run-rail');
  const railLabel=el('p','rail-label','最近任务');
  const railNav=el('nav');railNav.setAttribute('aria-label','最近任务');rail.replaceChildren(railLabel,railNav);
  const header=el('header','sm-header');
  const headline=el('div'), back=btn('← 运行总览','sm-back',()=>navigate(null));
  const title=el('h1','','运行总览'), subtitle=el('p','sub');
  const connection=el('span','sm-connection','正在连接');connection.setAttribute('role','status');
  headline.append(back,title,subtitle);header.append(headline,connection);
  const overview=el('section'), detail=el('section','sm-detail');
  const totals=el('dl','sm-summary');
  const counters=['进行中','已完成','任务总数'].map(label=>{
    const cell=el('div'),count=el('dd','','—');cell.append(el('dt','',label),count);totals.append(cell);return count;
  });
  const filters=el('div','sm-filters');filters.setAttribute('role','group');filters.setAttribute('aria-label','任务类型');
  const filterNodes=new Map(Object.entries(kinds).map(([id,name])=>{
    const b=btn(name,'chip',()=>{filter=id;render();});
    const count=el('span','sm-filter-count');b.append(count);filters.append(b);return [id,{b,count}];
  }));
  const list=el('div','sm-run-list');list.setAttribute('aria-label','运行任务');
  overview.append(totals,filters,list);root.replaceChildren(header,overview,detail);
  const rowCache=new Map();
  let channels=new Map(), sorted=[], selected=null, filter='all', visible=false, generation=0;
  let received=false, connected=false, polling=false, stream=null, pending=null, timer=null, retryStreamAfter=0;
  let trainingView=null, gameView=null, detailNodes=null, mountedId=null, railKey='', crumb='';
  function statusText(){
    return connected?(polling?'自动刷新':'实时更新'):received?'连接中断 · 显示上次记录':'等待本机日志';
  }
  function card(c) {
    const p=resultProgress(c), step=stepRecord(c);
    const loss=step?.mean_training_loss??c.latest?.loss;
    const summary=[phaseText(c),finite(loss)?'损失 '+loss.toFixed(4):''].filter(Boolean).join(' · ');
    const key=JSON.stringify([c.kind,c.name,c.status,summary,stamp(c),p,leadingBoard(c)]);
    let entry=rowCache.get(c.id);
    if(!entry){entry={node:btn('','sm-run',()=>navigate(c.id))};rowCache.set(c.id,entry);}
    if(entry.key===key)return entry.node;
    entry.key=key;
    const node=entry.node, identity=el('span','sm-run-name');
    identity.append(el('strong','',c.name||c.id),el('span','sm-small',[kindName(c),summary].filter(Boolean).join(' · ')));
    const progress=el('span','sm-run-progress');fillProgress(progress,p);
    const state=el('span','sm-state');badge(state,c);
    const time=el('time','sm-time',formatTime(stamp(c)));if(stamp(c))time.dateTime=stamp(c);
    const meta=el('span','sm-run-meta');meta.append(state,time);
    node.replaceChildren(glimpse(c),identity,progress,meta,el('span','sm-arrow','›'));
    node.dataset.channel=c.id;node.setAttribute('aria-label','查看'+kindName(c)+'任务 '+(c.name||c.id));
    return node;
  }
  function syncList(container,nodes) {
    const keep=new Set(nodes);
    for(const child of [...container.children])if(!keep.has(child))child.remove();
    nodes.forEach((node,index)=>{if(container.children[index]!==node)container.insertBefore(node,container.children[index]||null);});
  }
  function renderRail() {
    const recent=sorted.slice(0,5);
    if(selected && channels.has(selected) && !recent.some(c=>c.id===selected))recent.push(channels.get(selected));
    const key=JSON.stringify([selected,recent.map(c=>[c.id,c.name,c.kind])]);
    if(key!==railKey){
      railKey=key;
      railNav.replaceChildren(...recent.map(c=>{
        const b=btn(c.name||c.id,'task',()=>navigate(c.id));b.title=kindName(c)+' · '+(c.name||c.id);
        if(c.id===selected)b.setAttribute('aria-current','page');return b;
      }));
    }
    rail.hidden=!visible||!recent.length;
  }
  function renderOverview() {
    const counts={all:sorted.length};
    for(const c of sorted)counts[c.kind]=(counts[c.kind]||0)+1;
    [sorted.filter(active).length,sorted.filter(finished).length,sorted.length].forEach((n,i)=>{
      counters[i].textContent=received?number(n):'—';
    });
    for(const [id,{b,count}] of filterNodes){
      b.setAttribute('aria-pressed',String(id===filter));count.textContent=received?number(counts[id]||0):'';
    }
    const entries=sorted.filter(c=>filter==='all'||c.kind===filter);
    if(entries.length)syncList(list,entries.map(card));
    else list.replaceChildren(el('p','sm-empty',!received?statusText():filter==='all'?'还没有可读取的运行记录。':'当前没有这类任务。'));
  }
  function clearDetail() {
    trainingView?.destroy?.();trainingView=null;
    gameView?.destroy();gameView=null;
    detailNodes=null;mountedId=null;detail.replaceChildren();
  }
  function mount(c) {
    clearDetail();
    mountedId=c.id;
    const summary=el('section','sm-detail-summary'), status=el('span','sm-state'), phase=el('span','sm-small');
    const top=el('div','sm-detail-top');top.append(status,phase);
    const progress=el('div','sm-detail-progress'), metrics=el('div','sm-detail-metrics');
    summary.append(top,progress,metrics);
    const main=el('div','sm-task-content'), timing=el('section','panel sm-facts'), log=el('section','panel sm-events');
    detail.replaceChildren(summary,main,timing,log);
    detailNodes={summary,status,phase,progress,metrics,main,timing,log};
    if(c.kind==='training' && onOpenTraining)trainingView=onOpenTraining(c,main);
    if(c.mode==='gameplay')gameView=gameplay(c.id,main);
  }
  function renderDetail(c) {
    if(mountedId!==c.id||(c.mode==='gameplay')!==Boolean(gameView))mount(c);
    const nodes=detailNodes;
    badge(nodes.status,c);nodes.phase.textContent=[kindName(c),phaseText(c)].filter(Boolean).join(' · ');
    const p=resultProgress(c);fillProgress(nodes.progress,p);nodes.progress.hidden=!p;
    const metrics=generalMetrics(c);
    nodes.metrics.replaceChildren(metricList(metrics));nodes.metrics.hidden=!metrics.length;
    const timings=Object.entries(c.timings||{}).filter(([key,v])=>finite(v)&&/_(seconds|ms)$/.test(key));
    nodes.timing.hidden=!timings.length;
    if(timings.length)nodes.timing.replaceChildren(el('h2','','计时明细'),metricList(timings.map(([key,v])=>[timingLabel(key),number(v)+(key.endsWith('_ms')?' ms':' 秒')])));
    const recent=events(c).filter(e=>e && typeof e==='object').slice(-12).reverse();
    const list=el('ol','sm-event-list');
    for(const e of recent){
      const line=el('li'),time=el('time','sm-small',formatTime(e.timestamp||e.updated_at));
      if(e.timestamp)time.dateTime=e.timestamp;
      line.append(time,el('span','',historyItem(e)));list.append(line);
    }
    nodes.log.replaceChildren(el('h2','','最近记录'),recent.length?list:el('p','sm-small','等待本任务写入事件。'));
    trainingView?.update?.(c);
    gameView?.update(c);
    if(!trainingView&&!gameView){
      const board=validBoard(c.latest?.board);
      if(board){
        const section=el('section','panel sm-latest-board'), grid=el('div','board');drawBoard(grid,board,'最新盘面');
        section.append(el('h2','','最新盘面'),grid);nodes.main.replaceChildren(section);
      }else nodes.main.replaceChildren();
    }
  }
  function render() {
    if(!visible)return;
    const c=selected && channels.get(selected);
    const heading=c?(c.name||c.id):selected?(received?'找不到这项任务':'正在读取任务'):'运行总览';
    title.textContent=heading;
    if(crumb!==heading){crumb=heading;onTitleChange?.(heading);}
    subtitle.textContent=c?'最后更新 '+formatTime(stamp(c)):selected?'':sorted.length?'按运行状态与最近更新时间排序。':'';
    subtitle.hidden=!subtitle.textContent;
    connection.textContent=statusText();connection.dataset.online=String(connected);
    back.hidden=!selected;overview.hidden=Boolean(selected);detail.hidden=!selected;
    renderRail();
    if(!selected){if(detailNodes)clearDetail();renderOverview();return;}
    if(c){renderDetail(c);return;}
    clearDetail();detail.append(el('p','sm-empty',received?'当前日志目录里没有这个任务，可以返回总览查看现有记录。':'正在读取本机运行目录…'));
  }
  function open(id) {
    const next=typeof id==='string'&&id?id:null;
    if(selected!==next){selected=next;clearDetail();}
    render();
  }
  function navigate(id){open(id);onNavigate?.(id);window.scrollTo({top:0,behavior:'instant'});}
  function receive(payload,isPolling) {
    if(!Array.isArray(payload?.channels))throw Error('Invalid monitor snapshot');
    const all=new Map();
    for(const c of payload.channels)if(c && typeof c.id==='string' && !all.has(c.id))all.set(c.id,c);
    channels=all;
    for(const id of rowCache.keys())if(!all.has(id))rowCache.delete(id);
    const rank=c=>active(c)?2:c.status==='paused'?1:0;
    sorted=[...all.values()].sort((a,b)=>rank(b)-rank(a)||((Date.parse(stamp(b))||0)-(Date.parse(stamp(a))||0))||a.id.localeCompare(b.id));
    received=true;connected=true;polling=isPolling;render();
  }
  function disconnect(){
    clearTimeout(timer);timer=null;
    pending?.abort();pending=null;
    stream?.close();stream=null;
  }
  async function load(ticket) {
    if(!visible||ticket!==generation)return;
    const controller=new AbortController();pending=controller;
    const deadline=setTimeout(()=>controller.abort(),6000);
    let success=false;
    try{
      const res=await fetch('/channels',{signal:controller.signal,credentials:'same-origin',cache:'no-store'});
      if(!res.ok)throw Error(String(res.status));
      const payload=await res.json();
      if(ticket!==generation)return;
      receive(payload,true);success=true;
    }catch{if(ticket===generation){connected=false;render();}}
    finally{clearTimeout(deadline);if(pending===controller)pending=null;}
    if(ticket!==generation||!visible)return;
    const retry=()=>{timer=setTimeout(()=>load(ticket),2400);};
    if(!success || typeof EventSource!=='function' || Date.now()<retryStreamAfter){retry();return;}
    let source;
    try{source=new EventSource('/channels/events');}catch{retry();return;}
    stream=source;
    function fallback(){
      if(stream!==source)return;
      source.close();stream=null;polling=true;
      // A busy stream endpoint can still serve snapshots. Poll while its listener slot clears.
      retryStreamAfter=Date.now()+10000;
      render();load(ticket);
    }
    source.addEventListener('channels',event=>{
      if(stream!==source||ticket!==generation)return;
      try{receive(JSON.parse(event.data),false);}catch{fallback();}
    });
    source.addEventListener('error',fallback);
  }
  function show(next) {
    next=Boolean(next);if(visible===next)return;
    visible=next;generation++;disconnect();connected=false;
    root.hidden=!visible;rail.hidden=!visible;
    if(visible){render();load(generation);}
    else clearDetail();
  }

  function gameplay(id,mount) {
    const panel=el('section','panel sm-match');
    const title=el('div','sm-paired'), library=el('a','sm-back','回看目录 →');
    library.href='#review';title.append(el('h2','','同种子模型对照'),library);
    const seedHead=el('div','sm-seed-head'), seedTitle=el('h3'), seedValue=el('strong'), count=el('span','sm-small');
    seedTitle.append(el('span','','种子'),seedValue);
    const seedNav=el('div','sm-seed-nav');
    const previousSeed=btn('‹','sm-seed-arrow',()=>moveSeed(-1)), nextSeed=btn('›','sm-seed-arrow',()=>moveSeed(1));
    previousSeed.setAttribute('aria-label','上一个种子');nextSeed.setAttribute('aria-label','下一个种子');
    seedNav.append(count,previousSeed,nextSeed);seedHead.append(seedTitle,seedNav);
    const strip=el('div','sm-seed-strip'), ruler=el('div','sm-seed-ruler');
    strip.setAttribute('role','group');strip.setAttribute('aria-label','全部测评种子');strip.append(ruler);
    const transport=el('div','sm-transport'), actions=el('div','sm-transport-actions');
    const play=btn('▶ 播放','btn solid sm-play',togglePlayback);
    const prev=btn('‹','btn sm-step-btn',()=>jump((following?total:cursor)-1));
    const next=btn('›','btn sm-step-btn',()=>jump((following?total:cursor)+1));
    for(const [b,label] of [[prev,'上一步'],[next,'下一步']]){b.setAttribute('aria-label',label);b.title=label;}
    const speed=el('select','sm-speed');speed.setAttribute('aria-label','播放速度');
    for(const n of [.5,1,2,4,8]){
      const o=el('option','',n+'×');o.value=String(n);speed.append(o);
    }
    speed.value='1';speed.onchange=()=>{if(playing)schedule();};
    const follow=btn('返回实况','chip',()=>choose(liveSeed(),true));
    const before=btn('落子前盘面','chip',()=>{showBefore=!showBefore;before.setAttribute('aria-pressed',String(showBefore));drawModels();});
    before.setAttribute('aria-pressed','false');
    const turnText=el('output','sm-turn-readout'), timeline=el('input','sm-timeline');
    timeline.type='range';timeline.min='0';timeline.max='0';timeline.value='0';timeline.step='1';
    timeline.setAttribute('aria-label','回放步数');timeline.oninput=()=>jump(Number(timeline.value));
    const tools=el('div','sm-view-tools');tools.append(before,follow);
    actions.append(play,prev,next,speed,turnText,tools);transport.append(actions,timeline);
    const caption=el('p','sm-small'), retry=btn('重新读取轨迹','sm-back',()=>loadPage(cursor,true));
    const note=el('div','sm-record-note'), models=el('div','sm-models');note.append(caption,retry);
    panel.append(title,seedHead,strip,transport,note,models);mount.append(panel);

    let current,selected=null,catalog=[],group=[],following=false,showBefore=false;
    let cursor=0,total=0,playing=false,timer=null,seekVersion=0,request=null,error='',dead=false;
    const slots=new Map(),ticks=new Map(),pages=new Map();
    function rebuildRuler(){
      ticks.clear();
      const marks=document.createDocumentFragment();
      catalog.forEach((seed,i)=>{
        const b=el('button','sm-seed-tick');b.type='button';b.value=String(seed);b.tabIndex=-1;
        b.setAttribute('aria-label','种子 '+seed);b.setAttribute('aria-pressed','false');
        b.title='种子 '+seed+' · 第 '+(i+1)+' 个';
        b.append(el('i','sm-seed-mark'));
        if(i===0||(i+1)%10===0){
          const tickLabel=el('span','sm-seed-number',String(i+1));tickLabel.setAttribute('aria-hidden','true');b.append(tickLabel);
        }
        ticks.set(seed,b);marks.append(b);
      });
      ruler.replaceChildren(marks);
    }
    function markSelected(previous){
      const old=ticks.get(previous),tick=ticks.get(selected);
      if(old){old.tabIndex=-1;old.setAttribute('aria-pressed','false');}
      if(!tick)return;
      tick.tabIndex=0;tick.setAttribute('aria-pressed','true');
      const index=catalog.indexOf(selected);
      seedValue.textContent=String(selected);count.textContent='第 '+(index+1)+' / '+catalog.length+' 个';
      previousSeed.disabled=index===0;nextSeed.disabled=index===catalog.length-1;
      const middle=tick.offsetLeft+tick.offsetWidth/2;
      if(middle<strip.scrollLeft+8||middle>strip.scrollLeft+strip.clientWidth-8){
        strip.scrollLeft=middle-strip.clientWidth/2;
      }
    }
    function moveSeed(offset){
      const i=Math.max(0,Math.min(catalog.length-1,catalog.indexOf(selected)+offset));
      if(catalog[i]!==undefined)choose(catalog[i],false);
    }
    strip.onclick=e=>{const b=e.target.closest('.sm-seed-tick');if(b)choose(Number(b.value),false);};
    strip.onkeydown=e=>{
      const offset={ArrowLeft:-1,ArrowRight:1,Home:-catalog.length,End:catalog.length}[e.key];
      if(offset===undefined)return;
      e.preventDefault();moveSeed(offset);ticks.get(selected)?.focus({preventScroll:true});
    };
    const sequence=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
    const ongoing=()=>active(current)||['paused','stale','waiting','ready'].includes(current?.status);
    const liveFrames=()=>(current?.episodes||[]).filter(e=>e.seed===selected);
    const liveSeed=()=>current.episodes?.find(e=>!e.stop&&group.includes(e.seed))?.seed??group[0]??catalog[0];
    const pageSize=()=>Math.min(128,Math.max(8,Math.floor(512/(current.arms?.length||1))));
    const pageStart=turn=>1+Math.floor((Math.max(1,turn)-1)/pageSize())*pageSize();
    function cancelRequest(){
      if(request){clearTimeout(request.timeout);request.controller.abort();request=null;}
    }
    function pageReady(turn){
      const page=pages.get(pageStart(turn));
      if(!page)return false;
      return !liveFrames().some(e=>{
        const recorded=page.tracks.find(t=>t.arm===e.arm)?.total_turns||0;
        return recorded<Math.min(e.turn,Math.max(1,turn));
      });
    }
    function loadPage(turn,force=false){
      if(dead||selected===null)return Promise.resolve(false);
      const start=pageStart(turn);
      if(request?.start===start)return request.promise;
      if(!force&&pageReady(turn))return Promise.resolve(true);
      cancelRequest();error='';
      const job={start,controller:new AbortController()};request=job;
      job.timeout=setTimeout(()=>job.controller.abort(),15000);
      job.promise=(async()=>{
        try{
          const query=new URLSearchParams({channel:id,seed:String(selected),start:String(start),limit:String(pageSize())});
          const response=await fetch('/gameplay/replay?'+query,{cache:'no-store',credentials:'same-origin',signal:job.controller.signal});
          if(!response.ok)throw Error(String(response.status));
          const page=await response.json();
          if(dead||request!==job)return false;
          if(page.channel!==id||page.seed!==selected||page.start!==start||!Array.isArray(page.tracks)||
             !Number.isSafeInteger(page.total_turns)||page.total_turns<0)throw Error('Invalid replay');
          pages.delete(start);pages.set(start,page);
          if(pages.size>8)pages.delete(pages.keys().next().value);
          total=Math.max(page.total_turns,...liveFrames().map(e=>e.turn));
          cursor=Math.min(cursor,total);
          return true;
        }catch{
          if(request===job&&!dead){error='轨迹暂时无法读取。';pause();}
          return false;
        }finally{
          clearTimeout(job.timeout);
          if(request===job){request=null;renderMatch();}
        }
      })();
      renderMatch();return job.promise;
    }
    function pause(){clearTimeout(timer);timer=null;playing=false;}
    function schedule(){
      clearTimeout(timer);timer=null;
      if(!playing||dead)return;
      if(cursor>=total){pause();renderMatch();return;}
      timer=setTimeout(()=>seek(cursor+1),560/Number(speed.value));
    }
    async function seek(turn){
      const version=++seekVersion;
      cursor=Math.max(0,Math.min(total,turn));renderMatch();
      if(await loadPage(cursor) && version===seekVersion && !dead && playing)schedule();
    }
    function jump(turn){pause();following=false;seek(turn);}
    function togglePlayback(){
      if(playing){pause();renderMatch();return;}
      if(!total||dead)return;
      const nextTurn=following||cursor>=total?0:cursor;
      following=false;playing=true;seek(nextTurn);
    }
    function choose(seed,live){
      if(!ticks.has(seed))return;
      if(selected===seed && following===live)return;
      const previous=selected;
      pause();seekVersion++;cancelRequest();pages.clear();error='';
      selected=seed;following=live;cursor=0;total=Math.max(0,...liveFrames().map(e=>e.turn));
      markSelected(previous);renderMatch();
      if(!following)loadPage(0);
    }
    function probabilityList(episode) {
      const probs=episode.probabilities;
      if(!probs || typeof probs!=='object')return null;
      const section=el('div','sm-probs');
      const head=el('div','sm-paired sm-small');head.append(el('span','','方向概率'),el('span','','0–100%'));section.append(head);
      for(const [action,direction] of Object.entries(directions)){
        const line=el('div','sm-prob'),value=probs[action],valid=finite(value)&&value>=0&&value<=1;
        line.dataset.chosen=String(episode.action===action);
        const track=el('span','sm-prob-track'),fill=el('i');
        fill.style.width=100*(valid?value:0)+'%';track.append(fill);track.setAttribute('aria-hidden','true');
        line.append(el('span','',direction),track,el('span','sm-prob-value',valid?percent(value):'不可走'));section.append(line);
      }
      return section;
    }
    function fillSlot(slot,arm,episode,label,empty) {
      const key=JSON.stringify([selected,showBefore,label,empty,episode]);
      if(key===slot.key)return;
      slot.key=key;
      const head=el('div','sm-paired'),recordedStop=episode?.stop;
      const ending=known({terminal:'自然结束',target:'达到 2048',capped:'步数上限',cap:'步数上限',stopped:'已停止'},recordedStop);
      head.append(el('h3','',arm==='official'?'官方基线':arm));
      if(episode)head.append(el('span','sm-small',ending||label));
      slot.node.replaceChildren(head);
      if(!episode){slot.node.append(el('p','sm-small sm-board-empty',empty));return;}
      const board=el('div','board'), cells=validBoard(showBefore?episode.before:episode.after);
      const score=el('div','sm-score');
      const beforeScore=showBefore&&episode.turn>0;
      score.append(el('span','sm-small',beforeScore?'落子后得分':'得分'),el('strong','',number(episode.score)));
      slot.node.append(score,el('div','sm-small','第 '+number(episode.turn)+' 步 · 最大 '+number(cells?Math.max(...cells):episode.max_tile)));
      if(cells){drawBoard(board,cells,showBefore?'落子前盘面':'当前盘面');slot.node.append(board);}
      const choice=known(directions,episode.action), conf=episode.probabilities?.[episode.action];
      if(choice)slot.node.append(el('p','sm-decision','选择 '+choice+(finite(conf)?' · '+percent(conf):'')));
      const probabilities=probabilityList(episode);if(probabilities)slot.node.append(probabilities);
    }
    function replayFrame(track,start){
      if(!track?.total_turns||track.status==='invalid')return null;
      if(cursor===0){
        const cells=track.frames[0]?.before;
        if(!cells)return null;
        return {seed:selected,turn:0,before:cells,after:cells,score:0,max_tile:Math.max(...cells.flat())};
      }
      const turn=Math.min(cursor,track.total_turns);
      return turn<start?track.last_frame:track.frames[turn-start];
    }
    function drawModels() {
      const arms=current.arms||[],frames=liveFrames(),start=pageStart(cursor),page=pages.get(start);
      for(const arm of slots.keys())if(!arms.includes(arm))slots.delete(arm);
      const nodes=arms.map(arm=>{
        if(!slots.has(arm))slots.set(arm,{node:el('article','sm-model')});
        return slots.get(arm).node;
      });
      syncList(models,nodes);
      for(const arm of arms){
        const track=page?.tracks.find(t=>t.arm===arm);
        const frame=following?frames.find(e=>e.arm===arm):replayFrame(track,start);
        const label=following&&ongoing()?'实况':cursor===0?'开局':'回放';
        const empty=error||(!following&&request&&!track?'正在读取轨迹…':track?.status==='invalid'?'轨迹校验未通过':'等待落子记录');
        fillSlot(slots.get(arm),arm,frame,label,empty);
      }
    }
    function renderMatch(){
      if(dead||!current)return;
      const position=following?total:cursor,mode=following?(ongoing()?'实况':'末步'):'回放';
      seedHead.hidden=strip.hidden=transport.hidden=!catalog.length;
      play.textContent=playing?'Ⅱ 暂停':'▶ 播放';play.setAttribute('aria-label',playing?'暂停':'播放');
      play.title=following||cursor>=total?'从开局播放':'从当前步播放';play.disabled=!total;
      prev.disabled=!total||position===0;next.disabled=!total||position>=total;
      timeline.max=String(total);timeline.value=String(position);timeline.disabled=!total;
      timeline.setAttribute('aria-valuetext','第 '+position+' 步，共 '+total+' 步');
      turnText.textContent=mode+' · 第 '+number(position)+' / '+number(total)+' 步';
      follow.hidden=following||!ongoing()||!group.length;
      retry.hidden=!error;
      const batch=current.batch;
      const batchNote=following&&batch?'第 '+number(batch.number)+' / '+number(batch.total)+' 批 · 本批 '+group.length+' 个种子':'';
      const reused=pages.get(pageStart(cursor))?.tracks.some(t=>t.reused);
      caption.textContent=error||(!catalog.length?'等待测评写入种子与模型。':!total&&request?'正在读取轨迹…':
        !total?'这个种子尚无已记录的落子。':batchNote||(reused?'包含已核验的复用轨迹':''));
      note.hidden=!caption.textContent&&!error;
      drawModels();
    }
    function update(c) {
      const previousGroup=group,wasOngoing=current&&ongoing();current=c;
      const nextGroup=Array.isArray(c.batch?.seeds)?c.batch.seeds:[];
      const nextCatalog=[...new Set([...(c.seed_catalog||[]),...nextGroup])].filter(Number.isSafeInteger);
      const newCatalog=!sequence(nextCatalog,catalog);
      if(newCatalog){catalog=nextCatalog;rebuildRuler();}
      group=nextGroup;
      if(newCatalog||!sequence(previousGroup,group)||wasOngoing!==ongoing()){
        for(const seed of previousGroup)ticks.get(seed)?.classList.remove('sm-in-batch');
        if(ongoing())for(const seed of group)ticks.get(seed)?.classList.add('sm-in-batch');
        if(newCatalog)markSelected(selected);
      }
      if(!catalog.length){
        pause();seekVersion++;cancelRequest();pages.clear();selected=null;total=cursor=0;renderMatch();return;
      }
      if(!ticks.has(selected)||(following&&!group.includes(selected))){
        choose(ongoing()?liveSeed():catalog[0],ongoing());return;
      }
      const recorded=liveFrames();
      total=Math.max(total,...recorded.map(e=>e.turn));
      if(following&&!ongoing()){
        following=false;cursor=total;loadPage(cursor);return;
      }
      const leftBatch=previousGroup.includes(selected)&&!group.includes(selected);
      if(!following&&!request&&!error&&(leftBatch||wasOngoing&&!ongoing()||!pageReady(cursor))){
        loadPage(cursor,true);return;
      }
      renderMatch();
    }
    function onVisibility(){if(document.hidden&&playing){pause();renderMatch();}}
    document.addEventListener('visibilitychange',onVisibility);
    return {update,destroy(){
      dead=true;pause();seekVersion++;cancelRequest();pages.clear();
      document.removeEventListener('visibilitychange',onVisibility);
      slots.clear();
    }};
  }
  return {show,open,finish:()=>show(false)};
}
