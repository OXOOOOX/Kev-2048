function createStudioTraining(channel, mount) {
  const doc=mount.ownerDocument;
  const make=(tag,cls,text)=>{
    const node=doc.createElement(tag);
    if(cls)node.className=cls;
    if(text!=null)node.textContent=text;
    return node;
  };
  const button=(label,cb,cls='btn')=>{
    const node=make('button',cls,label);node.type='button';node.onclick=cb;return node;
  };
  const select=label=>{
    const wrap=make('label','',label), node=make('select');
    node.setAttribute('aria-label',label);wrap.append(node);controls.append(wrap);return node;
  };
  const options=(node,items,value)=>{
    node.replaceChildren(...items.map(([key,label])=>{const opt=make('option','',label);opt.value=key;return opt;}));
    node.value=value;
  };
  const directions={up:'↑ 上',down:'↓ 下',left:'← 左',right:'→ 右'};
  const initial=new URLSearchParams(location.search);
  const panel=make('section','training-records panel');
  const head=make('div','tr-head'), title=make('h2','','训练盘面'), count=make('span','tr-note');
  head.append(title,count);
  const controls=make('div','tr-controls');
  const trialSelect=select('候选配置'), epochSelect=select('训练轮次');
  const play=button('逐盘播放',toggle,'btn solid');controls.append(play);
  const speed=select('倍速');
  options(speed,[[.5,'0.5×'],[1,'1×'],[2,'2×'],[4,'4×'],['logged','按训练速度']],'1');
  speed.title='1× 时每张盘面停留 1.5 秒';
  const latest=button('跟到最新批次',()=>{follow=true;showNewest();});controls.append(latest);
  const note=make('p','tr-note','读取训练日志…');note.setAttribute('role','status');
  const chartHead=make('div','tr-axes'),coverage=make('span','tr-note');chartHead.append(make('h3','','训练步'),coverage,make('span','tr-note','每步用一批盘面更新一次参数，点柱查看'));
  const scroller=make('div','tr-track-scroll'),track=make('div','tr-track');
  track.setAttribute('role','group');track.setAttribute('aria-label','训练步');scroller.append(track);
  const layout=make('div','tr-layout'),wallPart=make('div'),wallTitle=make('h3'),wall=make('div','tr-grid');
  wall.setAttribute('role','group');wall.setAttribute('aria-label','选择盘面');wallPart.append(wallTitle,wall);
  const details=make('div'),boardHead=make('div','tr-board-head'),indexText=make('h3'),rowText=make('span','tr-note');
  boardHead.append(indexText,rowText);
  const decision=make('div','tr-decision'),board=make('div','tr-board'),answers=make('div');
  board.setAttribute('role','img');
  const choice=label=>{const wrap=make('div','tr-choice'),value=make('strong');wrap.append(make('span','',label),value);answers.append(wrap);return value;};
  const teacher=choice('老师走法'),candidate=choice('更新前候选首选');
  decision.append(board,answers);
  const probTitle=make('div','tr-head');probTitle.append(make('h3','','候选四向概率'),make('span','tr-note','0–100%'));
  const probs=make('div'),source=make('p','tr-note');
  const probRows=Object.entries(directions).map(([key,label])=>{
    const line=make('div','tr-prob'),meter=make('meter'),out=make('output');
    meter.min=0;meter.max=1;meter.setAttribute('aria-label',label);
    line.append(make('span','',label),meter,out);probs.append(line);return {key,line,meter,out};
  });
  const metrics=make('p','tr-metric');
  details.append(boardHead,decision,probTitle,probs,source);layout.append(wallPart,details);
  panel.append(head,controls,note,chartHead,scroller,layout,metrics);mount.replaceChildren(panel);
  const controller=new AbortController(), cache=new Map();
  let batches=[],epochs=[],group=[],meta=null,detail=null,idx=0,trial='',epoch=null;
  let timer=null,playing=false,ended=false,dead=false,sampleTicket=0,indexTicket=0,signal=null,retry=null;
  const active=()=>['running','loading','waiting','paused'].includes(channel.status);
  let first=true,follow=!initial.has('step')&&!initial.has('epoch')&&active(), axisSignature='',lastSaved=0,rate=null;
  const showNote=(message,error=false)=>{note.textContent=message;note.hidden=!message;note.classList.toggle('tr-error',error);};
  function storePosition(force=true){
    if(!force&&performance.now()-lastSaved<350)return;
    const url=new URL(location.href);
    if(url.searchParams.get('run')!==channel.id)return;
    lastSaved=performance.now();
    for(const [key,value] of Object.entries({trial,epoch,step:meta?.step,row:detail?.samples[idx]?.row})){
      if(value==null||value==='')url.searchParams.delete(key);else url.searchParams.set(key,String(value));
    }
    history.replaceState(history.state,'',url);
  }
  async function request(path,key){
    const query=new URLSearchParams({channel:channel.id});if(key)query.set('key',key);
    const response=await fetch(path+'?'+query,{cache:'no-store',signal:controller.signal});
    if(!response.ok)throw Error('训练日志读取失败（'+response.status+'）');
    return response.json();
  }
  const cell=(n,large)=>{
    const tile=make(large?'span':'i','tr-cell',large&&n?n:'');tile.dataset.v=n;tile.setAttribute('aria-hidden','true');return tile;
  };
  function playLabel(){play.textContent=playing?'暂停回放':ended?'重播本轮':'逐盘播放';}
  function pause(){playing=false;clearTimeout(timer);timer=null;playLabel();}
  function journalPace(){
    if(!meta)return null;
    const timed=group.filter(b=>b.size>0&&b.timestamp&&Number.isFinite(Date.parse(b.timestamp)));
    if(timed.length<2)return null;
    let end=timed.findIndex(b=>b.step>=meta.step);
    if(end<0)end=timed.length-1;
    if(end===0)end=1;
    const from=timed[end-1],to=timed[end],steps=to.step-from.step;
    const seconds=(Date.parse(to.timestamp)-Date.parse(from.timestamp))/1000;
    if(seconds<=0||steps<=0)return null;
    const span=group.filter(b=>b.step>from.step&&b.step<=to.step);
    let boards;
    if(span.length===steps&&span.every(b=>b.size>0))boards=span.reduce((n,b)=>n+b.size,0);
    else if(from.size===to.size||to.step===to.steps)boards=(steps-1)*from.size+to.size;
    return boards?{rate:boards/seconds,from:from.step,to:to.step}:null;
  }
  function drawMetrics(){
    const report=epochs.find(r=>r.trial===trial&&r.epoch===epoch);
    const fields=[];
    if(Number.isFinite(meta?.loss))fields.push('本轮截至第 '+meta.step+' 步的平均训练损失 '+meta.loss.toFixed(4)+(meta.loss_source==='replayed'?'（历史复算）':''));
    for(const [key,label] of [['train_loss','轮末训练损失'],['dev_loss','轮末验证损失'],['dev_objective','轮末验证目标'],['dev_best_rate','验证集命中率']]){
      if(Number.isFinite(report?.[key]))fields.push(label+' '+(key==='dev_best_rate'?(report[key]*100).toFixed(2)+'%':report[key].toFixed(4)));
    }
    if(!fields.length&&Number.isFinite(report?.loss))fields.push('日志记录损失 '+report.loss.toFixed(4));
    if(Number.isFinite(report?.train_hits)&&report.train_count>0)fields.push('训练集命中 '+report.train_hits+' / '+report.train_count);
    const estimate=meta?.boards_per_second?null:journalPace();
    rate=meta?.boards_per_second||estimate?.rate||null;
    const paceOption=speed.querySelector('option[value="logged"]');paceOption.disabled=!rate;
    if(!rate&&speed.value==='logged')speed.value='1';
    if(rate)fields.push((estimate?'按第 '+estimate.from+'–'+estimate.to+' 步日志估计 ':'日志吞吐 ')+rate.toLocaleString('zh-CN',{maximumFractionDigits:1})+' 盘/秒');
    if(speed.value==='logged'&&rate>12)fields.push('逐盘显示上限 12 盘/秒');
    metrics.textContent=fields.join(' · ');metrics.hidden=!fields.length;
  }
  function present(){
    const s=detail?.samples[idx];layout.hidden=!s;
    if(!s)return;
    const length=detail.samples.length;
    indexText.textContent='盘面 '+(idx+1)+' / '+length;rowText.textContent='数据行 '+s.row;
    board.replaceChildren(...s.board.flat().map(n=>cell(n,true)));board.setAttribute('aria-label',s.board.map(r=>r.join('、')).join('。'));
    teacher.textContent=s.best_actions?.map(a=>directions[a]).join('、')||'未记录';
    candidate.textContent=directions[s.candidate?.choice]||'未记录';
    probs.hidden=probTitle.hidden=!s.candidate;
    for(const {key,line,meter,out} of probRows){
      const p=s.candidate?.probabilities[key];meter.value=p??0;out.textContent=p==null?'不可走':(p*100).toFixed(2)+'%';
      line.dataset.chosen=String(s.candidate?.choice===key);
    }
    const captions=[];
    if(detail.board_source==='reconstructed')captions.push('盘面由已核对的采样顺序还原');
    if(s.candidate_source==='replayed')captions.push('候选概率由历史权重复算');
    else if(s.candidate)captions.push('候选概率来自本步更新前的训练日志');
    source.textContent=captions.join(' · ');source.hidden=!source.textContent;
    const prev=wall.querySelector('[aria-pressed=true]');if(prev)prev.setAttribute('aria-pressed','false');
    wall.children[idx].setAttribute('aria-pressed','true');
  }
  function drawWall(){
    const samples=detail?.samples||[];
    wallTitle.textContent=meta?'第 '+meta.step+' 步盘面 · '+samples.length+' 张':'';
    if(meta?.size>samples.length)wallTitle.textContent+=' / 整批 '+meta.size+' 张';
    wall.replaceChildren(...samples.map((s,i)=>{
      const b=button('',()=>{follow=false;ended=false;pause();idx=i;present();storePosition();},'tr-mini');
      b.setAttribute('aria-label','盘面 '+(i+1)+'，数据行 '+s.row);b.setAttribute('aria-pressed','false');
      b.append(...s.board.flat().map(n=>cell(n,false)));return b;
    }));present();
  }
  function drawAxis(){
    chartHead.hidden=scroller.hidden=!group.length;
    const last=group.reduce((n,b)=>Math.max(n,b.steps||b.step),1),signature=group.map(x=>x.key).join(',')+'|'+last;
    coverage.textContent='本轮可看 '+group.filter(b=>b.sample_count>0).length+' / '+last+' 步';
    if(signature!==axisSignature){
      axisSignature=signature;
      const positions=group.map(x=>(x.step-1)/Math.max(1,last-1));
      const minGap=positions.slice(1).reduce((v,p,i)=>Math.min(v,p-positions[i]),1);
      const gap=last<100?28:34;
      track.style.minWidth=positions.length>1?Math.min(40000,Math.ceil(gap/Math.max(.0001,minGap)))+'px':'0';
      const marks=[];
      for(let i=0,n=Math.min(last,64);i<n;i++){
        const mark=make('i','tr-tick');mark.style.left=100*i/Math.max(1,n-1)+'%';mark.setAttribute('aria-hidden','true');marks.push(mark);
      }
      group.forEach((batch,i)=>{
        const b=button('',()=>{follow=false;pause();selectBatch(batch);},'tr-step');b.dataset.key=batch.key;
        b.setAttribute('aria-label','第 '+batch.step+' 步');b.style.left=100*positions[i]+'%';
        b.append(make('i'),make('span','',batch.step));marks.push(b);
      });
      track.replaceChildren(...marks);
    }
    for(const b of track.querySelectorAll('button')){
      const selected=b.dataset.key===meta?.key;b.setAttribute('aria-pressed',String(selected));
      if(selected&&scroller.scrollWidth>scroller.clientWidth){
        const bounds=scroller.getBoundingClientRect(),btn=b.getBoundingClientRect();
        if(btn.left<bounds.left)scroller.scrollLeft+=btn.left-bounds.left;
        else if(btn.right>bounds.right)scroller.scrollLeft+=btn.right-bounds.right;
      }
    }
  }
  function selectors(){
    const trials=[...new Set([...batches,...epochs].map(r=>r.trial))];
    options(trialSelect,trials.map(t=>[t,t||'默认配置']),trial);trialSelect.parentElement.hidden=trials.length<2;
    const allEpochs=[...new Set([...batches,...epochs].filter(r=>r.trial===trial).map(r=>r.epoch))].sort((a,b)=>a-b);
    if(!allEpochs.includes(epoch))epoch=allEpochs[0]??null;
    options(epochSelect,allEpochs.map(e=>[e,e===0?'训练前（基线）':'第 '+e+' 轮']),String(epoch));epochSelect.parentElement.hidden=!allEpochs.length;
    group=batches.filter(b=>b.trial===trial&&b.epoch===epoch).sort((a,b)=>a.step-b.step);
  }
  function schedule(){
    clearTimeout(timer);if(playing)timer=setTimeout(nextBoard,speed.value==='logged'?1000/Math.min(rate,12):1500/Number(speed.value));
  }
  function nextBoard(){
    if(!playing||!detail)return;
    if(idx+1<detail.samples.length){idx++;present();storePosition(false);schedule();return;}
    const next=group.slice(group.findIndex(b=>b.key===meta.key)+1).find(b=>b.sample_count>0);
    if(next){selectBatch(next);return;}
    ended=true;pause();showNote('本轮已播完');storePosition();
  }
  function toggle(){
    follow=false;
    if(playing){pause();storePosition();return;}
    if(!detail?.samples.length)return;
    playing=true;playLabel();
    if(ended){ended=false;selectBatch(group.find(b=>b.sample_count>0));}
    else schedule();
  }
  async function selectBatch(next,row=null){
    clearTimeout(timer);timer=null;const ticket=++sampleTicket;
    meta=next||null;detail=null;idx=0;ended=false;layout.hidden=true;play.disabled=true;
    drawAxis();drawMetrics();playLabel();
    if(!meta){pause();showNote(epochs.length?'这份日志记录了轮次指标，本轮没有逐盘样本。':'尚未记录训练批次或轮次指标。');storePosition();return;}
    if(!meta.sample_count){pause();showNote('此步记录了参数更新，日志中没有可回放的盘面。');storePosition();return;}
    showNote('读取本步盘面…');
    try{
      const cacheKey=meta.key+'-'+meta.revision;
      let received=cache.get(cacheKey);
      if(!received){received=await request('/training/batch',meta.key);}
      if(dead||ticket!==sampleTicket)return;
      if(!cache.has(cacheKey)){cache.set(cacheKey,received);if(cache.size>8)cache.delete(cache.keys().next().value);}
      detail=received;idx=Math.max(0,detail.samples.findIndex(s=>String(s.row)===String(row)));
      drawWall();play.disabled=!detail.samples.length;
      if(play.disabled){pause();showNote('本步的盘面暂不可读。');return;}
      showNote('');storePosition();schedule();
    }catch(error){if(!dead&&ticket===sampleTicket){pause();showNote(error.message,true);}}
  }
  function showNewest(){
    const b=batches.at(-1),r=b||epochs.at(-1);
    if(r){trial=r.trial;epoch=r.epoch;}
    selectors();pause();selectBatch(b);
  }
  async function reload(){
    const ticket=++indexTicket;
    try{
      const result=await request('/training/batches');
      if(dead||ticket!==indexTicket)return;
      batches=result.batches;epochs=result.epochs;count.textContent=batches.length+' 个训练批次';
      latest.hidden=!active()||!batches.length;
      const old=meta;let next=old&&batches.find(b=>b.key===old.key);
      if(first){
        const selected=batches.find(b=>String(b.epoch)===initial.get('epoch')&&String(b.step)===initial.get('step')&&
          (!initial.has('trial')||b.trial===initial.get('trial')));
        const selectedEpoch=epochs.find(e=>String(e.epoch)===initial.get('epoch')&&(!initial.has('trial')||e.trial===initial.get('trial')));
        next=selected||(follow?batches.at(-1):batches[0]);
        const choice=selected||selectedEpoch||next||epochs.at(-1);if(choice){trial=choice.trial;epoch=choice.epoch;}
        selectors();first=false;await selectBatch(group.find(b=>b.key===next?.key)||group[0],selected?initial.get('row'):null);return;
      }
      if(follow&&batches.length&&batches.at(-1).key!==old?.key){showNewest();return;}
      selectors();
      if(!next||next.revision!==old?.revision||!detail&&next.sample_count){await selectBatch(next||group[0],detail?.samples[idx]?.row);return;}
      meta=next;drawAxis();drawMetrics();
    }catch(error){
      if(dead||ticket!==indexTicket)return;showNote(error.message,true);clearTimeout(retry);retry=setTimeout(reload,4000);
    }
  }
  trialSelect.onchange=()=>{follow=false;pause();trial=trialSelect.value;epoch=null;selectors();selectBatch(group[0]);};
  epochSelect.onchange=()=>{follow=false;pause();epoch=Number(epochSelect.value);selectors();selectBatch(group[0]);};
  speed.onchange=()=>{drawMetrics();if(playing)schedule();};
  function visibility(){if(doc.hidden&&playing){pause();storePosition();}}
  doc.addEventListener('visibilitychange',visibility);
  function update(next){
    channel=next;const stamp=next.log_revision||[next.updated_at,next.training_step?.optimizer_step,next.status].join('|');
    if(signal!==stamp){signal=stamp;reload();}
  }
  play.disabled=true;layout.hidden=chartHead.hidden=scroller.hidden=latest.hidden=true;update(channel);
  return {update,destroy(){dead=true;pause();clearTimeout(retry);controller.abort();doc.removeEventListener('visibilitychange',visibility);}};
}
