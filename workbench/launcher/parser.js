/* Independent offline verifier; never evaluates imported data as code. */
(function (root) {
  'use strict';
  // A standard 4×4 board can hold at most 4*(2**16-1) tile points.
  // Each move spawns at least 2. This bounds a real recording, not play itself.
  const MAX_TURNS = 131068, MAX_BYTES = 128 * 1024 * 1024, MAX_ROWS = MAX_TURNS + 2, MAX_TRACKS = 128;
  const STANDARD_BYTES = 16 * 1024 * 1024, STANDARD_ROWS = 25000, STANDARD_TURNS = 10000;
  const DIRS = ['up','down','left','right'];
  const END = new Map([['terminal','terminal'],['target','target'],['cap','cap'],['capped','cap'],['stopped','stopped'],['error','incomplete'],['incomplete','incomplete']]);
  const EVENTS = new Set(['ready','loading','running','paused','move',...END.keys()]);
  let imports = 0;
  function fail(s) { throw new Error(s); }
  function obj(x) { return x !== null && typeof x === 'object' && !Array.isArray(x); }
  function has(o,k) { return Object.prototype.hasOwnProperty.call(o,k); }
  function eq(a,b) { return a.length === b.length && a.every((x,i) => x === b[i]); }
  function str(x) {
    return typeof x === 'string' && x.trim() && x.length <= 100 && !/[\x00-\x1f\x7f]/.test(x);
  }
  function tile(x) {
    return Number.isSafeInteger(x) && (x===0 || (x>=2 && x<=2**30 && Number.isInteger(Math.log2(x))));
  }
  function board(raw) {
    if (!Array.isArray(raw)) fail('缺少可验证的 4×4 棋盘');
    let b = raw;
    if (raw.length === 4 && raw.every(row => Array.isArray(row) && row.length === 4)) b = raw.flat();
    if (b.length !== 16 || !b.every(tile)) fail('棋盘必须包含 16 个合法方块值');
    return b.slice();
  }
  const lines = Object.fromEntries(DIRS.map(d => [d, Array.from({length:4}, (_,i) =>
    Array.from({length:4}, (_,j) => d==='up' ? 4*j+i : d==='down' ? 4*(3-j)+i :
      d==='left' ? 4*i+j : 4*i+3-j))]));
  function slide(b, a) {
    const c = Array(16).fill(0);
    let gain = 0;
    for (const idx of lines[a]) {
      const ts = idx.map(i=>b[i]).filter(Boolean);
      let dest = 0;
      for (let i = 0; i < ts.length; i++) {
        let v = ts[i];
        if (v === ts[i+1]) { v *= 2; i++; gain += v; }
        c[idx[dest++]] = v;
      }
    }
    return { b:c, gain };
  }
  function legal(b) { return DIRS.filter(d => !eq(slide(b,d).b,b)); }
  function decode(text) {
    text = text.trimStart().replace(/^\uFEFF/, '').trim();
    if (!text || !/^[{[]/.test(text)) fail('只支持 JSON 或 JSONL 逐步记录');
    let x, parsed = false;
    try { x = JSON.parse(text); parsed = true; } catch (_) {}
    if (parsed) {
      if (Array.isArray(x)) return x;
      if (obj(x)) {
        const recognized = ['moves','events','rows','frames'].filter(k => Array.isArray(x[k]));
        if (recognized.length > 1) fail('结构含多个回放数组，无法区分');
        if (recognized.length === 1) {
          return x[recognized[0]].map(row => {
            if (!obj(row)) return row;
            const out={...row};
            for (const k of ['seed','policy','profile','run_id','cap','mode']) {
              if (has(x,k) && has(row,k) && x[k] !== row[k]) fail('分组字段与步骤冲突');
              if (has(x,k) && !has(row,k)) out[k]=x[k];
            }
            return out;
          });
        }
        if (Array.isArray(x.before) && (Array.isArray(x.after) || Array.isArray(x.board))) return [x];
      }
      fail('缺少逐步记录，不能使用摘要生成回放');
    }
    const ls=text.split(/\r?\n/);
    if (ls.length > MAX_ROWS) fail('记录行数超限');
    return ls.flatMap((s,i) => {
      if (!s.trim()) return [];
      try { return [JSON.parse(s)]; } catch (_) { fail('第 '+(i+1)+' 行不是完整 JSON'); }
    });
  }
  function stopOf(x) {
    if (x===null || x===undefined || x==='') return null;
    if (!END.has(x)) fail('未知结束类型');
    return END.get(x);
  }
  function stopFits(stop, terminal, max, previousMax, mode) {
    if (mode==='endless' && (stop==='target' || stop==='cap')) return false;
    if (stop === 'target') return max >= 2048 && previousMax < 2048;
    return (stop === 'terminal') === terminal;
  }
  function parseTrace(text, sourceName='导入记录') {
    if (typeof text !== 'string') fail('只接受本地文本回放');
    const bytes = text.length > MAX_BYTES ? Infinity : new TextEncoder().encode(text).byteLength;
    if (bytes > MAX_BYTES) fail('文件超过 128 MiB');
    const rows=decode(text);
    if (!rows.length || rows.length > MAX_ROWS) fail('空记录或记录数超限');
    const large = bytes > STANDARD_BYTES || rows.length > STANDARD_ROWS;
    const sourceId='import-'+(++imports);
    const file=(String(sourceName).split(/[\\/]/).pop()||'导入记录').replace(/[\x00-\x1f\x7f]/g,'').slice(0,160);
    const groups=new Map();
    let line=0;
    for (const record of rows) {
      line++;
      try {
        const r=record;
        if (!obj(r)) fail('步骤必须为对象');
        const endless=r.mode==='endless';
        if (large && !endless) fail('普通回放最多支持 16 MiB、25000 行');
        if (endless && r.cap!==null) fail('无尽模式不能设置步数上限');
        if (has(r,'type') && !EVENTS.has(r.type)) fail('未知事件类型');
        // A server's ready greeting is not a run and has no trusted trajectory.
        if (r.type==='ready' && !r.run_id && r.turn===0 && !has(r,'action')) continue;
        if (!(Number.isSafeInteger(r.seed) && r.seed>=0) && !str(r.seed)) fail('缺少有效种子');
        const p=has(r,'policy') ? r.policy : has(r,'profile') ? r.profile : 'native';
        if (!str(p)) fail('无效策略名');
        if (has(r,'run_id') && !str(r.run_id)) fail('无效运行编号');
        const key=JSON.stringify([r.seed,p,r.run_id??null]);
        let g=groups.get(key);
        if (!g) {
          if (groups.size>=MAX_TRACKS) fail('对局轨道数超限');
          g={id:sourceId+'-game-'+groups.size,sourceId,file,seed:r.seed,label:p,policy:p,runId:r.run_id??null,frames:[],
             mode:endless?'endless':null,status:'incomplete',cap:null,ended:false,seq:null,opening:null};
          groups.set(key,g);
        }
        if (endless!==(g.mode==='endless')) fail('同一对局的玩法不一致');
        if (has(r,'seq')) {
          if (!Number.isSafeInteger(r.seq) || r.seq<0 || (g.seq!==null && r.seq<=g.seq)) fail('事件序号没有严格递增');
          g.seq=r.seq;
        }
        if (has(r,'cap') && r.cap!==null) {
          if (!Number.isSafeInteger(r.cap) || r.cap<1 || r.cap>STANDARD_TURNS || (g.cap!==null && g.cap!==r.cap)) fail('无效或互相矛盾的步数上限');
          g.cap=r.cap;
        }
        if (has(r,'type') && r.type!=='move') {
          if (has(r,'action') || !Number.isSafeInteger(r.turn) || r.turn<0 || !Array.isArray(r.board)) fail('无效状态事件');
          const current=g.frames.at(-1);
          const reported=board(r.board);
          if (current) {
            if (r.turn!==current.p || !eq(reported,current.b) || (has(r,'score') && r.score!==current.s) ||
                (has(r,'tile') && r.tile!==current.m)) fail('运行状态与已记录棋盘不一致');
          } else {
            if (r.turn!==0 || r.score!==0 || (g.opening && !eq(reported,g.opening))) fail('首个状态必须为开局');
            g.opening=reported;
          }
          const final=stopOf(END.has(r.type)?r.type:null);
          if (final) {
            if (!current) {
              if (final==='stopped' || final==='incomplete') {g.ended=true; continue;}
              fail('无可重放的走子');
            }
            if (g.ended && final!==g.status) fail('结束事件相互矛盾');
            if (!stopFits(final,!legal(current.b).length,current.m,g.frames.at(-2)?.m,g.mode)) fail('结束类型与棋局状态不一致');
            if (final==='cap' && g.cap!==null && current.p!==g.cap) fail('结束步数不是上限');
            if (stopOf(r.stop)!==null && stopOf(r.stop)!==final) fail('结束标记相互矛盾');
            g.status=final; g.ended=true;
          } else if (g.ended) fail('结束后还有运行事件');
          continue;
        }
        if (g.ended) fail('结束后还有走子');
        const turn=r.turn;
        if (!Number.isSafeInteger(turn) || turn<1 || turn>(endless?MAX_TURNS:STANDARD_TURNS) || (g.cap!==null && turn>g.cap)) fail('无效步数');
        const prev=board(r.before), next=board(has(r,'after') ? r.after : r.board);
        if (has(r,'after') && has(r,'board') && !eq(next,board(r.board))) fail('两种结果棋盘冲突');
        if (!g.frames.length) {
          if (turn!==1 || prev.filter(Boolean).length!==2 || prev.some(v=>v!==0 && v!==2 && v!==4)) fail('首步缺失或不是标准开局');
          if (g.opening && !eq(prev,g.opening)) fail('首步和开局不一致');
          g.frames.push({p:0,b:prev,s:0,m:Math.max(...prev),a:null,g:0,n:null});
        }
        const current=g.frames.at(-1);
        if (turn!==current.p+1 || !eq(prev,current.b)) fail('步骤不连续或前后棋盘不一致');
        const action=r.action;
        if (!DIRS.includes(action)) fail('未知方向');
        const moved=slide(prev,action);
        if (eq(prev,moved.b)) fail('不能移动的方向不是合法一步');
        const diffs=next.map((v,i)=>v===moved.b[i]?-1:i).filter(i=>i>=0);
        if (diffs.length!==1 || moved.b[diffs[0]]!==0 || ![2,4].includes(next[diffs[0]])) fail('滑动后必须仅在空格生成一个 2 或 4');
        const n=diffs[0];
        if (has(r,'spawn_index') && r.spawn_index!==n) fail('新方块位置错误');
        if (has(r,'spawn')) {
          if (!(Array.isArray(r.spawn) && r.spawn.length===3 && r.spawn[0]===Math.floor(n/4) && r.spawn[1]===n%4 && r.spawn[2]===next[n])) fail('新方块详情错误');
        }
        const score=current.s+moved.gain, max=Math.max(...next);
        if (r.score!==score || !Number.isSafeInteger(score)) fail('累计分数与实际合并不一致');
        for (const k of ['gained','gain']) if (has(r,k) && r[k]!==moved.gain) fail('本步得分与实际合并不一致');
        for (const k of ['tile','max_tile']) if (has(r,k) && r[k]!==max) fail('最大方块与棋盘不一致');
        const available=legal(next);
        if (has(r,'terminal') && (typeof r.terminal!=='boolean' || r.terminal!==!available.length)) fail('终局标记与可走方向矛盾');
        const stop=stopOf(r.stop ?? (['capped','cap','stopped','target'].includes(r.status)?r.status:null));
        if (stop && !stopFits(stop,!available.length,max,current.m,g.mode)) fail('结束标记与棋盘矛盾');
        if (stop==='cap' && g.cap!==null && turn!==g.cap) fail('标记截断的步数不是上限');
        g.frames.push({p:turn,b:next,s:score,m:max,a:action,g:moved.gain,n});
        if (stop || !available.length || (g.cap!==null && turn===g.cap)) {
          g.status=stop || (!available.length?'terminal':'cap');
          g.ended=true;
        }
      } catch (err) { fail('记录 '+line+'：'+err.message); }
    }
    const out=[...groups.values()].filter(g=>g.frames.length>1);
    if (!out.length) fail('没有可验证的完整走子');
    if (out.length!==groups.size) fail('文件混有不能回放的运行');
    return out.map(({id,file,sourceId,seed,label,policy,runId,cap,mode,frames,status}) => ({id,file,sourceId,seed,label,policy,runId,cap,mode,frames,status}));
  }
  const api={parseTrace,MAX_BYTES,MAX_ROWS,MAX_TRACKS,MAX_TURNS};
  root.KevTrace=api;
  if (typeof module !== 'undefined' && module.exports) module.exports=api;
})(typeof globalThis==='object'?globalThis:this);
