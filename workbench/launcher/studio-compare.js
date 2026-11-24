/* On-demand same-seed comparisons from the workbench's evaluation directory. */
function createStudioCompare({get, makeButton, serviceAvailable, offlinePeriod, readArchivedSeed,
                              selectSeed, onClear, parseTrace}) {
  const periodList = get('compare-periods'), seedList = get('seeds');
  const refreshButton = get('compare-refresh'), feedback = get('compare-feedback');
  const pageSize = 256, maxTurns = 10000;
  const offline = !serviceAvailable && location.protocol === 'file:';
  let visible = false, periods = null, current = null, currentSeed = null;
  let requestedRun = null, requestedSeed = null, displayed = false;
  let catalogJob = null, replayJob = null;

  function message(text) {
    feedback.textContent = text;
    feedback.hidden = !text;
  }
  async function readJSON(url, signal) {
    const response = await fetch(url, {credentials:'same-origin', cache:'no-store', signal});
    if (!response.ok) throw Error('读取失败');
    return response.json();
  }
  function clearBoard() {
    displayed = false;
    onClear();
  }
  function mark(container, field, value) {
    let selected;
    for (const button of container.children) {
      const active = button.dataset[field] === String(value);
      button.setAttribute('aria-pressed', String(active));
      if (active) selected = button;
    }
    if (selected && container.scrollWidth > container.clientWidth) {
      const rect = selected.getBoundingClientRect(), strip = container.getBoundingClientRect();
      if (rect.left < strip.left) container.scrollLeft += rect.left - strip.left;
      else if (rect.right > strip.right) container.scrollLeft += rect.right - strip.right;
    }
  }
  function navigate(runId, seed) {
    const url = new URL(location.href);
    url.hash = 'compare';
    url.searchParams.set('run', runId);
    if (seed != null) url.searchParams.set('seed', seed);
    else url.searchParams.delete('seed');
    for (const field of ['epoch','step','trial','row','summary']) url.searchParams.delete(field);
    if (url.href !== location.href && serviceAvailable) history.pushState(null, '', url);
    route(runId, seed);
  }
  const comparable = p => p.arms.length >= 2;
  function drawPeriods() {
    const choices = periods.filter(comparable);
    const label = p => {
      const round = /^round[ _-]?(\d+)(?:[ _-]|$)/i.exec(p.name);
      return round ? '第 ' + Number(round[1]) + ' 轮' : p.name;
    };
    const counts = new Map();
    for (const p of choices) counts.set(label(p), (counts.get(label(p)) || 0) + 1);
    periodList.replaceChildren(...choices.map(p => {
      const button = makeButton(counts.get(label(p)) === 1 ? label(p) : p.name, 'chip', () => navigate(p.id, null));
      button.dataset.run = p.id;
      button.title = p.name + ' · ' + p.arms.length + ' 个模型 · ' + p.seeds.length + ' 个种子';
      return button;
    }));
  }
  function drawSeeds(period) {
    seedList.replaceChildren(...(period?.seeds || []).map(seed => {
      const button = makeButton(String(seed), 'chip', () => navigate(period.id, seed));
      button.dataset.seed = String(seed);
      button.title = '种子 ' + seed;
      return button;
    }));
    get('seed-catalog-label').textContent = period ? '种子 · ' + period.seeds.length + ' 个' : '种子';
  }
  function choose(remember, refresh = false) {
    if (!periods) return;
    const eligible = periods.filter(comparable);
    const period = requestedRun == null ? eligible.find(p => p.id === remember?.id) || eligible[0]
      : periods.find(p => p.id === requestedRun) || periods.find(p => p.aliases?.includes(requestedRun));
    const available = period && comparable(period) ? period : null;
    const seed = requestedSeed == null
      ? available?.seeds.find(s => available.id === remember?.id && s === remember.seed) ?? available?.seeds[0]
      : available?.seeds.find(s => String(s) === requestedSeed);
    const changed = current?.id !== available?.id || currentSeed !== (seed ?? null);
    if (changed || !available) {
      replayJob?.abort();
      replayJob = null;
      clearBoard();
    }
    if (current !== available) drawSeeds(available);
    current = available;
    currentSeed = seed ?? null;
    mark(periodList, 'run', current?.id);
    mark(seedList, 'seed', currentSeed);
    get('compare-period-info').textContent = (current ? current.name + ' · ' + current.arms.length + ' 个模型 · ' : '') +
      '共 ' + eligible.length + ' 期对照';
    if (!current) {
      if (period) message(period.name + ' 目前只有 ' + period.arms.length + ' 个模型。可在上方选择已有同种子对照的期次。');
      else if (requestedRun) message('这条链接对应的期次在当前测评目录中找不到。可选择上方期次，或点击「刷新」。');
      else message('尚未发现有多个模型的整局测评。新测评记录写入后，点击「刷新」即可查看。');
    } else if (currentSeed == null) {
      message(requestedSeed == null ? '本期正在等待种子记录。' : '本期没有记录链接中的种子，请从种子条选择。');
    } else if (changed || refresh || (!displayed && !replayJob)) {
      loadSeed(current, currentSeed, !changed && displayed);
    }
  }
  function route(runId, seedParam) {
    const nextRun = runId || null, nextSeed = seedParam == null ? null : String(seedParam);
    if (!visible || requestedRun === nextRun && requestedSeed === nextSeed) return;
    requestedRun = nextRun;
    requestedSeed = nextSeed;
    choose();
  }
  async function refreshCatalog(manual) {
    catalogJob?.abort();
    const job = new AbortController();
    catalogJob = job;
    refreshButton.disabled = true;
    const runAtStart = requestedRun, seedAtStart = requestedSeed;
    const remember = manual && current ? {id:current.id, seed:currentSeed} : null;
    message(manual ? '正在刷新测评目录…' : '正在读取测评目录…');
    try {
      if (!serviceAvailable && !offline) {
        message('请从本机工作台打开同种子对照，以载入测评目录。');
        return;
      }
      const data = serviceAvailable ? await readJSON('/replay/comparisons', job.signal) : {periods: offlinePeriod ? [offlinePeriod] : []};
      if (job.signal.aborted) return;
      if (!Array.isArray(data?.periods)) throw Error('目录格式无效');
      periods = data.periods.filter(p => typeof p?.id === 'string' && typeof p.name === 'string' &&
        Array.isArray(p.arms) && Array.isArray(p.seeds));
      drawPeriods();
      choose(requestedRun === runAtStart && requestedSeed === seedAtStart ? remember : null, true);
    } catch {
      if (!job.signal.aborted) message('测评目录读取失败。' + (displayed ? '当前保留上次读取的盘面。' : '') + '点击「刷新」重试。');
    } finally {
      if (catalogJob === job) {
        catalogJob = null;
        refreshButton.disabled = !serviceAvailable;
      }
    }
  }

  function validDecision(row) {
    const probs = row.probabilities;
    if (!probs || typeof probs !== 'object' || Array.isArray(probs) || !Object.hasOwn(probs, row.action)) return false;
    const values = Object.entries(probs);
    return values.every(([d,p]) => ['up','down','left','right'].includes(d) && Number.isFinite(p) &&
      p >= 0 && p <= 1 && p <= probs[row.action]) &&
      Math.abs(values.reduce((sum,[,p]) => sum+p, 0) - 1) < .002;
  }
  function recordedModel(arm, seed, rows, reused) {
    const games = parseTrace(JSON.stringify(rows), arm);
    if (games.length !== 1 || games[0].seed !== seed || games[0].frames.length !== rows.length + 1) throw Error('步数不符');
    const frames = games[0].frames.slice(1).map((f,i) => ({
      board:rows[i].after, action:f.a, score:f.s, tile:f.m, gained:f.g, probs:rows[i].probabilities,
      spawn:[Math.floor(f.n/4), f.n%4, f.b[f.n]]
    }));
    const stop = rows.at(-1).stop;
    return {id:arm, opening:rows[0].before, frames, reused, endStatus:stop === 'capped' ? 'cap' : stop ?? null};
  }
  async function readSource(channel, seed, arms, borrowed, signal) {
    const states = new Map(arms.map(arm => [arm, {rows:[], total:null}]));
    const pending = state => !state.problem && (state.total == null || state.rows.length < state.total);
    for (let start = 1; [...states.values()].some(pending); start += pageSize) {
      const query = new URLSearchParams({channel, seed, start, limit:pageSize});
      let page;
      try {
        page = await readJSON('/gameplay/replay?' + query, signal);
        if (page.channel !== channel || page.seed !== seed || page.start !== start || !Array.isArray(page.tracks) ||
            !Number.isSafeInteger(page.total_turns) || page.total_turns < 0 ||
            page.next_start !== (start + pageSize <= page.total_turns ? start + pageSize : null)) throw Error('分页不符');
      } catch (error) {
        if (signal.aborted) throw error;
        for (const state of states.values()) if (pending(state)) state.problem = '读取失败，请刷新重试';
        break;
      }
      for (const [arm,state] of states) {
        if (!pending(state)) continue;
        const matches = page.tracks.filter(t => t?.arm === arm), track = matches[0];
        if (!track) { state.problem = '本种子无轨迹'; continue; }
        if (matches.length === 1 && track.status === 'waiting') { state.problem = '等待记录'; continue; }
        const total = track.total_turns, count = Math.max(0, Math.min(pageSize, total-start+1));
        if (matches.length !== 1 || track.status !== 'recorded' || !Number.isSafeInteger(total) || total < start ||
            total > maxTurns || total > page.total_turns || total < (state.total ?? 0) ||
            !Array.isArray(track.frames) || track.frames.length !== count ||
            !track.frames.every((f,i) => f?.seed === seed && f.turn === start+i && validDecision(f))) {
          state.problem = '轨迹校验失败';
          continue;
        }
        // A running writer may append steps between requests. Replay the size seen on page one.
        state.total ??= total;
        state.reused = borrowed || Boolean(track.reused);
        state.rows.push(...track.frames.slice(0, state.total-state.rows.length));
      }
    }
    for (const [arm,state] of states) {
      if (!state.problem) {
        try { state.model = recordedModel(arm, seed, state.rows, state.reused); }
        catch { state.problem = '轨迹校验失败'; }
      }
      delete state.rows;
    }
    return states;
  }
  async function collect(period, seed, signal) {
    const states = await readSource(period.id, seed, period.direct_arms || period.arms, false, signal);
    const sources = new Map();
    const refs = period.references?.[String(seed)] || {};
    for (const arm of period.arms) {
      const source = Object.hasOwn(refs, arm) && refs[arm];
      if (typeof source !== 'string' || source === period.id || states.get(arm)?.model) continue;
      if (!sources.has(source)) sources.set(source, []);
      sources.get(source).push(arm);
    }
    await Promise.all([...sources].map(async ([source, arms]) => {
      const archived = await readSource(source, seed, arms, true, signal);
      for (const [arm,state] of archived) states.set(arm, state);
    }));
    const models = [], missing = [];
    let focus = null, milestoneTurn = null;
    for (const arm of period.arms) {
      const state = states.get(arm), model = state?.model;
      if (!model) {
        missing.push(arm + '（' + (state?.problem || '本种子无轨迹') + '）');
        continue;
      }
      models.push(model);
      const index = model.frames.findIndex(f => f.tile >= 512);
      if (index >= 0 && (milestoneTurn == null || index+1 < milestoneTurn)) {
        focus = arm;
        milestoneTurn = index+1;
      }
    }
    const openings = new Set(models.map(m => JSON.stringify(m.opening)));
    if (openings.size > 1) throw Error('同种子模型的开局盘面不一致。');
    return {result:{seed, models, focus:focus ?? models[0]?.id, milestoneTurn}, missing};
  }
  async function loadSeed(period, seed, preserve) {
    replayJob?.abort();
    const job = new AbortController();
    replayJob = job;
    message('正在读取种子 ' + seed + ' 的盘面…');
    try {
      const {result, missing} = serviceAvailable ? await collect(period, seed, job.signal)
        : {result:readArchivedSeed(seed), missing:[]};
      if (job.signal.aborted) return;
      if (result.models.length) {
        selectSeed({...result, periodId:period.id, archived:!serviceAvailable, embedded:Boolean(period.embedded)}, {preserve});
        displayed = true;
      } else clearBoard();
      const info = result.models.length ? '已载入 ' + result.models.length + ' 个模型。' : '这个种子尚无可播放的盘面。';
      message(missing.length ? info + missing.join('、') : '');
    } catch (error) {
      if (job.signal.aborted) return;
      clearBoard();
      message('种子 ' + seed + ' 的回放读取失败。' + error.message + ' 点击「刷新」重试。');
    } finally {
      if (replayJob === job) replayJob = null;
    }
  }
  function hide() {
    visible = false;
    catalogJob?.abort();
    replayJob?.abort();
    catalogJob = replayJob = null;
  }
  function show(runId, seedParam) {
    hide();
    visible = true;
    requestedRun = runId || null;
    requestedSeed = seedParam == null ? null : String(seedParam);
    current = currentSeed = periods = null;
    clearBoard();
    periodList.replaceChildren();
    drawSeeds(null);
    get('compare-period-info').textContent = '';
    refreshCatalog(false);
  }
  refreshButton.onclick = () => { if (visible) refreshCatalog(true); };
  return {show, hide, route};
}
