/* Local run bundles: bounded list previews and validated playback for selected runs. */
(function (root) {
  'use strict';
  const trace = root.KevTrace || (typeof require === 'function' ? require('./parser.js') : null);
  const MAX_METADATA_BYTES = 1024 * 1024;
  const MILESTONE_TILES = [512, 1024, 2048];
  const SUMMARY = /(^|[-_.])(benchmark|bench|summary|progress|timing|metrics|train|dev|loss|report|lock)([-_.]|$)/;
  const ARM = /^[a-z0-9][a-z0-9-]{0,63}$/;
  const obj = x => x !== null && typeof x === 'object' && !Array.isArray(x);
  const given = x => x !== null && x !== undefined;
  const pathOf = f => String(f.webkitRelativePath || f.name || '未命名文件').replaceAll('\\', '/');
  const basename = p => p.split('/').at(-1);
  const dirname = p => p.includes('/') ? p.slice(0, p.lastIndexOf('/')) : '';
  const isCompanion = f => /^(run|result|identity)\.json$/i.test(basename(pathOf(f)));
  const isSummary = f => isCompanion(f) || SUMMARY.test(basename(pathOf(f)).toLowerCase());
  const sizeOK = (f, limit) => Number.isSafeInteger(f.size) && f.size > 0 && f.size <= limit;
  const errorAt = (file, message) => new Error('「' + pathOf(file) + '」：' + message);
  function isTraceCandidate(file) {
    const p = pathOf(file).toLowerCase(), pieces = p.split('/'), name = pieces.at(-1);
    if (!/\.(json|jsonl|ndjson)$/.test(name) || isSummary(file)) return false;
    if (/(^|\/)implementation\/configs\//.test(p)) return false;
    if (/(^|[-_.])(moves?|traces?|replays?|events?|gameplay|games?)([-_.]|$)/.test(name)) return true;
    const stem = name.replace(/\.(json|jsonl|ndjson)$/, '');
    if (pieces.slice(0, -1).includes('traces') && /^(d4|wide|stock|[a-z0-9]+[-_][a-z0-9_-]+)$/.test(stem)) return true;
    return pieces.slice(0, -1).includes('live') && /^[0-9a-f]{32}$/.test(stem);
  }
  function describeFiles(input, {direct = false} = {}) {
    const files = Array.from(input), dirs = new Map();
    for (const file of files) {
      const dir = dirname(pathOf(file));
      if (!dirs.has(dir)) dirs.set(dir, {traces: [], results: [], runs: [], identities: []});
      const group = dirs.get(dir), name = basename(pathOf(file)).toLowerCase();
      if (name === 'result.json') group.results.push(file);
      else if (name === 'run.json') group.runs.push(file);
      else if (name === 'identity.json') group.identities.push(file);
      else if (direct ? /\.(json|jsonl|ndjson)$/i.test(name) && !isSummary(file) : isTraceCandidate(file)) group.traces.push(file);
    }
    return [...dirs.values()].flatMap(group => group.traces.map(file => {
      const resultFile = group.results[0] || null, runFile = group.runs[0] || null;
      const identityFile = group.traces.length === 1 ? group.identities[0] || null : null;
      const companions = [resultFile, runFile, identityFile].filter(Boolean);
      const ambiguous = companions.length && (group.traces.length > 1 || group.results.length > 1 ||
        group.runs.length > 1 || (identityFile && group.identities.length > 1));
      const bad = !sizeOK(file, trace.MAX_BYTES) ? file : companions.find(f => !sizeOK(f, MAX_METADATA_BYTES));
      const limit = bad === file ? trace.MAX_BYTES : MAX_METADATA_BYTES;
      const disabledReason = bad ? pathOf(bad) + ' 为空或超过 ' + (limit / 1048576) + ' MiB 上限' : '';
      return {file, resultFile, runFile, identityFile, path: pathOf(file),
        bytes: [file, ...companions].reduce((sum, f) => sum + (Number.isSafeInteger(f.size) && f.size >= 0 ? f.size : 0), 0),
        disabled: Boolean(bad), disabledReason,
        error: ambiguous ? '同目录存在多份记录或同名摘要，无法唯一关联；请分别选择每次运行的文件夹或只导入逐步记录。' : null};
    }));
  }
  async function readText(file, maxBytes) {
    if (!sizeOK(file, maxBytes)) throw errorAt(file, '文件为空或超过 ' + (maxBytes / 1048576) + ' MiB 上限');
    let text;
    try { text = await file.text(); } catch (_) { throw errorAt(file, '无法读取本地文件'); }
    if (typeof text !== 'string' || text.length > maxBytes || new TextEncoder().encode(text).byteLength > maxBytes)
      throw errorAt(file, '文件内容超过 ' + (maxBytes / 1048576) + ' MiB 上限');
    return text;
  }
  async function readObject(file) {
    const text = await readText(file, MAX_METADATA_BYTES);
    let value;
    try { value = JSON.parse(text.replace(/^\uFEFF/, '')); } catch (_) { throw errorAt(file, '摘要不是完整 JSON'); }
    if (!obj(value)) throw errorAt(file, '摘要必须是 JSON 对象');
    return value;
  }
  function label(value, file, field) {
    if (!given(value)) return null;
    if (typeof value !== 'string' || !value.trim() || value.length > 160 || /[\x00-\x1f\x7f]/.test(value))
      throw errorAt(file, field + ' 必须是非空短文本');
    return value.trim();
  }
  function hash(value, file, field) {
    if (!given(value)) return null;
    if (typeof value !== 'string' || !/^[a-f\d]{64}$/i.test(value)) throw errorAt(file, field + ' 必须是 SHA-256 指纹');
    return value.toLowerCase();
  }
  function cap(value, file) {
    if (!given(value)) return null;
    if (!Number.isSafeInteger(value) || value < 1 || value > trace.MAX_TURNS) throw errorAt(file, '无效步数上限');
    return value;
  }
  const armTitle = arm => arm === 'official' ? '官方基线' : arm;
  function episodeModel(metadata, file) {
    const arm = metadata.arm;
    if (typeof arm !== 'string' || !ARM.test(arm)) throw errorAt(file, '缺少有效模型名称');
    const runIdentity = hash(metadata.run_identity, file, 'run_identity');
    const headSha = hash(metadata.checkpoint_head_sha256, file, 'checkpoint_head_sha256');
    if (!runIdentity || !headSha) throw errorAt(file, '缺少评测或检查点指纹');
    return {arm, runIdentity, headSha, title: armTitle(arm), device: label(metadata.device, file, 'device')};
  }
  function checkMetadata(reported, expected, file) {
    for (const [key, name] of [['headSha','检查点指纹'], ['device','设备'], ['cap','步数上限'],
        ['arm','模型名称'], ['runIdentity','评测指纹']]) {
      if (given(reported[key]) && given(expected[key]) && reported[key] !== expected[key])
        throw errorAt(file, name + '与运行清单冲突');
    }
  }
  function verifyEpisode(identity, entry, games, info) {
    if (identity?.metadata?.split !== 'gameplay_dev') return info;
    const file = entry.identityFile, limit = cap(identity.cap, file);
    const [game] = games;
    if (identity.schema_version !== 1) throw errorAt(file, '不支持的单局清单版本');
    if (games.length !== 1 || identity.seed !== game.seed) throw errorAt(file, '单局清单种子与逐步记录不一致');
    const step = game.frames.at(-1).p;
    if (!limit || step > limit || (given(game.cap) && game.cap !== limit) || (game.status === 'cap' && step !== limit))
      throw errorAt(file, '单局清单的步数上限与逐步记录不一致');
    const model = {...episodeModel(identity.metadata, file), cap: limit};
    checkMetadata(model, info, file);
    return {...info, ...model, device: model.device || info.device};
  }
  async function matchesTraceHash(raw, result, file) {
    const expected = hash(result.moves_sha256, file, 'moves_sha256');
    if (!expected) throw errorAt(file, '缺少逐步记录指纹');
    const subtle = root.crypto?.subtle ||
      (typeof module !== 'undefined' && module.exports ? require('node:crypto').webcrypto.subtle : null);
    if (!subtle) throw errorAt(file, '当前浏览器环境无法校验逐步记录指纹');
    const bytes = new Uint8Array(await subtle.digest('SHA-256', new TextEncoder().encode(raw)));
    const observed = Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('');
    if (observed !== expected) throw errorAt(file, '逐步记录的 SHA-256 指纹与摘要不一致');
  }
  function localRef(value, file, field) {
    if (!given(value)) return null;
    if (typeof value !== 'string' || !value || /[\x00-\x1f\x7f]/.test(value)) throw errorAt(file, field + ' 必须是同目录相对文件名');
    const normalized = value.replaceAll('\\', '/').replace(/^\.\//, '');
    if (normalized.includes('/') || normalized === '.' || normalized === '..' || normalized.includes(':'))
      throw errorAt(file, field + ' 只支持本次运行同目录中的文件');
    return normalized;
  }
  const seedKey = seed => JSON.stringify(seed);
  function validateRun(run, entry, games) {
    if (!run) return {title: null, headSha: null, device: null, cap: null};
    const file = entry.runFile;
    if (run.schema_version !== 1 || run.kind !== 'play') throw errorAt(file, '仅支持 schema_version=1、kind=play 的运行清单');
    if (games.some(g => (g.mode === 'endless') !== (run.mode === 'endless')) ||
        (run.mode === 'endless' && run.max_moves_per_game !== null))
      throw errorAt(file, '玩法或步数上限与逐步记录冲突');
    if (new Set(games.map(g => seedKey(g.seed))).size !== games.length)
      throw errorAt(file, '同一种子包含多条策略轨道，无法唯一关联清单');
    const runId = label(run.run_id, file, 'run_id');
    if (runId && games.some(g => given(g.runId) && g.runId !== runId)) throw errorAt(file, '运行编号与逐步记录冲突');
    if (given(run.files) && !obj(run.files)) throw errorAt(file, 'files 必须是对象');
    for (const [key, target] of [['moves', entry.file], ['result', entry.resultFile]]) {
      const ref = localRef(run.files?.[key], file, 'files.' + key);
      if (ref && target && ref !== basename(pathOf(target))) throw errorAt(file, 'files.' + key + ' 与所选文件不一致');
    }
    if (given(run.model) && !obj(run.model)) throw errorAt(file, 'model 必须是对象');
    const model = run.model || {};
    for (const key of ['base_weights_sha256', 'adapter_sha256']) hash(model[key], file, 'model.' + key);
    const headSha = hash(model.head_sha256, file, 'model.head_sha256');
    const title = label(run.variant, file, 'variant') || label(run.label, file, 'label') || label(model.profile, file, 'model.profile');
    const device = label(run.device, file, 'device'), limit = cap(run.max_moves_per_game, file);
    if (given(run.seeds)) {
      if (!Array.isArray(run.seeds)) throw errorAt(file, 'seeds 必须是数组');
      if (run.seeds.length) {
        const expected = new Set(games.map(g => seedKey(g.seed))), reported = new Set(run.seeds.map(seedKey));
        if (reported.size !== run.seeds.length || expected.size !== reported.size || [...expected].some(k => !reported.has(k)))
          throw errorAt(file, '清单种子与逐步记录不一致');
      }
    }
    if (limit && games.some(g => given(g.cap) && g.cap !== limit)) throw errorAt(file, '步数上限与逐步记录冲突');
    if (limit && games.some(g => g.frames.at(-1).p > limit)) throw errorAt(file, '记录步数超过清单上限');
    if (limit && games.some(g => g.status === 'cap' && g.frames.at(-1).p !== limit)) throw errorAt(file, '结束步数与清单上限冲突');
    return {title, headSha, device, cap: limit};
  }
  async function reconcile(result, entry, games, info, raw) {
    if (!result) return {games, headSha: info.headSha, device: info.device, title: info.title};
    const file = entry.resultFile;
    const single = result.split === 'gameplay_dev';
    const limit = cap(result.cap, file);
    const model = single ? episodeModel(result, file) : {
      headSha: hash(result.checkpoint_head_sha256, file, 'checkpoint_head_sha256'),
      device: label(result.device, file, 'device'),
    };
    checkMetadata({...model, cap: limit}, info, file);
    let rows = result.games;
    if (single) {
      if (result.schema_version !== 1 || !limit || games.length !== 1 || given(result.games))
        throw errorAt(file, '整局开发评测需要一份有效的单局摘要');
      if (!['terminal', 'capped', 'target'].includes(result.stop))
        throw errorAt(file, '单局摘要缺少有效结束原因');
      rows = [{...result, tile: result.max_tile, stop: result.stop === 'capped' ? 'cap' : result.stop}];
    } else if (info.arm) {
      throw errorAt(file, '单局清单与该摘要格式冲突');
    }
    if (!Array.isArray(rows) || rows.length !== games.length) throw errorAt(file, '摘要对局数与逐步记录不一致');
    const bySeed = new Map();
    for (const row of rows) {
      if (!obj(row)) throw errorAt(file, 'games 中的每局必须为对象');
      const key = seedKey(row.seed);
      if (bySeed.has(key)) throw errorAt(file, '摘要含重复种子，无法唯一关联');
      bySeed.set(key, row);
    }
    const seen = new Set();
    const merged = games.map(g => {
      const key = seedKey(g.seed), row = bySeed.get(key), end = g.frames.at(-1);
      if (seen.has(key)) throw errorAt(file, '同一种子包含多条策略轨道，无法唯一关联摘要');
      seen.add(key);
      if (!row || row.turn !== end.p || row.score !== end.s || row.tile !== end.m)
        throw errorAt(file, '种子 ' + String(g.seed) + ' 的末步、分数或最大方块与逐步记录不一致');
      if (!['terminal', 'cap', 'stopped'].includes(row.stop) && !(single && row.stop === 'target'))
        throw errorAt(file, '种子 ' + String(g.seed) + ' 缺少有效结束原因');
      const terminal = g.status === 'terminal';
      if (row.stop !== 'target' && (row.stop === 'terminal') !== terminal)
        throw errorAt(file, '种子 ' + String(g.seed) + ' 的结束原因与棋盘冲突');
      if (single && ((row.stop === 'target') !== (end.m >= 2048)
          || (row.stop === 'target' && g.frames.at(-2).m >= 2048)))
        throw errorAt(file, '种子 ' + String(g.seed) + ' 的 2048 达成记录与棋盘冲突');
      if (g.status !== 'incomplete' && row.stop !== g.status) throw errorAt(file, '种子 ' + String(g.seed) + ' 的结束原因与逐步记录冲突');
      const maximum = limit || info.cap;
      if (maximum && given(g.cap) && g.cap !== maximum) throw errorAt(file, '种子 ' + String(g.seed) + ' 的步数上限与逐步记录冲突');
      if (maximum && end.p > maximum) throw errorAt(file, '种子 ' + String(g.seed) + ' 的步数超过摘要上限');
      if (row.stop === 'cap' && (!maximum || end.p !== maximum)) throw errorAt(file, '种子 ' + String(g.seed) + ' 的结束步数不是已报告上限');
      return {...g, status: row.stop};
    });
    if (single) await matchesTraceHash(raw, result, file);
    return {games: merged, headSha: model.headSha || info.headSha, device: model.device || info.device,
      title: model.title || info.title};
  }
  async function loadRun(entry) {
    if (entry.error) throw errorAt(entry.file, entry.error);
    const raw = await readText(entry.file, trace.MAX_BYTES);
    let games;
    try { games = trace.parseTrace(raw, pathOf(entry.file)); }
    catch (error) { throw errorAt(entry.file, String(error.message || error)); }
    const run = entry.runFile ? await readObject(entry.runFile) : null;
    const result = entry.resultFile ? await readObject(entry.resultFile) : null;
    const identity = entry.identityFile ? await readObject(entry.identityFile) : null;
    const info = verifyEpisode(identity, entry, games, validateRun(run, entry, games));
    const resolved = await reconcile(result, entry, games, info, raw);
    const directory = dirname(pathOf(entry.file)), parent = basename(directory);
    const arm = basename(dirname(directory));
    const armPath = games.length === 1 && games[0].policy === 'native' && parent === String(games[0].seed) && ARM.test(arm);
    const title = resolved.title || (armPath ? armTitle(arm) : parent) ||
      (games.length === 1 && games[0].policy !== 'native' ? games[0].label : null) || basename(pathOf(entry.file));
    return {path: pathOf(entry.file), title, headSha: resolved.headSha, device: resolved.device,
      games: resolved.games.map(g => ({...g, runTitle: title, headSha: resolved.headSha, device: resolved.device}))};
  }
  const previewNumber = (value, limit = Number.MAX_SAFE_INTEGER) =>
    Number.isSafeInteger(value) && value >= 0 && value <= limit ? value : null;
  const previewLabel = value => typeof value === 'string' && value.trim() && value.length <= 160 &&
    !/[\x00-\x1f\x7f]/.test(value) ? value.trim() : null;
  const previewSeed = value => previewNumber(value) ?? (previewLabel(value) && value.length <= 100 ? value : null);
  const previewArm = value => typeof value === 'string' && ARM.test(value) ? armTitle(value) : null;
  const validTile = value => Number.isSafeInteger(value) && (value === 0 ||
    (value >= 2 && value <= 2 ** 30 && Number.isInteger(Math.log2(value))));
  function previewTile(row) {
    const value = row.max_tile ?? row.tile;
    if (validTile(value)) return value;
    const board = row.after ?? row.board;
    if (!Array.isArray(board)) return null;
    const flat = board.length === 4 && board.every(line => Array.isArray(line) && line.length === 4) ? board.flat() : board;
    return flat.length === 16 && flat.every(validTile) ? Math.max(...flat) : null;
  }
  function reportedTime(row) {
    for (const key of ['ended_at_utc','ended_at','completed_at','saved_at','timestamp','started_at_utc','started_at']) {
      const value = row?.[key];
      if (typeof value !== 'string' || value.length > 40 || !/^\d{4}-\d{2}-\d{2}T/.test(value) || !/(Z|[+-]\d{2}:?\d{2})$/i.test(value)) continue;
      const time = Date.parse(value);
      if (Number.isFinite(time)) return new Date(time).toISOString();
    }
    return null;
  }
  async function previewObject(file) {
    if (!file) return null;
    try { return await readObject(file); } catch (_) { return null; }
  }
  function* previewLines(text) {
    for (const [line] of text.matchAll(/[^\n]+/g)) {
      if (!line.trim()) continue;
      try { yield JSON.parse(line); } catch (_) {}
    }
  }
  function previewRows(raw) {
    const text = raw.trimStart();
    let document;
    try { document = JSON.parse(text); } catch (_) { return [previewLines(text), {}]; }
    if (Array.isArray(document)) return [document, {}];
    if (obj(document)) {
      const series = ['moves','events','rows','frames'].find(key => Array.isArray(document[key]));
      return [series ? document[series] : [document], series ? document : {}];
    }
    return [[], {}];
  }
  function previewTrack(row, header) {
    const name = row.policy ?? row.profile ?? header.policy ?? header.profile;
    const id = row.run_id ?? header.run_id;
    return {policy: previewLabel(name) ? name : null, runId: previewLabel(id) ? id : null};
  }
  async function previewTrace(file) {
    let raw;
    try { raw = await readText(file, trace.MAX_BYTES); } catch (_) { return []; }
    const [rows, header] = previewRows(raw), games = new Map();
    const headerTime = reportedTime(header);
    for (const row of rows) {
      if (!obj(row)) continue;
      const seed = previewSeed(row.seed ?? header.seed), turns = previewNumber(row.turn, trace.MAX_TURNS);
      if (seed === null || !turns) continue;
      const labels = previewTrack(row, header), policy = labels.policy ?? 'native', runId = labels.runId;
      const key = JSON.stringify([seed, policy, runId]), previous = games.get(key);
      if (previous ? turns < previous.turns : games.size >= trace.MAX_TRACKS) continue;
      const tile = previewTile(row), milestones = previous?.milestones ?? {};
      const consecutive = previous ? previous.consecutive && (turns === previous.turns || turns === previous.turns + 1) : turns === 1;
      const milestonePrefix = consecutive && (tile !== null || turns === previous?.turns);
      if (milestonePrefix && tile !== null) {
        for (const target of MILESTONE_TILES) {
          if (tile >= target && milestones[target] === undefined) milestones[target] = turns;
        }
      }
      games.set(key, {seed, turns, policy, runId, score: previewNumber(row.score) ?? previous?.score ?? null,
        max_tile: tile ?? previous?.max_tile ?? null, time: reportedTime(row) ?? previous?.time ?? headerTime,
        milestones, consecutive: milestonePrefix});
    }
    return [...games.values()];
  }
  const needsMilestones = game => !game.turns || game.score === null || !game.max_tile || game.max_tile >= MILESTONE_TILES[0];
  function reconcilePreview(game, tracks) {
    if (!needsMilestones(game)) return game;
    const matches = tracks.filter(track => track.seed === game.seed &&
      (!game.policy || game.policy === track.policy) && (!game.runId || game.runId === track.runId) &&
      (game.turns === null || game.turns === track.turns) && (game.score === null || game.score === track.score) &&
      (game.max_tile === null || game.max_tile === track.max_tile));
    if (matches.length !== 1) return game;
    const [track] = matches;
    return {...game, policy: track.policy, runId: track.runId, milestones: track.milestones,
      turns: game.turns ?? track.turns, score: game.score ?? track.score, max_tile: game.max_tile ?? track.max_tile,
      time: game.time ?? track.time};
  }
  async function describeReplay(entry) {
    const file = entry.file, modified = file.lastModified;
    const modifiedTime = Number.isFinite(modified) && modified > 0 && modified <= 8640000000000000
      ? new Date(modified).toISOString() : null;
    const dir = dirname(entry.path), parent = basename(dir), arm = basename(dirname(dir));
    const fallback = parent && parent !== 'traces' ? parent : basename(entry.path).replace(/\.[^.]+$/, '');
    const [run, result, identity] = !entry.disabled && !entry.error
      ? await Promise.all([entry.runFile, entry.resultFile, entry.identityFile].map(previewObject)) : [];
    const title = (result?.split === 'gameplay_dev' && previewArm(result.arm)) ||
      (identity?.metadata?.split === 'gameplay_dev' && previewArm(identity.metadata.arm)) ||
      previewLabel(run?.variant) || previewLabel(run?.label) || previewLabel(run?.model?.profile);
    const runTime = reportedTime(result) || reportedTime(run) || reportedTime(identity) || reportedTime(identity?.metadata);
    const summary = result?.split === 'gameplay_dev' ? [result] : result?.games;
    const seen = new Set();
    let games = (Array.isArray(summary) ? summary.slice(0, trace.MAX_TRACKS) : []).flatMap(row => {
      if (!obj(row)) return [];
      const seed = previewSeed(row.seed);
      if (seed === null || seen.has(seed)) return [];
      seen.add(seed);
      return [{seed, turns: previewNumber(row.turn, trace.MAX_TURNS), score: previewNumber(row.score),
        max_tile: previewTile(row), time: reportedTime(row), ...previewTrack(row, result)}];
    });
    if (!entry.disabled && !entry.error && (!games.length || games.some(needsMilestones))) {
      const observed = await previewTrace(file);
      games = games.length ? games.map(game => reconcilePreview(game, observed)) : observed;
    }
    const known = new Set(games.map(game => game.seed));
    for (const value of Array.isArray(run?.seeds) ? run.seeds.slice(0, trace.MAX_TRACKS) : [identity?.seed]) {
      const seed = previewSeed(value);
      if (games.length >= trace.MAX_TRACKS) break;
      if (seed !== null && !known.has(seed)) { games.push({seed}); known.add(seed); }
    }
    if (!games.length) games.push({seed: null});
    return games.map(game => {
      const {seed, turns = null, score = null, max_tile = null, policy = null, runId = null, milestones = {}} = game;
      const model = title || (parent === String(seed) && previewArm(arm)) ||
        (policy !== 'native' && policy) || fallback || '未命名模型';
      const time = game.time ?? runTime;
      const single = result?.split === 'gameplay_dev' && result.seed === seed && ARM.test(result.arm ?? '') &&
        /^[0-9a-f]{64}$/.test(result.run_identity ?? '') && /^[0-9a-f]{64}$/.test(result.moves_sha256 ?? '');
      return {model, seed, turns, score, max_tile, policy, runId, milestones,
        ...(single ? {replayIdentity: [result.run_identity,result.arm,seed,result.moves_sha256].join('/')} : {}),
        time: time ?? modifiedTime, timeSource: time ? 'reported' : modifiedTime ? 'modified' : null};
    });
  }
  const api = {pathOf, isTraceCandidate, describeFiles, describeReplay, loadRun, MAX_METADATA_BYTES};
  root.KevRuns = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis === 'object' ? globalThis : this);
