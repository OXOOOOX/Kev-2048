/* Local run bundles: enumerate filenames first, read only an explicitly selected run. */
(function (root) {
  'use strict';
  const trace = root.KevTrace || (typeof require === 'function' ? require('./parser.js') : null);
  const MAX_METADATA_BYTES = 1024 * 1024;
  const SUMMARY = /(^|[-_.])(benchmark|bench|summary|progress|timing|metrics|train|dev|loss|report|lock)([-_.]|$)/;
  const obj = x => x !== null && typeof x === 'object' && !Array.isArray(x);
  const given = x => x !== null && x !== undefined;
  const pathOf = f => String(f.webkitRelativePath || f.name || '未命名文件').replaceAll('\\', '/');
  const basename = p => p.split('/').at(-1);
  const dirname = p => p.includes('/') ? p.slice(0, p.lastIndexOf('/')) : '';
  const isCompanion = f => /^(run|result)\.json$/i.test(basename(pathOf(f)));
  const isSummary = f => isCompanion(f) || SUMMARY.test(basename(pathOf(f)).toLowerCase());
  const sizeOK = (f, limit) => Number.isSafeInteger(f.size) && f.size > 0 && f.size <= limit;
  const errorAt = (file, message) => new Error('「' + pathOf(file) + '」：' + message);
  function isTraceCandidate(file) {
    const p = pathOf(file).toLowerCase(), pieces = p.split('/'), name = pieces.at(-1);
    if (!/\.(json|jsonl|ndjson)$/.test(name) || isSummary(file)) return false;
    if (/(^|[-_.])(moves?|traces?|replays?|events?|gameplay|games?)([-_.]|$)/.test(name)) return true;
    const stem = name.replace(/\.(json|jsonl|ndjson)$/, '');
    if (pieces.slice(0, -1).includes('traces') && /^(d4|wide|stock|[a-z0-9]+[-_][a-z0-9_-]+)$/.test(stem)) return true;
    return pieces.slice(0, -1).includes('live') && /^[0-9a-f]{32}$/.test(stem);
  }
  function describeFiles(input, {direct = false} = {}) {
    const files = Array.from(input), dirs = new Map();
    for (const file of files) {
      const dir = dirname(pathOf(file));
      if (!dirs.has(dir)) dirs.set(dir, {traces: [], results: [], runs: []});
      const group = dirs.get(dir), name = basename(pathOf(file)).toLowerCase();
      if (name === 'result.json') group.results.push(file);
      else if (name === 'run.json') group.runs.push(file);
      else if (direct ? /\.(json|jsonl|ndjson)$/i.test(name) && !isSummary(file) : isTraceCandidate(file)) group.traces.push(file);
    }
    return [...dirs.values()].flatMap(group => group.traces.map(file => {
      const resultFile = group.results[0] || null, runFile = group.runs[0] || null;
      const companions = [resultFile, runFile].filter(Boolean);
      const ambiguous = companions.length && (group.traces.length > 1 || group.results.length > 1 || group.runs.length > 1);
      const bad = !sizeOK(file, trace.MAX_BYTES) ? file : companions.find(f => !sizeOK(f, MAX_METADATA_BYTES));
      const limit = bad === file ? trace.MAX_BYTES : MAX_METADATA_BYTES;
      const disabledReason = bad ? pathOf(bad) + ' 为空或超过 ' + (limit / 1048576) + ' MiB 上限' : '';
      return {file, resultFile, runFile, path: pathOf(file),
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
  function reconcile(result, entry, games, info) {
    if (!result) return {games, headSha: info.headSha, device: info.device};
    const file = entry.resultFile;
    const headSha = hash(result.checkpoint_head_sha256, file, 'checkpoint_head_sha256');
    const device = label(result.device, file, 'device'), limit = cap(result.cap, file);
    if (headSha && info.headSha && headSha !== info.headSha) throw errorAt(file, '检查点指纹与运行清单冲突');
    if (device && info.device && device !== info.device) throw errorAt(file, '设备与运行清单冲突');
    if (limit && info.cap && limit !== info.cap) throw errorAt(file, '步数上限与运行清单冲突');
    if (!Array.isArray(result.games) || result.games.length !== games.length) throw errorAt(file, '摘要对局数与逐步记录不一致');
    const bySeed = new Map();
    for (const row of result.games) {
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
      if (!['terminal', 'cap', 'stopped'].includes(row.stop)) throw errorAt(file, '种子 ' + String(g.seed) + ' 缺少有效结束原因');
      const terminal = g.status === 'terminal';
      if ((row.stop === 'terminal') !== terminal) throw errorAt(file, '种子 ' + String(g.seed) + ' 的结束原因与棋盘冲突');
      if (g.status !== 'incomplete' && row.stop !== g.status) throw errorAt(file, '种子 ' + String(g.seed) + ' 的结束原因与逐步记录冲突');
      const maximum = limit || info.cap;
      if (maximum && given(g.cap) && g.cap !== maximum) throw errorAt(file, '种子 ' + String(g.seed) + ' 的步数上限与逐步记录冲突');
      if (maximum && end.p > maximum) throw errorAt(file, '种子 ' + String(g.seed) + ' 的步数超过摘要上限');
      if (row.stop === 'cap' && (!maximum || end.p !== maximum)) throw errorAt(file, '种子 ' + String(g.seed) + ' 的结束步数不是已报告上限');
      return {...g, status: row.stop};
    });
    return {games: merged, headSha: headSha || info.headSha, device: device || info.device};
  }
  async function loadRun(entry) {
    if (entry.error) throw errorAt(entry.file, entry.error);
    const raw = await readText(entry.file, trace.MAX_BYTES);
    let games;
    try { games = trace.parseTrace(raw, pathOf(entry.file)); }
    catch (error) { throw errorAt(entry.file, String(error.message || error)); }
    const run = entry.runFile ? await readObject(entry.runFile) : null;
    const result = entry.resultFile ? await readObject(entry.resultFile) : null;
    const info = validateRun(run, entry, games), resolved = reconcile(result, entry, games, info);
    const parent = basename(dirname(pathOf(entry.file)));
    const title = info.title || parent || (games.length === 1 && games[0].policy !== 'native' ? games[0].label : null) || basename(pathOf(entry.file));
    return {path: pathOf(entry.file), title, headSha: resolved.headSha, device: resolved.device,
      games: resolved.games.map(g => ({...g, runTitle: title, headSha: resolved.headSha, device: resolved.device}))};
  }
  const api = {pathOf, isTraceCandidate, describeFiles, loadRun, MAX_METADATA_BYTES};
  root.KevRuns = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis === 'object' ? globalThis : this);
