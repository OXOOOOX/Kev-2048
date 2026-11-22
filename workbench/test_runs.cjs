'use strict';
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const trace = require('./launcher/parser.js'), runs = require('./launcher/runs.js');
const sample = fs.readFileSync(path.join(__dirname, 'samples/sample-events.jsonl'), 'utf8');
const nativeRows = sample.trim().split('\n').map(line => {
  const row = JSON.parse(line);
  delete row.cap; delete row.stop; delete row.profile;
  return row;
});
const native = nativeRows.map(row => JSON.stringify(row)).join('\n');
const game = trace.parseTrace(native)[0], end = game.frames.at(-1);
const hashA = 'a'.repeat(64), hashB = 'b'.repeat(64);
const summary = () => ({checkpoint_head_sha256: hashA, device: 'cuda', cap: 4,
  games: [{seed: game.seed, turn: end.p, score: end.s, tile: end.m, stop: 'cap'}]});
const manifest = () => ({schema_version: 1, kind: 'play', variant: 'Stock 训练版', status: 'planned',
  files: {moves: 'moves.jsonl', result: 'result.json'}, seeds: [game.seed], max_moves_per_game: 4,
  device: 'cuda', model: {head_sha256: hashA}});
let reads = 0, checks = 0;
function file(p, content, extra = {}) {
  const text = typeof content === 'string' ? content : JSON.stringify(content);
  return {name: p.split('/').at(-1), webkitRelativePath: p.includes('/') ? p : '',
    size: Buffer.byteLength(text), lastModified: 1, text: async () => {reads++; return text;}, ...extra};
}
function bundle(result = summary(), run = manifest()) {
  return [file('runs/a/moves.jsonl', native), ...(result ? [file('runs/a/result.json', result)] : []),
    ...(run ? [file('runs/a/run.json', run)] : [])];
}
async function load(files, options) { return runs.loadRun(runs.describeFiles(files, options)[0]); }
async function test(name, fn) { await fn(); checks++; }
(async () => {
  await test('enumeration is lazy and budgets include companions', async () => {
    const files = bundle(), before = reads, [entry] = runs.describeFiles(files);
    assert.equal(reads, before);
    assert.equal(entry.bytes, files.reduce((n, f) => n + f.size, 0));
    assert.equal(entry.resultFile, files[1]); assert.equal(entry.runFile, files[2]);
    assert.equal(entry.disabled, false); assert.equal(entry.path, 'runs/a/moves.jsonl');
    const loaded = await runs.loadRun(entry);
    assert.equal(reads, before + 3); assert.equal(loaded.title, 'Stock 训练版');
    assert.equal(loaded.headSha, hashA); assert.equal(loaded.device, 'cuda');
    assert.equal(loaded.games[0].status, 'cap'); assert.equal(loaded.games[0].runTitle, loaded.title);
    assert.equal(loaded.games[0].frames.length, 5);
  });
  await test('native traces and explicit ends remain compatible without summaries', async () => {
    const unknown = await load([file('runs/old/moves.jsonl', native)]);
    assert.equal(unknown.games[0].status, 'incomplete'); assert.equal(unknown.title, 'old');
    assert.equal(unknown.headSha, null); assert.equal(unknown.device, null);
    const recorded = await load([file('sample-events.jsonl', sample)]);
    assert.equal(recorded.games[0].status, 'cap'); assert.equal(recorded.title, 'frozen_v2');
    const onlyManifest = await load(bundle(null));
    assert.equal(onlyManifest.games[0].status, 'incomplete'); // planned/status never manufactures a stop
  });
  await test('valid JSON and unknown direct filenames are discoverable', async () => {
    assert.equal(runs.isTraceCandidate(file('runs/x/replay.json', nativeRows)), true);
    assert.equal(runs.isTraceCandidate(file('runs/x/moves.json', nativeRows)), true);
    assert.equal(runs.isTraceCandidate(file('runs/x/benchmark.json', {})), false);
    assert.equal(runs.isTraceCandidate(file('runs/x/progress.jsonl', '{}')), false);
    assert.equal(runs.isTraceCandidate(file('runs/x/run.json', {})), false);
    assert.equal(runs.describeFiles([file('custom.json', nativeRows)]).length, 0);
    const custom = await load([file('custom.json', nativeRows)], {direct: true});
    assert.equal(custom.games.length, 1);
    assert.equal((await load([file('runs/x/replay.json', nativeRows)])).games[0].frames.length, 5);
  });
  await test('same basename imports are safe only without ambiguous companions', async () => {
    const two = [file('moves.jsonl', native), file('moves.jsonl', native)];
    const entries = runs.describeFiles(two, {direct: true});
    assert.equal(entries.length, 2); assert(entries.every(e => !e.error));
    await Promise.all(entries.map(runs.loadRun));
    const withSummary = runs.describeFiles([...two, file('result.json', summary())], {direct: true});
    const before = reads;
    await assert.rejects(runs.loadRun(withSummary[0]), /moves\.jsonl.*无法唯一关联/);
    assert.equal(reads, before);
    const mixed = runs.describeFiles([...two, file('result.json', summary()), file('runs/good/moves.jsonl', native)]);
    assert.equal(mixed.length, 3); assert.equal(mixed[2].error, null);
    await runs.loadRun(mixed[2]);
  });
  await test('metadata is only paired within the chosen directory', async () => {
    const [entry] = runs.describeFiles([file('runs/a/moves.jsonl', native), file('runs/b/result.json', summary())]);
    assert.equal(entry.resultFile, null);
    const wrong = manifest(); wrong.files.moves = '../a/moves.jsonl';
    await assert.rejects(load(bundle(summary(), wrong)), /runs\/a\/run\.json.*同目录/);
    const renamed = manifest(); renamed.files.moves = 'different.jsonl';
    await assert.rejects(load(bundle(summary(), renamed)), /files\.moves.*不一致/);
  });
  await test('summary tail values and seed types must match', async () => {
    for (const field of ['turn', 'score', 'tile']) {
      const result = summary(); result.games[0][field]++;
      await assert.rejects(load(bundle(result)), /runs\/a\/result\.json.*不一致/);
    }
    const wrongSeed = summary(); wrongSeed.games[0].seed = String(game.seed);
    await assert.rejects(load(bundle(wrongSeed)), /不一致/);
    const missing = summary(); missing.games = [];
    await assert.rejects(load(bundle(missing)), /对局数/);
    const extra = summary(); extra.games.push({...extra.games[0], seed: 123});
    await assert.rejects(load(bundle(extra)), /对局数/);
  });
  await test('end reasons cannot override the board or recorded endings', async () => {
    const wrongTerminal = summary(); wrongTerminal.games[0].stop = 'terminal';
    await assert.rejects(load(bundle(wrongTerminal)), /结束原因与棋盘冲突/);
    const wrongCap = summary(); wrongCap.cap = 5;
    await assert.rejects(load(bundle(wrongCap, null)), /结束步数不是已报告上限/);
    const noCap = summary(); delete noCap.cap;
    await assert.rejects(load(bundle(noCap, null)), /结束步数不是已报告上限/);
    const stopped = summary(); stopped.games[0].stop = 'stopped';
    assert.equal((await load(bundle(stopped))).games[0].status, 'stopped');
    const files = bundle(stopped); files[0] = file('runs/a/moves.jsonl', sample);
    await assert.rejects(load(files), /结束原因与逐步记录冲突/);
    const unknown = summary(); unknown.games[0].stop = 'target';
    await assert.rejects(load(bundle(unknown)), /有效结束原因/);
  });
  await test('hashes and manifest facts must agree', async () => {
    const mismatched = manifest(); mismatched.model.head_sha256 = hashB;
    await assert.rejects(load(bundle(summary(), mismatched)), /指纹.*冲突/);
    const upper = manifest(); upper.model.head_sha256 = hashA.toUpperCase();
    assert.equal((await load(bundle(summary(), upper))).headSha, hashA);
    const invalid = manifest(); invalid.model.head_sha256 = 'unknown';
    await assert.rejects(load(bundle(summary(), invalid)), /SHA-256/);
    const seeds = manifest(); seeds.seeds = [1];
    await assert.rejects(load(bundle(summary(), seeds)), /清单种子/);
    const device = manifest(); device.device = 'cpu';
    await assert.rejects(load(bundle(summary(), device)), /设备.*冲突/);
    const version = manifest(); version.schema_version = 2;
    await assert.rejects(load(bundle(summary(), version)), /schema_version/);
    const template = JSON.parse(fs.readFileSync(path.join(__dirname, 'run.template.json'), 'utf8'));
    assert.equal((await load(bundle(summary(), template))).games[0].status, 'cap');
  });
  await test('duplicate seed policies never silently inherit a single summary', async () => {
    const multiple = [...nativeRows.map(r => ({...r, policy: 'first'})), ...nativeRows.map(r => ({...r, policy: 'second'}))];
    const result = summary(); result.games.push({...result.games[0], seed: 999});
    const files = bundle(result, null); files[0] = file('runs/a/moves.jsonl', multiple);
    await assert.rejects(load(files), /多条策略轨道/);
    result.games[1].seed = game.seed;
    const duplicate = bundle(result, null); duplicate[0] = file('runs/a/moves.jsonl', multiple);
    await assert.rejects(load(duplicate), /重复种子/);
    const runOnly = bundle(null); runOnly[0] = file('runs/a/moves.jsonl', multiple);
    await assert.rejects(load(runOnly), /多条策略轨道/);
  });
  await test('run-only identity and limits cannot contradict trace metadata', async () => {
    const wrongRun = manifest(); wrongRun.run_id = 'different-run';
    await assert.rejects(load(bundle(null, wrongRun)), /运行编号.*冲突/);
    const rightRun = manifest(); rightRun.run_id = nativeRows[0].run_id;
    assert.equal((await load(bundle(null, rightRun))).games[0].runId, nativeRows[0].run_id);
    const wrongLimit = manifest(); wrongLimit.max_moves_per_game = 8;
    const files = bundle(null, wrongLimit); files[0] = file('runs/a/moves.jsonl', sample);
    await assert.rejects(load(files), /上限.*冲突|结束步数.*冲突/);
    const declaredEight = nativeRows.map(row => ({...row, cap: 8}));
    const mismatch = bundle(); mismatch[0] = file('runs/a/moves.jsonl', declaredEight);
    await assert.rejects(load(mismatch), /上限.*冲突/);
  });
  await test('size guards and read errors identify the full relative path', async () => {
    const bad = file('runs/broken/moves.jsonl', 'not JSON');
    await assert.rejects(load([bad]), /runs\/broken\/moves\.jsonl/);
    const oversize = file('runs/a/run.json', {}, {size: runs.MAX_METADATA_BYTES + 1});
    const [entry] = runs.describeFiles([file('runs/a/moves.jsonl', native), oversize]);
    assert.equal(entry.disabled, true); assert.match(entry.disabledReason, /runs\/a\/run\.json/);
    await assert.rejects(runs.loadRun(entry), /run\.json.*超过 1 MiB/);
    const empty = runs.describeFiles([file('runs/empty/moves.jsonl', '')])[0];
    assert.equal(empty.disabled, true);
    const invalid = bundle(); invalid[1] = file('runs/a/result.json', '{');
    await assert.rejects(load(invalid), /runs\/a\/result\.json.*完整 JSON/);
  });
  let localRuns = 0;
  for (const name of ['official-kev', 'stock-01-cuda']) {
    const dir = path.join(__dirname, '..', 'runs', 'stock-gameplay-cuda', name);
    if (!fs.existsSync(path.join(dir, 'moves.jsonl')) || !fs.existsSync(path.join(dir, 'result.json'))) continue;
    await test('existing CUDA run ' + name, async () => {
      const files = ['moves.jsonl', 'result.json'].map(base => file('runs/' + name + '/' + base, fs.readFileSync(path.join(dir, base), 'utf8')));
      const loaded = await load(files);
      assert.equal(loaded.games.length, 5); assert(loaded.games.every(g => g.status === 'terminal'));
      assert.equal(loaded.device, 'cuda'); assert.match(loaded.headSha, /^[a-f\d]{64}$/);
      assert.equal(loaded.title, name);
    });
    localRuns++;
  }
  console.log('PASS run imports: ' + checks + ' scenarios; ' + localRuns + ' local CUDA bundles; lazy enumeration, identity/tail/stop matching, ambiguity rejection, legacy and JSON support.');
})().catch(error => {console.error(error); process.exitCode = 1;});
