'use strict';
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path');
const {createHash} = require('node:crypto');
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
const sha256 = data => createHash('sha256').update(data).digest('hex');
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
function gameplayFixture(arm = 'official', moves, tail) {
  if (!moves) {
    const rows = nativeRows.map(({type, run_id, seq, board, tile, decision_ms, ...row}) =>
      ({...row, after: board, max_tile: tile, stop: null}));
    rows.at(-1).stop = 'capped';
    tail = rows.at(-1);
    moves = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
  }
  const {seed, turn, score, max_tile, stop} = tail;
  const cap = stop === 'capped' ? turn : 5000;
  const metadata = {arm, split: 'gameplay_dev', run_identity: hashB, checkpoint_head_sha256: hashA,
    device: 'cuda', dtype: 'fp32', temperature: 1};
  return {moves, result: {...metadata, schema_version: 1, seed, cap, turn, score, max_tile, stop,
    moves_sha256: sha256(moves)}, identity: {schema_version: 1, seed, cap, metadata}};
}
function gameplayFiles(fixture, dir = '') {
  return [file(dir + 'moves.jsonl', fixture.moves), file(dir + 'result.json', fixture.result),
    file(dir + 'identity.json', fixture.identity)];
}
async function load(files, options) { return runs.loadRun(runs.describeFiles(files, options)[0]); }
async function preview(files) { return runs.describeReplay(runs.describeFiles(files)[0]); }
function expectedMilestones(game) {
  return Object.fromEntries([512, 1024, 2048].flatMap(tile => {
    const frame = game.frames.find(row => row.m >= tile);
    return frame ? [[tile, frame.p]] : [];
  }));
}
async function test(name, fn) { await fn(); checks++; }
(async () => {
  await test('folder previews use one-row and multi-game summaries without loading moves', async () => {
    const reported = '2026-01-31T20:00:00+08:00';
    const fixture = gameplayFixture(); fixture.result.ended_at_utc = reported;
    const files = gameplayFiles(fixture, `runs/gameplay/official/${game.seed}/`), before = reads;
    const [one] = await preview(files);
    assert.deepEqual(one, {model:'官方基线', seed:game.seed, turns:end.p, score:end.s, max_tile:end.m,
      replayIdentity:[hashB,'official',game.seed,fixture.result.moves_sha256].join('/'),
      time:'2026-01-31T12:00:00.000Z', timeSource:'reported', policy:null, runId:null, milestones:{}});
    assert.equal(reads, before + 2);
    const result = summary(), run = manifest();
    result.games.push({seed:25123, turn:40, score:256, tile:32});
    const two = await preview(bundle(result, run));
    assert.deepEqual(two.map(row => [row.model, row.seed, row.turns, row.score, row.max_tile]), [
      ['Stock 训练版', game.seed, end.p, end.s, end.m], ['Stock 训练版', 25123, 40, 256, 32],
    ]);
    assert.equal(two[1].timeSource, 'modified');
    assert.equal(reads, before + 4);
  });
  await test('trace previews preserve model choices on the same seed', async () => {
    const rows = ['first', 'second'].flatMap(policy => nativeRows.map(row => ({...row, policy})));
    const files = [file('traces/stock.jsonl', rows.map(row => JSON.stringify(row)).join('\n'))];
    const listed = await preview(files);
    assert.deepEqual(listed.map(row => [row.model, row.seed, row.policy, row.runId, row.score]), [
      ['first', game.seed, 'first', nativeRows[0].run_id, end.s],
      ['second', game.seed, 'second', nativeRows[0].run_id, end.s],
    ]);
    assert.deepEqual((await load(files)).games.map(row => row.policy), ['first', 'second']);
  });
  await test('milestones follow exact trace tracks and require an observed opening', async () => {
    const times = (policy, run_id, tiles) => tiles.map((tile, i) => ({
      seed:123, policy, run_id, turn:i+1, score:16000, max_tile:tile,
    }));
    const rows = [...times('first', 'one', [4, 512, 512, 1024, 2048, 2048]),
      ...times('first', 'two', [4, 256, 512, 1024, 2048, 2048]),
      ...times('second', 'one', [4, 256, 256, 512, 1024, 2048]),
      ...times('partial', 'three', [4, 256, 512, 1024, 2048]).slice(1)];
    const moves = file('runs/a/moves.jsonl', rows.map(row => JSON.stringify(row)).join('\n'));
    assert.deepEqual((await preview([moves])).map(row => row.milestones), [
      {512:2, 1024:4, 2048:5}, {512:3, 1024:4, 2048:5}, {512:4, 1024:5, 2048:6}, {},
    ]);
    const tail = {seed:123, turn:6, score:16000, tile:2048};
    const withResult = row => [moves, file('runs/a/result.json', {games:[row]})];
    assert.deepEqual((await preview(withResult(tail)))[0].milestones, {});
    const [specific] = await preview(withResult({...tail, policy:'first', run_id:'two'}));
    assert.deepEqual([specific.policy, specific.runId, specific.milestones], ['first', 'two', {512:3, 1024:4, 2048:5}]);
    assert.deepEqual((await preview(withResult({...tail, policy:'first', run_id:'two', score:16004})))[0].milestones, {});
    const [unknown] = await preview(withResult({seed:123, policy:'first', run_id:'one'}));
    assert.deepEqual([unknown.turns, unknown.score, unknown.max_tile, unknown.milestones],
      [6, 16000, 2048, {512:2, 1024:4, 2048:5}]);
  });
  await test('large summary-free traces include early games, and retain unobserved manifest seeds', async () => {
    const timestamp = '2026-01-30T23:09:31.000Z';
    const first = {seed:123, turn:490, tile:2048, score:20320, timestamp};
    const later = nativeRows.map(row => ({...row, tile:undefined, detail:'x'.repeat(600 * 1024)}));
    const source = file('runs/a/moves.jsonl', [first, ...later].map(row => JSON.stringify(row)).join('\n'));
    assert(source.size > 2 * 1024 * 1024 && source.size < trace.MAX_BYTES);
    const manifestSeeds = {...manifest(), seeds:[123, game.seed, 456]};
    const rows = await preview([source, file('runs/a/run.json', manifestSeeds)]);
    assert.deepEqual(rows.map(row => [row.seed, row.turns, row.max_tile, row.timeSource]), [
      [123, 490, 2048, 'reported'], [game.seed, end.p, end.m, 'modified'], [456, null, null, 'modified'],
    ]);
    assert.equal(rows[0].time, timestamp);
  });
  await test('preview errors are recoverable and loading still validates the trace', async () => {
    const files = [file('runs/unreadable/moves.jsonl', 'invalid', {lastModified:0}),
      file('runs/unreadable/result.json', '{')];
    assert.deepEqual(await preview(files), [{model:'unreadable', seed:null, turns:null, score:null,
      max_tile:null, time:null, timeSource:null, policy:null, runId:null, milestones:{}}]);
    await assert.rejects(load(files), /moves\.jsonl/);
  });
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
  await test('single gameplay summaries and identity companions import in both modes', async () => {
    const fixture = gameplayFixture();
    for (const [dir, direct] of [['runs/gameplay/official/' + game.seed + '/', false], ['', true]]) {
      const files = gameplayFiles(fixture, dir), before = reads;
      const entries = runs.describeFiles(files, {direct});
      assert.equal(reads, before);
      assert.equal(runs.isTraceCandidate(files[2]), false);
      assert.equal(entries.length, 1);
      assert.equal(entries[0].path, dir + 'moves.jsonl');
      assert.equal(entries[0].bytes, files.reduce((n, f) => n + f.size, 0));
      const loaded = await runs.loadRun(entries[0]);
      assert.equal(loaded.title, '官方基线');
      assert.equal(loaded.headSha, hashA);
      assert.equal(loaded.device, 'cuda');
      assert.equal(loaded.games[0].runTitle, loaded.title);
      assert.equal(loaded.games[0].status, 'cap');
      assert.equal(loaded.games[0].frames.length, 5);
    }
    const files = gameplayFiles(fixture, 'runs/gameplay/official/' + game.seed + '/');
    assert.equal((await load(files.slice(0, 2))).title, '官方基线');
    const config = file('runs/gameplay/implementation/configs/gameplay-head-2000.json', {seed_group_size: 8});
    assert.equal(runs.describeFiles([...files, config]).length, 1);
    const foreign = file('runs/other/identity.json', fixture.identity);
    const foreignEntries = runs.describeFiles([files[0], foreign]);
    assert.equal(foreignEntries.length, 1);
    assert.equal((await runs.loadRun(foreignEntries[0])).headSha, null);
  });
  await test('all three models retain their names when viewing the same seed', async () => {
    const models = [['official', '官方基线'], ['lr8e-5-kl003', 'lr8e-5-kl003'], ['lr8e-5-kl015', 'lr8e-5-kl015']];
    const folders = models.flatMap(([arm]) =>
      gameplayFiles(gameplayFixture(arm), 'runs/gameplay/' + arm + '/' + game.seed + '/'));
    const loaded = await Promise.all(runs.describeFiles(folders).map(runs.loadRun));
    assert.deepEqual(loaded.map(run => run.title), models.map(([, title]) => title));
    for (const [arm, title] of models) {
      const [moves, , identity] = gameplayFiles(gameplayFixture(arm));
      const identityOnly = await load([moves, identity], {direct: true});
      assert.equal(identityOnly.title, title);
      assert.equal(identityOnly.headSha, hashA);
      const movePath = 'runs/gameplay/' + arm + '/' + game.seed + '/moves.jsonl';
      assert.equal((await load([file(movePath, gameplayFixture(arm).moves)])).title, title);
    }
  });
  await test('single gameplay summaries validate input identity, tail, caps and move file digest', async () => {
    const dir = 'runs/gameplay/official/' + game.seed + '/';
    for (const field of ['seed', 'turn', 'score', 'max_tile', 'cap']) {
      const fixture = gameplayFixture();
      fixture.result[field]++;
      await assert.rejects(load(gameplayFiles(fixture, dir)), /result\.json.*不一致|result\.json.*冲突/);
    }
    for (const [field, value] of [['checkpoint_head_sha256', hashB], ['arm', 'wrong-arm'], ['run_identity', hashA]]) {
      const fixture = gameplayFixture();
      fixture.result[field] = value;
      await assert.rejects(load(gameplayFiles(fixture, dir)), /result\.json.*不一致|result\.json.*冲突/);
    }
    const fixture = gameplayFixture(), files = gameplayFiles(fixture, dir);
    files[0] = file(dir + 'moves.jsonl', '\n' + fixture.moves);
    await assert.rejects(load(files), /SHA-256|哈希|校验|指纹/);
    fixture.result.stop = 'target';
    await assert.rejects(load(gameplayFiles(fixture, dir)), /目标|2048|结束/);
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
  const calibration = path.join(__dirname, '..', 'runs', 'preparation-20261123', 'calibration', 'd3', '2611230000', 'moves.jsonl');
  if (fs.existsSync(calibration)) {
    await test('first 2048 target is preserved as the end of the replay', async () => {
      const moves = fs.readFileSync(calibration, 'utf8'), tail = JSON.parse(moves.trimEnd().split('\n').at(-1));
      assert.equal(tail.stop, 'target');
      const loaded = await load(gameplayFiles(gameplayFixture('official', moves, tail), 'runs/target/'));
      assert.equal(loaded.games[0].status, 'target');
      assert.equal(loaded.games[0].frames.at(-1).m, 2048);
      assert(loaded.games[0].frames.slice(0, -1).every(frame => frame.m < 2048));
      const following = sample.trimEnd().split('\n').map(line =>
        JSON.stringify({...JSON.parse(line), detail:'x'.repeat(600 * 1024)}));
      const combined = [file('traces/moves.jsonl', [moves.trimEnd(), ...following].join('\n'))];
      assert(combined[0].size > 2 * 1024 * 1024);
      const listed = await preview(combined);
      assert.deepEqual(listed.map(row => [row.seed, row.max_tile]), [[tail.seed, 2048], [game.seed, end.m]]);
      assert.deepEqual(listed[0].milestones, expectedMilestones(loaded.games[0]));
      assert.deepEqual((await load(combined)).games.map(row => row.status), ['target', 'cap']);
    });
  }
  let gameplayRuns = 0, milestoneRuns = 0, originalFiles = 0;
  const evalDir = path.join(__dirname, '..', 'runs', 'gameplay-dev-head2000-20261123');
  const reportPath = path.join(evalDir, 'result.json');
  if (fs.existsSync(reportPath) && JSON.parse(fs.readFileSync(reportPath, 'utf8')).status === 'completed') {
    await test('all completed gameplay development traces import without changing originals', async () => {
      const originals = new Map();
      const readOriginal = diskPath => {
        const bytes = fs.readFileSync(diskPath);
        originals.set(diskPath, sha256(bytes));
        return bytes;
      };
      readOriginal(reportPath);
      const manifest = JSON.parse(readOriginal(path.join(evalDir, 'resolved-config.json')));
      assert.equal(manifest.seeds.length, 100);
      const arms = Object.keys(manifest.checkpoints);
      assert.equal(arms.length, 3);
      for (const arm of arms) for (const seed of manifest.seeds) {
        const [moves, result, identity] = ['moves.jsonl', 'result.json', 'identity.json'].map(base => {
          const relative = path.join(arm, String(seed), base);
          return file('runs/gameplay-dev-head2000-20261123/' + relative.replaceAll('\\', '/'),
            readOriginal(path.join(evalDir, relative)).toString('utf8'));
        });
        const summary = JSON.parse(await result.text());
        const entries = runs.describeFiles([moves, result, identity]);
        assert.equal(entries.length, 1);
        const loaded = await runs.loadRun(entries[0]);
        assert.equal(loaded.title, arm === 'official' ? '官方基线' : arm);
        assert.equal(loaded.headSha, summary.checkpoint_head_sha256);
        assert.equal(loaded.device, summary.device);
        assert.equal(loaded.games.length, 1);
        const game = loaded.games[0], end = game.frames.at(-1);
        assert.equal(game.seed, seed);
        assert.equal(game.status, summary.stop === 'capped' ? 'cap' : summary.stop);
        assert.deepEqual([end.p, end.s, end.m], [summary.turn, summary.score, summary.max_tile]);
        const before = reads, [listed] = await runs.describeReplay(entries[0]);
        assert.deepEqual(listed.milestones, expectedMilestones(game));
        const eligible = summary.max_tile >= 512;
        assert.equal(reads - before, eligible ? 3 : 2);
        milestoneRuns += Number(eligible);
        gameplayRuns++;
      }
      assert.equal(gameplayRuns, 300);
      for (const [diskPath, before] of originals) assert.equal(sha256(fs.readFileSync(diskPath)), before, diskPath);
      originalFiles = originals.size;
    });
  }
  console.log('PASS run imports: ' + checks + ' scenarios; ' + localRuns + ' local CUDA bundles; ' +
    gameplayRuns + ' gameplay development bundles, ' + milestoneRuns + ' with milestones; ' + originalFiles + ' originals byte-identical.');
})().catch(error => {console.error(error); process.exitCode = 1;});
