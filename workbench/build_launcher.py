"""Bundle monitoring, local replay and live inference as one browser workspace."""
from pathlib import Path

if __package__:
    from .build_tile_materials import BUNDLE_END, BUNDLE_START, material_styles, refresh_workbench_styles
else:
    from build_tile_materials import BUNDLE_END, BUNDLE_START, material_styles, refresh_workbench_styles

HERE = Path(__file__).resolve().parent
LAUNCH = HERE / "launcher"
base = (LAUNCH / "replay-template.html").read_text(encoding="utf-8")
parser = (LAUNCH / "parser.js").read_text(encoding="utf-8")
runs = (LAUNCH / "runs.js").read_text(encoding="utf-8")
app = (LAUNCH / "app.js").read_text(encoding="utf-8")
motion = (LAUNCH / "motion.js").read_text(encoding="utf-8")
for script in (parser, runs, app, motion):
    if "</script" in script.lower():
        raise ValueError("Unexpected HTML in application source")
if app.count("/* SHARED_MOTION */") != 1:
    raise ValueError("Missing or duplicate replay animation placeholder")
if any(base.count(key) != 1 for key in ("/* TRACE_PARSER */", "/* REPLAY_APP */")):
    raise ValueError("Missing or duplicate replay placeholders")
base = (base.replace("/* TRACE_PARSER */", parser + "\n" + runs)
            .replace("/* REPLAY_APP */", app.replace("/* SHARED_MOTION */", motion)))
refresh_workbench_styles()
css = (LAUNCH / "controls.css").read_text(encoding="utf-8")
css += "\n" + "\n".join((BUNDLE_START, material_styles(motion=True), BUNDLE_END)) + "\n"
js = (LAUNCH / "controls.js").read_text(encoding="utf-8")
play = (LAUNCH / "play.js").read_text(encoding="utf-8")
model_info = (LAUNCH / "play-model-info.js").read_text(encoding="utf-8")
if any("</script" in script.lower() for script in (js, play, model_info)):
    raise ValueError("Unexpected HTML in workspace controls")
for marker, script in (("/* SHARED_MOTION */", motion), ("/* PLAY_VIEW */", model_info + "\n" + play)):
    if js.count(marker) != 1:
        raise ValueError("Missing or duplicate workspace control placeholder: " + marker)
    js = js.replace(marker, script)
tags = {
 "connect-src 'none';": "connect-src 'self';",
 "<title>Kev 2048 即时回看</title>": "<title>Kev 2048 本地工作台</title>",
 "模型实验 / 即时回看": "模型实验 / 本地工作台",
 "</style>": css + "</style>",
 "</header>": '''</header>
<nav class="arcade-nav" aria-label="工作台模式"><div role="tablist" aria-label="工作台模式">
<button type="button" role="tab" id="arcade-monitor-tab" aria-selected="true" aria-controls="arcade-monitor-pane">训练控制台</button>
<button type="button" role="tab" id="arcade-review-tab" aria-selected="false" aria-controls="arcade-review-pane" tabindex="-1">回看</button>
<button type="button" role="tab" id="arcade-play-tab" aria-selected="false" aria-controls="arcade-play-pane" tabindex="-1">即时游玩</button>
</div></nav>
<section id="arcade-monitor-pane" class="monitor-pane" role="tabpanel" aria-labelledby="arcade-monitor-tab">
<div class="monitor-header"><div><h1>训练控制台</h1><p id="monitor-description">查看训练计划中的训练、测评、推理和测速进展。</p></div>
<span class="pill" id="monitor-connection" role="status">尚未联接</span></div>
<div class="monitor-filters" aria-label="筛选观察来源">
<button type="button" data-monitor-filter="all" aria-pressed="true">全部</button>
<button type="button" data-monitor-filter="training" aria-pressed="false">训练</button>
<button type="button" data-monitor-filter="evaluation" aria-pressed="false">测评</button>
<button type="button" data-monitor-filter="inference" aria-pressed="false">推理</button>
<button type="button" data-monitor-filter="benchmark" aria-pressed="false">测速</button>
</div>
<div id="monitor-sources"><div class="card monitor-empty">尚未收到观察来源。</div></div>
<div id="monitor-gameplay"></div>
</section>
<div id="arcade-review-pane" role="tabpanel" aria-labelledby="arcade-review-tab" hidden>''',
 "</main>": '''</div>
<section id="arcade-play-pane" class="play-pane" role="tabpanel" aria-labelledby="arcade-play-tab" hidden>
<div class="monitor-header"><div><h1>即时游玩</h1><p>选一个或多个模型，共用同一个种子，各自推理落子。</p></div></div>
<form id="play-form" class="card play-form">
<fieldset id="play-selection" disabled><legend>模型</legend><div id="play-model-options" class="play-options"></div></fieldset>
<div class="play-actions">
  <label class="play-seed">种子（可选）<input id="play-seed" name="seed" type="number" min="0" max="2147483647" step="1" placeholder="留空随机生成" disabled></label>
  <button id="play-start" class="picker primary" type="submit" disabled>开始新局</button>
  <button id="play-stop" class="picker secondary" type="button" disabled>停止游玩</button>
</div>
<p id="play-feedback" class="monitor-meta play-feedback" role="status">等待连接本机推理服务。</p>
</form>
<section id="play-session-view" class="play-session" aria-labelledby="play-session-title" hidden>
<div class="play-session-header"><h2 id="play-session-title"></h2><span id="play-session-state" class="pill"></span>
  <button id="play-save" class="picker secondary play-save" type="button" aria-pressed="false" aria-describedby="play-save-note" disabled>☆ 星标保存</button>
</div>
<p id="play-save-note" class="monitor-meta play-save-note" role="status"></p>
<div class="card play-timeline" role="group" aria-label="对局时间轴">
  <div class="play-timeline-head">
    <output id="play-cursor" for="play-seek" aria-live="off"></output>
    <div class="play-timeline-actions">
      <button id="play-pause" class="picker secondary" type="button" disabled>暂停推理</button>
      <button id="play-latest" class="picker secondary" type="button" disabled>回到最新</button>
    </div>
  </div>
  <input id="play-seek" type="range" min="0" max="0" step="1" value="0" aria-label="选择回看步数" disabled>
  <div class="play-timeline-actions">
    <button id="play-prev" class="picker secondary" type="button" disabled>上一步</button>
    <button id="play-review" class="picker secondary" type="button" disabled>播放回看</button>
    <button id="play-next" class="picker secondary" type="button" disabled>下一步</button>
  </div>
</div>
<div id="play-boards" class="play-boards" data-tile-motion></div>
</section>
</section>
</main>''',
 "</body>": "<script>\n" + js + "\n</script>\n</body>",
}
for source, target in tags.items():
    if base.count(source) != 1:
        raise ValueError(f"Unrecognized base HTML marker: {source}")
    base = base.replace(source, target)
out = HERE / "index.html"
out.write_bytes(base.encode("utf-8"))
print(out.name, out.stat().st_size)
