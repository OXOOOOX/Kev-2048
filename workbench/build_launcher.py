"""Bundle the read-only monitor and local-file replay as one browser workspace."""
from pathlib import Path

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
css = (LAUNCH / "controls.css").read_text(encoding="utf-8")
js = (LAUNCH / "controls.js").read_text(encoding="utf-8")
assert "</script" not in js.lower()
tags = {
 "connect-src 'none';": "connect-src 'self';",
 "<title>Kev 2048 即时回看</title>": "<title>Kev 2048 本地工作台</title>",
 "模型实验 / 即时回看": "模型实验 / 本地工作台",
 "</style>": css + "</style>",
 "</header>": '''</header>
<nav class="arcade-nav" aria-label="工作台模式"><div role="tablist" aria-label="工作台模式">
<button type="button" role="tab" id="arcade-monitor-tab" aria-selected="true" aria-controls="arcade-monitor-pane">监控总览</button>
<button type="button" role="tab" id="arcade-review-tab" aria-selected="false" aria-controls="arcade-review-pane" tabindex="-1">即时回看</button>
</div></nav>
<section id="arcade-monitor-pane" class="monitor-pane" role="tabpanel" aria-labelledby="arcade-monitor-tab">
<div class="monitor-header"><div><h1>训练与推理监控</h1><p id="monitor-description">只读查看实际来源的最近进展。</p></div>
<span class="pill" id="monitor-connection" role="status">尚未联接</span></div>
<div class="monitor-filters" aria-label="筛选观察来源">
<button type="button" data-monitor-filter="all" aria-pressed="true">全部</button>
<button type="button" data-monitor-filter="training" aria-pressed="false">训练</button>
<button type="button" data-monitor-filter="inference" aria-pressed="false">推理</button>
<button type="button" data-monitor-filter="benchmark" aria-pressed="false">测速</button>
</div>
<div id="monitor-sources"><div class="card monitor-empty">尚未收到观察来源。</div></div>
</section>
<div id="arcade-review-pane" role="tabpanel" aria-labelledby="arcade-review-tab" hidden>''',
 "</main>": "</div>\n</main>",
 "</body>": "<script>\n" + js + "\n</script>\n</body>",
}
for source, target in tags.items():
    if base.count(source) != 1:
        raise ValueError(f"Unrecognized base HTML marker: {source}")
    base = base.replace(source, target)
out = HERE / "index.html"
out.write_bytes(base.encode("utf-8"))
print(out.name, out.stat().st_size)
