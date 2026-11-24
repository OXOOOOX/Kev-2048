"""Bundle the shared board artwork for the live stylesheet and portable, offline pages."""
import base64
from pathlib import Path
from tempfile import NamedTemporaryFile


HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "launcher"
BUNDLE_START = "/* Begin bundled tile materials */"
BUNDLE_END = "/* End bundled tile materials */"


def material_styles(*, motion):
    css = (LAUNCHER / "tile-materials.css").read_text(encoding="utf-8")
    if not motion:
        css = css.split("/* Moving reflections */", 1)[0]
    for name in ("tile-materials.webp", "tile-opal-base.webp", "tile-opal-light.webp"):
        reference = f'url("{name}")'
        if reference in css:
            data = base64.b64encode((LAUNCHER / name).read_bytes()).decode("ascii")
            css = css.replace(reference, f'url("data:image/webp;base64,{data}")')
    return css.strip()


def refresh_block(path, start, end, css):
    original = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    prefix, opening, remainder = original.partition(start.encode())
    _, closing, suffix = remainder.partition(end.encode())
    if not opening or not closing:
        raise ValueError(f"Missing material block in {path}")
    block = css.encode("utf-8").replace(b"\n", newline)
    updated = prefix + opening + newline + block + newline + closing + suffix
    if updated == original:
        return
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(updated)
        if path.read_bytes() != original:
            raise RuntimeError(f"{path} changed during the build")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def refresh_workbench_styles():
    # The running local server already exposes this stylesheet across workbench modes.
    refresh_block(LAUNCHER / "studio-monitor.css", BUNDLE_START, BUNDLE_END, material_styles(motion=True))
    batches = HERE / "training-batches.html"
    if batches.is_file():
        refresh_block(batches, '<style id="tile-materials">', "</style>", material_styles(motion=False))


if __name__ == "__main__":
    refresh_workbench_styles()
    refresh_block(HERE / "index.html", BUNDLE_START, BUNDLE_END, material_styles(motion=True))
