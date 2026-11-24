"""Record head fits started by standalone package drivers without editing their frozen source."""

import argparse
from functools import wraps
from importlib.machinery import PathFinder
from pathlib import Path
import sys
import time


class _HeadFitImport:
    """Wait for the shared fitter's normal import, then observe calls to fit only."""

    def __init__(self, out):
        self.out = out
        self.started = time.monotonic()

    def find_spec(self, fullname, path, target=None):
        if fullname != __package__ + ".head_fit":
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec and hasattr(spec.loader, "exec_module"):
            spec.loader = _HeadFitLoader(spec.loader, self)
        return spec


class _HeadFitLoader:
    def __init__(self, loader, finder):
        self.loader = loader
        self.finder = finder

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def exec_module(self, module):
        module.__spec__.loader = module.__loader__ = self.loader
        self.loader.exec_module(module)
        fit = module.fit
        sys.meta_path.remove(self.finder)
        out, started = self.finder.out, self.finder.started

        @wraps(fit)
        def recorded_fit(*args, **kwargs):
            from .training_progress import watch_training
            with watch_training(out, "standalone-head", started=started):
                return fit(*args, **kwargs)

        module.fit = recorded_fit


def watch_module_command(original_argv, argv):
    if not argv or argv[0] != "-m" or len(original_argv) <= len(argv):
        return
    # The unconsumed user arguments are the suffix in both lists. The token
    # immediately before them is the module name (or a combined -mNAME).
    module = original_argv[-len(argv)]
    if module.startswith("-"):
        module = module.partition("m")[2]
    # The main CLI already scopes observers around its individual training commands.
    if not module.startswith(__package__ + ".") or module == __package__ + ".cli":
        return

    # Leave full validation and help to the original CLI. argparse also handles
    # its accepted abbreviations, --out=PATH, whitespace, and repeated options.
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--out", type=Path)
    try:
        options, _ = parser.parse_known_args(argv[1:])
    except argparse.ArgumentError:
        return
    if options.help or options.out is None:
        return

    # Keep existing custom importers ahead of the default filesystem finder.
    if PathFinder in sys.meta_path:
        sys.meta_path.insert(sys.meta_path.index(PathFinder), _HeadFitImport(options.out.absolute()))
