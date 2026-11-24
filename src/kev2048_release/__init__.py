"""Kev 2048 training and evaluation tools."""

import sys as _sys

# CPython sets argv[0] to "-m" while resolving a module's parent package.
if _sys.argv[:1] == ["-m"]:
    from .training_entry import watch_module_command as _watch_module_command
    _watch_module_command(_sys.orig_argv, _sys.argv)
