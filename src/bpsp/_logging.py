"""Console logging for the pipeline.

Stage progress and result tables are reported through the ``bpsp`` logger. By default
they are written, unformatted, to the *current* ``sys.stdout`` so output appears in
Jupyter/Colab cells exactly as the original notebook printed it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

PACKAGE_LOGGER = "bpsp"


class _CurrentStdoutHandler(logging.StreamHandler):
    """Stream handler that always writes to the current ``sys.stdout``.

    Jupyter kernels and output-capturing tools replace ``sys.stdout`` after import;
    resolving it on every record keeps output visible in all of them.
    """

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    @property
    def stream(self) -> TextIO:
        return sys.stdout

    @stream.setter
    def stream(self, _value: TextIO) -> None:
        pass


def configure_logging(
    level: int | str = "INFO", fmt: str = "%(message)s", logfile: str | None = None
) -> logging.Logger:
    """Send ``bpsp`` log records to stdout (and optionally a file).

    Calling it again changes the level and format; a log file is attached only once.

    Parameters
    ----------
    level:
        Logging level, e.g. ``"INFO"`` (default) or ``"WARNING"`` for quiet runs.
    fmt:
        Record format. The default reproduces the notebook's plain printed output.
    logfile:
        Optional path; records are also appended there with timestamps.

    Notes
    -----
    Records do not propagate to the root logger, so stage output is not printed twice
    in environments that configure root logging. To route ``bpsp`` output into your own
    logging setup instead, attach handlers to ``logging.getLogger("bpsp")`` before the
    first run; the default console handler is then not added.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    console = next((h for h in logger.handlers if isinstance(h, _CurrentStdoutHandler)), None)
    if console is None:
        console = _CurrentStdoutHandler()
        logger.addHandler(console)
    console.setFormatter(logging.Formatter(fmt))
    if logfile:
        target = str(Path(logfile).resolve())
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == target for h in logger.handlers):
            fh = logging.FileHandler(target, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
            logger.addHandler(fh)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def ensure_logging() -> None:
    """Add the default console handler unless ``bpsp`` logging is already configured."""
    if not logging.getLogger(PACKAGE_LOGGER).handlers:
        configure_logging()
