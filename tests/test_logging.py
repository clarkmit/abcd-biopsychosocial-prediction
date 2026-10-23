import logging

import pytest

from bpsp import configure_logging
from bpsp._logging import _CurrentStdoutHandler, ensure_logging


@pytest.fixture
def bpsp_logger():
    logger = logging.getLogger("bpsp")
    saved = (list(logger.handlers), logger.level, logger.propagate)
    logger.handlers.clear()
    yield logger
    for handler in logger.handlers:
        if handler not in saved[0]:
            handler.close()
    logger.handlers[:], logger.level, logger.propagate = saved


def test_records_go_to_the_current_stdout_unformatted(bpsp_logger, capsys):
    configure_logging()
    logging.getLogger("bpsp.analysis.runner").info("Shape after preprocessing: (210, 522)")
    assert capsys.readouterr().out == "Shape after preprocessing: (210, 522)\n"


def test_reconfiguring_updates_level_and_format_without_duplicating_handlers(bpsp_logger, tmp_path, capsys):
    log = tmp_path / "run.log"
    configure_logging(logfile=str(log))
    configure_logging(level="WARNING", fmt="[%(levelname)s] %(message)s", logfile=str(log))
    ours = [h for h in bpsp_logger.handlers if isinstance(h, (_CurrentStdoutHandler, logging.FileHandler))]
    assert len(ours) == 2  # one console handler, one file handler
    logging.getLogger("bpsp").info("hidden")
    logging.getLogger("bpsp").warning("shown")
    assert capsys.readouterr().out == "[WARNING] shown\n"
    assert "shown" in log.read_text() and "hidden" not in log.read_text()


def test_user_handlers_are_respected(bpsp_logger):
    mine = logging.NullHandler()
    bpsp_logger.addHandler(mine)
    ensure_logging()
    assert mine in bpsp_logger.handlers
    assert not any(isinstance(h, _CurrentStdoutHandler) for h in bpsp_logger.handlers)
