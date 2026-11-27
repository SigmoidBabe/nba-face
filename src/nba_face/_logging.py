"""Instance-local debug output independent of the host's log handlers."""
import logging


def configured_logger(name, config):
    debug = config.get("debug", False)
    if not isinstance(debug, bool):
        raise ValueError("config.debug must be a boolean")
    if not debug:
        return logging.getLogger(name)

    # A dedicated logger avoids changing Core's levels/handlers or duplicating
    # output when multiple runtime and registration backends share a worker.
    logger = logging.Logger(name, level=logging.DEBUG)
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    logger.addHandler(handler)
    return logger
