"""Low-volume lifecycle diagnostics. Never log PCM, credentials or transcripts."""
import logging
import os
import time

logger = logging.getLogger("piuu")


def configure_logging():
    level = os.getenv("PIUU_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s",
                                  datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False


def event(name, identity=None, **fields):
    if not logger.isEnabledFor(logging.INFO):
        return
    context = {}
    if identity is not None:
        context.update(c=f"{identity.client:016x}", s=f"{identity.server:016x}",
                       turn=identity.turn, resp=identity.response)
    context.update(fields)
    # repr keeps external strings on one line, including IDs containing newlines.
    details = " ".join(f"{key}={value!r}" for key, value in context.items())
    logger.info("%s %s", name, details)
