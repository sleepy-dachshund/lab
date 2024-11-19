import logging
import logging.config
import logging.handlers
import atexit
import queue

logging_config = {
  "version": 1,
  "disable_existing_loggers": False,
  "formatters": {
    "simple": {
      "format": "%(levelname)s: %(message)s"
    },
    "detailed": {
      "format": "[%(levelname)s|%(module)s|L%(lineno)d] %(asctime)s: %(message)s",
      "datefmt": "%Y-%m-%dT%H:%M:%S%z"
    }
  },
  "handlers": {
    "stderr": {
      "class": "logging.StreamHandler",
      "level": "WARNING",
      "formatter": "simple",
      "stream": "ext://sys.stderr"
    },
    "file": {
      "class": "logging.handlers.RotatingFileHandler",
      "level": "DEBUG",
      "formatter": "detailed",
      "filename": "encoder_v2.log",
      "maxBytes": 10_000_000,
      "backupCount": 3
    }
  },
  "loggers": {
    "root": {
      "level": "DEBUG",
      "handlers": [
        "stderr",
        "file"
      ]
    }
  }
}


def setup_logging(log_file: str = "encoder_v2.log") -> logging.Logger:
    logging_config["handlers"]["file"]["filename"] = log_file
    logging.config.dictConfig(logging_config)
    return logging.getLogger(__name__)


# Example
if __name__ == "__main__":
    logger = setup_logging()
