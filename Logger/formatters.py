import logging


class ColorFormatter(logging.Formatter):
    """Custom formatter for colorized console output"""

    def __init__(self, fmt='%(asctime)s - %(levelname)s - %(message)s', colors=None):
        super().__init__(fmt)
        self.COLORS = colors or {
            'DEBUG': '\033[37m',  # White
            'INFO': '\033[36m',  # Cyan
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',  # Red
            'CRITICAL': '\033[41m',  # Red background
        }
        self.RESET = '\033[0m'

    def format(self, record):
        log_message = super().format(record)
        level_name = record.levelname
        return f"{self.COLORS.get(level_name, self.RESET)}{log_message}{self.RESET}"