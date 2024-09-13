import logging
from colorama import Fore, Style

def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def log_fancy(logger, message, level=logging.INFO, color=Fore.WHITE, style=Style.NORMAL):
    colored_message = f"{style}{color}{message}{Style.RESET_ALL}"
    if level == logging.INFO:
        logger.info(colored_message)
    elif level == logging.WARNING:
        logger.warning(colored_message)
    elif level == logging.ERROR:
        logger.error(colored_message)
    elif level == logging.DEBUG:
        logger.debug(colored_message)
