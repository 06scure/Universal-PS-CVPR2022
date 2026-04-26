import os
import sys
import logging
from datetime import datetime

def setup_logger(outdir: str) -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    log_file = os.path.join(outdir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(filename=log_file, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        force=True
    )
    return logging.getLogger()
