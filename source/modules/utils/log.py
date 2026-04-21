import os
import sys
import logging

def setup_logger(outdir: str) -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    log_file = os.path.join(outdir, 'test.log')
    file_handler = logging.FileHandler(filename=log_file, mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler],
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name=__name__)
