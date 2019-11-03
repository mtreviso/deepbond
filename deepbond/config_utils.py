import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def configure_output(output_dir):
    if output_dir is None:
        output_time = time.strftime('%Y-%m-%d_%H:%M:%S')
        output_path = Path('runs', output_time)
        output_path.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_path)
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def configure_device(gpu_id):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)


def configure_logger(debug, output_dir):
    logging.Formatter.converter = time.gmtime
    logging.Formatter.default_msec_format = '%s.%03d'
    log_format = '[%(asctime)s] %(levelname)s: %(message)s'
    if logging.getLogger().handlers:
        log_formatter = logging.Formatter(log_format)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(log_formatter)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

    log_level = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(log_level)
    if output_dir is not None:
        fh = logging.FileHandler(os.path.join(output_dir, 'out.log'))
        fh.setLevel(log_level)
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def empty_cache(gpu_id):
    if gpu_id is not None:
        torch.cuda.empty_cache()
