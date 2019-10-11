import argparse
import logging

from deepbond import config_utils
from deepbond import opts
from deepbond import predict
from deepbond import train

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='DeepBond')
parser.add_argument('task', type=str, choices=['train', 'predict'])
opts.general_opts(parser)
opts.preprocess_opts(parser)
opts.model_opts(parser)
opts.train_opts(parser)
opts.predict_opts(parser)


if __name__ == '__main__':
    options = parser.parse_args()
    options.output_dir = config_utils.configure_output(options.output_dir)
    config_utils.configure_logger(options.debug, options.output_dir)
    config_utils.configure_seed(options.seed)
    config_utils.configure_device(options.gpu_id)
    logger.info('Output directory is: {}'.format(options.output_dir))

    if options.task == 'train':
        train.run(options)
    elif options.task == 'predict':
        predict.run(options)
