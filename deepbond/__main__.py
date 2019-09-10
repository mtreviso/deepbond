import argparse
import logging
from pprint import pformat

from deeptagger import config_utils
from deeptagger import opts
from deeptagger import predict
from deeptagger import train

parser = argparse.ArgumentParser(description='DeepTagger')
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

    if options.task == 'train':
        logging.info('Running options:\n{}'.format(pformat(vars(options))))
        logging.info('Output directory is: {}'.format(options.output_dir))
        train.run(options)
    elif options.task == 'predict':
        predict.run(options)
