import logging
from pathlib import Path

from deepbond.stats import BestValueEpoch


def get_line_bar(template_head):
    line_head = list('-' * len(template_head))
    bar_indexes = [i for i, c in enumerate(template_head) if c == '|']
    for i in bar_indexes:
        line_head[i] = '+'
    return ''.join(line_head)


class Reporter:
    """
    Simple class to print stats on the screen using logging.info and
    optionally, tensorboard.

    Args:
        output_dir (str): Path location to save tensorboard artifacts.
        use_tensorboard (bool): Wheter to log stats on tensorboard server.
    """
    def __init__(self, output_dir, use_tensorboard):
        self.tb_writer = None
        if use_tensorboard:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                logging.error('Please install `tensorboardx` package first. '
                              'See `tensorboard` section in README.md '
                              'for more information on tensorboard logging.')
            self.tb_writer = SummaryWriter(output_dir)
            logging.info('Starting tensorboard logger...')
            logging.info('Type `tensorboard --logdir runs/` in your terminal '
                         'to see live stats (`tensorflow` package required).')
        self.mode = None
        self.epoch = None
        self.output_dir = output_dir
        self.template_head =  'Loss    (val / epoch) | '
        self.template_head += 'Prec.     '
        self.template_head += 'Rec.    '
        self.template_head += 'F1     (val / epoch) | '
        self.template_head += 'SER    (val / epoch) | '
        self.template_head += 'MCC    (val / epoch) |'
        self.template_line = get_line_bar(self.template_head)
        self.template_body = '{:7.4f} ({:.4f} / {:2d}) |'
        self.template_body += '{:7.4f}  '
        self.template_body += '{:7.4f}  '
        self.template_body += '{:7.4f} ({:.4f} / {:2d}) |'
        self.template_body += '{:7.4f} ({:.4f} / {:2d}) |'
        self.template_body += '{:7.4f} ({:.4f} / {:2d}) |'
        self.template_footer = '---'

    def set_mode(self, mode):
        self.mode = mode

    def set_epoch(self, epoch):
        self.epoch = epoch

    def show_head(self):
        logging.info(self.template_head)
        logging.info(self.template_line)

    def show_footer(self):
        logging.info(self.template_footer)

    def show_stats(self, stats_dict):
        logging.info(
            self.template_body.format(
                stats_dict['loss'],
                stats_dict['best_loss'].value,
                stats_dict['best_loss'].epoch,
                stats_dict['prec_rec_f1'][0],
                stats_dict['prec_rec_f1'][1],
                stats_dict['prec_rec_f1'][2],
                stats_dict['best_prec_rec_f1'].value[2],
                stats_dict['best_prec_rec_f1'].epoch,
                stats_dict['ser'],
                stats_dict['best_ser'].value,
                stats_dict['best_ser'].epoch,
                stats_dict['mcc'],
                stats_dict['best_mcc'].value,
                stats_dict['best_mcc'].epoch,
            )
        )

    def report_progress(self, i, nb_iters, loss):
        print('Loss ({}/{}): {:.4f}'.format(i, nb_iters, loss), end='\r')
        if self.tb_writer is not None:
            j = (self.epoch - 1) * nb_iters + i
            mode_metric = '{}/{}'.format(self.mode, 'moving_loss')
            self.tb_writer.add_scalar(mode_metric, loss, j)

    def report_stats(self, stats_dict):
        self.show_head()
        self.show_stats(stats_dict)
        self.show_footer()
        if self.tb_writer is not None:
            for metric, value in stats_dict.items():
                if isinstance(value, BestValueEpoch):
                    continue
                if metric == 'prec_rec_f1':
                    mm_0 = '{}/{}'.format(self.mode, 'precision')
                    mm_1 = '{}/{}'.format(self.mode, 'recall')
                    mm_2 = '{}/{}'.format(self.mode, 'f1')
                    self.tb_writer.add_scalar(mm_0, value[0], self.epoch)
                    self.tb_writer.add_scalar(mm_1, value[1], self.epoch)
                    self.tb_writer.add_scalar(mm_2, value[2], self.epoch)
                else:
                    mode_metric = '{}/{}'.format(self.mode, metric)
                    self.tb_writer.add_scalar(mode_metric, value, self.epoch)

    def report_stats_history(self, stats_history):
        self.show_head()
        for stats_dict in stats_history:
            self.show_stats(stats_dict)
        self.show_footer()

    def close(self):
        if self.tb_writer is not None:
            all_scalars_path = Path(self.output_dir, 'all_scalars.json')
            self.tb_writer.export_scalars_to_json(str(all_scalars_path))
            self.tb_writer.close()
