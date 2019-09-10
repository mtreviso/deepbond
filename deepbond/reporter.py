import logging
from pathlib import Path

from deeptagger.stats import BestValueEpoch


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
        self.template_head = 'Loss    (val / epoch) | '
        self.template_head += 'Acc    (val / epoch) | '
        self.template_head += 'Acc oov train  (val / epoch) | '
        self.template_head += 'Acc oov emb  (val / epoch) | '
        self.template_head += 'Acc sent.  (val / epoch) |'
        self.template_line = get_line_bar(self.template_head)
        self.template_body = '{:7.4f} ({:.4f} / {:2d}) |'
        self.template_body += '{:7.4f} ({:.4f} / {:2d}) |'
        self.template_body += '{:15.4f} ({:.4f} / {:2d}) |'
        self.template_body += '{:13.4f} ({:.4f} / {:2d}) |'
        self.template_body += '{:11.4f} ({:.4f} / {:2d}) |'
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
                stats_dict['acc'],
                stats_dict['best_acc'].value,
                stats_dict['best_acc'].epoch,
                stats_dict['acc_oov'],
                stats_dict['best_acc_oov'].value,
                stats_dict['best_acc_oov'].epoch,
                stats_dict['acc_emb'],
                stats_dict['best_acc_emb'].value,
                stats_dict['best_acc_emb'].epoch,
                stats_dict['acc_sent'],
                stats_dict['best_acc_sent'].value,
                stats_dict['best_acc_sent'].epoch
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
