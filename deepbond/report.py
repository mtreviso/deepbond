import logging


def report_progress(i, n, loss):
    print('Loss ({}/{}): {:.4f}'.format(i, n, loss), end='\r')


def get_line_bar(template_head):
    line_head = list('-' * len(template_head))
    bar_indexes = [i for i, c in enumerate(template_head) if c == '|']
    for i in bar_indexes:
        line_head[i] = '+'
    return ''.join(line_head)


def report_head():
    template_head = 'Loss    (val / epoch) | '
    template_head += 'Acc    (val / epoch) | '
    template_head += 'Acc oov train  (val / epoch) | '
    template_head += 'Acc oov emb  (val / epoch) | '
    template_head += 'Acc sent.  (val / epoch) |'
    template_line = get_line_bar(template_head)
    logging.info(template_head)
    logging.info(template_line)


def _report_stats(loss, best_loss_value, best_loss_epoch,
                  acc, best_acc_value, best_acc_epoch,
                  acc_oov, best_acc_oov_value, best_acc_oov_epoch,
                  acc_emb, best_acc_emb_value, best_acc_emb_epoch,
                  acc_sent, best_acc_sent_value, best_acc_sent_epoch):
    template_body = '{:7.4f} ({:.4f} / {:2d}) |'
    template_body += '{:7.4f} ({:.4f} / {:2d}) |'
    template_body += '{:15.4f} ({:.4f} / {:2d}) |'
    template_body += '{:13.4f} ({:.4f} / {:2d}) |'
    template_body += '{:11.4f} ({:.4f} / {:2d}) |'
    logging.info(
        template_body.format(loss, best_loss_value, best_loss_epoch,
                             acc, best_acc_value, best_acc_epoch,
                             acc_oov, best_acc_oov_value, best_acc_oov_epoch,
                             acc_emb, best_acc_emb_value, best_acc_emb_epoch,
                             acc_sent, best_acc_sent_value, best_acc_sent_epoch
                             ))


def report_stats(stats):
    report_head()
    _report_stats(stats.get_loss(),
                  stats.best_loss.value,
                  stats.best_loss.epoch,
                  stats.get_acc(),
                  stats.best_acc.value,
                  stats.best_acc.epoch,
                  stats.get_acc_oov(),
                  stats.best_acc_oov.value,
                  stats.best_acc_oov.epoch,
                  stats.get_acc_emb(),
                  stats.best_acc_emb.value,
                  stats.best_acc_emb.epoch,
                  stats.get_acc_sentence(),
                  stats.best_acc_sent.value,
                  stats.best_acc_sent.epoch)


def report_stats_final(stats_history):
    report_head()
    for stats_dict in stats_history:
        _report_stats(stats_dict['loss'],
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
                      stats_dict['best_acc_sent'].epoch)
    logging.info('---')
