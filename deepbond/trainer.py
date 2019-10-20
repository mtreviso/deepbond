import logging
import time
from pathlib import Path

import torch

from deepbond import models
from deepbond import optimizer
from deepbond import scheduler
from deepbond.reporter import Reporter
from deepbond.stats import Stats

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        train_iter,
        model,
        optimizer,
        scheduler,
        options,
        dev_iter=None,
        test_iter=None
    ):
        self.model = model
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.test_iter = test_iter
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = options.epochs
        self.output_dir = options.output_dir
        self.dev_checkpoint_epochs = options.dev_checkpoint_epochs
        self.save_checkpoint_epochs = options.save_checkpoint_epochs
        self.save_best_only = options.save_best_only
        self.early_stopping_patience = options.early_stopping_patience
        self.restore_best_model = options.restore_best_model
        self.current_epoch = 1
        self.train_stats_history = []
        self.dev_stats_history = []
        self.test_stats_history = []
        self.final_report = options.final_report
        tags_vocab = train_iter.dataset.fields['tags'].vocab.stoi
        self.train_stats = Stats(tags_vocab)
        self.dev_stats = Stats(tags_vocab)
        self.test_stats = Stats(tags_vocab)
        self.reporter = Reporter(options.output_dir, options.tensorboard)

    def train(self):
        # Perform an evaluation on dev set if it is available
        if self.dev_iter is not None:
            logger.info('Evaluating before training...')
            self.reporter.set_epoch(0)
            self.dev_epoch()

        # Perform an evaluation on test set if it is available
        if self.test_iter is not None:
            logger.info('Testing before training...')
            self.reporter.set_epoch(0)
            self.test_epoch()

        start_time = time.time()
        for epoch in range(self.current_epoch, self.epochs + 1):
            logger.info('Epoch {} of {}'.format(epoch, self.epochs))

            self.reporter.set_epoch(epoch)
            self.current_epoch = epoch

            # Train a single epoch
            logger.info('Training...')
            self.train_epoch()

            # Perform an evaluation on dev set if it is available
            if self.dev_iter is not None:
                # Only perform if a checkpoint was reached
                if (self.dev_checkpoint_epochs > 0
                        and epoch % self.dev_checkpoint_epochs == 0):
                    logger.info('Evaluating...')
                    self.dev_epoch()

            # Perform an evaluation on test set if it is available
            if self.test_iter is not None:
                logger.info('Testing...')
                self.test_epoch()

            # Only save if an improvement has occurred
            if self.save_best_only and self.dev_iter is not None:
                if self.dev_stats.best_prec_rec_f1.epoch == epoch:
                    logger.info('F1 improved on epoch {}'.format(epoch))
                    self.save(epoch)
            else:
                # Otherwise, save if a checkpoint was reached
                if (self.save_checkpoint_epochs > 0
                        and epoch % self.save_checkpoint_epochs == 0):
                    self.save(epoch)

            # Stop training before the total number of epochs
            if self.early_stopping_patience > 0 and self.dev_iter is not None:
                # Only stop if the desired patience epochs was reached
                passed_epochs = epoch - self.dev_stats.best_prec_rec_f1.epoch
                if passed_epochs == self.early_stopping_patience:
                    logger.info('Training stopped! No improvements on F1 '
                                'after {} epochs'.format(passed_epochs))
                    if self.restore_best_model:
                        if self.dev_stats.best_prec_rec_f1.epoch < epoch:
                            self.restore_epoch(self.dev_stats.best_prec_rec_f1.epoch)  # NOQA
                    break

            # Restore best model if early stopping didnt occur for final epoch
            if epoch == self.epochs and self.dev_iter is not None:
                if self.restore_best_model:
                    if self.dev_stats.best_prec_rec_f1.epoch < epoch:
                        self.restore_epoch(self.dev_stats.best_prec_rec_f1.epoch)  # NOQA

        elapsed = time.time() - start_time
        hms = time.strftime("%Hh:%Mm:%Ss", time.gmtime(elapsed))
        logger.info('Training ended after {}'.format(hms))

        if self.final_report:
            logger.info('Training final report: ')
            self.reporter.report_stats_history(self.train_stats_history)
            if self.dev_iter:
                logger.info('Dev final report: ')
                self.reporter.report_stats_history(
                    self.dev_stats_history, start=0
                )
            if self.test_iter:
                logger.info('Test final report: ')
                self.reporter.report_stats_history(
                    self.test_stats_history, start=0
                )
        self.reporter.close()

    def train_epoch(self):
        self.reporter.set_mode('train')
        self.train_stats.reset()
        self._train()
        self.train_stats_history.append(self.train_stats.to_dict())
        self.reporter.report_stats(self.train_stats.to_dict())

    def dev_epoch(self):
        self.reporter.set_mode('dev')
        self.dev_stats.reset()
        self._eval(self.dev_iter, self.dev_stats)
        self.dev_stats_history.append(self.dev_stats.to_dict())
        self.reporter.report_stats(self.dev_stats.to_dict())

    def test_epoch(self):
        self.reporter.set_mode('test')
        self.test_stats.reset()
        self._eval(self.test_iter, self.test_stats)
        self.test_stats_history.append(self.test_stats.to_dict())
        self.reporter.report_stats(self.test_stats.to_dict())

    def _train(self):
        self.model.train()
        for i, batch in enumerate(self.train_iter, start=1):

            # basic training steps:
            self.model.zero_grad()
            pred = self.model(batch)
            loss = self.model.loss(pred, batch.tags)
            loss.backward()
            self.optimizer.step()

            # keep stats object updated:
            pred_tags = self.model.predict_classes(batch)
            self.train_stats.update(loss.item(), pred_tags, batch.tags)

            # report current loss to the user:
            acum_loss = self.train_stats.get_loss()
            self.reporter.report_progress(i, len(self.train_iter), acum_loss)

        self.train_stats.calc(self.current_epoch)

        # scheduler.step() after training
        self.scheduler.step()

    def _eval(self, ds_iterator, stats):
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(ds_iterator, start=1):

                # basic prediction steps:
                pred = self.model(batch)
                loss = self.model.loss(pred, batch.tags)

                # keep stats object updated:
                pred_tags = self.model.predict_classes(batch)
                stats.update(loss.item(), pred_tags, batch.tags)

                # report current loss to the user:
                acum_loss = stats.get_loss()
                self.reporter.report_progress(i, len(ds_iterator), acum_loss)

        stats.calc(self.current_epoch)

    def save(self, current_epoch):
        epoch_dir = 'epoch_{}'.format(current_epoch)
        output_path = Path(self.output_dir, epoch_dir)
        output_path.mkdir(exist_ok=True)
        logger.info('Saving training state to {}'.format(output_path))
        models.save(output_path, self.model)
        optimizer.save(output_path, self.optimizer)
        scheduler.save(output_path, self.scheduler)

    def load(self, directory):
        logger.info('Loading training state from {}'.format(directory))
        models.load_state(directory, self.model)
        optimizer.load_state(directory, self.optimizer)
        scheduler.load_state(directory, self.scheduler)

    def restore_epoch(self, epoch):
        epoch_dir = 'epoch_{}'.format(epoch)
        self.load(str(Path(self.output_dir, epoch_dir)))

    def resume(self, epoch):
        self.restore_epoch(epoch)
        self.current_epoch = epoch
        self.train()
