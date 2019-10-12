import logging
from pathlib import Path

from deepbond import iterator
from deepbond import models
from deepbond import optimizer
from deepbond import opts
from deepbond import scheduler
from deepbond.dataset import dataset, fields
from deepbond.trainer import Trainer

logger = logging.getLogger(__name__)


def run(options):
    logger.info('Running with options: {}'.format(options))

    words_field = fields.WordsField()
    tags_field = fields.TagsField()
    fields_tuples = [('words', words_field), ('tags', tags_field)]

    logger.info('Building train corpus: {}'.format(options.train_path))
    train_dataset = dataset.build(options.train_path, fields_tuples, options)

    logger.info('Building train iterator...')
    train_iter = iterator.build(train_dataset,
                                options.gpu_id,
                                options.train_batch_size,
                                is_train=True)

    dev_dataset = None
    dev_iter = None
    if options.dev_path is not None:
        logger.info('Building dev dataset: {}'.format(options.dev_path))
        dev_dataset = dataset.build(options.dev_path, fields_tuples, options)
        logger.info('Building dev iterator...')
        dev_iter = iterator.build(dev_dataset,
                                  options.gpu_id,
                                  options.dev_batch_size,
                                  is_train=False)

    test_dataset = None
    test_iter = None
    if options.test_path is not None:
        logger.info('Building test dataset: {}'.format(options.test_path))
        test_dataset = dataset.build(options.test_path, fields_tuples, options)
        logger.info('Building test iterator...')
        test_iter = iterator.build(test_dataset,
                                   options.gpu_id,
                                   options.dev_batch_size,
                                   is_train=False)

    datasets = [train_dataset, dev_dataset, test_dataset]
    datasets = list(filter(lambda x: x is not None, datasets))

    # BUILD
    if not options.load:
        logger.info('Building vocabulary...')
        fields.build_vocabs(fields_tuples, train_dataset, datasets, options)
        loss_weights = None
        if options.loss_weights == 'balanced':
            loss_weights = train_dataset.get_loss_weights()
        logger.info('Building model...')
        model = models.build(options, fields_tuples, loss_weights)
        logger.info('Building optimizer...')
        optim = optimizer.build(options, model.parameters())
        logger.info('Building scheduler...')
        sched = scheduler.build(options, optim)

    # OR LOAD
    else:
        logger.info('Loading vocabularies...')
        fields.load_vocabs(options.load, fields_tuples)
        logger.info('Loading model...')
        model = models.load(options.load, fields_tuples, options.gpu_id)
        logger.info('Loading optimizer...')
        optim = optimizer.load(options.load, model.parameters())
        logger.info('Loading scheduler...')
        sched = scheduler.load(options.load, optim)

    # STATS
    logger.info('Word vocab size: {}'.format(len(words_field.vocab)))
    logger.info('Tag vocab size: {}'.format(len(tags_field.vocab) - 1))
    logger.info('Number of training examples: {}'.format(len(train_dataset)))
    if dev_dataset:
        logger.info('Number of dev examples: {}'.format(len(dev_dataset)))
    if test_dataset:
        logger.info('Number of test examples: {}'.format(len(test_dataset)))

    logger.info('Model info: ')
    logger.info(str(model))
    logger.info('Optimizer info: ')
    logger.info(str(optim))
    logger.info('Scheduler info: ')
    logger.info(str(sched))

    # TRAIN
    logger.info('Building trainer...')
    trainer = Trainer(train_iter, model, optim, sched, options,
                      dev_iter=dev_iter, test_iter=test_iter)

    if options.resume_epoch and options.load is None:
        logger.info('Resuming training...')
        trainer.resume(options.resume_epoch)

    trainer.train()

    # SAVE
    if options.save:
        logger.info('Saving path: {}'.format(options.save))
        config_path = Path(options.save)
        config_path.mkdir(parents=True, exist_ok=True)
        logger.info('Saving config options...')
        opts.save(config_path, options)
        logger.info('Saving vocabularies...')
        fields.save_vocabs(config_path, fields_tuples)
        logger.info('Saving model...')
        models.save(config_path, model)
        logger.info('Saving optimizer...')
        optimizer.save(config_path, optim)
        logger.info('Saving scheduler...')
        scheduler.save(config_path, sched)

    return fields_tuples, model, optim, sched
