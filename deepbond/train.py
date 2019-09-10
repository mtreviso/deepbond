import logging
from pathlib import Path

from deeptagger.dataset import dataset, fields
from deeptagger import features
from deeptagger import iterator
from deeptagger import models
from deeptagger import optimizer
from deeptagger import scheduler
from deeptagger import opts
from deeptagger.trainer import Trainer


def run(options):
    words_field = fields.WordsField()
    tags_field = fields.TagsField()
    fields_tuples = [('words', words_field), ('tags', tags_field)]
    fields_tuples += features.build(options)

    logging.info('Building train corpus: {}'.format(options.train_path))
    train_dataset = dataset.build(options.train_path, fields_tuples, options)
    logging.info('Building train iterator...')
    train_iter = iterator.build(train_dataset,
                                options.gpu_id,
                                options.train_batch_size,
                                is_train=True)

    dev_dataset = None
    dev_iter = None
    if options.dev_path is not None:
        logging.info('Building dev dataset: {}'.format(options.dev_path))
        dev_dataset = dataset.build(options.dev_path, fields_tuples, options)
        logging.info('Building dev iterator...')
        dev_iter = iterator.build(dev_dataset,
                                  options.gpu_id,
                                  options.dev_batch_size,
                                  is_train=False)

    test_dataset = None
    test_iter = None
    if options.test_path is not None:
        logging.info('Building test dataset: {}'.format(options.test_path))
        test_dataset = dataset.build(options.test_path, fields_tuples, options)
        logging.info('Building test iterator...')
        test_iter = iterator.build(test_dataset,
                                   options.gpu_id,
                                   options.dev_batch_size,
                                   is_train=False)

    datasets = [train_dataset, dev_dataset, test_dataset]
    datasets = list(filter(lambda x: x is not None, datasets))
    if options.load:
        logging.info('Loading vocabularies...')
        fields.load_vocabs(options.load, fields_tuples)
        logging.info('Word vocab size: {}'.format(len(words_field.vocab)))
        logging.info('Tag vocab size: {}'.format(len(tags_field.vocab)))
        logging.info('Loading model...')
        model = models.load(options.load, fields_tuples)
        logging.info('Loading optimizer...')
        optim = optimizer.load(options.load, model.parameters())
        logging.info('Loading scheduler...')
        sched = scheduler.load(options.load, optim)
    else:
        logging.info('Building vocabulary...')
        fields.build_vocabs(fields_tuples, train_dataset, datasets, options)
        logging.info('Word vocab size: {}'.format(len(words_field.vocab)))
        logging.info('Tag vocab size: {}'.format(len(tags_field.vocab)))
        logging.info('Building model...')
        model = models.build(options, fields_tuples)
        logging.info('Building optimizer...')
        optim = optimizer.build(options, model.parameters())
        logging.info('Building scheduler...')
        sched = scheduler.build(options, optim)

    logging.info('Building trainer...')
    trainer = Trainer(train_iter, model, optim, sched, options,
                      dev_iter=dev_iter, test_iter=test_iter)

    if options.resume_epoch and options.load is None:
        logging.info('Resuming training...')
        trainer.resume(options.resume_epoch)

    trainer.train()

    if options.save:
        logging.info('Saving path: {}'.format(options.save))
        config_path = Path(options.save)
        config_path.mkdir(parents=True, exist_ok=True)
        logging.info('Saving config options...')
        opts.save(config_path, options)
        logging.info('Saving vocabularies...')
        fields.save_vocabs(config_path, fields_tuples)
        logging.info('Saving model...')
        models.save(config_path, model)
        logging.info('Saving optimizer...')
        optimizer.save(config_path, optim)
        logging.info('Saving scheduler...')
        scheduler.save(config_path, sched)

    return fields_tuples, model, optim, sched
