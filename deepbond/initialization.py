from torch import nn


def init_xavier(module, constant=0., dist='uniform', **kwargs):
    for name, param in module.named_parameters():
        if param.dim() == 1:
            nn.init.constant_(param, constant)
        else:
            if dist == 'uniform':
                nn.init.xavier_uniform_(param)
            elif dist == 'normal':
                nn.init.xavier_normal_(param, **kwargs)
            else:
                raise Exception('distribution {} not found'.format(dist))


def init_kaiming(
    module, constant=0., dist='uniform', **kwargs
):
    for name, param in module.named_parameters():
        if param.dim() == 1:
            nn.init.constant_(param, constant)
        else:
            if dist == 'uniform':
                nn.init.kaiming_uniform_(param)
            elif dist == 'normal':
                nn.init.kaiming_normal_(param, **kwargs)
            else:
                raise Exception('distribution {} not found'.format(dist))
