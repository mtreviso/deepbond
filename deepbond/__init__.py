# -*- coding: utf-8 -*-
"""
deepbond
~~~~~~~~~~~~~~~~~~~

Deep neural approach to Boundary and Disfluency Detection.

:copyright: (c) 2016 by Marcos Treviso
:licence: MIT, see LICENSE for more details
"""
from __future__ import absolute_import, unicode_literals
import logging


# Generate your own AsciiArt at:
# patorjk.com/software/taag/#f=Calvin%20S&t=DeepBond
__banner__ = r"""
╔╦╗┌─┐┌─┐┌─┐╔╗ ┌─┐┌┐┌┌┬┐
 ║║├┤ ├┤ ├─┘╠╩╗│ ││││ ││ by Marcos Treviso & Christopher Shulby & Sandra Aluísio
═╩╝└─┘└─┘┴  ╚═╝└─┘┘└┘─┴┘
"""

__prog__ = "deepbond"
__title__ = 'DeepBond'
__summary__ = 'Deep neural approach to Boundary and Disfluency Detection.'
__uri__ = 'https://www.github.com/mtreviso/deepbond'

__version__ = '0.0.1-alpha'

__author__ = 'Marcos Treviso'
__email__ = 'marcostreviso@usp.br'

__license__ = 'MIT'
__copyright__ = 'Copyright 2017 Marcos Treviso'

# the user should dictate what happens when a logging event occurs
logging.getLogger(__name__).addHandler(logging.NullHandler())



# imports
from . import dataset
from . import error_analysis
from . import features
from . import helpers
from . import models

from . import loader
from . import log
from . import statistics
from . import utils
from . import train
from . import pipeline
