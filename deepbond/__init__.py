"""
deeptagger
~~~~~~~~~~~~~~~~~~~

Part-of-speech tagger based on Deep Learning.

:copyright: (c) 2018 by Marcos Treviso
:licence: MIT, see LICENSE for more details
"""

# Generate your own AsciiArt at:
# patorjk.com/software/taag/#f=Calvin%20S&t=DeepTagger
__banner__ = r"""
╔╦╗┌─┐┌─┐┌─┐╔╦╗┌─┐┌─┐┌─┐┌─┐┬─┐
 ║║├┤ ├┤ ├─┘ ║ ├─┤│ ┬│ ┬├┤ ├┬┘
═╩╝└─┘└─┘┴   ╩ ┴ ┴└─┘└─┘└─┘┴└─
"""

__prog__ = "deeptagger"
__title__ = 'DeepTagger'
__summary__ = 'Part-of-speech tagger based on Deep Learning.'
__uri__ = 'https://github.com/mtreviso/deeptagger'

__version__ = '0.0.1'

__author__ = 'Marcos Treviso'
__email__ = 'marcostreviso@usp.br'

__license__ = 'MIT'
__copyright__ = 'Copyright 2019 Marcos Treviso'

from .predicter import Predicter  # NOQA
from .tagger import Tagger  # NOQA
from .trainer import Trainer  # NOQA
