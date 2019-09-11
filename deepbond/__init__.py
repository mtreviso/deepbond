"""
deepbond
~~~~~~~~~~~~~~~~~~~

Deep neural approach to Boundary and Disfluency Detection.

:copyright: (c) 2019 by Marcos Treviso
:licence: MIT, see LICENSE for more details
"""

# Generate your own AsciiArt at:
# patorjk.com/software/taag/#f=Calvin%20S&t=DeepBond
__banner__ = r"""
╔╦╗┌─┐┌─┐┌─┐╔╗ ┌─┐┌┐┌┌┬┐
 ║║├┤ ├┤ ├─┘╠╩╗│ ││││ ││
═╩╝└─┘└─┘┴  ╚═╝└─┘┘└┘─┴┘
"""

__prog__ = "deepbond"
__title__ = 'DeepBond'
__summary__ = 'Deep neural approach to Boundary and Disfluency Detection'
__uri__ = 'https://github.com/mtreviso/deepbond'

__version__ = '0.0.2'

__author__ = 'Marcos Treviso'
__email__ = 'marcostreviso@usp.br'

__license__ = 'MIT'
__copyright__ = 'Copyright 2019 Marcos Treviso'

from .predicter import Predicter  # NOQA
from .tagger import Tagger  # NOQA
from .trainer import Trainer  # NOQA
