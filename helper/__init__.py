""" Module for various helper class or functions."""


from .lazy import lazy_property
from .log import get_logger

from os.path import sep
def ossep(s):
    """ Change a path using '/' to a path
    using current os directories separation.
    """
    return s.replace('/', sep)
