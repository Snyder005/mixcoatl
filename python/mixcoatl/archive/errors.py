"""Errors specific to MixCOATL.
"""
class Error(Exception):
    """Base class for other exceptions."""
    pass

class MissingKeyword(Error):
    """Raised when a required keyword is missing from query."""
    pass
