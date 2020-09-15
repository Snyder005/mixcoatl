class Error(Exception):
    """Base class for other exceptions."""
    pass

class MissingKeyword(Error):
    """Raised when an SQL query is missing a required keyword."""
    pass

class AlreadyExists(Error):
    """Raised when an object already exists in the SQL database."""
    pass
