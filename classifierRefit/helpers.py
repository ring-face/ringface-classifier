
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class MultiFaceError(Error):
    """Exception raised for more than one face in the input file.

     """

    def __init__(self, filename):
        self.filename = filename