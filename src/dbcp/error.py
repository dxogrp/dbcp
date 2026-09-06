class InitiationError(Exception):
    """Exception raised for errors during initialization process."""


class SolveError(Exception):
    """Exception raised for errors during solving process."""


class DBCPError(Exception):
    """Exception raised when trying to solve a non-DBCP problem."""
