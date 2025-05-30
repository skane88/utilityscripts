"""
Holds exceptions for the concrete modules
"""


class ConcreteError(Exception):
    pass


class CCAAError(Exception):
    pass


class CCAANoLoadError(CCAAError):
    pass


class CCAALoadNotFoundError(CCAAError):
    pass
