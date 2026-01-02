class AMKError(Exception):
    """Base error for allesmusterklassifizierer."""


class ConfigError(AMKError):
    """Raised when configuration is invalid."""


class DataError(AMKError):
    """Raised when dataset loading/format is invalid."""


class ValidationError(AMKError):
    """Raised when validation strategy fails."""


class ClassifierError(AMKError):
    """Raised when classifier configuration or execution fails."""
