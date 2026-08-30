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


class AuditError(AMKError):
    """Raised when an audit policy rejects a dataset."""


class SchemaError(DataError):
    """Raised when tabular data does not conform to the declared schema."""


class ArtifactError(AMKError):
    """Raised when a persisted artifact is invalid or incompatible."""
