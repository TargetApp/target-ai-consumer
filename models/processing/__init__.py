from enum import Enum, auto


class ProcessingModelType(Enum):
    """Processing model type enumeration.

    Parameters
    ----------
    Enum : EnumMeta
        Enumeration metaclass.
    """

    CLASSIFICATION = auto()
    SEGMENTATION = auto()
