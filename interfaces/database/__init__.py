from abc import ABC, abstractmethod

from models import Model, ModelCategory
from models.processing import ProcessingModelType
from models.validation import ValidationModelType

from .. import Interface


class Database(Interface, ABC):
    """Database interface.

    Parameters
    ----------
    ABC : _abc.ABCMeta
        Abstract base class.
    """

    @abstractmethod
    def __init__(self, module_settings: dict):
        pass

    @abstractmethod
    def update_classification_report(
        self, report_id: int, disease_id: int, severity_id: int
    ):
        """Update a classification report.

        Parameters
        ----------
        report_id : int
            Report ID.
        disease_id : int
            Disease ID.
        severity_id : int
            Severity ID.
        """
        pass

    @abstractmethod
    def update_segmentation_report(
        self, report_id: int, stress_ratio: float, severity_id: int
    ):
        """Update a segmentation report.

        Parameters
        ----------
        report_id : int
            Report ID.
        stress_ratio : float
            Stress ratio.
        severity_id : int
            Severity ID.
        """
        pass

    @property
    @abstractmethod
    def enabled_models(self) -> dict[int, Model]:
        """Get enabled models from the database and return them as a dictionary with the model ID as the key.

        Returns
        -------
        dict[int, Model]
            Enabled models.
        """
        pass

    @property
    @abstractmethod
    def model_category_dict(self) -> dict[str, int]:
        """Get the model category dictionary from the database and return it as a dictionary with the model category as the key and the model category ID as the value."""
        pass

    @property
    @abstractmethod
    def model_type_dict(self) -> dict[str, int]:
        """Get the model type dictionary from the database and return it as a dictionary with the model type as the key and the model type ID as the value."""
        pass

    def insert_model(
        self,
        model_category: ModelCategory,
        model_type: ProcessingModelType | ValidationModelType,
        subtype: str,
        module: str,
        class_name: str,
        version: str,
        enabled: bool,
    ) -> int:
        """Template method to insert a model into the database.

        Parameters
        ----------
        model_category : ModelCategory
            Model category.
        model_type : ModelType
            Model type.
        subtype : str
            Indicates the directory where the model's module is located.
        module : str
            Indicates the  module where the model's class is defined.
        class_name : str
            Indicates the the class that defines the model.
        version : str
            Semantic version of the model.
        enabled : bool
            Indicates if the model is enabled.

        Returns
        -------
        int
            Model ID.
        """
        ModelCategory.validate_model_category_type(model_category, model_type)
        return self._insert_model(
            model_category, model_type, subtype, module, class_name, version, enabled
        )

    @abstractmethod
    def _insert_model(
        self,
        model_category: ModelCategory,
        model_type: ProcessingModelType | ValidationModelType,
        subtype: str,
        module: str,
        class_name: str,
        version: str,
        enabled: bool,
    ) -> int:
        """Insert a model into the database.

        Parameters
        ----------
        model_category : ModelCategory
            Model category.
        model_type : ModelType
            Model type.
        subtype : str
            Indicates the directory where the model's module is located.
        module : str
            Indicates the  module where the model's class is defined.
        class_name : str
            Indicates the the class that defines the model.
        version : str
            Semantic version of the model.
        enabled : bool
            Indicates if the model is enabled.

        Returns
        -------
        int
            Model ID.
        """
        pass

    @abstractmethod
    def insert_classification_report(
        self, user_id: int, image_id: int, model_id: int
    ) -> int:
        """Insert a classification report into the database.

        Parameters
        ----------
        user_id : int
            User ID.
        image_id : int
            Image ID.
        model_id : int
            Model ID.

        Returns
        -------
        int
            Report ID.
        """
        pass

    @abstractmethod
    def insert_segmentation_report(
        self, user_id: int, image_id: int, model_id: int, has_mask: bool = False
    ) -> int:
        """Insert a segmentation report into the database.

        Parameters
        ----------
        user_id : int
            User ID.
        image_id : int
            Image ID.
        model_id : int
            Model ID.
        has_mask : bool, optional
            Indicates if the report has a mask, by default False

        Returns
        -------
        int
            Report ID.
        """
        pass

    @abstractmethod
    def insert_image(self, user_id: int, filename: str) -> int:
        """Insert an image into the database.

        Parameters
        ----------
        user_id : int
            User ID.
        filename : str
            Image filename.

        Returns
        -------
        int
            Image ID.
        """
        pass

    @abstractmethod
    def update_report_validity(
        self, report_id: int, report_type: ProcessingModelType, valid: bool
    ):
        """Update a report's validity.

        Parameters
        ----------
        report_id : int
            Report ID.
        report_type : ProcessingModelType
            Report type.
        valid : bool
            Validity.
        """
        pass
