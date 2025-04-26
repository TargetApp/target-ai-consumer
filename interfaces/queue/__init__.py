from abc import ABC, abstractmethod
from dataclasses import dataclass

from models.processing import ProcessingModelType

from .. import Interface


@dataclass
class QueueElement:
    """Queue element dataclass."""

    id: int
    model_id: int
    report_id: int
    report_type: ProcessingModelType
    image: bytes
    generate_mask: bool = False


class Queue(Interface, ABC):
    """Queue interface.

    Parameters
    ----------
    ABC : _abc.ABCMeta
        Abstract base class.
    """

    @abstractmethod
    def __init__(self, module_settings: dict):
        pass

    def enqueue_to_processing_queue(
        self,
        image_id: int,
        model_id: int,
        model_type: ProcessingModelType,
        report_id: int,
        image: bytes,
        generate_mask: bool | None = None,
    ):
        """Template method to enqueue an element to the processing queue.

        Parameters
        ----------
        image_id : int
            Image ID.
        model_id : int
            Model ID.
        model_type : ModelType
            Model type.
        report_id : int
            Report ID.
        image : bytes
            Image.
        generate_mask : bool | None, optional
            Indicates if a segmentation mask should be generated. Does not apply to classification models, by default None.

        Raises
        ------
        ValueError
            If generate_mask is not provided for segmentation models.
        """
        if model_type == ProcessingModelType.SEGMENTATION and generate_mask is None:
            raise ValueError("generate_mask must be provided for segmentation models")
        self._enqueue_to_processing_queue(
            image_id, model_id, model_type, report_id, image, generate_mask
        )

    @abstractmethod
    def _enqueue_to_processing_queue(
        self,
        image_id: int,
        model_id: int,
        model_type: ProcessingModelType,
        report_id: int,
        image: bytes,
        generate_mask: bool | None = None,
    ):
        """Enqueue an element to the processing queue.

        Parameters
        ----------
        image_id : int
            Image ID.
        model_id : int
            Model ID.
        model_type : ModelType
            Model type.
        report_id : int
            Report ID.
        image : bytes
            Image.
        generate_mask : bool | None, optional
            Indicates if a segmentation mask should be generated. Does not apply to classification models, by default None.
        """
        pass

    @abstractmethod
    def dequeue_from_processing_queue(self) -> QueueElement | None:
        """Dequeue an element.

        Returns
        -------
        QueueElement | None
            The dequeued element or None if the queue is empty.
        """
        pass

    @abstractmethod
    def processing_queue_has_elements(self) -> bool:
        """Check if the queue has elements.

        Returns
        -------
        bool
            True if the queue has elements, False otherwise.
        """
        pass

    def enqueue_to_validation_queue(
        self,
        image_id: int,
        validation_model_id: int,
        processing_model_id: int,
        processing_model_type: ProcessingModelType,
        report_id: int,
        image: bytes,
        generate_mask: bool | None = None,
    ):
        """Template method to enqueue an element to the validation queue.

        Parameters
        ----------
        image_id : int
            Image ID.
        validation_model_id : int
            Validation model ID.
        processing_model_id : int
            Processing model ID.
        processing_model_type : ModelType
            Processing model type.
        report_id : int
            Report ID.
        image : bytes
            Image.
        generate_mask : bool | None, optional
            Indicates if a segmentation mask should be generated. Does not apply to classification models, by default None.

        Raises
        ------
        ValueError
            If generate_mask is not provided for segmentation models.
        """
        if (
            processing_model_type == ProcessingModelType.SEGMENTATION
            and generate_mask is None
        ):
            raise ValueError("generate_mask must be provided for segmentation models")
        self._enqueue_to_validation_queue(
            image_id,
            validation_model_id,
            processing_model_id,
            processing_model_type,
            report_id,
            image,
            generate_mask,
        )

    @abstractmethod
    def _enqueue_to_validation_queue(
        self,
        image_id: int,
        validation_model_id: int,
        processing_model_id: int,
        processing_model_type: ProcessingModelType,
        report_id: int,
        image: bytes,
        generate_mask: bool | None = None,
    ):
        """Enqueue an element to the validation queue.

        Parameters
        ----------
        image_id : int
            Image ID.
        validation_model_id : int
            Validation model ID.
        processing_model_id : int
            Processing model ID.
        processing_model_type : ModelType
            Processing model type.
        report_id : int
            Report ID.
        image : bytes
            Image.
        generate_mask : bool | None, optional
            Indicates if a segmentation mask should be generated. Does not apply to classification models, by default None.
        """
        pass

    @abstractmethod
    def dequeue_from_validation_queue(self) -> QueueElement | None:
        """Dequeue an element from the validation queue.

        Returns
        -------
        QueueElement | None
            The dequeued element or None if the queue is empty.
        """
        pass

    @abstractmethod
    def validation_queue_has_elements(self) -> bool:
        """Check if the validation queue has elements.

        Returns
        -------
        bool
            True if the queue has elements, False otherwise.
        """
        pass

    @abstractmethod
    def update_buffer(self, element_id: int, validation_result: bool):
        """Update the buffer.

        Parameters
        ----------
        element_id : int
            Element ID.
        validation_result : bool
            Validation result.
        """
        pass

    @staticmethod
    def process_report_id(
        classification_report_id: int, segmentation_report_id: int
    ) -> tuple[int, ProcessingModelType]:
        """Process the report ID.

        Parameters
        ----------
        classification_report_id : int
            Classification report ID.
        segmentation_report_id : int
            Segmentation report ID.

        Returns
        -------
        tuple[int, ProcessingModelType]
            Report ID and report type.
        """
        report_id: int
        report_type: ProcessingModelType
        match (classification_report_id, segmentation_report_id):
            case (classification_report_id, None):
                report_id = classification_report_id
                report_type = ProcessingModelType.CLASSIFICATION
            case (None, segmentation_report_id):
                report_id = segmentation_report_id
                report_type = ProcessingModelType.SEGMENTATION
            case (None, None):
                raise ValueError("No report ID found")
            case (_, _):
                raise ValueError("Multiple report IDs found")
        return report_id, report_type
