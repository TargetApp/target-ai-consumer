import tomllib
from argparse import ArgumentParser
from enum import Enum, auto
from time import sleep
from typing import Callable

from core import ModelWrapper
from core.processing.classification import Classification
from core.processing.segmentation import Segmentation
from core.validation.coffee_leaf_occ import CoffeeLeafOCC
from interfaces import InterfaceFactory
from interfaces.database import Database
from interfaces.queue import Queue
from interfaces.storage import Storage
from models import Model, ModelCategory
from models.processing import ProcessingModelType
from models.validation import ValidationModelType


class RunningMode(Enum):
    VALIDATION = auto()
    PROCESSING = auto()
    BOTH = auto()


def cli():
    """CLI for main.py

    Returns
    -------
    ArgumentParser
        The CLI parser.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "settings",
        type=lambda x: tomllib.load(open(x, "rb")),
        help="Path to the settings file",
    )
    parser.add_argument(
        "running_mode",
        choices=[x.name.lower() for x in RunningMode],
        help="Running mode",
    )
    parser.description = """Main script for the processing pipeline."""
    return parser.parse_args()


def load_validation_model(model: Model, weights: bytes) -> ModelWrapper:
    """Load a validation model.

    Parameters
    ----------
    model : Model
        Model to load.
    weights : bytes
        Weights of the model.

    Returns
    -------
    ModelWrapper
        Loaded model.
    """
    model_class: type[ModelWrapper]
    match model.type:
        case ValidationModelType.COFFEE_LEAF_OCC:
            model_class = CoffeeLeafOCC
        case _:
            raise ValueError(f"Invalid validation model type: {model.type}")
    return model_class(model, weights)


def load_processing_model(model: Model, weights: bytes) -> ModelWrapper:
    """Load a processing model.

    Parameters
    ----------
    model : Model
        Model to load.
    weights : bytes
        Weights of the model.

    Returns
    -------
    ModelWrapper
        Loaded model.
    """
    model_class: type[ModelWrapper]
    match model.type:
        case ProcessingModelType.CLASSIFICATION:
            model_class = Classification
        case ProcessingModelType.SEGMENTATION:
            model_class = Segmentation
        case _:
            raise ValueError(f"Invalid processing model type: {model.type}")
    return model_class(model, weights)


def load_model(model: Model, weights: bytes) -> ModelWrapper:
    """Load a model.

    Parameters
    ----------
    model : Model
        Model to load.
    weights : bytes
        Weights of the model.

    Returns
    -------
    ModelWrapper
        Loaded model.
    """
    load_function: Callable[[Model, bytes], ModelWrapper]
    match model.category:
        case ModelCategory.VALIDATION:
            load_function = load_validation_model
        case ModelCategory.PROCESSING:
            load_function = load_processing_model
        case _:
            raise ValueError(f"Invalid model category: {model.category}")
    return load_function(model, weights)


def load_proper_models(
    database: Database,
    storage: Storage,
    running_mode: RunningMode,
) -> dict[int, ModelWrapper]:
    """Load enabled models.

    Parameters
    ----------
    database : DatabaseInterface
        Database interface.
    storage : StorageInterface
        Storage interface.
    running_mode : RunningMode
        Running mode.

    Returns
    -------
    dict[int, ModelWrapper]
        Loaded models.

    Raises
    ------
    ValueError
        If the model type is invalid.
    """
    models: dict[int, ModelWrapper] = dict()
    enabled_models = database.enabled_models
    match running_mode:
        case RunningMode.VALIDATION:
            models = {
                id: load_validation_model(model, storage.retrieve_weights(id))
                for id, model in enabled_models.items()
                if model.category == ModelCategory.VALIDATION
            }
        case RunningMode.PROCESSING:
            models = {
                id: load_processing_model(model, storage.retrieve_weights(id))
                for id, model in enabled_models.items()
                if model.category == ModelCategory.PROCESSING
            }
        case RunningMode.BOTH:
            models = {
                id: load_model(model, storage.retrieve_weights(id))
                for id, model in enabled_models.items()
                if model.category
                in (ModelCategory.VALIDATION, ModelCategory.PROCESSING)
            }
        case _:
            raise ValueError(f"Invalid running mode: {running_mode}")
    return models


def consume_processing_queue_elements(
    database: Database,
    queue: Queue,
    storage: Storage,
    models: dict[int, ModelWrapper],
):
    """Consume elements from the processing queue and process them.

    Parameters
    ----------
    database : DatabaseInterface
        Database interface.
    queue : QueueInterface
        Queue interface.
    storage : StorageInterface
        Storage interface.
    models : dict[int, ModelWrapper]
        Loaded models.

    Raises
    ------
    ValueError
        If the model type is invalid.
    """
    while queue.processing_queue_has_elements():
        element = queue.dequeue_from_processing_queue()
        if element is None:
            return
        model = models[element.model_id]
        match element.report_type:
            case ProcessingModelType.CLASSIFICATION:
                disease, severity = model(element.image)
                database.update_classification_report(
                    element.report_id,
                    disease,
                    severity,
                )
            case ProcessingModelType.SEGMENTATION:
                stress_ratio, severity, mask = model(
                    element.image, element.generate_mask
                )
                database.update_segmentation_report(
                    element.report_id,
                    stress_ratio,
                    severity,
                )
                if mask is not None:
                    storage.store_mask(mask, element.report_id)
            case _:
                raise ValueError(f"Invalid model type: {model.__class__.__name__}")


def consume_validation_queue_elements(
    database: Database,
    queue: Queue,
    models: dict[int, ModelWrapper],
):
    """Consume elements from the validation queue and validate them.

    Parameters
    ----------
    database : DatabaseInterface
        Database interface.
    queue : QueueInterface
        Queue interface.
    storage : StorageInterface
        Storage interface.
    models : dict[int, ModelWrapper]
        Loaded models.

    Raises
    ------
    ValueError
        If the model type is invalid.
    """
    while queue.validation_queue_has_elements():
        element = queue.dequeue_from_validation_queue()
        if element is None:
            return
        model = models[element.model_id]
        validity = bool(model(element.image)[0])
        queue.update_buffer(element.id, validity)
        database.update_report_validity(
            element.report_id, element.report_type, validity
        )


if __name__ == "__main__":
    args = cli()
    interface_factory = InterfaceFactory(args.settings)
    database = interface_factory.get_interface(Database)  # type: ignore
    queue = interface_factory.get_interface(Queue)  # type: ignore
    storage = interface_factory.get_interface(Storage)  # type: ignore
    running_mode = RunningMode[args.running_mode.upper()]
    models = load_proper_models(database, storage, running_mode)
    polling_interval = args.settings["polling_interval"]
    match running_mode:
        case RunningMode.VALIDATION:
            while True:
                if not queue.validation_queue_has_elements():
                    sleep(polling_interval)
                    continue
                consume_validation_queue_elements(database, queue, models)
        case RunningMode.PROCESSING:
            while True:
                if not queue.processing_queue_has_elements():
                    sleep(polling_interval)
                    continue
                consume_processing_queue_elements(database, queue, storage, models)
        case RunningMode.BOTH:
            while True:
                validation_queue_has_elements = queue.validation_queue_has_elements()
                processing_queue_has_elements = queue.processing_queue_has_elements()
                if (
                    not validation_queue_has_elements
                    and not processing_queue_has_elements
                ):
                    sleep(polling_interval)
                    continue
                if validation_queue_has_elements:
                    consume_validation_queue_elements(database, queue, models)
                if processing_queue_has_elements:
                    consume_processing_queue_elements(database, queue, storage, models)
        case _:
            raise ValueError(f"Invalid running mode: {args.running_mode}")
