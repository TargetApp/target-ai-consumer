import os
import tomllib
from argparse import ArgumentParser

from interfaces import InterfaceFactory
from interfaces.database import Database
from interfaces.queue import Queue
from interfaces.storage import Storage
from models import ProcessingModelType


def cli():
    """CLI for enqueue_report.py

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
    parser.add_argument("user_id", type=int, help="User ID")
    parser.add_argument(
        "image", type=lambda x: open(x, "rb"), help="Path to the image file"
    )
    parser.add_argument("processing_model_id", type=int, help="Processing model ID")
    parser.add_argument(
        "-v", "--validation-model-id", type=int, help="Validation model ID"
    )
    subparser = parser.add_subparsers(
        title="processing_model_type",
        required=True,
        description="Model category",
        dest="processing_model_type",
    )
    subparser.add_parser("classification", help="Classification processing model type")
    segmentation_parser = subparser.add_parser(
        "segmentation", help="Segmentation processing model type"
    )
    segmentation_parser.add_argument(
        "-g", "--generate-mask", action="store_true", help="Generate mask flag"
    )
    parser.description = "Enqueue a report to processing queue"
    return parser.parse_args()


def enqueue_report(
    database: Database,
    queue: Queue,
    storage: Storage,
    user_id: int,
    filename: str,
    image: bytes,
    processing_model_id: int,
    processing_model_type: ProcessingModelType,
    generate_mask: bool = False,
    validation_model_id: int | None = None,
) -> int:
    """Enqueue a report.

    Parameters
    ----------
    database : DatabaseInterface
        Database interface.
    queue : QueueInterface
        Queue interface.
    storage : StorageInterface
        Storage interface.
    user_id : int
        User ID.
    filename : str
        Filename.
    image : bytes
        Image bytes.
    processing_model_id : int
        Model ID.
    processing_model_type : ProcessingModelType
        Model type.
    generate_mask : bool
        Indicates if a mask should be generated, by default False.
    validation_model_id : int, optional
        Validation model ID, by default None.

    Returns
    -------
    int
        Report ID.
    """
    _processing_model_type = database.enabled_models[processing_model_id].type
    if processing_model_type != _processing_model_type:
        raise ValueError(
            f"Model type mismatch: {processing_model_type} expected, "
            f"{_processing_model_type} found"
        )
    image_id = database.insert_image(user_id, filename)
    storage.store_image(image, image_id)
    match processing_model_type:
        case ProcessingModelType.CLASSIFICATION:
            if generate_mask:
                raise ValueError("Classification models do not generate masks")
            report_id = database.insert_classification_report(
                user_id, image_id, processing_model_id
            )
        case ProcessingModelType.SEGMENTATION:
            report_id = database.insert_segmentation_report(
                user_id, image_id, processing_model_id, generate_mask
            )
        case _:
            raise ValueError(f"Invalid model type: {processing_model_type}")
    if validation_model_id is None:
        queue.enqueue_to_processing_queue(
            image_id,
            processing_model_id,
            processing_model_type,
            report_id,
            image,
            generate_mask,
        )
    elif isinstance(validation_model_id, int):
        queue.enqueue_to_validation_queue(
            image_id,
            validation_model_id,
            processing_model_id,
            processing_model_type,
            report_id,
            image,
            generate_mask,
        )
    else:
        raise TypeError(f"Invalid validation model ID: {validation_model_id}")
    return report_id


if __name__ == "__main__":
    args = cli()
    interface_factory = InterfaceFactory(args.settings)
    database = interface_factory.get_interface(Database)  # type: ignore
    queue = interface_factory.get_interface(Queue)  # type: ignore
    storage = interface_factory.get_interface(Storage)  # type: ignore
    processing_model_type = ProcessingModelType[args.processing_model_type.upper()]
    generate_mask = (
        args.generate_mask
        if processing_model_type == ProcessingModelType.SEGMENTATION
        else False
    )
    enqueue_report(
        database,
        queue,
        storage,
        args.user_id,
        os.path.basename(args.image.name),
        args.image.read(),
        args.processing_model_id,
        processing_model_type,
        generate_mask,
        args.validation_model_id,
    )
