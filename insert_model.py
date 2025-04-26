import tomllib
from argparse import ArgumentParser

from interfaces import InterfaceFactory
from interfaces.database import Database
from interfaces.storage import Storage
from models import ModelCategory, ProcessingModelType, ValidationModelType


def cli():
    """CLI for insert_model.py

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
    subparser = parser.add_subparsers(
        title="category", dest="category", required=True, description="Model category"
    )
    validation_parser = subparser.add_parser("validation", help="Validation model")
    validation_parser.add_argument(
        "type",
        choices=[x.name.lower() for x in ValidationModelType],
        help="Model type",
    )
    processing_parser = subparser.add_parser("processing", help="Processing model")
    processing_parser.add_argument(
        "type",
        choices=[x.name.lower() for x in ProcessingModelType],
        help="Model type",
    )
    for parser_ in (validation_parser, processing_parser):
        parser_.add_argument("subtype", type=str, help="Model subtype")
        parser_.add_argument("module", type=str, help="Model module")
        parser_.add_argument("class_name", type=str, help="Model class name")
        parser_.add_argument("version", type=str, help="Model version")
        parser_.add_argument(
            "weights",
            type=lambda x: open(x, "rb").read(),
            help="Path to the weights file",
        )
        parser_.add_argument(
            "-e", "--enabled", action="store_true", help="Model enabled"
        )
    parser.description = "Insert a model"
    return parser.parse_args()


def insert_model(
    database: Database,
    storage: Storage,
    model_category: ModelCategory,
    model_type: ValidationModelType | ProcessingModelType,
    subtype: str,
    module: str,
    class_name: str,
    version: str,
    enabled: bool,
    weights: bytes,
):
    """Insert a model.

    Parameters
    ----------
    database : DatabaseInterface
        Database interface.
    storage : StorageInterface
        Storage interface.
    model_category : ModelCategory
        Model category.
    model_type : ValidationModelType | ProcessingModelType
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
    weights : bytes
        Weights of the model.
    """
    model_id = database.insert_model(
        model_category, model_type, subtype, module, class_name, version, enabled
    )
    storage.store_weights(weights, model_id)


if __name__ == "__main__":
    args = cli()
    interface_factory = InterfaceFactory(args.settings)
    database = interface_factory.get_interface(Database)  # type: ignore
    storage = interface_factory.get_interface(Storage)  # type: ignore
    model_category = ModelCategory[args.category.upper()]
    model_type = model_category.value[args.type.upper()]
    insert_model(
        database,
        storage,
        model_category,
        model_type,
        args.subtype,
        args.module,
        args.class_name,
        args.version,
        args.enabled,
        args.weights,
    )
