from abc import ABC, abstractmethod
from importlib import import_module
from typing import Type, TypeVar


class Interface(ABC):
    """Metaclass for interfaces.

    Parameters
    ----------
    ABC : _abc.ABCMeta
        Abstract base class.
    """

    @abstractmethod
    def __init__(self, module_settings: dict):
        """Initialize the interface.

        Parameters
        ----------
        module_settings : dict
            Module settings.
        """
        pass


T = TypeVar("T", bound="Interface")


class InterfaceFactory:
    def __init__(self, settings: dict):
        self.settings = settings["interfaces"]

    def get_interface(self, interface_class: Type[T]) -> T:
        interface_name = interface_class.__name__.lower()
        interface_settings = self.settings[interface_name]
        module_name = interface_settings["module"]
        module_settings = interface_settings[module_name]
        class_name = module_settings["class"]
        module = import_module(f"interfaces.{interface_name}.{module_name}")
        interface_class = getattr(module, class_name)
        return interface_class(module_settings)
