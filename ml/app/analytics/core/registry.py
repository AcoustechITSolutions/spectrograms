#!/usr/bin/env

class Registry:
    """
    The selector/factory class for specified entities group.
    The name of group passes into constructor and all registered entities
    belongs to this group.
    """
    def __init__(self, name):
        """
        Constructor
        :param name: name of the group
        """
        self._name = name
        self._register_dict = {}

    def _register(self, name, obj):
        if name in self._register_dict:
            raise KeyError(f'{name} is already registered in {self._name}')
        self._register_dict[name] = obj

    def register(self, name):
        """
        Registries class definition, should be used as decorator in class definition
        :param name: name under which class should be saved
        :return redecorated function
        """
        def wrap(obj):
            self._register(name, obj)
            return obj
        return wrap

    def get(self, name):
        """
        Get instance of the class by name
        :param name: name of the entity to take
        :return instance of the class by name
        """
        return self._register_dict[name]
