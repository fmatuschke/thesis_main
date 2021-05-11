import yaml
import os
import collections
import types

FILE_NAME = os.path.abspath(os.path.realpath(__file__))
FILE_PATH = os.path.dirname(FILE_NAME)


def tupleware(obj):
    # https://stackoverflow.com/a/51653724
    if isinstance(obj, dict):
        fields = sorted(obj.keys())
        namedtuple_type = collections.namedtuple(
            typename='TWare',
            field_names=fields,
            rename=True,
        )
        field_value_pairs = collections.OrderedDict(
            (str(field), tupleware(obj[field])) for field in fields)
        try:
            return namedtuple_type(**field_value_pairs)
        except TypeError:
            # Cannot create namedtuple instance so fallback to dict (invalid attribute names)
            return dict(**field_value_pairs)
    elif isinstance(obj, (list, set, tuple, frozenset)):
        return [tupleware(item) for item in obj]
    else:
        return obj


class RecursiveNamespace(types.SimpleNamespace):

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))


def get_namespace():
    with open(os.path.join(FILE_PATH, 'parameter.yaml'), 'r') as stream:
        config = yaml.safe_load(stream)
        return RecursiveNamespace(**config)


def get_tupleware():
    with open(os.path.join(FILE_PATH, 'parameter.yaml'), 'r') as stream:
        config = yaml.safe_load(stream)
        return tupleware(config)
