import os
import yaml

class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

global_config = AttrDict()
default_config = {'Global': {'debug': False, }}
def merge_config(config):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in config.items():
        if "." not in key:
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in global_config
            ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                global_config.keys(), sub_keys[0])
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    merge_config(default_config)
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    merge_config(yaml.load(open(file_path, 'rb'), Loader=yaml.Loader))
    return global_config
