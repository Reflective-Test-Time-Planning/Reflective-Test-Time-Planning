import absl.flags
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags


def define_flags(**kwargs):
    output = {}
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
            output[key] = val
        elif isinstance(val, tuple):
            assert len(val) == 2 or len(val) == 3
            if len(val) == 3:
                default_val, flag_type, help_str = val
            else:
                default_val, flag_type = val
                help_str = ""
            assert flag_type in ["integer", "string", "bool", "float"], f"Type `{flag_type}` not supported"
            assert isinstance(help_str, str)
            getattr(absl.flags, f"DEFINE_{flag_type}")(key, default_val, help_str)
            output[key] = default_val
        else:
            raise ValueError(f"Bad define for flag `{key}`")
    return output


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if prefix is not None:
            next_prefix = '{}.{}'.format(prefix, key)
        else:
            next_prefix = key
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=next_prefix))
        else:
            output[next_prefix] = val
    return output
