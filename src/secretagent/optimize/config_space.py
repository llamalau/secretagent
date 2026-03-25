"""Define a space of possible configuration changes.
"""

import copy
from functools import reduce
from itertools import product
from pprint import pprint
from pydantic import BaseModel
from typing import Any
import yaml

from secretagent import config

class ConfigSpace(BaseModel):
    variants: dict[str, list[Any]] = {}
    
    def __iter__(self):
        """Iterates over configuration deltas.

        A configuration delta is a dict D that can be passed to config
        via the context manager "config.configuration(**D)" or via
        config.configure(**D)
        """
        # combine the variant values all possible ways
        for value_choices in product(*self.variants.values()):
            # pair the parameters up with the values
            param_bindings = list(zip(self.variants, value_choices))
            # convert from something like (llm.model, 'gpt5') to a nested
            # dict for the heirarchical parameter, like {'llm':{'model':'gpt5'}}
            bindings_as_dicts = [self._expand_heirarchy(p,v) for p, v in param_bindings]
            # combine the parameter bindings into a single dictionary
            yield reduce(lambda d1, d2: dict(**d1, **d2), bindings_as_dicts)

    def _expand_heirarchy(self, dotted_param, value):
        if '.' not in dotted_param:
            return {dotted_param: value}
        else:
            first, rest = dotted_param.split('.', 1)
            return {first: self._expand_heirarchy(rest, value)}

    @staticmethod
    def load(yaml_file: str):
        """Load a ConfigSpace from a yaml file."""
        with open(yaml_file, 'r') as fp:
            return ConfigSpace(**yaml.safe_load(fp))

    def save(self, yaml_file: str):
        """Save to a yaml file."""
        with open(yaml_file, 'w') as fp:
            yaml.dump(self.model_dump(), fp, default_flow_style=False)

# smoketest
if __name__ == '__main__':
    cs = ConfigSpace(
        base_config_path='foo.yaml',
        variants={
            'name': ['fred'],
            'llm.model': ['big', 'small'],
            'ptool.extract': [
                {'method':'direct','fn':'extract_fn'} ,
                {'method': 'simulate', 'llm.model': 'huge'}]
        })
    config.configure(spacetest=True)
    for i, cfg in enumerate(cs):
        print(f' config {i+1} '.center(60, '-'))
        print(cfg)
        with config.configuration(**cfg):
            print(config.to_dotlist(config.GLOBAL_CONFIG))
    cs.save('/tmp/foo.yaml')
    print(ConfigSpace.load('/tmp/foo.yaml'))
