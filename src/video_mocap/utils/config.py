from typing import Dict
import yaml

from mergedeep import merge

def load_config(
    filename: str,
):
    with open(filename, "r") as stream:
        try:
            output = yaml.safe_load(stream)
            if output["parent"] is not None:
                parent_output = load_config(output["parent"])
                output = merge({}, parent_output, output)
            return output
        except yaml.YAMLError as error:
            print(error)
            return None
