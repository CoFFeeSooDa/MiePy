import json
import numpy as np


def find_attribute(target_object:object, parent:str):
            # Get attributes (list) for the target object
            all_attributes = dir(target_object)
            # Get the attributes of an base object
            base_object_attributes = dir(object)
            # Missing attributes in a base object
            base_object_attributes.extend(['__dict__', '__module__', '__weakref__'])
            # Filter the attributes
            user_defined_attributes = [attr for attr in all_attributes if attr not in base_object_attributes]

            for attr in user_defined_attributes:
                #print(type(getattr(target_object,attr)))
                if isinstance(getattr(target_object,attr), list) \
                        or isinstance(getattr(target_object,attr), str) \
                        or isinstance(getattr(target_object,attr), float) \
                        or isinstance(getattr(target_object,attr), int) \
                        or isinstance(getattr(target_object,attr), np.ndarray):
                    print(parent + '.' + str(attr))
                    continue
                elif isinstance(getattr(target_object,attr), object):
                    parent1 = parent + '.' + str(attr)
                    find_attribute(getattr(target_object,attr), parent1)

def read_json(path):
            with open(path) as f:
                # Load inputs
                inputs = json.load(f)
                # Convert inputs dictionaries to an object
                return inputs