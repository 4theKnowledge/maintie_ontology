import random
import string
import json
import re

# TODO:
# - Need to capture that has_part and contains (and their inverses) cannot hold between two of the same individual classes.


def convert_format(input_string, include_brackets: bool = True):
    # Split the string by '/' and take the last part
    last_part = input_string.split("/")[-1]

    # Insert spaces before capital letters and convert to lowercase
    formatted_string = re.sub(r"(?<!^)(?=[A-Z])", " ", last_part).lower()

    return f"<{formatted_string}>" if include_brackets else f"{formatted_string}"


def transform_data(data, include_brackets: bool = True):
    result = []

    for item in data:
        # Transform the current item
        transformed_item = {
            "onto_name": item["name"],
            "annotation_name": convert_format(item["name"], include_brackets),
            "children": transform_data(
                item["children"], include_brackets
            ),  # Recursively transform children
        }
        result.append(transformed_item)

    return result


def process_maintie_scheme():
    with open("./inputs/maintie_scheme.json", "r") as f:
        scheme = json.load(f)

    entity_scheme = scheme["entity"]
    relation_scheme = scheme["relation"]

    relation_scheme_transformed = transform_data(
        relation_scheme, include_brackets=False
    )
    entity_scheme_transformed = transform_data(entity_scheme)
    return entity_scheme_transformed, relation_scheme_transformed


def find_individual_by_label(onto, label, entity_class):
    # Helper function to find an individual by label
    for individual in onto.get_instances_of(entity_class):
        if label in individual.label:
            return individual
    return None


def generate_short_id(length=8):
    # Generates a random string of uppercase letters and digits
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def flatten_entity_classes(entity_classes):
    flat_dict = {}

    for class_data in entity_classes:
        onto_name = class_data["onto_name"]
        annotation_name = class_data["annotation_name"]

        # Add the ontology name and annotation name to the flat dictionary
        flat_dict[annotation_name] = onto_name

        # If there are children, recursively flatten them
        children = class_data.get("children", [])
        if children:
            child_dict = flatten_entity_classes(children)
            flat_dict.update(child_dict)

    return flat_dict