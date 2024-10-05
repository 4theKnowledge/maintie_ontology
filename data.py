from utils import flatten_entity_classes, process_maintie_scheme

RELATION_CLASSES = {
    "has_part": {
        "domain": ["PhysicalObject"],
        "range": ["PhysicalObject"],
    },
    "is_part_of": {
        "inverse_of": "has_part",
        "domain": ["PhysicalObject"],
        "range": ["PhysicalObject"],
    },
    # "is_a_property": {"parent": "is_a", "domain": ["Property"], "range": ["Property"]},
    # "is_a_process": {"parent": "is_a", "domain": ["Process"], "range": ["Process"]},
    "has_agent": {},
    "is_agent_of": {"inverse_of": "has_agent"},
    # "state_has_agent": {
    #     "parent": "has_agent",
    #     "domain": ["State"],
    #     "range": ["PhysicalObject"],
    # },
    # "state_is_agent_of": {"inverse_of": "state_has_agent", "parent": "is_agent_of"},
    # "activity_has_agent": {
    #     "parent": "has_agent",
    #     "domain": ["Activity"],
    #     "range": ["PhysicalObject"],
    # },
    # "activity_is_agent_of": {
    #     "inverse_of": "activity_has_agent",
    #     "parent": "is_agent_of",
    # },
    # "process_has_agent": {
    #     "parent": "has_agent",
    #     "domain": ["Process"],
    #     "range": ["PhysicalObject"],
    # },
    # "process_is_agent_of": {"inverse_of": "process_has_agent", "parent": "is_agent_of"},
    "has_patient": {},
    "is_patient_of": {"inverse_of": "has_patient"},
    # "state_has_patient_physicalobject": {
    #     "parent": "has_patient",
    #     "domain": ["State"],
    #     "range": ["PhysicalObject"],
    # },
    # "physicalobject_is_patient_of_state": {
    #     "parent": "is_patient_of",
    #     "inverse_of": "state_has_patient_physicalobject",
    # },
    # "state_has_patient_activity": {
    #     "parent": "has_patient",
    #     "domain": ["State"],
    #     "range": ["Activity"],
    # },
    # "activity_has_patient_physicalobject": {
    #     "parent": "has_patient",
    #     "domain": ["Activity"],
    #     "range": ["PhysicalObject"],
    # },
    # "physicalobject_is_patient_of_activity": {
    #     "parent": "is_patient_of",
    #     "inverse_of": "activity_has_patient_physicalobject",
    # },
    # "activity_has_patient_activity": {
    #     "parent": "has_patient",
    #     "domain": ["Activity"],
    #     "range": ["Activity"],
    # },
    # "activity_has_patient_state": {
    #     "parent": "has_patient",
    #     "domain": ["Activity"],
    #     "range": ["State"],
    # },
    # "activity_has_patient_process": {
    #     "parent": "has_patient",
    #     "domain": ["Activity"],
    #     "range": ["Process"],
    # },
    # "process_has_patient_physicalobject": {
    #     "parent": "has_patient",
    #     "domain": ["Process"],
    #     "range": ["PhysicalObject"],
    # },
    "contains": {
        "domain": ["PhysicalObject"],
        "range": ["PhysicalObject"],
    },
    "is_contained_by": {
        "inverse_of": "contains",
        "domain": ["PhysicalObject"],
        "range": ["PhysicalObject"],
    },
    "has_property": {
        "domain": ["PhysicalObject"],
        "range": ["Property"],
    },
    "is_property_of": {
        "inverse_of": "has_property",
        "domain": ["Property"],
        "range": ["PhysicalObject"],
    },
}


PROPERTY_CHAIN_AXIOMS = {
    "can_be_patient_of_state_due_to_part": [
        "has_part",
        "physicalobject_is_patient_of_state",
    ],
    "can_be_patient_of_activity_due_to_part": [
        "has_part",
        "physicalobject_is_patient_of_activity",
    ],
}


ENTITY_CLASSES, _ = process_maintie_scheme()

RELATION_PATTERN_MAPPING = {
    (
        "<state>",
        "has_patient",
        "<physical object>",
    ): "state_has_patient_physicalobject",
    ("<state>", "has_patient", "<activity>"): "state_has_patient_activity",
    (
        "<process>",
        "has_patient",
        "<physical object>",
    ): "process_has_patient_physicalobject",
    (
        "<activity>",
        "has_patient",
        "<physical object>",
    ): "activity_has_patient_physicalobject",
    ("<activity>", "has_patient", "<activity>"): "activity_has_patient_activity",
    ("<activity>", "has_patient", "<state>"): "activity_has_patient_state",
    ("<activity>", "has_patient", "<process>"): "activity_has_patient_process",
    ("<state>", "has_agent", "<physical object>"): "state_has_agent",
    ("<activity>", "has_agent", "<physical object>"): "activity_has_agent",
    ("<process>", "has_agent", "<physical object>"): "process_has_agent",
    # ("<process>", "is_a", "<process>"): "is_a_process",
    # ("<activity>", "is_a", "<activity>"): "is_a_activity",
    # ("<state>", "is_a", "<state>"): "is_a_state",
    # ("<physical object>", "is_a", "<physical object>"): "is_a_physical_object",
    # ("<property>", "is_a", "<property>"): "is_a_property",
}

ACTIVITY_CLASSES = flatten_entity_classes(
    [x for x in ENTITY_CLASSES if x["annotation_name"] == "<activity>"]
)
STATE_CLASSES = flatten_entity_classes(
    [x for x in ENTITY_CLASSES if x["annotation_name"] == "<state>"]
)
PROCESS_CLASSES = flatten_entity_classes(
    [x for x in ENTITY_CLASSES if x["annotation_name"] == "<process>"]
)
PHYSICAL_OBJECT_CLASSES = flatten_entity_classes(
    [x for x in ENTITY_CLASSES if x["annotation_name"] == "<physical object>"]
)
PROPERTY_CLASSES = flatten_entity_classes(
    [x for x in ENTITY_CLASSES if x["annotation_name"] == "<property>"]
)

# Create a mapping from child class names to their parent class names
child_to_parent_class_name = {a: "<activity>" for a in ACTIVITY_CLASSES}
child_to_parent_class_name.update({s: "<state>" for s in STATE_CLASSES})
child_to_parent_class_name.update({p: "<process>" for p in PROCESS_CLASSES})
child_to_parent_class_name.update(
    {po: "<physical object>" for po in PHYSICAL_OBJECT_CLASSES}
)
child_to_parent_class_name.update({pr: "<property>" for pr in PROPERTY_CLASSES})
