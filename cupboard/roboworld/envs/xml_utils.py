from typing import Dict
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom
import numpy as np
from datetime import datetime


def array2string(a: np.ndarray):
    assert a.ndim == 1
    return np.array2string(a, separator=" ")[1:-1]


def set_attribute(elem, attr_key, attr_val):
    v = attr_val
    if not isinstance(v, str):
        if isinstance(v, list) or isinstance(v, tuple):
            v = array2string(np.array(v))
        elif isinstance(v, np.ndarray):
            v = array2string(v)
        elif isinstance(v, bool):
            v = str(v).lower()
        elif isinstance(v, int) or isinstance(v, float):
            v = str(v)
        else:
            raise ValueError(f"Unsupported value type `{type(v)}` for key `{attr_key}`.")
    elem.set(attr_key, v)


def set_attributes(elem: Element, attributes: Dict):
    for k, v in attributes.items():
        set_attribute(elem, k, v)


def create_element(tag, attributes=None, parent=None):
    if parent is None:
        elem = Element(tag)
    else:
        elem = SubElement(parent, tag)
    if attributes is not None:
        set_attributes(elem, attributes)
    return elem


def write_xml_file(root_elem, filename):
    xmlstr = minidom.parseString(tostring(root_elem)).toprettyxml(indent="  ")
    with open(filename, "w+") as f:
        f.write(xmlstr)


class XmlMaker(object):

    def __init__(self):
        self.root = Element("mujoco")  # root
        timestamp = datetime.now().strftime("%H:%M:%S, %m/%d/%Y")
        self.root.append(Comment(f"Automatically generated @ {timestamp}."))

        # compiler
        compiler = create_element("compiler", parent=self.root, attributes={
            "angle": "radian",
            "meshdir": ".",
            "autolimits": "true",
        })

        # include
        create_element("include", parent=self.root, attributes=dict(file="franka/panda.xml"))
        create_element("include", parent=self.root, attributes=dict(file="scene.xml"))

        # asset
        self.asset = SubElement(self.root, "asset")

        # worldbody
        self.worldbody = SubElement(self.root, "worldbody")

        # equality
        self.equality = SubElement(self.root, "equality")

    def add_object(self, body: Element):
        self.worldbody.append(body)

    def add_objects(self, bodies):
        for body in bodies:
            self.add_object(body)

    def add_equality(self, equality: Element):
        self.equality.append(equality)

    def add_asset(self, asset: Element):
        self.asset.append(asset)

    def write_to_file(self, filename):
        write_xml_file(self.root, filename)
