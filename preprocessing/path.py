import os,sys,inspect

def add_parent_to_path():
    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parent = os.path.dirname(current)
    sys.path.insert(0,parent)