from gen_metadata import gen_metadata_safe
from pycallgraph import PyCallGraph, Config
from pycallgraph.output import GraphvizOutput


with PyCallGraph(output=GraphvizOutput(), config=Config()):
    gen_metadata_safe('/usr/local/bgnn/tulane/UWZM-F-0003318.JPG')
