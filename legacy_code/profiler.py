from gen_metadata import gen_metadata_safe
from pycallgraph import PyCallGraph, Config
from pycallgraph.output import GraphvizOutput


with PyCallGraph(output=GraphvizOutput(output_file='output.jpg'), config=Config(max_depth=4)):
    gen_metadata_safe('/usr/local/bgnn/tulane/UWZM-F-0003318.JPG')
