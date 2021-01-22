"""
    Contains constants shared across the project.
"""

import os
import enum


BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')
CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'checkpoints')

DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
CORA_PATH = os.path.join(DATA_DIR_PATH, 'cora')

# Make sure these exist as the rest of the code assumes it
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)


class DatasetType(enum.Enum):
    CORA = 0


class GraphVisualizationTool(enum.Enum):
    NETWORKX = 0,
    IGRAPH = 1


network_repository_cora_url = r'http://networkrepository.com/graphvis.php?d=./data/gsm50/labeled/cora.edges'
