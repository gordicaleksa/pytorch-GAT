"""
    Contains constants shared across the project.
"""

import os
import enum


class DatasetType(enum.Enum):
    CORA = 0


class GraphVisualizationTool(enum.Enum):
    NETWORKX = 0,
    IGRAPH = 1


# Support for 3 different GAT implementations - we'll profile each one of these in playground.py
class LayerType(enum.Enum):
    IMP1 = 0,
    IMP2 = 1,
    IMP3 = 2


BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'binaries')
CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'checkpoints')
DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
CORA_PATH = os.path.join(DATA_DIR_PATH, 'cora')  # this is checked-in no need to make a directory

# Make sure these exist as the rest of the code assumes it
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

#
# Cora specific information
#

# Thomas Kipf et al. first used this split in GCN paper and later Petar Veličković et al. in GAT paper
CORA_TRAIN_RANGE = [0, 140]
CORA_VAL_RANGE = [140, 140+500]
CORA_TEST_RANGE = [1708, 1708+1000]
CORA_NUM_INPUT_FEATURES = 1433
CORA_NUM_CLASSES = 7

network_repository_cora_url = r'http://networkrepository.com/graphvis.php?d=./data/gsm50/labeled/cora.edges'





