
BATCH_SIZE = 4  
INPUT_SEQUENCE_LENGTH = 16
INPUT_SHAPE = (INPUT_SEQUENCE_LENGTH, 1)
OUTPUT_SEQUENCE_LENGTH = 1
TEST_SIZE_IN_TSCV = 2

LEARNING_RATE = 2e-4
N_SPLIT_TRAIN = 50
N_SPLIT_INFERENCE = 2

LAYERS_LIST = [16, 32, 8, 4]
ACTIVATION = 'sigmoid'
LOSS_FUNC = 'mean_squared_error'
EPOCH = 2
JUMP = 2

PATH_TO_DATA = 'data.xlsx'
PATH_TO_LAST_CHKPOINT = 'last_epoch.h5'
PATH_TO_BEST_CHKPOINT = 'best_model1.h5'
PATH_TO_LOG = 'history'
PATH_TO_CONFIG_JSON = 'configs/config.json'
PATH_TO_CONTRACT_JSON = 'configs/contract.json'

PREDICT_URL = "https://api.thegraph.com/subgraphs/name/pancakeswap/prediction-v2"
