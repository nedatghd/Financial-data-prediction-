import json
from typing import List
import requests
import pandas as pd
from hparams import *
import numpy as np

with open(PATH_TO_CONFIG_JSON) as f:
    configs = json.load(f)

with open(PATH_TO_CONTRACT_JSON) as f:
    contract = json.load(f)

def get_data(last_epoch_num=150730, current_epochNum:int=152262) -> pd.DataFrame:
    step = 1000

    duration = current_epochNum - last_epoch_num
    if duration<0:
        raise TypeError("the current_epochNum should be  greater than last_epoch_num")

    outputs = pd.DataFrame({})
    for pa in range(last_epoch_num, current_epochNum+1, step):
         
        rounds = np.arange(pa, min(pa+step, current_epochNum+1)).tolist()

        print(f'Progress ----- from {rounds[0]} to {rounds[-1]}')
        body_data = {
            'query': """query MyQuery($id_in: [ID!] = "") {
                rounds(where: {id_in: $id_in}, first: 1000) {
                    id
                    epoch
                    position
                    failed
                    bearAmount
                    bullAmount
                    closePrice
                    closeAt
                    lockAt
                    lockPrice
                    startAt
                    totalAmount
                    failed
                  }
                }""",
            "variables": {"id_in": rounds}
        }

        result = requests.post(PREDICT_URL, json=body_data).json()
        roundData = result['data']['rounds']
        outputs = outputs.append(roundData)

    return outputs


def get_current_epoch() -> int:
    body_data = {
        'query': """query MyQuery {
          rounds( orderBy: epoch, orderDirection: desc, first: 1) {
            id
          }
        }"""
    }
    result = requests.post(PREDICT_URL, json=body_data).json()
    return int(result['data']['rounds'][0]["id"])

if __name__== "__main__":
    current_epochNum = get_current_epoch()
    data = get_data(current_epochNum=current_epochNum)
    print(data.shape)    