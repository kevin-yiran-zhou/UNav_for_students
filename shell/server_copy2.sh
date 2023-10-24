#!/bin/bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script="$CURRENT_DIR/../src/server_copy2.py"
server_config="$CURRENT_DIR/../configs/server.yaml"
hloc_config="$CURRENT_DIR/../configs/hloc.yaml"

conda activate UNav

CUDA_VISIBLE_DEVICES=1 python $script -s $server_config -l $hloc_config
