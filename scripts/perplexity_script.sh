#!/usr/bin/env bash

SAVE_FOLDER=$1
utterances_to_generate=$2

for encoding_type in "FLC" "VLC"; do
    echo "================================" >> logger.txt
    echo "encoding_type ${encoding_type}" >> logger.txt
    
    for m in 1; do
        echo "beats_per_token ${m}" >> logger.txt
        
        python src/perplexity_exp.py \
            --model_path distilgpt2 \
            --data_path data/wikitext-2 \
            --out_folder ${SAVE_FOLDER} \
            --encoding_type ${encoding_type} \
            --cuda \
            --len_of_private_code 100 \
            --utterances_to_generate ${utterances_to_generate} \
            --m ${m}
    done
done