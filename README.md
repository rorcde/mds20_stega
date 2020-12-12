# mds20_stega
This is the repository for the Models of Sequence Data 2020 Edition for the project [RNN-Stega: Linguistic Steganography Based on Recurrent Neural Networks](http://static.tongtianta.site/paper_pdf/899f6470-c222-11e9-9474-00163e08bb86.pdf)

---
| Path  | Description
| :---  | :----------
| repos | existing implementation
| &boxvr;&nbsp; [RNN-Stega](https://github.com/YangzlTHU/RNN-Stega) | Tensorflow implementation of RNN-Stega.
| &boxvr;&nbsp; [word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model) | Torch code for training language model and text generating.
| &boxvr;&nbsp; [steganography-lstm-master](https://github.com/tbfang/steganography-lstm) | Torch code for another steganographic paper "Generating Steganographic Text with LSTMs" which uses another information-encoding algorithm for encrypted messages exchanging

---
Scripts for Transformers:

Generate pure text via pretrained transformer(without steganography)
```console
!python src/generate_transformer.py \
--model_path {PATH to HUGGINGFACE Model or your pretrained one} \
--data_path data/wikitext-2 \
--out_path experiment/generated.json \
--seed 42 \
--cuda \
--utterances_to_generate 100 \
--sentences_per_unique_start 5 \
--do_sample \
--top_k 0 \
--top_p 0.8 \
--max_length 50 \
--min_length 15
```

Script which generates texts via GPT2, encoding private message inside, using Fixed Length Coding and Variable Length Coding. Experiments use a range of bits per token.

```bash
bash scipts/perplexity_script.sh folder_to_save_exp number_of_generated_utt_per_each_experiment
```

Replication of table $3$ from paper:

| beats per token | FLC(perplexity)    | VLC(perplexity)    | beats per token(VLC) |
|---|--------------------|--------------------|----------------------|
| 1 | 3.8646043681999744 | 3.8366896652604727 | 1.0101010101010102   |
| 2 | 7.205916203983316  | 5.0784163112266905 | 1.8108396863625664   |
| 3 | 12.104180800837643 | 5.934022751040277  | 2.414234325583641    |
| 4 | 21.022630540129374 | 7.320870174625539  | 3.044510747122937    |
| 5 | 35.79009683580573  | 9.008582112131112  | 3.62515860068878     |
| 6 | 57.72525323057455  | 11.123461032485856 | 4.184625685232456    |

---
Scripts for LSTM(failed experiments so far...):

Train model on data corpus:
```console
!python train.py \
--data data/wikitext-2/ \
--model LSTM \
--emsize 800 \
--nhid 800 \
--nlayers 3 \
--epochs 35 \
--batch_size 200 \
--bptt 50 \
--dropout 0.2 \
--seed 42 \
--log-interval 100 \
--cuda
```

Script for basic steganography algorithm, described in the paper. However, maybe, it consists some bugs(or the quality of our language model is extremely low). The information hiding algorithm is based on variable-length coding(VLC), based on a Huffman tree.
```console
!python steganography.py \
--data data/wikitext-2/ \
--model LSTM \
--emsize 800 \
--nhid 800 \
--nlayers 3 \
--checkpoint lstm_wikitext2.pt \
--seed 42 \
--cuda \
--len_of_generation 40 \
--bit_num 2 \
--utterances_to_generate 100 \
--bit_stream_path data/experiment/bit_stream.txt \
--save_path data/experiment/
```

Generate text, using a pretrained model. The script itself does not solve the problems associated with steganography but can be used to evaluate the language model.
```console
!python generate.py \
--data ./data/wikitext-2 \
--model LSTM \
--emsize 800 \
--nhid 800 \
--nlayers 3 \
--dropout 0.2 \
--checkpoint lstm_wikitext2.pt \
--outf generated_sample.txt \
--words 5000 \
--seed 42 \
--cuda
```

[Link to LSTM checkpoint](https://drive.google.com/file/d/1KALhEWSYobpav_eDgn58Otjob09fpy4m/view?usp=sharing) on Wikitext-2: $800$-dimensional vectors, $3$ LSTM hidden layers , $800$ LSTM units, $20$ epochs, Adam optimizer, lr = $1e{-4}$, linear scheduling.

---

|                              Tasks to do                              | Status |
|:---------------------------------------------------------------------:|:------:|
|                         FLC encoding algorithm                        |    ✅   |
|                            Code refactoring                           |    🌚   |
|          Add attacks and their metrics(table 5 in the paper)          |    ❌   |
|   Natural Language Generation metrics(perplexity, maybe some other)   |    ✅   |
|                    New encoding scheme from Notion                    |    ❌   |
| Improve Language model: train LSTM on a larger text corpus/ take GPT? |    ✅   |
