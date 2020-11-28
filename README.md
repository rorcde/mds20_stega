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
Scripts:
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
|                         FLC encoding algorithm                        |    ❌   |
|                            Code refactoring                           |    ❌   |
|          Add attacks and their metrics(table 5 in the paper)          |    ❌   |
|   Natural Language Generation metrics(perplexity, maybe some other)   |    ❌   |
|                    New encoding scheme from Notion                    |    ❌   |
| Improve Language model: train LSTM on a larger text corpus/ take GPT? |    ❌   |
