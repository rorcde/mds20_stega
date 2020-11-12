## Ideas for the implementation

1. The authors generate one word at random from the list of keywords, and then embed bits in all subsequent words. 
For example, we can choose the number $k$, and after the first word, simply generate $k$ words (choose the best in terms of conditional probabilities),
and only then embed bits in the subsequent words.
Thus, Information Hidden Capacity should decrease, but Imperceptibility may increase.

2. The proposed model uses a Candidate Pool Size $m$.
We can somehow dynamically change this $m$ depending on the length of a sentence or text. For example, we can take a smaller value of $m$ at the beginning of a sentence and
then increase it

3. One possible step to improve the quality of generation can be using a pretrained transformer model(e.g. GPT), which guarantees high-level text generating from the box. It will significantly save our computational resources and can increase metrics. Also, [transformer package](https://github.com/huggingface/transformers) has a great base of code and model checkpoints for experiments.