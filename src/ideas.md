## Ideas for the implementation

<ul>
<li>The authors generate one word at random from the list of keywords, and then embed bits in all subsequent words. 
For example, we can choose the number $k$, and after the first word, simply generate $k$ words (choose the best in terms of condition probabilities),
and only then embed bits in the subsequent words.
Thus, Information Hidden Capacity should decrease, but Imperceptibility may increase.</li>
<li>The proposed model uses a Candidate Pool Size $m$.
We can somehow dynamically change this $m$ depending on the length of a sentence or text. For example, we can take a smaller value of $m$ at the beggining of a sentence and
then increase it
</li>
</ul>
