# Propaganda Detection (Datathon Hack the news) #


### What is the task about? ###

* Binary classification of news articles as propagandistic of not.

### Method ###


For solving this task, we use the following embeddings:
<ul>
 	<li>Dependency Based Embeddings  (<a href="http://www-users.cs.york.ac.uk/~suresh/papers/dep_embeddings_naacl2016.pdf">Komninos, A., &amp; Manandhar, S. (2016)</a>) - <a href="https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/komninos_english_embeddings.gz">Download</a></li>
 	<li>Distributed Representations of Words (<a href="https://arxiv.org/pdf/1301.3781.pdf">Mikolov, T., Chen, K., Corrado, G., &amp; Dean, J. (2013)</a>) - <a href="https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing">Download</a></li>
 	<li>Global Vectors for Word Representation (<a href="https://nlp.stanford.edu/pubs/glove.pdf">Pennington, J., Socher, R., &amp; Manning, C. (2014)</a>) - <a href="http://nlp.stanford.edu/data/glove.6B.zip">Download</a></li>
 	<li>Dependency-based word embeddings (<a href="http://www.aclweb.org/anthology/P14-2050">(Levy, Omer, and Yoav Goldberg (2014)</a>) - <a href="https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/levy_english_dependency_embeddings.gz">Download</a></li>
 	<li>GloVe + Sentiment Embeddings (urban dictionary) (<a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&amp;arnumber=7296633">Tang, Duyu, et al. (2016)</a>) - <a href="https://drive.google.com/file/d/15sJxZSAo2kBxTh8CJv9XVgo4ZE4hcuoo/view">Download</a></li>
 	<li>FastText embeddings (<a href="https://arxiv.org/pdf/1612.03651.pdf">Joulin, Armand, et al (2016)</a>) - <a href="https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip">Download</a></li>
</ul>
And also following architectures:
<ul>
 	<li>CNN</li>
 	<li>CNN + LSTM</li>
 	<li>LSTM</li>
</ul>
We tested every combination of embeddings and architecture.
We found out that the best performance was achieved by the cnn model combined with Komninos embeddings.

### Requiments ###

Other than the embeddings above we use:
<ul>
 	<li>Keras - <a href="https://github.com/keras-team/keras">Github</a></li>
 	<li>scikit-learn - <a href="https://scikit-learn.org/stable/">Site</a></li>
 	<li>NumPy - <a href="http://www.numpy.org/">Site</a></li> 
</ul>