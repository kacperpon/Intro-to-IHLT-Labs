**Feature Techniques
- *Semantic*
	- **WordNet Path Similarity**: Captures word-level semantic relationships
	- **WordNet Average Similarity**: Accounts for synonymy and related terms
- *Lexical*
	- **Jaccard Similarity**: Overlap of unique words
	- **Longest Common Subsequence**: Quantifies shared word sequences
- *Structural*
	- **POS Tag Overlap**: Analyses grammatical similarity
	- **Sentence Length Ratio**: Handles paraphrases with length variations


| Feature                    | Preprocessing Needed                                                                   |
|----------------------------|----------------------------------------------------------------------------------------|
| WordNet Path Similarity    | Tokenization, Lowercasing, Lemmatization.                                              |
| WordNet Average Similarity | Tokenization, Lowercasing, Lemmatization.                                              |
| Jaccard Similarity         | Tokenization, Lowercasing, Punctuation Removal (optional: Stopword Removal).           |
| Longest Common Subsequence | Lowercasing.                                                                           |
| POS Tag Overlap            | Tokenization, Lowercasing, POS Tagging.                                                |
| Sentence Length Ratio      | Tokenization,                                                                          |


---

Keep pre-processing pipeline as fluid as possible.

Be prepared to make changes! [remove features if they perform poorly + replace with another]

Is 6 enough? (2 in each category.. do we want / need more?)
- Missing any suited for paraphrase detection?