# TODO List for STS Project

## 1. Setup and Preparation
- [ ] Install required Python libraries:
  - [ ] `spacy`
  - [ ] `scikit-learn`
  - [ ] `pandas`
  - [ ] `numpy`
  - [ ] `matplotlib`
  - [ ] `scipy`
  - [ ] `nltk`
- [ ] Download and install SpaCy's small language model: `python -m spacy download en_core_web_sm`
- [x] Create project directory structure:
  - `data/`
  - `notebooks/`
  - `src/`
  - `results/`
  - `slides/`
- [ ] Review `train` and `test` datasets to understand input-output formats.
- [ ] Preprocess the data:
  - [ ] Tokenization and lemmatization (e.g., using SpaCy or TreeTagger).
  - [ ] Remove stopwords where necessary.

---

## 2. Exploratory Data Analysis (EDA)
- [ ] Analyze dataset:
  - [ ] Check sentence lengths and token patterns.
  - [ ] Plot label distributions (0–5 scale).
- [ ] Visualize data:
  - [ ] Create histograms for label distributions.
  - [ ] Plot scatter diagrams of lexical similarity vs. labels.

---

## 3. String-Based Similarity Measures
- [ ] Implement the following string-based techniques:
  - [ ] Longest common subsequence.
  - [ ] Longest common substring.
  - [ ] Greedy string tiling.
- [ ] Implement n-gram comparisons:
  - [ ] Character n-grams (2 to 4).
  - [ ] Word n-grams (1 to 4, using Jaccard and containment measures).
- [ ] Evaluate string-based features:
  - [ ] Compute Pearson correlation for results.
  - [ ] Analyze where these techniques succeed or fail.

---

## 4. Semantic Similarity Measures
- [ ] Implement pairwise word similarity:
  - [ ] Use WordNet-based measures (e.g., Resnik, Jiang-Conrath).
  - [ ] Aggregate pairwise word similarity scores using IDF weighting.
- [ ] Implement Explicit Semantic Analysis (ESA):
  - [ ] Use Wikipedia or Wiktionary as vector space resources.
- [ ] Add Distributional Thesaurus-based similarity:
  - [ ] Compute similarity using dependency-parsed sentences.
- [ ] Evaluate semantic features:
  - [ ] Compute Pearson correlation for each technique.
  - [ ] Compare to string-based methods.

---

## 5. Text Expansion Mechanisms
- [ ] Add lexical substitution:
  - [ ] Implement a lexical substitution system to provide synonyms for frequent nouns.
- [ ] Implement statistical machine translation (SMT):
  - [ ] Perform back-translation using bridge languages (e.g., English → German → English).
  - [ ] Combine original text and back-translations for similarity computation.
- [ ] Evaluate text expansion features:
  - [ ] Analyze the impact of substitution and SMT on performance.

---

## 6. Structural and Stylistic Features
- [ ] Implement structural measures:
  - [ ] POS n-grams (2 to 4).
  - [ ] Stopword n-grams (2 to 4).
- [ ] Implement stylistic measures:
  - [ ] Function word frequencies.
  - [ ] Statistical text properties (e.g., type-token ratio).
- [ ] Evaluate structural and stylistic features:
  - [ ] Compute Pearson correlation and analyze results.

---

## 7. Feature Combination and Model Training
- [ ] Combine all similarity features:
  - [ ] Log-transform feature values.
  - [ ] Use linear regression (e.g., WEKA or scikit-learn).
- [ ] Train the model:
  - [ ] Use 10-fold cross-validation on the `train` dataset.
  - [ ] Evaluate on a held-out test set.
- [ ] Optimize feature selection:
  - [ ] Experiment with combinations of ~20 best features.
  - [ ] Remove redundant features.

---

## 8. Evaluation
- [ ] Evaluate final model:
  - [ ] Compute Pearson correlation for the `test` dataset.
  - [ ] Compare results with SemEval-2012 benchmarks.
- [ ] Document comparison with baseline and other candidates.

---

## 9. Analysis and Visualization
- [ ] Perform error analysis:
  - [ ] Identify cases where lexical or semantic features fail independently.
  - [ ] Highlight improvements from combined features.
- [ ] Create visualizations:
  - [ ] Bar charts or line graphs comparing methods.
  - [ ] Scatterplots or confusion matrices for insights.

---

## 10. Presentation Preparation
- [ ] Create slides:
  - [ ] Cover objectives, dataset, methods, results, analysis, and conclusions.
- [ ] Rehearse presentation:
  - [ ] Prepare clear explanations for methods and results.
  - [ ] Anticipate likely questions from peers and prepare answers.

---

## 11. Participation
- [ ] Attend all peer presentations.
- [ ] Prepare 1–2 insightful questions per session:
  - Example: "How did you address the limitations of lexical similarity methods?"
- [ ] Be ready to answer questions about your approach and analysis.

---

## 12. Submission
- [ ] Prepare submission files:
  - [ ] Ensure the Jupyter Notebook is cleaned and named `sts-[Student1]-[Student2].ipynb`.
  - [ ] Finalize presentation slides as `sts-[Student1]-[Student2].pdf`.
- [ ] Verify:
  - [ ] Code runs without errors.
  - [ ] All deliverables meet submission requirements.
- [ ] Submit before **December 12th**.

---

## General Notes
- [ ] Focus on achieving strong **analysis and conclusions**.
- [ ] Ensure performance beats the baseline (31% Pearson).
- [ ] Review Task 6 benchmarks and ensure your system aligns with expectations.
- [ ] Aim for active participation during presentations to maximize marks.
