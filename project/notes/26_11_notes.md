Raw data is pre-processed before any feature / measure technique can be applied.

Suggestion for this: a pipeline!

e.g.:

folder structure as follows:
(inside 'data/train' and 'data/test')
- 0.1_raw
- 0.2_preprocessed

some preprocess .py or .ipynb file, takes in raw data from 0.1_raw and outputs fully preprocessed data in 0.2_preprocessed
 -> key is data can be preprocessed in 1 run of file -> easily reproducible and adjustable. 

---

### Training

Once data is pre-processed, **apply feature techniques -> need to decide on what ones.**

Get scores as output for each sentence pair for each feature technique

Input 1: Combine results from each technique for each pair into a vector.
Input 2: Gold standard scores corresponding to each sentence pair

(Input 2 doesnt go directly inside the model, but is an input in the training process)

Model uses input 1 to predict a similarity score, error is calculated based on this prediction against input 2 -> model adjusts parameters to minimise error.


---

### Testing

Model is now trained.

Same process for data as 'Training'.

Difference here is we don't include the gold standard score as an input.

Trained model outputs gold standard scores for each sentence pair -> we can compare accuracy of these scores against the expected scores in test folder.

### Evaluation

Compute Pearson correlation.

