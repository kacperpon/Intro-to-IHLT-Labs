# Semantic Textual Similarity Project (IHLT)

## Project Weightage
- **30%**: Content of project.
- **10%**: Questions we ask during other presentations.
- **10%**: How we answer questions from others.
- **Overall**: **50% of CA**.

---

## Task Overview
- **Goal**: Detect paraphrases and assign similarity labels to text pairs.
- **Labels**:
  - 5: Completely equivalent (same meaning).
  - 4: Mostly equivalent (minor unimportant differences).
  - 3: Roughly equivalent (some important info differs/missing).
  - 2: Not equivalent but share some details.
  - 1: Not equivalent but same topic.
  - 0: Completely different topics.
- Labels are **subjective** and created by humans.

---

## Data Usage
- **`trial` File**:
  - Used for prototyping.
  - Contains errors but useful to understand input-output formats.
  - Not provided; labs will act as our "trial".
- **`train` File**:
  - Main dataset to exploit for model training.
- **`test` File**:
  - Contains unlabeled data for evaluation.
  - Labels are generated as part of the task.

---

## Key Rules
1. **No resources or models created after 2012**:
   - No embeddings (e.g., BERT, Word2Vec).
   - Pre-trained embedding models are **not allowed**.
   - Tools like SpaCy are permitted but without embeddings.
2. Need **analysis and conclusions** for all experiments.
3. Only **slides and Jupyter Notebook** are required for submission.
   - No PDF report needed.

---

## Expected Workflow
1. **Trial Phase**:
   - Use lab sessions to simulate `trial` set.
   - Build and debug models here.
2. **Training Phase**:
   - Train models using the `train` dataset.
3. **Evaluation Phase**:
   - Test models on the `test` dataset.
   - Use benchmarks provided in the "Task 6 Results" table for comparison.

---

## Tips for Success
- Benchmarks (Slide 6):
  - **Baseline**: Avoid going below baseline (it’s simple to beat).
  - Results > **0.7562 Pearson** score are outstanding.
- **Analysis**:
  - Analysis is key; focus on detailed comparisons and insights.
  - Refer to papers for high-quality analysis examples.
- **Collaboration**:
  - Attend all presentations.
  - Ask **1-2 well-thought-out questions** per person to earn full marks.
  - Answer questions clearly to maximize marks.

---

## Deliverables
- **Deadline**: December 12th.
- **Submission**:
  - A large, structured Jupyter Notebook split into readable sections.
  - Presentation slides summarizing:
    1. Objectives.
    2. Methodologies.
    3. Results.
    4. Analysis.
    5. Conclusions.
- **Evaluation Benchmarks**:
  - Use "Task 6 Results" table for insights on model techniques and comparison.

---

## General Notes
- Achieving **10/10 overall** is rare but:
  - Scoring **10/10 for analysis** is possible with thorough effort.
- Most groups score **9 or above** based on historical results.
- Test models and results must align with the guidelines on **Slide 7**.

---
