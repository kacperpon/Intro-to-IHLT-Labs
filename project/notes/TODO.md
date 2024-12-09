# TODO

- [x] Save preprocessed files
- [x] Load saved preprocessed files to use
- [x] Grid search for each model
- [ ] Adjust (or add to) pipeline to accurately map work from "Statement" slide of brief
- [ ] Data analysis [e.g.: plot history of model]
- [ ] Define types for each feature [lexical, semantic or syntatic]
- [ ] [Optional] Feature selection
- [x] Refactor + tidy code and files
    - [x] Finish feature extraction file
    - [x] Preprocess file
- [x] Deal with failed combinations in MLP gracefully
- [ ] Define README.md
- [ ] Write analysis and conclusion
- [ ] Finish presentation slides


ASK PAU:
- [ ] Run feature extraction pipeline again to ensure refactor didn't mess stuff up
- [ ] Run RF (and maybe other) model(s) with additional parameters in param grid
    # TODO: Run this grid with beefier hardware, it's too slow for me to test
    # Define the grid of hyperparameters to search
    # param_grid = {
    #     'n_estimators': [100, 200, 300],  
    #     'max_depth': [None, 10, 20, 30], 
    #     'min_samples_split': [2, 5, 10], 
    #     'min_samples_leaf': [1, 2, 4],   
    #     'max_features': ['sqrt', 'log2'],
    #     'bootstrap': [True, False],      
    # }