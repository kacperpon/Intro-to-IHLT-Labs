# TODO

- [x] Save preprocessed files
- [x] Load saved preprocessed files to use
- [x] Grid search for each model
- [ ] Adjust (or add to) pipeline to accurately map work from "Statement" slide of brief
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
- [ ] Data analysis [e.g.: plot history of model]
- [ ] [Optional] Feature selection
- [x] Refactor + tidy code and files
    - [ ] Finish feature extraction file
    - [ ] Preprocess file
- [x] Deal with failed combinations in MLP gracefully
- [ ] Error handling throughout
- [ ] Define README.md
- [ ] Write analysis and conclusion
- [ ] Finish presentation slides