# TODO

- [x] Save preprocessed files
- [x] Load saved preprocessed files to use
- [x] Grid search for each model
- [ ] Adjust (or add to) pipeline to accurately map work from "Statement" slide of brief
- [ ] Run RF (and maybe other) model(s) with additional parameters in param grid
    - [ ] [Optional] Load project into Colab to run on GPU
- [ ] Initial data analysis
- [ ] (Attempt) Feature selection
- [ ] Refactor + tidy code and files
- [ ] Write analysis and conclusion
- [ ] Finish presentation slides
- [ ] Deal with failed combinations in MLP gracefully
- [ ] For RF:
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
- [ ] Error handling throughout
- [ ] Reinstate plot history for training models
- [ ] Define README.md