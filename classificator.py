import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import config
from calibration import asr_for_both_conditions
from utils import balance_conditions

CLASSIFIERS = {
    'Logistic_Regression': LogisticRegression()
}

PARAMETERS = {
    'Logistic_Regression': {
        'classifier__C': np.logspace(-3, 3, 7),
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    }
}


df_all = asr_for_both_conditions(fname_calibfile=config.FNAME_CALIBFILE,
                                 fname_rest=config.FNAME_REST,
                                 fname_task=config.FNAME_TASK)

df = balance_conditions(df_all, 'condition')

X = df.drop(['channel', 'condition'], axis=1)
y = df['condition']  # .replace({'norm': 0, 'sch': 1}).values

############################################################
#                       GRID SEARCH                        #
############################################################
# Prepare a dictionary to store the best models after grid search
best_models = {}

# Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Loop through each classifier and its parameters for grid search
for name, clf in CLASSIFIERS.items():
    best_models[name] = {}
    # Create a pipeline with the classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf) #('scaler', StandardScaler()), ('selector', SelectKBest(f_classif))
    ])

    # Perform grid search with stratified cross-validation
    grid_search = GridSearchCV(pipeline, PARAMETERS[name], cv=cv, scoring='accuracy')

    # Fit the grid search
    grid_search.fit(X, y)

    # Store the best model
    best_models[name]['model'] = grid_search.best_estimator_
    best_models[name]['score'] = grid_search.best_score_
    best_models[name]['params'] = grid_search.best_params_


for name, model in best_models.items():
    print(f"Best model for {name}: {model}")
