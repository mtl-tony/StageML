# StagedML
ML algorithms are constantly used for Supervised learning problems in both regression and classification. 

There is constantly a tradeoff of interpretability vs performance. This repo aims to have a model architecture in which this tradeoff can be accurately measured as we iteratively train up from fully interpretable models to fully complex models. This model can also be fit in stages and in different subsets of the data to ensure certain features can be seen first.

## Modelling Philosophy
The logic behind doing boosting in stages like in stageml repo is to expand on the concept of boosting.

It allows training models incrementally in an iterative fashion. Typically all features are looked at and analyzed at the same time in which we apply a score to a validation set until boosting stops. However the goal of this package is to sepearte these different portions of boosting. The result allows for being able to identify to which extent that interactions are truly required. 

Each stage focuses modeling power on the residuals (errors) from the previous stage's predictions. This forces diversity in the kinds of patterns each stage learns. This does not only allow model developers to analyze individual vs interaction impact but also features in an order of relevance. An example would be for a model to fit specific features that are always available for the whole population, and then features that are only available at times for a diffrent subset.

Staging breaks the boosting process into discrete "chunks" that are easier to optimize, monitor, and understand compared to one massive model. Intermediate stage predictions can be used for applications that require progressive results, not just a final model.

Failed stages can be identified and addressed earlier. Model performance converges over stages rather than all at once.

## Models Implemented
Currently models are only available for a subset of models

- GLM (Generalized Linear Models from sklearn.linear_model)
- GAM (Generalized Additive Models from interpretml ExplainableBoostingMachines)
- GA2M (Generalized Additive Models with Pairwise Interactions (Fitting only the pairwise interactions) from interpretml ExplainableBoostingMachines)
- GBM (Gradient Boosting Machines from lightGBM)
