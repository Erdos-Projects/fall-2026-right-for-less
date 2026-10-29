# Step 4: Final Test

## What this step does

The final test comes after the policy selection on validation set. Here we freeze all the training parameters and hyperparameter $\lambda$ (see Step 3: Policy Evaluation for details). Specifically, given the training results in Step 2 (Router Training), we choose the E5 embedding and the kNN model for score and cost predictions. For each question $x$ in the test set, we embed it by E5 embedding to $\phi(x)$ and perform kNN model predictions for its accuracy $\hat{p}_m(x)$ and cost $\hat{c}_m(x)$ by each candidate LLM (indexed by $m$). We then calculate the overall score $S_m(x;\lambda)$ and choose the LLM that minimizes this score, $\pi_{\lambda}(x)$.

The resulting $\pi_{\lambda}(x)$'s on all the test sample $x$'s are the result of the final test. Before evluating the test result, note that there is no single metric indicating how well the router performs. There is an intrinsic tradeoff between accuracy and cost. Generally, if the router tends to select the most accurate LLM, the average cost will be higher, and vice versa. It is important that the router should combine all the candidate LLMs in a reasonable way so that it is **not strictly worse than any one of the candidate LLMs**. Specifically, we choose GPT-4 as a reference since it is regarded a good LLM among the 11 candidates in the dataset. **As a baseline, the router should not be less accurate while cost more than the reference model**.

Besides the accuracy and cost, some other KPIs of the router are considered and calculated (see kpis.md for details).

## How to run this step

There is one python Notebook, Final_Test_kNN_E5_Frozen_Policy_KPIs_TieFix.ipynb, in this directory. To run this Notebook, several steps should be finished beforehand, which are listed below.

### Make sure embeddings are done

In the directory 1_data_preparation/, the following three Notebooks should be run in order to create the embedded vectors of the test set:

1. step1_TFIDF_complete.ipynb

2. step1_e5_complete.ipynb

3. loading_template.py

The resulting vectors will be automatically saved into 1_data_preparation/data/, which will be used by further steps. Note that even though we do not use TF-IDF embeddings in the test, we still need to run step1_TFIDF_complete.ipynb because the modeling Notebook (see below) deals with both embeddings.

### Make sure models are built

After embeddings are created, the Notebook TF-IDF_E5_Score_Classification_Router.ipynb in the directory 2_Router_Training/ should be run to build the models for accuracy and cost predictions. The model parameters are stored in 2_Router_Training/experiment_ouputs/.

### (Optional) Policy selection

Since the model parameters will be used in the test step, it is essential that the model has to be built and the 2_Router_Training/TF-IDF_E5_Score_Classification_Router.ipynb has to be called. However, the only information needed from the policy selection step is the single number $\lambda$. In the notebook Final_Test_kNN_E5_Frozen_Policy_KPIs_TieFix.ipynb in this directory, $\lambda$ has been initialized to the correct value corresponding to kNN model and E5 embeddings. However, it is a good practice to reproduce this result by yourself -- so you are recommended to run the policy selection program on your local PC.

In the directory 3_policy_evaluation, find the Notebook viz_notebook.ipynb and set model_name = 'knn' and embed_name = 'e5'. Run the notebook and obtain the $\lambda$ at the end. Note that this notebook selects $\lambda$ from a grid, so it is possible that two independent runs with different models or embeddings result in the same $\lambda$!

Note that you are also encouraged to try different embeddings or models in this step and it is possible to obtain different $\lambda$. However, the notebook Final_Test_kNN_E5_Frozen_Policy_KPIs_TieFix.ipynb in the current directory only applies to kNN model and E5 embeddings. Future updates might introduce other notebooks for different model and embedding pairs.

### Finally, run the test

In the Notebook Final_Test_kNN_E5_Frozen_Policy_KPIs_TieFix.ipynb, set the FROZEN_LAMBDA to be the $\lambda$ you obtain in the last step (it should be the same as the initialized value). Then run the notebook.

The Notebook analyzed the test result by comparing the accuracy and cost of the router to the reference GPT-4 model. It also summarizes the distribution of LLMs that the router turns to while dealing with the questions in the test set.