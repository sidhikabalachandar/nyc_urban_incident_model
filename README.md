# Overview
We share here a multiview, multioutput GNN-based model for an urban incident prediction task. Our model is summarized in the figure below. Our model uses two sources of data:
 - First, we provide observations of the ground truth incident state via *government inspections* which generate *ratings* for neighborhoods. For example, New York City conducts street inspections for every street and rates them from 1-10. Importantly, these inspections are only conducted for some incident types and neighborhoods and are thus sparsely observed.
 - We also provide another source of data: frequently observed, biased proxies of the incident state, via crowdsourced *reports* of incidents. Unlike ratings, indicators of whether reports are made are observed across all incident types, all neighborhoods, and multiple points in time.
We share our data publicly [here](https://github.com/sidhikabalachandar/nyc_urban_incident_data).

Using this data, we train our model to simultaneously predict ground-truth ratings using learned node and type embeddings and infer how the likelihood of reporting varies by demographics, conditional on ground-truth. Our model's novel contribution lies in adapting GNN architectures for biased data settings by connecting multi-view datasets through a multi-output loss function.

![Picture1](https://github.com/user-attachments/assets/71b1f9c9-fd91-46d4-b416-3beec8491079)

# Repository structure
In the `model` folder we provide our model and code to train and evaluate the model. This folder contains the following files:
 - `model/model.py` provides our model
 - `model/data.py` provides our dataloader and custom data sampler
 - `model/create_graph.ipynb` creates a networkx graph of New York City census tracts
 - `model/train.py` provides code to train the model
 - `model/test.py` provides code to evaluate a trained model

In the `data` folder we provide code to generate semisynthetic data. This data uses real reporting data and real demographic data, and synthetically generates ratings.

In the `results` folder we provide code to reproduce each figure and table in our paper.

In the `figures` folder we provide the files of each figure in our paper. 
