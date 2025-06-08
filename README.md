# Overview
We share here a multiview, multioutput GNN-based model for an urban incident prediction task. Our model is summarized in the figure below. Our model uses two sources of data for this prediction task.
 - First, we provide observations of the ground truth incident state via *government inspections* which generate *ratings* for neighborhoods. For example, New York City conducts street inspections for every street and rates them from 1-10. Importantly, these inspections are only conducted for some incident types and neighborhoods and are thus sparsely observed.
 - We also provide another source of data: frequently observed, biased proxies of the incident state, via crowdsourced *reports* of incidents. Unlike ratings, indicators of whether reports are made are observed across all incident types, all neighborhoods, and multiple points in time.

Using this data, we train our model to simultaneously predict ground-truth ratings using learned node and type embeddings and infer how the likelihood of reporting varies by demographics, conditional on ground-truth. Our model's novel contribution lies in adapting GNN architectures for biased data settings by connecting multi-view datasets through a multi-output loss function.

[model.pdf](https://github.com/user-attachments/files/20646996/model.pdf)


# Repository structure
