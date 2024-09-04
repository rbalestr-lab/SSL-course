# Guillotine regularization

## Guillotine Regularization: Why removing layers is needed to improve generalization in Self-Supervised Learning ([link](https://arxiv.org/pdf/2206.13378))

One unexpected technique that emerged in recent years consists in training a Deep Network (DN) with a Self-Supervised Learning (SSL) method, and using this network on downstream tasks but with its last few projector layers entirely removed. This trick of throwing away the projector is actually critical for SSL methods to display competitive performances on ImageNet for which more than 30 percentage points can be gained that way. This is a little vexing, as one would hope that the network layer at which invariance is explicitly enforced by the SSL criterion during training (the last projector layer) should be the one to use for best generalization performance downstream. But it seems not to be, and this study sheds some light on why. This trick, which we name Guillotine Regularization (GR), is in fact a generically applicable method that has been used to improve generalization performance in transfer learning scenarios. In this work, we identify the underlying reasons behind its success and show that the optimal layer to use might change significantly depending on the training setup, the data or the downstream task. Lastly, we give some insights on how to reduce the need for a projector in SSL by aligning the pretext SSL task and the downstream task.


## Toy experiment on [dsprites](https://github.com/google-deepmind/dsprites-dataset) dataset

- run the training pipeline using the [guillotine.py](./guillotine.py) Python file
    - This will generate 4 .csv files with the results (assumes you have 1 GPU available)
- read the saved save using the [reader.ipynb](./reader.ipynb) Jupyter notebook
    - This will produce the two heatmaps to compare performances per-task with and without guillotine