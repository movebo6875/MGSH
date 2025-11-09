## Meta-Guided Sample Reweighting for Robust Cross-Modal Hashing Retrieval with Noisy labels

PyTorch implementation for Meta-Guided Sample Reweighting for Robust Cross-Modal Hashing Retrieval with Noisy labels. (AAAI 2026). 

## MGSH framework

![MGSH_framework](figure\MGSH_framework.png)Overview of the MGSH framework. The framework consists of three main components: (1) Feature Extraction: A dual-stream encoder processes training and meta samples to obtain modality-specific representations. (2) Bi-level Network Architecture: Based on the meta-importance weights, the main model \( \mathcal{F}_{\Theta} \) computes a robust hashing loss function, while the meta model \( \mathcal{G}_{\Phi} \) reweights the samples and provides the updated weights to the main model. (3) Meta Pipeline: The meta-learning process updates the parameters of both models, enabling robust sample reweighting and adaptive margin adjustment.


## Train with Our Model
Before running the main script, you need to generate the `.h5` file and the noise. To do this, run `tools.py` and `generate.py`:
```bash
python ./utils/tools.py
python ./noise/generate.py
```
Then in our model, we split a small meta-clean dataset from the `.h5` file :

```bash
python ./meta/generate_meta.py
```

You can set `META_RATE` to change the meta-dataset size to train meta-network.

Once the `.h5` file and noise are generated, you can run the main script `MGSH.py` for MIRFlickr-25K dataset:

```bash
python MGSH.py   --Lambda 0.7  --r 0.5 --margin 0.5 --tau 0.5
```
We have already provided the trained model under 50% noise in 64-bit on MIRFlickr-25K dataset. You can download the model dataset from [here](https://drive.google.com/drive/folders/1Zk1dZIAUlUKcKscPPhgm6hJvSFz6QkjF).

## Experiment Results:
![experiment](figure\experiment.png)

