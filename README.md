# PyTorch RecSys

This repo contains a PyTorch implementation of ALS algorithm for Collaborative Filtering and its extentions &mdash; eALS and iALS. Additionally, it provides the implementation of NGCF &mdash; Neural Graph for Collaborative Filtering.

Project [presentation](https://github.com/Mr6one/recsys-project-2023/blob/main/presentation.pdf)  
Project [report](https://github.com/Mr6one/recsys-project-2023/blob/main/report.pdf) 
## Installation 

```bash
git clone https://github.com/Mr6one/recsys-project-2023.git && cd recsys-project-2023
pip install -r requirements.txt
```

## Usage

### Basics
For all models you need simply provide **a single matrix X of user-item interactions** in any sparse format supported by scipy.sparse. For NGCF you need additionally provide the number of users and items.
```python
from src.models import ALS # eALS, iALS, NGCF


model = ALS().fit(X) # fit the model
model.recommend(user_ids, N=5) # generate top 5 recommendations for users
```

### GPU Utilization
All models support GPU and may be easily transfered between devices

```python
model = ALS(device='cuda').fit(X) # create and fit model on cuda
model = model.cpu() # switch to cpu
model = model.cuda() # back to gpu
```

### Checkpointing
You can save and load the pretrained models. **Note:** the models are allocated to cpu before saving, so don't forget to switch back to cuda after loading.

```python
model = iALS(device='cuda').fit(X) # train model on large dataset with high-end GPU
model.save('ials.pkl') # save model
model = iALS.from_checkpoint('ials.pkl').cuda() # use it on weak laptop
```

### Logging
Moreover, the NGCF model supports tensorboard for logging and you can track the training process with following command

```bash
tensorboard --logdir=lightning_logs --port=6009
```
Then go to localhost:6009

## Results

Metrics obtained on the MovieLens dataset 

![image](https://user-images.githubusercontent.com/67689354/227789667-6e65c2a5-e165-4b52-a0ae-184f82bae2ea.png)

Metrics obtained on the Yelp dataset

![image](https://user-images.githubusercontent.com/67689354/227789707-1c89d1e0-11fc-438e-b858-a50f58f2ff34.png)

Time complexities graphs for ALS, eALS, iALS models

<img src="https://user-images.githubusercontent.com/67689354/227789623-64e5824f-b235-4bc4-a1f3-e15fb081541e.png"  width="100%">

For full reuslts see the [notebook](https://github.com/Mr6one/recsys-project-2023/blob/main/notebooks/experiments.ipynb). The most straingtforward way to reproduce our results is

```bash
python main.py --config ./configs/{model_name}_base.json
```

Or use the [pretrained models](https://github.com/Mr6one/recsys-project-2023/tree/main/weights). Don't forget to choose the appropriate dataset.
