# PyTorch RecSys

This repo contains a pytorch implementation of ALS algorithm for Collaborative Filtering and its extentions &mdash; eALS and iALS. Additionally, it provides the implementation of NGCF &mdash; Neural Graph for Collaborative Filtering.

## Installation 

```bash
git clone https://github.com/Mr6one/recsys-project-2023.git && cd recsys-project-2023
pip install -r requirements.txt
```

## Usage

### Basics
For all models you need simply provide a single matrix X of user-item interactions in any sparse format supported by scipy.sparse. For NGCF you need additionally provide the number of users and items.
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

### Checkpoints
You can save and load the pretrained models. **Note:** models are allocated to cpu before saving, so don't forget to switch back to cuda after loading.

```
model = iALS(device='cuda').fit(X) # train model on large dataset with high-end GPU
model.save('./weights/ials.pkl') # save model
model = iALS.from_checkpoint('./weights/ials.pkl').cuda() # use it on weak laptop
```

### Logging
Moreover, the NGCF model supports tensorboard for logging and you can track the training process with following command

```bash
tensorboard --logdir=lightning_logs --port=6009
```
Then go to localhost:6009.

## Results

TODO: add visualization

The most straingtforward way to reproduce our results is

```bash
python main.py --config ./configs/{model_name}_base.json
```

Don't forget to chose the appropriate dataset.
