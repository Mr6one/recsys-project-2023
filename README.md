# PyTorch RecSys

This repo contains a pytorch implementation of ALS algorithm for Collaborative Filtering and its extentions &mdash; eALS and iALS. Additionally, it provides the implementation of NGCF &mdash; Neural Graph for Collaborative Filtering.

## Installation 

```bash
git clone https://github.com/Mr6one/recsys-project-2023.git && cd recsys-project-2023
pip install -r requirements.txt
```

## Usage

For all models you need simply provide a single matrix X of user-item interactions in any sparse format supported by scipy.sparse. For NGCF you need additionally provide the number of users and items.
```python
from src.models import ALS # eALS, iALS, NGCF


model = ALS().fit(X) # fit the model
model.recommend(user_ids, N=5) # generate top 5 recommendations for users
```

All models support GPU and you can easily switch the devices

```python
model = model.cpu() # switch to cpu
model = model.cuda() # switch to gpu
```
