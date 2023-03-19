from src.utils import parse_args
from src.models import models
from src.datasets.utils import load_dataset


def main(config=None):
    if config is None:
        config = parse_args()

    _, _, _, interactions_matrix, _ = load_dataset(config['dataset'], data_path='./data')

    if config['model'] == 'ngcf':
        n_users, n_items = interactions_matrix.shape
        config['model_args']['n_users'] = n_users
        config['model_args']['n_items'] = n_items

    model = models[config['model']](**config['model_args'])
    model.fit(interactions_matrix)


if __name__ == '__main__':
    main()
