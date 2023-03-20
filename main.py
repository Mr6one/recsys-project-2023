from src.utils import parse_args
from src.models import models
from src.datasets.utils import load_dataset


def main(config=None):
    if config is None:
        config = parse_args()

    _, testset, holdout, training_matrix, data_description = load_dataset(config['dataset'], data_path='./data')

    if config['model'] == 'ngcf':
        n_users, n_items = training_matrix.shape
        config['model_args']['n_users'] = n_users
        config['model_args']['n_items'] = n_items

    model = models[config['model']](**config['model_args'])
    model.fit(training_matrix, [holdout, testset, data_description])


if __name__ == '__main__':
    main()
