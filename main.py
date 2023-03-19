
'''
from torch.utils.data import DataLoader

from src.utils import parse_args
from src.models import Net
from src.trainers import create_trainer
from src.datasets import Dataset


def main(config=None):
    if config is None:
        config = parse_args()

    training_dataset, validation_dataset, _ = Dataset(**config['data']).data
    batch_size = config['trainer'].pop('batch_size')

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    config['model']['learning_rate'] = config['trainer'].pop('learning_rate')
    model = Net.create_from_config(config['model'])
    
    trainer = create_trainer(model, **config['trainer'])
    trainer.fit(train_dataloaders=training_dataloader, val_dataloaders=validation_dataloader)


if __name__ == '__main__':
    main()
'''
