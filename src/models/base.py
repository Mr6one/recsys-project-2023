import pickle


class BaseModel:
    def __init__(self):
        self.device = 'cpu'

    @staticmethod
    def from_checkpoint(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model
    
    def _save_model(self, save_path):
        self.cpu()
        with open(save_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.device == 'cuda':
            self.cuda()
    
    def save(self, path):
        self._save_model(path)

    def _to_device(self):
        pass

    def cpu(self):
        self._to_device('cpu')
        return self

    def cuda(self):
        self._to_device('cuda')
        return self
