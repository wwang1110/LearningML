import torch

class MLPDataset(torch.utils.data.Dataset):

    def __init__(self, len):
        self.len = len

    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = torch.rand(20).float()
        item['labels'] = torch.rand(3).float()
        #item['labels'] = torch.Tensor(3)
        return item

    def __len__(self):
        return self.len