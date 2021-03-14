import torch


class QnABatcher(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        srcs, tgts = zip(*batch)
        S = torch.LongTensor(srcs)
        T = torch.LongTensor(tgts)

        return S.to(self.device), T.to(self.device)