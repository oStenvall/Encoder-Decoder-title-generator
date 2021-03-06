import torch


class TitleQuestionBatcher(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        srcs, tgts, tgts_gt = zip(*batch)
        S = torch.LongTensor(srcs)
        T = torch.LongTensor(tgts)
        T_gt = torch.LongTensor(tgts_gt)

        return S.to(self.device), T.to(self.device), T_gt.to(self.device)
