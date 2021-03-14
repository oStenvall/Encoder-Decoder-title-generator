class QnABatcher(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        srcs, tgts = zip(*batch)


        return srcs.to(self.device), tgts.to(self.device)