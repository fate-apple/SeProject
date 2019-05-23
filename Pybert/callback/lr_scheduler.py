

class BertLR(object):
    def __init__(self,optimizer,lr,t_total,warmup):
        self.lr = lr;
        self.optimizer = optimizer
        self.t_total = t_total
        self.warmup  = warmup

    def warmup_mode(self,x,warmup=0.002):
        #Linear
        if x< warmup:
            return x/warmup
        return 1-x

    def batch_step(self,training_step):
        lr_schedule = self.warmup_mode(training_step/self.t_total,self.warmup)
        lr_step = self.lr * lr_schedule
        for param_group in self.optimizer.param_groups:
            param_group['lr']  =lr_step
