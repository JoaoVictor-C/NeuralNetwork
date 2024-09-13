import numpy as np

class LearningRateScheduler:
    def __init__(self, config):
        self.config = config['learning_rate']

    def get_lr_scheduler(self):
        scheduler_type = self.config.get('lr_scheduler', 'constant')
        if scheduler_type == 'step':
            return self.step_scheduler
        elif scheduler_type == 'exponential':
            return self.exponential_scheduler
        elif scheduler_type == 'cyclical':
            return self.cyclical_scheduler
        elif scheduler_type == 'cosine_annealing':
            return self.cosine_annealing_scheduler
        else:
            return self.constant_scheduler

    def step_scheduler(self, epoch):
        if self.config['lr_use_warm_up'] and epoch < self.config['lr_warm_up_epochs']:
            return self.warm_up_scheduler(epoch)
        return self.config['learning_rate'] * (self.config['lr_decay'] ** (epoch // self.config['lr_step']))

    def exponential_scheduler(self, epoch):
        if self.config['lr_use_warm_up'] and epoch < self.config['lr_warm_up_epochs']:
            return self.warm_up_scheduler(epoch)
        return self.config['learning_rate'] * (self.config['lr_decay'] ** epoch)

    def cyclical_scheduler(self, epoch):
        if self.config['lr_use_warm_up'] and epoch < self.config['lr_warm_up_epochs']:
            return self.warm_up_scheduler(epoch)
        cycle = np.floor(1 + epoch / (2 * self.config['lr_step_size']))
        x = np.abs(epoch / self.config['lr_step_size'] - 2 * cycle + 1)
        return self.config['lr_min'] + (self.config['learning_rate'] - self.config['lr_min']) * np.maximum(0, (1 - x))

    def cosine_annealing_scheduler(self, epoch):
        if self.config['lr_use_warm_up'] and epoch < self.config['lr_warm_up_epochs']:
            return self.warm_up_scheduler(epoch)
        return self.config['lr_min'] + (self.config['learning_rate'] - self.config['lr_min']) * (1 + np.cos(np.pi * epoch / self.config['lr_T_max'])) / 2

    def warm_up_scheduler(self, epoch):
        if epoch < self.config['lr_warm_up_epochs']:
            return self.config['lr_min'] + (self.config['learning_rate'] - self.config['lr_min']) * epoch / self.config['lr_warm_up_epochs']
        return self.config['learning_rate']

    def constant_scheduler(self, epoch):
        return self.config['learning_rate']
