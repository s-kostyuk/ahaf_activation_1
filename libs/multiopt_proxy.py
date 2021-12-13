import torch


class MultiOptProxy:
    def __init__(self, opt_lambda, sched_lambda, net=None, trainer=None):
        self._opt_lambda = opt_lambda
        self._sched_lambda = sched_lambda
        self._opts = []
        self._scheds = []
        self._trainer = trainer

        if net is not None:
            self._create_for_net(net, trainer)

    def _create_for_net(self, net, trainer=None):
        if trainer is None:
            params_sets = [net.parameters()]
        elif trainer == "dsu3":
            params_sets = net.dsu_param_sets
        elif trainer == "dsu4":
            params_sets = net.dsu2_param_sets
        else:
            params_sets = [net.parameters()]

        for s in reversed(params_sets):
            self._create_for(s)

    def _create_for(self, param_set):
        opt = self._opt_lambda(param_set)
        self._opts.append(opt)
        if self._sched_lambda:
            sched = self._sched_lambda(opt)
            self._scheds.append(sched)

        return len(self._opts) - 1

    def get_opt(self, idx=None):
        if self._trainer == "dsu3" or self._trainer == "dsu4":
            return self._opts[idx]

        return self._opts[0]

    def sched_step(self):
        for s in self._scheds:
            s.step()

    def get_last_lr(self):
        return self._scheds[0].get_last_lr()
