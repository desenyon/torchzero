"""Optimizers: SGD, Adam, AdamW; gradient clipping; LR schedules.

Equations
---------
SGD (momentum optional):
    v <- mu * v + g                (PyTorch-style dampening=0)
    p <- p - lr * v                (+ weight decay: p <- p - lr*wd*p first)

Adam (Kingma & Ba 2015):
    m <- beta1*m + (1-beta1)*g
    v <- beta2*v + (1-beta2)*g^2
    mhat <- m / (1-beta1^t);  vhat <- v / (1-beta2^t)
    p <- p - lr * mhat / (sqrt(vhat) + eps)

AdamW (Loshchilov & Hutter 2019, decoupled weight decay):
    p <- p - lr*wd*p               (before the Adam update)
    then the Adam update above.
"""

from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = [p for p in params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_dict(self):
        return {
            "optimizer": type(self).__name__,
            "hyperparameters": {key: getattr(self, key) for key in self._hyperparameters},
            "param_state": [
                {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in state.items()}
                for state in self._state_list()
            ],
        }

    def load_state_dict(self, sd):
        if sd.get("optimizer", type(self).__name__) != type(self).__name__:
            raise ValueError("optimizer type mismatch")
        states = sd["param_state"]
        current = self._state_list()
        if len(states) != len(current):
            raise ValueError("optimizer state count mismatch")
        for state, saved in zip(current, states):
            if set(state) != set(saved):
                raise ValueError("optimizer state keys mismatch")
            for key, value in saved.items():
                if np.asarray(value).shape != np.asarray(state[key]).shape:
                    raise ValueError(f"optimizer buffer shape mismatch: {key}")
        hyper = sd.get("hyperparameters", {})
        if hyper and set(hyper) != set(self._hyperparameters):
            raise ValueError("optimizer hyperparameter mismatch")
        for state, saved in zip(current, states):
            for key, value in saved.items():
                # _state_list may construct temporary dicts; mutate the real
                # buffers in place so SGD's velocity is actually restored.
                if isinstance(state[key], np.ndarray):
                    state[key][...] = value
                else:
                    state[key] = value
        for key, value in hyper.items():
            setattr(self, key, value)

    def _state_list(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    _hyperparameters = ("lr", "momentum", "weight_decay", "dampening", "nesterov")

    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0,
                 dampening=0.0, nesterov=False):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        self._velocity = [np.zeros_like(p.data) for p in self.params]
        if nesterov and momentum <= 0:
            raise ValueError("nesterov momentum requires momentum > 0")

    def _state_list(self):
        return [{"velocity": v} for v in self._velocity]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.astype(p.data.dtype)
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            if self.momentum:
                v = self._velocity[i]
                v[:] = self.momentum * v + (1 - self.dampening) * g
                if self.nesterov:
                    g = g + self.momentum * v
                else:
                    g = v
            p.data -= self.lr * g


class Adam(Optimizer):
    _hyperparameters = ("lr", "beta1", "beta2", "eps", "weight_decay")

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False):
        super().__init__(params)
        self.lr = lr
        if amsgrad:
            raise ValueError("amsgrad is not implemented; use standard Adam or AdamW")
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]

    def _state_list(self):
        return [{"m": m, "v": v} for m, v in zip(self._m, self._v)] + \
               [{"t": np.array(self.t)}]

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        self.t = int(sd["param_state"][-1]["t"])

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1 - b1 ** self.t
        bc2 = 1 - b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.astype(np.float64)
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            m, v = self._m[i], self._v[i]
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            mhat = m / bc1
            vhat = v / bc2
            p.data -= (self.lr * mhat / (np.sqrt(vhat) + self.eps)
                       ).astype(p.data.dtype)


class AdamW(Adam):
    """Adam with decoupled L2 weight decay applied directly to parameters."""

    def step(self):
        wd = self.weight_decay
        if wd:
            for p in self.params:
                if p.grad is not None:
                    p.data -= self.lr * wd * p.data
        # run the pure Adam update without its own weight decay
        saved_wd = self.weight_decay
        self.weight_decay = 0.0
        super().step()
        self.weight_decay = saved_wd


def clip_grad_norm(params, max_norm):
    """Global-norm gradient clipping across parameters. Returns the total norm
    before clipping."""
    params = list(params)
    total_sq = 0.0
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(None)
            continue
        g = p.grad.astype(np.float64)
        grads.append(g)
        total_sq += float((g * g).sum())
    total_norm = float(np.sqrt(total_sq))
    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / total_norm
        for p, g in zip(params, grads):
            if g is not None:
                p.grad = (scale * g).astype(p.data.dtype)
    return total_norm


class LambdaLR:
    """Learning rate schedule via a function of the current step."""

    def __init__(self, optimizer, lr_lambda):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.lr_lambda = lr_lambda
        self.last_step = 0

    def step(self):
        self.last_step += 1
        self.optimizer.lr = self.base_lr * self.lr_lambda(self.last_step)

    def state_dict(self):
        return {"base_lr": self.base_lr, "last_step": self.last_step,
                "lr": self.optimizer.lr}

    def load_state_dict(self, state):
        if not isinstance(state["last_step"], int) or state["last_step"] < 0:
            raise ValueError("invalid scheduler step")
        self.base_lr = float(state["base_lr"])
        self.last_step = state["last_step"]
        self.optimizer.lr = float(state["lr"])


def warmup_cosine(step, warmup_steps, total_steps, min_ratio=0.05):
    """Classic transformer schedule: linear warmup then cosine decay."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    import math
    return min_ratio + (1 - min_ratio) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, progress)))
