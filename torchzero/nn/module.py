"""Neural network module system: Parameter, Module, containers, layers.

Parameter registration is automatic: any ``Parameter`` assigned as an
attribute (or inside ModuleList / dict attributes) of a Module is discovered
by traversal. state_dict/load_state_dict support nested modules and lists.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import pickle

import numpy as np

from ..tensor.tensor import Tensor


class Parameter(Tensor):
    """A Tensor that is a learnable module parameter."""

    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)


class Module:
    def __init__(self):
        self.training = True
        self._parameters = {}

    # ---------------------------------------------------------- registration
    def _named_members(self):
        """Yield (name, obj) for all direct attribute members of interest."""
        for name in sorted(vars(self)):
            if name.startswith("_"):
                continue
            yield name, getattr(self, name)

    def parameters(self) -> list:
        """All unique parameters reachable from this module."""
        seen = {}
        self._collect_parameters(self, seen)
        return list(seen.values())

    @staticmethod
    def _collect_parameters(module, seen):
        for name, value in vars(module).items():
            if name.startswith("_"):
                continue
            if isinstance(value, Parameter):
                seen.setdefault(id(value), value)
            elif isinstance(value, Module):
                Module._collect_parameters(value, seen)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Parameter):
                        seen.setdefault(id(item), item)
                    elif isinstance(item, Module):
                        Module._collect_parameters(item, seen)
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, Parameter):
                        seen.setdefault(id(item), item)
                    elif isinstance(item, Module):
                        Module._collect_parameters(item, seen)

    def named_parameters(self, prefix="") -> dict:
        from .module import ModuleList
        out = {}
        for name, value in sorted(vars(self).items()):
            if name.startswith("_"):
                continue
            if isinstance(value, Parameter):
                out[prefix + name] = value
            elif isinstance(value, ModuleList):
                for i, item in enumerate(value.modules):
                    for k, v in item.named_parameters(
                            f"{prefix}{name}.{i}.").items():
                        out[k] = v
            elif isinstance(value, Module):
                for k, v in value.named_parameters(
                        prefix + name + ".").items():
                    out[k] = v
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Parameter):
                        out[f"{prefix}{name}.{i}"] = item
                    elif isinstance(item, Module):
                        for k, v in item.named_parameters(
                                f"{prefix}{name}.{i}.").items():
                            out[k] = v
            elif isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, Parameter):
                        out[f"{prefix}{name}.{key}"] = item
                    elif isinstance(item, Module):
                        for k, v in item.named_parameters(
                                f"{prefix}{name}.{key}.").items():
                            out[k] = v
        return out

    def named_modules(self):
        """Yield each reachable module once, including list/dict containers."""
        seen = set()

        def visit(value, name):
            if isinstance(value, Module):
                if id(value) in seen:
                    return
                seen.add(id(value))
                yield name, value
                for key, child in sorted(vars(value).items()):
                    if not key.startswith("_"):
                        yield from visit(child, f"{name}.{key}" if name else key)
            elif isinstance(value, (list, tuple, dict)):
                items = value.items() if isinstance(value, dict) else enumerate(value)
                for key, child in items:
                    yield from visit(child, f"{name}.{key}")

        yield from visit(self, "")

    @contextmanager
    def evaluating(self):
        """Temporarily eval all modules, restoring mixed modes on exit."""
        modes = [(module, module.training) for _, module in self.named_modules()]
        self.eval()
        try:
            yield self
        finally:
            for module, training in modes:
                module.training = training

    def rng_state_dict(self):
        """Snapshot persistent random streams owned by modules (e.g. dropout)."""
        return {name: copy.deepcopy(module._rng.bit_generator.state)
                for name, module in self.named_modules() if hasattr(module, "_rng")}

    def load_rng_state_dict(self, state):
        streams = {name: module._rng for name, module in self.named_modules()
                   if hasattr(module, "_rng")}
        if set(state) != set(streams):
            raise ValueError("model RNG state names do not match")
        # Validate every state on a temporary generator before applying any.
        for name, rng in streams.items():
            probe = type(rng.bit_generator)()
            probe.state = copy.deepcopy(state[name])
        for name, rng in streams.items():
            rng.bit_generator.state = copy.deepcopy(state[name])

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self):
        self._set_mode(True)
        return self

    def eval(self):
        self._set_mode(False)
        return self

    def _set_mode(self, training):
        for _, module in self.named_modules():
            module.training = training

    # ----------------------------------------------------------- serialization
    def state_dict(self) -> dict:
        """Flat mapping of parameter name -> numpy array."""
        return {k: v.data.copy() for k, v in self.named_parameters().items()}

    def load_state_dict(self, state_dict: dict):
        params = self.named_parameters()
        missing = set(params) - set(state_dict)
        unexpected = set(state_dict) - set(params)
        if missing or unexpected:
            raise KeyError(
                f"state_dict mismatch; missing={sorted(missing)} "
                f"unexpected={sorted(unexpected)}")
        for name, array in state_dict.items():
            target = params[name]
            array = np.asarray(array)
            if array.shape != target.shape:
                raise ValueError(
                    f"shape mismatch for {name}: checkpoint "
                    f"{array.shape} vs model {target.shape}")
            target.data = array.astype(target.data.dtype)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.load_state_dict(pickle.load(f))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModuleList(Module):
    """A list of submodules with automatic registration."""

    def __init__(self, modules=None):
        super().__init__()
        self.modules = list(modules) if modules else []

    def append(self, module):
        self.modules.append(module)

    def extend(self, modules):
        self.modules.extend(modules)

    def __len__(self):
        return len(self.modules)

    def __getitem__(self, idx):
        return self.modules[idx]

    def __iter__(self):
        return iter(self.modules)

    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x


class Sequential(Module):
    """Chains layers, calling each in order."""

    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
