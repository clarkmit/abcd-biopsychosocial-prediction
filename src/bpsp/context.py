"""Run context: the explicit shared state of one pipeline session.

The pipeline was developed as a Colab notebook in which every cell read and wrote
one global namespace. In this package that namespace is a single, explicit object,
``ctx``: stage configuration (the former Colab form fields) and the results that
stages hand to each other (train/test splits, fitted models, metrics, archives)
live here, so that data flow between stages is visible and inspectable.

Typical use goes through :mod:`bpsp.pipeline`; ``ctx`` can be inspected after a
run, e.g. ``ctx.metrics_dict_full`` or ``ctx._batch_archive``.
"""

from __future__ import annotations

from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any


class UnsetContextName(AttributeError):
    """Raised when a stage reads a context entry that no earlier stage produced.

    The notebook raised ``NameError`` for an undefined global; an ``AttributeError``
    subclass keeps ``getattr``/``hasattr``/``copy`` working on the context object.
    """


class RunContext(dict):
    """A dict with attribute access, holding the state shared between stages."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise UnsetContextName(f"name '{name}' is not defined in the run context") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise UnsetContextName(f"name '{name}' is not defined in the run context") from None

    def __dir__(self) -> list[str]:  # tab completion: stored names and the methods
        return sorted(set(self.keys()) | set(dir(type(self))))

    def register(self, module: ModuleType) -> None:
        """Expose a stage module's definitions, as executing the notebook cell did.

        Later stages look some helpers up by name at run time; registering a stage
        before it runs reproduces the notebook's "defined once the cell has run"
        behavior.
        """
        for name, value in vars(module).items():
            if not (name.startswith("__") and name.endswith("__")):
                self[name] = value

    def reset(self) -> None:
        """Clear all state (start of a new session)."""
        self.clear()

    def variables(self) -> dict[str, Any]:
        """Data entries only (settings and results), without registered functions, classes or modules.

        Convenient for inspection, e.g. ``sorted(ctx.variables())``.
        """
        code = (ModuleType, FunctionType, BuiltinFunctionType, type)
        return {k: v for k, v in self.items() if not isinstance(v, code) and getattr(v, "__module__", None) != "typing"}


ctx = RunContext()

__all__ = ["RunContext", "UnsetContextName", "ctx"]
