"""Shared contracts for optional native backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn


class NativeBackendUnavailableError(RuntimeError):
    """Raised when a public API needs a native module that is not installed."""


def backend_unavailable(
    feature: str,
    module_name: str,
) -> NoReturn:
    """Raise the standard, actionable optional-backend error."""
    raise NativeBackendUnavailableError(
        f"{feature} requires optional native module {module_name!r}. "
        "Build the CUDA/C++ extensions as described in README.md."
    )


def unavailable_function(feature: str, module_name: str) -> Callable:
    """Return a public placeholder with the same failure contract on CPU-only installs."""

    def _unavailable(*args, **kwargs):
        del args, kwargs
        backend_unavailable(feature, module_name)

    _unavailable.__name__ = feature
    _unavailable.__qualname__ = feature
    _unavailable.__doc__ = (
        f"{feature} is unavailable because optional module {module_name!r} "
        "was not built."
    )
    return _unavailable


def is_missing_optional_module(exc: ImportError, module_name: str) -> bool:
    """Return True only when *module_name* itself is absent.

    Import errors raised from inside an installed extension (for example a
    missing dependent DLL) are deliberately not classified as an optional
    backend absence; hiding those would make broken installations look like
    valid CPU-only installations.
    """
    return isinstance(exc, ModuleNotFoundError) and exc.name == module_name


__all__ = [
    "NativeBackendUnavailableError",
    "backend_unavailable",
    "is_missing_optional_module",
    "unavailable_function",
]
