"""Preserve native capture;accept a stop request for an already-exited child.

Windows can return error5 from TerminateProcess after the target has exited.
Confirm GetExitCodeProcess is not STILL_ACTIVE before accepting that case.
The original debug loop/code object and exception handling remain unchanged.
"""

import ctypes
from ctypes import wintypes
from types import FunctionType

import rbpf_windows_debug as original


class ExitAwareKernel:
    def __init__(self, kernel):
        self.kernel = kernel
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def TerminateProcess(self, handle, status):
        if self.kernel.TerminateProcess(handle, status):
            return True
        error = ctypes.get_last_error()
        code = wintypes.DWORD(259)
        if error == 5 and self.kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != 259:
            return True
        ctypes.set_last_error(error)
        return False


def checked_api():
    return ExitAwareKernel(original.api())


debug_child = FunctionType(original.debug_child.__code__,
    dict(original.debug_child.__globals__, api=checked_api), original.debug_child.__name__)
