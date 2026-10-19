"""Explicit adaptive and full-capacity controls;inherited score implementation."""

from rbpf_register_lu import RegisterLUScorer


class CompactScratchScorer(RegisterLUScorer):
    def __init__(self, dll, capacity=512, kernel="register"):
        modes = {"original": 0, "generic": 1, "register": 2, "adaptive": 2, "synced": 3, "full": 4}
        if kernel not in modes:
            raise ValueError("Unknown compact scratch kernel")
        super().__init__(dll, capacity, kernel="register")
        code = self.dll.rbpf_register_select(self.handle, modes[kernel])
        if code:
            self.close()
            raise RuntimeError(f"Compact scratch selector failed:{code}")
