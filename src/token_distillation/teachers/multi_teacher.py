"""A shared-origin preview contract, not a trained fusion or batch dataset."""
from __future__ import annotations

from .khquant_teacher import KhquantTeacher


class MultiTeacherPreview:
    def __init__(self, source, ict_vendor_root, *, context_bars=256, swing_width=2, include_ict=True):
        self.w32 = KhquantTeacher(source, context_bars=context_bars)
        self.ict = None
        if include_ict:
            from .ict_teacher import IctTeacher
            self.ict = IctTeacher(source, ict_vendor_root, context_bars=context_bars, swing_width=swing_width)

    def prepare(self, symbol, as_of):
        w32 = self.w32.prepare(symbol, as_of)
        teachers = {"w32": w32}
        if self.ict is not None:
            ict = self.ict.prepare(symbol, as_of)
            for field in ("symbol", "as_of", "context_start", "context_end", "context_rows"):
                if w32[field] != ict[field]:
                    raise ValueError(f"Teachers do not share the same input window: {field}")
            teachers["ict"] = ict
        return {"schema": "causal_multi_teacher_preview_v1", "symbol": w32["symbol"],
                "as_of": w32["as_of"], "context_start": w32["context_start"],
                "context_end": w32["context_end"], "context_rows": w32["context_rows"],
                "teachers": teachers, "training_started": False,
                "training_dataset_built": False, "student_trained": False,
                "scope": "same_window_teacher_preview_only"}
