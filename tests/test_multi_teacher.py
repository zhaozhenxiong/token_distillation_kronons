import unittest
from unittest.mock import Mock, patch

from token_distillation.teachers.multi_teacher import MultiTeacherPreview


class SharedOriginTests(unittest.TestCase):
    def record(self):
        return {"symbol": "600519.SH", "as_of": "2026-10-16T15:00:00+08:00",
                "context_start": "2025-09-19", "context_end": "2026-10-16", "context_rows": 256}

    def bundle(self, w32, ict):
        with patch("token_distillation.teachers.multi_teacher.KhquantTeacher") as w32_class, \
             patch("token_distillation.teachers.ict_teacher.IctTeacher") as ict_class:
            w32_class.return_value.prepare.return_value = w32
            ict_class.return_value.prepare.return_value = ict
            return MultiTeacherPreview(Mock(), "/unused", context_bars=256)

    def test_two_teachers_must_share_exact_origin_and_context(self):
        for field, wrong in (("as_of", "2026-10-16T16:00:00+08:00"),
                             ("context_start", "2025-09-18"), ("context_rows", 255)):
            with self.subTest(field=field):
                ict = {**self.record(), field: wrong}
                with self.assertRaises(ValueError):
                    self.bundle(self.record(), ict).prepare("600519.SH", "2026-10-16")

    def test_joint_preview_never_claims_training_or_dataset_completion(self):
        result = self.bundle(self.record(), self.record()).prepare("600519.SH", "2026-10-16")
        self.assertEqual(set(result["teachers"]), {"w32", "ict"})
        self.assertFalse(result["training_started"])
        self.assertFalse(result["training_dataset_built"])
        self.assertFalse(result["student_trained"])


if __name__ == "__main__":
    unittest.main()
