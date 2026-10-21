"""Document information omitted by normalized token-only internal heads."""
import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location("normalization_identifiability",
    Path(__file__).resolve().parents[1] / "scripts/check_normalization_identifiability.py")
example = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(example)


def test_consistent_ohlcva_translation_has_identical_normalized_inputs_but_different_classes():
    history, future, shifted_history, shifted_future, time, left, right = example.counterexample()
    assert history.shape == shifted_history.shape == (128, 6)
    assert future.shape == shifted_future.shape == (5, 6)
    assert time.shape == (133, 5)
    for frame in (history, future, shifted_history, shifted_future):
        assert np.isfinite(frame).all() and (frame > 0).all()
        assert np.array_equal(frame[:, 0], frame[:, 3])
        assert np.array_equal(frame[:, 1], frame[:, 3] + 1)
        assert np.array_equal(frame[:, 2], frame[:, 3] - 1)
        assert np.array_equal(frame[:, 5], frame[:, 3] * frame[:, 4])
    for dtype in (np.float64, np.float32):
        assert np.array_equal(left.history.astype(dtype), right.history.astype(dtype))
        assert np.array_equal(left.future.astype(dtype), right.future.astype(dtype))
    left_return = future[-1, 3] / history[-1, 3] - 1
    right_return = shifted_future[-1, 3] / shifted_history[-1, 3] - 1
    assert example.direction_class(left_return) == 2
    assert example.direction_class(right_return) == 1
    assert np.array_equal(left.stats.std, right.stats.std)
    assert not np.array_equal(left.stats.mean, right.stats.mean)


def test_stored_normalization_stats_disambiguate_denormalized_prices_and_relative_targets():
    history, future, shifted_history, shifted_future, _, left, right = example.counterexample()
    # No clipping is active for this construction, so statistics recover the
    # original coordinates to numerical precision. Heads do not receive them.
    for raw, normalized, stats in ((history, left.history, left.stats),
                                   (future, left.future, left.stats),
                                   (shifted_history, right.history, right.stats),
                                   (shifted_future, right.future, right.stats)):
        restored = normalized * (stats.std + stats.epsilon) + stats.mean
        np.testing.assert_allclose(restored, raw, rtol=1e-15, atol=1e-9)
    a, b = 32, 64
    assert history[b, 3] / history[a, 3] != shifted_history[b, 3] / shifted_history[a, 3]
    assert np.sign(history[b, 3] - history[a, 3]) == np.sign(shifted_history[b, 3] - shifted_history[a, 3])
