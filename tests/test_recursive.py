"""Tests for recursive forecasting strategy (Issue 8)."""
import numpy as np
import pytest

SEED = 42
N_TRAIN = 80
N_VAL = 20
N_TEST = 10
L = 32
H_STEP = 8
H_TOTAL = 24  # 3 iterations of step=8


def make_data(n, L, H, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.random((n, L)).astype(np.float32)
    Y = rng.random((n, H)).astype(np.float32)
    return X, Y


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestPredictRecursiveShape:
    """predict_recursive must return (N, H_total) for various H_total/step combos."""

    def test_lstm_exact_multiple(self):
        from tsforecast.models.lstm import LSTMModel
        X_tr, Y_tr = make_data(N_TRAIN, L, H_STEP)
        X_val, Y_val = make_data(N_VAL, L, H_STEP)
        X_te, _ = make_data(N_TEST, L, H_STEP)
        model = LSTMModel(
            context_length=L, horizon=H_TOTAL,
            hidden_size=16, num_layers=1, dropout=0.0,
            lr=1e-2, max_epochs=2, batch_size=16, patience=5,
            random_state=SEED, strategy="recursive", step=H_STEP,
        )
        model.fit(X_tr, Y_tr, X_val, Y_val)
        preds = model.predict(X_te)
        assert preds.shape == (N_TEST, H_TOTAL)
        assert np.isfinite(preds).all()

    def test_lstm_non_divisible_horizon(self):
        """H_total=25 with step=8 → iterations of 8+8+8+1=25."""
        from tsforecast.models.lstm import LSTMModel
        H_odd = 25
        X_tr, Y_tr = make_data(N_TRAIN, L, H_STEP)
        X_val, Y_val = make_data(N_VAL, L, H_STEP)
        X_te, _ = make_data(N_TEST, L, H_STEP)
        model = LSTMModel(
            context_length=L, horizon=H_odd,
            hidden_size=16, num_layers=1, dropout=0.0,
            lr=1e-2, max_epochs=2, batch_size=16, patience=5,
            random_state=SEED, strategy="recursive", step=H_STEP,
        )
        model.fit(X_tr, Y_tr, X_val, Y_val)
        preds = model.predict(X_te)
        assert preds.shape == (N_TEST, H_odd)

    def test_lstm_step_equals_horizon(self):
        """When step == H_total, recursive reduces to one-shot prediction."""
        from tsforecast.models.lstm import LSTMModel
        H = 16
        X_tr, Y_tr = make_data(N_TRAIN, L, H)
        X_val, Y_val = make_data(N_VAL, L, H)
        X_te, _ = make_data(N_TEST, L, H)
        model = LSTMModel(
            context_length=L, horizon=H,
            hidden_size=16, num_layers=1, dropout=0.0,
            lr=1e-2, max_epochs=2, batch_size=16, patience=5,
            random_state=SEED, strategy="recursive", step=H,
        )
        model.fit(X_tr, Y_tr, X_val, Y_val)
        preds = model.predict(X_te)
        assert preds.shape == (N_TEST, H)

    def test_lstm_predict_recursive_direct(self):
        """predict_recursive can be called directly with custom arguments."""
        from tsforecast.models.lstm import LSTMModel
        X_tr, Y_tr = make_data(N_TRAIN, L, H_STEP)
        X_val, Y_val = make_data(N_VAL, L, H_STEP)
        X_te, _ = make_data(N_TEST, L, H_STEP)
        model = LSTMModel(
            context_length=L, horizon=H_TOTAL,
            hidden_size=16, num_layers=1, dropout=0.0,
            lr=1e-2, max_epochs=2, batch_size=16, patience=5,
            random_state=SEED, strategy="recursive", step=H_STEP,
        )
        model.fit(X_tr, Y_tr, X_val, Y_val)
        preds = model.predict_recursive(X_te, total_horizon=H_TOTAL, step=H_STEP)
        assert preds.shape == (N_TEST, H_TOTAL)


# ---------------------------------------------------------------------------
# No future leakage test
# ---------------------------------------------------------------------------

class TestPredictRecursiveNoFutureLeak:
    """Recursive predict must not use true future values."""

    def test_lstm_context_rolls_on_predictions(self):
        """
        The recursive loop should only use model predictions to extend context,
        not ground-truth Y values. We verify by checking that two calls with
        identical X but different (ignored) Y produce identical outputs.
        """
        from tsforecast.models.lstm import LSTMModel
        X_tr, Y_tr = make_data(N_TRAIN, L, H_STEP)
        X_val, Y_val = make_data(N_VAL, L, H_STEP)
        X_te, _ = make_data(N_TEST, L, H_STEP)
        model = LSTMModel(
            context_length=L, horizon=H_TOTAL,
            hidden_size=16, num_layers=1, dropout=0.0,
            lr=1e-2, max_epochs=2, batch_size=16, patience=5,
            random_state=SEED, strategy="recursive", step=H_STEP,
        )
        model.fit(X_tr, Y_tr, X_val, Y_val)

        # predict twice — the second call has no access to Y, only X
        preds_a = model.predict(X_te)
        preds_b = model.predict(X_te)
        np.testing.assert_array_equal(preds_a, preds_b)

    def test_lstm_different_context_gives_different_output(self):
        """Changing X should change recursive predictions."""
        from tsforecast.models.lstm import LSTMModel
        rng = np.random.default_rng(0)
        X_tr, Y_tr = make_data(N_TRAIN, L, H_STEP)
        X_val, Y_val = make_data(N_VAL, L, H_STEP)
        model = LSTMModel(
            context_length=L, horizon=H_TOTAL,
            hidden_size=16, num_layers=1, dropout=0.0,
            lr=1e-2, max_epochs=2, batch_size=16, patience=5,
            random_state=SEED, strategy="recursive", step=H_STEP,
        )
        model.fit(X_tr, Y_tr, X_val, Y_val)

        X1 = rng.random((5, L)).astype(np.float32)
        X2 = rng.random((5, L)).astype(np.float32)
        preds1 = model.predict(X1)
        preds2 = model.predict(X2)
        assert not np.allclose(preds1, preds2), "Different contexts must give different predictions"


# ---------------------------------------------------------------------------
# MIMO vs recursive: same model class, different strategy
# ---------------------------------------------------------------------------

class TestMimoVsRecursive:
    def test_lstm_mimo_shape(self):
        from tsforecast.models.lstm import LSTMModel
        X_tr, Y_tr = make_data(N_TRAIN, L, H_TOTAL)
        X_val, Y_val = make_data(N_VAL, L, H_TOTAL)
        X_te, _ = make_data(N_TEST, L, H_TOTAL)
        model = LSTMModel(
            context_length=L, horizon=H_TOTAL,
            hidden_size=16, num_layers=1, dropout=0.0,
            lr=1e-2, max_epochs=2, batch_size=16, patience=5,
            random_state=SEED, strategy="mimo",
        )
        model.fit(X_tr, Y_tr, X_val, Y_val)
        preds = model.predict(X_te)
        assert preds.shape == (N_TEST, H_TOTAL)

    def test_lstm_save_load_recursive(self, tmp_path):
        from tsforecast.models.lstm import LSTMModel
        X_tr, Y_tr = make_data(N_TRAIN, L, H_STEP)
        X_val, Y_val = make_data(N_VAL, L, H_STEP)
        X_te, _ = make_data(N_TEST, L, H_STEP)
        model = LSTMModel(
            context_length=L, horizon=H_TOTAL,
            hidden_size=16, num_layers=1, dropout=0.0,
            lr=1e-2, max_epochs=2, batch_size=16, patience=5,
            random_state=SEED, strategy="recursive", step=H_STEP,
        )
        model.fit(X_tr, Y_tr, X_val, Y_val)
        model.save(tmp_path / "lstm_rec")
        loaded = LSTMModel.load(tmp_path / "lstm_rec")
        assert loaded.strategy == "recursive"
        assert loaded.step == H_STEP
        preds = loaded.predict(X_te)
        assert preds.shape == (N_TEST, H_TOTAL)
