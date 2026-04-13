import numpy as np
import pytest

SEED = 42
N_TRAIN = 80
N_VAL = 20
N_TEST = 10
L = 16
H = 4


def make_data(n, L, H, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.random((n, L)).astype(np.float32)
    Y = rng.random((n, H)).astype(np.float32)
    return X, Y


def test_rf_smoke():
    from tsforecast.models.rf import RandomForestModel
    X_tr, Y_tr = make_data(N_TRAIN, L, H)
    X_val, Y_val = make_data(N_VAL, L, H)
    X_te, _ = make_data(N_TEST, L, H)
    model = RandomForestModel(n_estimators=10, random_state=SEED)
    model.fit(X_tr, Y_tr, X_val, Y_val)
    preds = model.predict(X_te)
    assert preds.shape == (N_TEST, H)
    assert np.isfinite(preds).all()


def test_rf_save_load(tmp_path):
    from tsforecast.models.rf import RandomForestModel
    X_tr, Y_tr = make_data(N_TRAIN, L, H)
    model = RandomForestModel(n_estimators=5, random_state=SEED)
    model.fit(X_tr, Y_tr)
    model.save(tmp_path / "rf_model")
    loaded = RandomForestModel.load(tmp_path / "rf_model")
    X_te, _ = make_data(N_TEST, L, H)
    np.testing.assert_array_almost_equal(model.predict(X_te), loaded.predict(X_te))


def test_lstm_smoke():
    from tsforecast.models.lstm import LSTMModel
    X_tr, Y_tr = make_data(N_TRAIN, L, H)
    X_val, Y_val = make_data(N_VAL, L, H)
    X_te, _ = make_data(N_TEST, L, H)
    model = LSTMModel(
        context_length=L, horizon=H,
        hidden_size=16, num_layers=1, dropout=0.0,
        lr=1e-2, max_epochs=2, batch_size=16, patience=5,
        random_state=SEED,
    )
    model.fit(X_tr, Y_tr, X_val, Y_val)
    preds = model.predict(X_te)
    assert preds.shape == (N_TEST, H)
    assert np.isfinite(preds).all()


def test_lstm_save_load(tmp_path):
    from tsforecast.models.lstm import LSTMModel
    X_tr, Y_tr = make_data(N_TRAIN, L, H)
    X_val, Y_val = make_data(N_VAL, L, H)
    model = LSTMModel(
        context_length=L, horizon=H,
        hidden_size=16, num_layers=1, dropout=0.0,
        lr=1e-2, max_epochs=2, batch_size=16, patience=5,
        random_state=SEED,
    )
    model.fit(X_tr, Y_tr, X_val, Y_val)
    model.save(tmp_path / "lstm_model")
    loaded = LSTMModel.load(tmp_path / "lstm_model")
    X_te, _ = make_data(N_TEST, L, H)
    np.testing.assert_array_almost_equal(model.predict(X_te), loaded.predict(X_te), decimal=5)


def test_lstmnet_forward():
    """_LSTMNet constructor and forward pass with the current 4-arg signature."""
    import torch
    from tsforecast.models.lstm import _LSTMNet

    net = _LSTMNet(output_horizon=4, hidden_size=16, num_layers=1, dropout=0.0)
    x = torch.randn(3, L)       # (B, context_length)
    out = net(x)                # expected (B, H, 1)
    assert out.shape == (3, 4, 1)
    assert torch.isfinite(out).all()


# PatchTST smoke test is marked slow -- requires transformers library
@pytest.mark.slow
def test_patchtst_smoke(tmp_path):
    from tsforecast.models.patchtst import PatchTSTModel
    X_tr, Y_tr = make_data(N_TRAIN, L, H)
    X_val, Y_val = make_data(N_VAL, L, H)
    X_te, _ = make_data(N_TEST, L, H)
    model = PatchTSTModel(
        context_length=L, horizon=H,
        patch_length=4, patch_stride=2,
        d_model=16, num_attention_heads=2, num_hidden_layers=1,
        ffn_dim=32, dropout=0.0,
        lr=1e-3, max_epochs=1, batch_size=16, patience=5,
        random_state=SEED,
        output_dir=str(tmp_path / "patchtst_tmp"),
    )
    model.fit(X_tr, Y_tr, X_val, Y_val)
    preds = model.predict(X_te)
    assert preds.shape == (N_TEST, H)
    assert np.isfinite(preds).all()
