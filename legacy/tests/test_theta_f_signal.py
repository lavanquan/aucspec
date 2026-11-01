from edge_specsim.verification_batcher import VerificationBatcher


def _bare():
    vb = VerificationBatcher.__new__(VerificationBatcher)
    vb.roofline_budget_model = None
    vb.verify_token_budget = 64
    vb.theta_f_signal = "none"
    vb.theta_f_fixed_ms = 0.0
    vb._theta_f_ema = None
    return vb


def test_none_returns_zero():
    vb = _bare()
    assert vb.estimate_verify_budget_signal(100)[2] == 0.0


def test_fixed_and_measured():
    vb = _bare()
    vb.theta_f_signal = "fixed"
    vb.theta_f_fixed_ms = 3.83
    assert vb.estimate_verify_budget_signal(100)[2] == 3.83
    vb.theta_f_signal = "measured"
    assert vb.estimate_verify_budget_signal(100)[2] == 0.0
    vb._theta_f_ema = 5.0
    assert vb.estimate_verify_budget_signal(100)[2] == 5.0
