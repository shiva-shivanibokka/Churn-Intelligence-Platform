"""
Tests for the FastAPI scoring endpoint.

The mock here has **five** segments, and that is the point of the file.

The previous version built a `segment_models` dict with exactly one entry,
"Champions". The endpoint it was testing computed the K-Means cluster, threw it
away and used `list(segment_models)[0]`, so it always answered "Champions" —
which, against a one-segment mock, is indistinguishable from working. Every
assertion passed while the endpoint returned the same segment for a
sixty-month five-star customer and a one-month complainer. The mock even set
`mock_kmeans.predict.return_value = np.array([0])`, so its author believed the
cluster was being used.

The same applied to `customer_type`: the endpoint passed a hardcoded
`uplift_score=0.0`, below the 0.05 Persuadable threshold, so only the two
negative-uplift quadrants were reachable — and
`test_customer_type_is_valid` asserted membership in a four-element set, which
two reachable values satisfy just as well as four.

So the tests below assert that the outputs *vary with the input*, not merely
that they are well-formed.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient

VALID_PAYLOAD = {
    "Tenure": 12.0,
    "CityTier": 1,
    "WarehouseToHome": 15.0,
    "HourSpendOnApp": 3.0,
    "NumberOfDeviceRegistered": 3,
    "SatisfactionScore": 3,
    "NumberOfAddress": 2,
    "Complain": 0,
    "OrderAmountHikeFromlastYear": 15.0,
    "CouponUsed": 1.0,
    "OrderCount": 3.0,
    "DaySinceLastOrder": 5.0,
    "CashbackAmount": 150.0,
    "PreferredLoginDevice": "Mobile Phone",
    "PreferredPaymentMode": "Debit Card",
    "Gender": "Male",
    "PreferedOrderCat": "Laptop & Accessory",
    "MaritalStatus": "Single",
}

SEGMENTS = ["Champions", "Loyal Customers", "At-Risk", "Price Sensitive", "Lapsed"]

# The features the real models are trained on. Named here so the mock cannot
# quietly accept a frame the real thing would reject.
CHURN_FEATURES = [
    "Tenure", "CityTier", "WarehouseToHome", "HourSpendOnApp",
    "NumberOfDeviceRegistered", "SatisfactionScore", "NumberOfAddress",
    "Complain", "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount",
    "DaySinceLastOrder", "CashbackAmount", "EngagementScore", "RecencySignal",
    "StickinessIndex", "SpendTrend", "SupportRiskScore", "DiscountSensitivity",
    "TenureStability", "WarehouseFriction",
]

CLUSTERING_FEATURES = [
    "EngagementScore", "RecencySignal", "StickinessIndex", "SpendTrend",
    "SupportRiskScore", "DiscountSensitivity", "TenureStability",
    "WarehouseFriction", "CityTier", "HourSpendOnApp", "OrderCount",
    "NumberOfDeviceRegistered", "SatisfactionScore",
]

UPLIFT_FEATURES = [
    "EngagementScore", "RecencySignal", "StickinessIndex", "SpendTrend",
    "SupportRiskScore", "TenureStability", "CityTier", "SatisfactionScore",
    "Complain",
]

NORMS = {
    "HourSpendOnApp": 10.0,
    "OrderCount": 50.0,
    "DaySinceLastOrder": 90.0,
    "NumberOfDeviceRegistered": 6.0,
    "NumberOfAddress": 10.0,
    "OrderAmountHikeFromlastYear": 40.0,
    "WarehouseToHome": 50.0,
}


def _make_mock_models(cluster: int = 0, uplift: float = 0.12):
    """
    Five segments, a K-Means whose answer actually matters, and a per-segment
    classifier that returns a different probability for each — so a bug that
    ignores the cluster shows up as the wrong number, not as silence.
    """
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 13))
    # A numpy array of names, because that is what sklearn actually stores.
    #
    # This was None at first, and it hid a live crash: the endpoint read the
    # attribute as `getattr(...) or [default]`, which raises "truth value of an
    # array with more than one element is ambiguous" for any real scaler. The
    # mock returned None, `None or [...]` is perfectly legal, and every test
    # passed against an endpoint that failed on every real request.
    mock_scaler.feature_names_in_ = np.array(CLUSTERING_FEATURES, dtype=object)

    mock_kmeans = MagicMock()
    mock_kmeans.predict.return_value = np.array([cluster])

    segment_models = {}
    for i, name in enumerate(SEGMENTS):
        clf = MagicMock()
        # A distinct, recognisable probability per segment.
        p = 0.1 + i * 0.2
        clf.predict_proba.return_value = np.array([[1 - p, p]])
        segment_models[name] = {
            "calibrated_clf": clf,
            "metrics": {"segment": name},
            "feature_cols": CHURN_FEATURES,
        }

    t_learner = MagicMock()
    # CausalML's convention is mu_1 - mu_0; serve.py negates it, so feed the
    # negation of the uplift we want back.
    t_learner.predict.return_value = np.array([[-uplift]])
    s_learner = MagicMock()
    s_learner.predict.return_value = np.array([[-uplift]])

    return {
        "segment_models": segment_models,
        "kmeans": mock_kmeans,
        "scaler": mock_scaler,
        "label_map": dict(enumerate(SEGMENTS)),
        "norms": NORMS,
        "dataset": "test-fixture",
        "uplift": {
            "learners": {"t_learner": t_learner, "s_learner": s_learner, "negated": True},
            "feature_cols": UPLIFT_FEATURES,
        },
    }


@pytest.fixture
def client():
    with patch("api.serve.get_models", return_value=_make_mock_models()):
        from api.serve import app
        yield TestClient(app)


def _client_for(**kwargs):
    """A TestClient whose models are configured per-test."""
    from api.serve import app
    return patch("api.serve.get_models", return_value=_make_mock_models(**kwargs)), app


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestReadinessEndpoint:
    def test_readiness_returns_ready_when_models_available(self, client):
        response = client.get("/readiness")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        # Readiness should say what it is ready to serve. "200 OK" alone does not
        # distinguish a full set of artifacts from a partial one.
        assert sorted(body["segments"]) == sorted(SEGMENTS)
        assert body["uplift_available"] is True


class TestScoreEndpoint:
    def test_score_returns_200(self, client):
        response = client.post("/score", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_score_response_has_required_fields(self, client):
        data = client.post("/score", json=VALID_PAYLOAD).json()
        for field in ("segment", "churn_probability", "churn_prediction",
                      "risk_tier", "customer_type", "uplift_score", "trained_on"):
            assert field in data

    def test_churn_probability_is_float_in_range(self, client):
        prob = client.post("/score", json=VALID_PAYLOAD).json()["churn_probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_risk_tier_is_valid(self, client):
        tier = client.post("/score", json=VALID_PAYLOAD).json()["risk_tier"]
        assert tier in ["Low Risk", "Medium Risk", "High Risk"]

    def test_customer_type_is_valid(self, client):
        ctype = client.post("/score", json=VALID_PAYLOAD).json()["customer_type"]
        assert ctype in ["Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"]

    def test_missing_required_field_returns_422(self, client):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "Tenure"}
        assert client.post("/score", json=bad).status_code == 422

    def test_invalid_satisfaction_score_returns_422(self, client):
        assert client.post("/score", json={**VALID_PAYLOAD, "SatisfactionScore": 10}).status_code == 422


class TestSegmentComesFromKMeans:
    """
    The regression this file exists for: the endpoint must return the segment
    the clustering model chose, not the first key of a dict.
    """

    @pytest.mark.parametrize("cluster,expected", list(enumerate(SEGMENTS)))
    def test_each_cluster_maps_to_its_own_segment(self, cluster, expected):
        patcher, app = _client_for(cluster=cluster)
        with patcher:
            data = TestClient(app).post("/score", json=VALID_PAYLOAD).json()
        assert data["segment"] == expected

    def test_segment_is_not_constant_across_clusters(self):
        seen = set()
        for cluster in range(len(SEGMENTS)):
            patcher, app = _client_for(cluster=cluster)
            with patcher:
                seen.add(TestClient(app).post("/score", json=VALID_PAYLOAD).json()["segment"])
        assert len(seen) == len(SEGMENTS), (
            f"Only {sorted(seen)} reachable — the endpoint is ignoring the cluster."
        )

    def test_probability_comes_from_the_matched_segments_model(self):
        """Each segment's own classifier must be the one that answers."""
        for cluster in range(len(SEGMENTS)):
            patcher, app = _client_for(cluster=cluster)
            with patcher:
                data = TestClient(app).post("/score", json=VALID_PAYLOAD).json()
            assert data["churn_probability"] == pytest.approx(0.1 + cluster * 0.2, abs=1e-6)

    def test_scaler_without_fitted_names_falls_back(self):
        """A scaler fitted on a bare array exposes no names; that must still work."""
        models = _make_mock_models()
        models["scaler"].feature_names_in_ = None
        from api.serve import app
        with patch("api.serve.get_models", return_value=models):
            assert TestClient(app).post("/score", json=VALID_PAYLOAD).status_code == 200

    def test_unknown_cluster_is_an_error_not_a_guess(self):
        """A cluster with no name must fail loudly rather than pick something."""
        patcher, app = _client_for(cluster=99)
        with patcher:
            response = TestClient(app).post("/score", json=VALID_PAYLOAD)
        assert response.status_code == 500
        assert "99" in response.json()["detail"]


class TestCustomerTypeUsesRealUplift:
    """
    `customer_type` was computed from a hardcoded `uplift_score=0.0`, so the two
    positive-uplift quadrants were unreachable for every request ever made.
    """

    def test_positive_uplift_and_high_churn_gives_persuadable(self):
        # cluster 4 -> p = 0.9, well above the 0.30 churn threshold
        patcher, app = _client_for(cluster=4, uplift=0.12)
        with patcher:
            data = TestClient(app).post("/score", json=VALID_PAYLOAD).json()
        assert data["uplift_score"] == pytest.approx(0.12, abs=1e-6)
        assert data["customer_type"] == "Persuadable"

    def test_positive_uplift_and_low_churn_gives_sure_thing(self):
        # cluster 0 -> p = 0.1, below the churn threshold
        patcher, app = _client_for(cluster=0, uplift=0.12)
        with patcher:
            data = TestClient(app).post("/score", json=VALID_PAYLOAD).json()
        assert data["customer_type"] == "Sure Thing"

    def test_negative_uplift_and_high_churn_gives_lost_cause(self):
        patcher, app = _client_for(cluster=4, uplift=-0.08)
        with patcher:
            data = TestClient(app).post("/score", json=VALID_PAYLOAD).json()
        assert data["customer_type"] == "Lost Cause"

    def test_all_four_quadrants_are_reachable(self):
        reached = set()
        for cluster in (0, 4):
            for uplift in (0.12, -0.08):
                patcher, app = _client_for(cluster=cluster, uplift=uplift)
                with patcher:
                    reached.add(TestClient(app).post("/score", json=VALID_PAYLOAD).json()["customer_type"])
        assert reached == {"Persuadable", "Sure Thing", "Lost Cause", "Sleeping Dog"}

    def test_missing_uplift_models_are_reported_not_defaulted(self):
        """
        Absent uplift models must produce "unknown", not a 0.0 substituted for a
        measurement — that substitution is what pinned every answer to a
        negative-uplift quadrant.
        """
        models = _make_mock_models()
        models["uplift"] = None
        from api.serve import app
        with patch("api.serve.get_models", return_value=models):
            data = TestClient(app).post("/score", json=VALID_PAYLOAD).json()
        assert data["uplift_score"] is None
        assert "Unknown" in data["customer_type"]
