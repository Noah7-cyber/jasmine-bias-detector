import pytest
from server import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def config(**changes):
    return dict(group_column="group", prediction_column="prediction", true_column="truth", minimum_group_size=1, **changes)


def test_wsgi_entrypoint_and_assets(client):
    from app import app as exported_app
    assert exported_app is app
    assert callable(app)
    for path in ("/", "/api/health", "/static/workbench.css", "/static/workbench.js", "/assets/plotly.min.js"):
        assert client.get(path).status_code == 200


@pytest.mark.parametrize("sample", ["compas", "adult", "s1", "s2", "s3", "s4", "s5", "s6"])
def test_offline_samples_available(client, sample):
    response = client.post("/api/inspect", json={"sample": sample})
    assert response.status_code == 200
    assert response.json["rows"] > 0
    assert response.headers["Cache-Control"] == "no-store"


def test_upload_manual_metrics_and_pdf(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = {"csv": "group,prediction,truth\nA,1,1\nA,1,0\nB,1,1\nB,0,0\n", "settings": config()}
    response = client.post("/api/analyze", json=data)
    assert response.status_code == 200
    assert response.json["result"]["metrics"]["demographic_parity_difference"] == 0.5
    assert response.json["result"]["metrics"]["disparate_impact_ratio"] == 0.5
    report = client.post("/api/report", json=data)
    assert report.status_code == 200
    assert report.data.startswith(b"%PDF")
    assert report.headers["Cache-Control"] == "no-store"
    assert list(tmp_path.iterdir()) == []
    # A fresh request cannot retrieve the previous user's upload.
    assert client.post("/api/analyze", json={"settings": config()}).status_code == 400


@pytest.mark.parametrize("data", [
    {}, {"sample": "../../app.py"}, {"csv": ""}, {"csv": "single\n1"},
    {"csv": "group,group\nA,B"}, {"sample": "compas", "settings": []},
    {"csv": "group,prediction,truth\nA,1,1\nB,0,0", "settings": config(score_threshold="bad")},
    {"csv": "group,prediction,truth\nA,1,1\nB,0,0", "settings": {"group_column": "group", "prediction_column": "group"}},
])
def test_invalid_inputs_are_recoverable(client, data):
    response = client.post("/api/analyze", json=data)
    assert response.status_code == 400
    assert "error" in response.json


def test_bad_json_and_payload_limit(client):
    assert client.post("/api/inspect", data="{bad", content_type="application/json").status_code == 400
    assert client.post("/api/inspect", json={"csv": "x" * (4 * 1024 * 1024)}).status_code == 413


def test_pdf_treats_uploaded_headers_and_labels_as_text(client):
    response = client.post("/api/report", json={
        "csv": "group,pred<iction>,truth\nA<tag>,1,1\nA<tag>,0,0\nB,1,0\nB,0,1\n",
        "settings": {"group_column": "group", "prediction_column": "pred<iction>", "true_column": "truth"},
    })
    assert response.status_code == 200
    assert response.data.startswith(b"%PDF")
