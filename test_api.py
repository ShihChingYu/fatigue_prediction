# test_api.py
from fastapi.testclient import TestClient
from main import app


def test_predict():
    # Using 'with' triggers the @app.on_event("startup") logic
    with TestClient(app) as client:
        payload = {
            "hr_history": [
                {"HRTIME": "2026-01-21T10:00:00", "HR": 70.0},
                {"HRTIME": "2026-01-21T10:15:00", "HR": 85.0},
            ],
            "sleep_history": [{"START": "2026-01-20T22:00:00", "END": "2026-01-21T06:00:00"}],
        }
        response = client.post("/predict", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")


if __name__ == "__main__":
    test_predict()
