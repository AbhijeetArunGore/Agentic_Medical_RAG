"""Locust load testing scenario for Agentic Medical RAG microservice.

Run with:
    locust -f benchmarks/locustfile.py --headless -u 10 -r 2 -t 30s --host http://localhost:8000
"""

from locust import HttpUser, task, between


class MedicalRAGUser(HttpUser):
    wait_time = between(1.0, 3.0)

    @task(4)
    def test_health(self):
        self.client.get("/health", name="/health")

    @task(3)
    def test_ready(self):
        self.client.get("/ready", name="/ready")

    @task(2)
    def test_query(self):
        payload = {
            "query": "Recommended treatments for rising creatinine in T2D patient.",
            "patient_id": "LOCUST-PAT-01",
        }
        self.client.post("/query", json=payload, name="/query")

    @task(1)
    def test_evaluate_patient(self):
        payload = {
            "patient_id": "LOCUST-PAT-02",
            "query": "Evaluate renal function and diabetes control",
        }
        self.client.post("/evaluate-patient", json=payload, name="/evaluate-patient")
