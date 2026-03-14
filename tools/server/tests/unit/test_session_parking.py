import pytest

from utils import *


server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    server.server_metrics = True


def get_metric(body: str, name: str) -> float:
    prefix = f"llamacpp:{name} "
    for line in body.splitlines():
        if line.startswith(prefix):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} not found")


def make_session_completion(session_key: str, prompt: str):
    return server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "cache_prompt": True,
        "metadata": {
            "session_key": session_key,
        },
    })


def test_hot_parked_session_reuse_avoids_cold_restore():
    global server
    server.start()

    res = make_session_completion("session-a", "What is the capital of France?")
    assert res.status_code == 200
    assert isinstance(res.body["content"], str)
    assert len(res.body["content"]) > 0
    assert res.body["timings"]["prompt_n"] >= 1

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert get_metric(metrics.body, "sessions_parked_hot") == 1
    assert get_metric(metrics.body, "sessions_parked_cold") == 0
    assert get_metric(metrics.body, "scheduler_restore_attempts_total") == 0

    res = make_session_completion("session-a", "What is the capital of Germany?")
    assert res.status_code == 200
    assert isinstance(res.body["content"], str)
    assert len(res.body["content"]) > 0
    assert res.body["timings"]["prompt_n"] >= 1

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert get_metric(metrics.body, "sessions_parked_hot") == 1
    assert get_metric(metrics.body, "sessions_parked_cold") == 0
    assert get_metric(metrics.body, "scheduler_restore_attempts_total") == 0


def test_hot_parked_session_spills_to_cold_when_displaced():
    global server
    server.n_slots = 1
    server.start()

    res = make_session_completion("session-a", "What is the capital of France?")
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] >= 1

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert get_metric(metrics.body, "sessions_parked_hot") == 0
    assert get_metric(metrics.body, "sessions_parked_cold") == 1
    assert get_metric(metrics.body, "scheduler_cold_park_total") == 1
    assert get_metric(metrics.body, "scheduler_cold_park_kv_bytes_total") > 0
    assert get_metric(metrics.body, "scheduler_cold_park_kv_copy_seconds_total") > 0

    res = make_session_completion("session-b", "What is the capital of Germany?")
    assert res.status_code == 200

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert get_metric(metrics.body, "sessions_parked_hot") == 0
    assert get_metric(metrics.body, "sessions_parked_cold") == 2
    assert get_metric(metrics.body, "scheduler_restore_attempts_total") == 0

    res = make_session_completion("session-a", "What is the capital of Germany?")
    assert res.status_code == 200
    assert isinstance(res.body["content"], str)
    assert len(res.body["content"]) > 0

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert get_metric(metrics.body, "sessions_parked_hot") == 0
    assert get_metric(metrics.body, "sessions_parked_cold") == 2
    assert get_metric(metrics.body, "scheduler_restore_attempts_total") == 1
    assert get_metric(metrics.body, "scheduler_restore_success_total") == 1
    assert get_metric(metrics.body, "scheduler_restore_kv_bytes_total") > 0
    assert get_metric(metrics.body, "scheduler_restore_kv_copy_seconds_total") > 0


def test_distinct_sessions_claim_distinct_hot_slots_before_reusing_cold_state():
    global server
    server.n_slots = 2
    server.start()

    res = make_session_completion("session-a", "What is the capital of France?")
    assert res.status_code == 200

    res = make_session_completion("session-b", "What is the capital of Germany?")
    assert res.status_code == 200

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert get_metric(metrics.body, "sessions_parked_hot") == 2
    assert get_metric(metrics.body, "sessions_parked_cold") == 0
    assert get_metric(metrics.body, "scheduler_restore_attempts_total") == 0

    res = make_session_completion("session-a", "What is the capital of Italy?")
    assert res.status_code == 200

    res = make_session_completion("session-b", "What is the capital of Spain?")
    assert res.status_code == 200

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert get_metric(metrics.body, "sessions_parked_hot") == 2
    assert get_metric(metrics.body, "sessions_parked_cold") == 0
    assert get_metric(metrics.body, "scheduler_restore_attempts_total") == 0
