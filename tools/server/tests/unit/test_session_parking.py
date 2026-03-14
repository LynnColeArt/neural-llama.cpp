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


def test_hot_resident_toggle_can_force_cold_parking():
    global server
    server.hot_resident_sessions = False
    server.start()

    res = make_session_completion("session-a", "What is the capital of France?")
    assert res.status_code == 200

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert get_metric(metrics.body, "sessions_parked_hot") == 0
    assert get_metric(metrics.body, "sessions_parked_cold") == 1


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


def test_disabling_empty_slot_preference_reintroduces_restore_churn():
    global server
    server.n_slots = 2
    server.prefer_empty_session_slots = False
    server.start()

    for session_key, prompt in [
        ("session-a", "What is the capital of France?"),
        ("session-b", "What is the capital of Germany?"),
        ("session-a", "What is the capital of Italy?"),
        ("session-b", "What is the capital of Spain?"),
    ]:
        res = make_session_completion(session_key, prompt)
        assert res.status_code == 200

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert get_metric(metrics.body, "scheduler_restore_attempts_total") >= 1


def test_continuity_tokens_replay_session_identity_without_explicit_metadata():
    global server
    server.continuity_tokens = True
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert "X-Neural-Continuity" in res.headers
    assert "Set-Cookie" in res.headers

    cookie = res.headers["Set-Cookie"].split(";", 1)[0]
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "cache_prompt": True,
    }, headers={
        "Cookie": cookie,
    })
    assert res.status_code == 200

    metrics = server.make_request("GET", "/metrics")
    assert metrics.status_code == 200
    assert get_metric(metrics.body, "sessions_parked_hot") == 1
    assert get_metric(metrics.body, "sessions_parked_cold") == 0
