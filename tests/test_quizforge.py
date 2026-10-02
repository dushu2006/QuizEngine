from __future__ import annotations

import csv
import io
import time

import pytest

from quizforge import VARIANTS, create_app
from quizforge.app import QuizState, StateStore
from quizforge.catalog import QUESTIONS, ZOOM_LEVELS
from quizengine.harness import build_runtime


@pytest.fixture()
def app():
    return create_app({"TESTING": True, "SECRET_KEY": "test-secret"})


@pytest.fixture()
def client(app):
    return app.test_client()


def test_home_gallery_and_health_render(client):
    home = client.get("/")
    gallery = client.get("/variants")
    assert home.status_code == 200
    assert b"QuizEngine \xe2\x80\x94 AI Computer-Use Agent" in home.data and b"Chaos lab" in home.data
    assert b"QuizForge" not in home.data
    assert gallery.status_code == 200
    assert b"19" in gallery.data and b"Check fixture parity" in gallery.data
    assert client.get("/health").json == {"status": "ok", "service": "quizengine-ui", "variants": 19}
    assert b"QuizEngine" in gallery.data and b"QuizForge" not in gallery.data


def test_every_variant_matches_quizengine_fixture_annotation(client):
    assert len(VARIANTS) >= 12
    for key in VARIANTS:
        response = client.get(f"/api/parity/{key}")
        assert response.status_code == 200, key
        assert response.json["parity"] is True, (key, response.json)
        assert response.json["mismatches"] == []


def test_answering_progress_repeat_protection_and_no_implicit_key(client):
    before = client.get("/api/session").json
    assert "correct_index" not in before["screen"]
    result = client.post("/api/answer", json={"option": 1})
    assert result.status_code == 200
    assert result.json["answered"] is True
    assert result.json["feedback"]["correct"] is True  # first question's answer is Paris (B)
    assert client.post("/api/answer", json={"option": 1}).status_code == 409
    assert client.post("/api/next").status_code == 200
    assert client.get("/api/session").json["screen"]["question"]["index"] == 1
    assert "correct_index" not in client.get("/api/session").json["screen"]


def test_answer_validation_and_navigation_requires_answer(client):
    assert client.post("/api/answer", json={"option": 99}).status_code == 400
    assert client.post("/api/answer", json={"option": "not-an-index"}).status_code == 400
    assert client.post("/api/next").status_code == 409


def test_settings_themes_zoom_and_chaos(client):
    result = client.post("/api/settings", json={"theme": "dark", "zoom": 2, "chaos": ["toast", "reorder"]})
    assert result.status_code == 200
    state = result.json
    assert state["settings"]["theme"] == "dark"
    assert state["settings"]["zoom"] == 2.0
    assert state["screen"]["theme"] == "dark"
    assert state["screen"]["zoom"] == 2.0
    assert state["screen"]["chaos"]["toast"] is True
    assert len(state["screen"]["options"]) == 4
    assert client.post("/api/settings", json={"theme": "infrared"}).status_code == 400
    assert client.post("/api/settings", json={"zoom": 3}).status_code == 400
    assert client.post("/api/settings", json={"chaos": ["captcha"]}).status_code == 400


def test_invalid_combined_settings_are_atomic(client):
    before = client.get("/api/session").json
    response = client.post("/api/settings", json={"theme": "dark", "zoom": 2, "chaos": ["captcha"]})
    assert response.status_code == 400
    after = client.get("/api/session").json
    assert after["settings"] == before["settings"]
    assert after["screen"]["theme"] == before["screen"]["theme"]
    assert after["screen"]["zoom"] == before["screen"]["zoom"]


def test_modal_notice_must_be_dismissed_before_answer(client):
    state = client.post("/api/settings", json={"variant": "popup_modal"}).json
    assert any(row["kind"] == "modal" for row in state["screen"]["overlays"])
    assert client.post("/api/answer", json={"option": 1}).status_code == 409
    dismissed = client.post("/api/overlay/dismiss").json
    assert not any(row["kind"] == "modal" for row in dismissed["screen"]["overlays"])
    assert client.post("/api/answer", json={"option": 1}).status_code == 200


def test_answer_key_is_explicit_csv_and_json_export(client):
    state = client.get("/api/session").json
    assert "correct_index" not in state["screen"]
    response = client.get("/api/answer-key?format=csv")
    assert response.status_code == 200
    assert "text/csv" in response.content_type
    rows = list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))
    assert len(rows) == 12
    assert rows[0]["correct_letter"] == "B"
    assert rows[0]["correct_text"] == "Paris"
    assert client.get("/api/answer-key?format=json").json["count"] == 12
    assert client.get("/api/answer-key?format=xml").status_code == 400


def test_full_18_layout_sweep_reaches_question_free_results_screen(client):
    started = client.post("/api/settings", json={"full_matrix": True}).json
    assert started["question_count"] == 18
    seen = []
    for _ in range(18):
        current = client.get("/api/session").json
        screen = current["screen"]
        seen.append(screen["variant"])
        assert screen["screen_role"] == "quiz"
        assert screen["question"]["text"]
        if any(overlay["kind"] == "modal" for overlay in screen["overlays"]):
            assert client.post("/api/overlay/dismiss").status_code == 200
            screen = client.get("/api/session").json["screen"]
        correct = screen["options"]
        # The answer key is an explicit export; compare the actual displayed
        # option permutation against that export, including chaos reorder.
        key = client.get("/api/answer-key?format=json").json["answer_key"]
        question = screen["question"]["text"]
        index = key[question]
        assert client.post("/api/answer", json={"option": index}).status_code == 200
        assert client.post("/api/next").status_code == 200
    assert len(seen) == 18
    assert len(set(seen)) == 18
    result = client.get("/api/session").json
    assert result["done"] is True
    assert result["screen"]["screen_role"] == "end_state"
    assert result["screen"]["options"] == []
    assert result["screen"]["navigation"]["next_label"] is None
    assert result["correct_count"] == 18


def test_state_store_is_bounded_and_evicts_least_recently_used():
    store = StateStore(max_sessions=2)
    first = store.get_or_create("first")
    first.theme = "dark"
    second = store.get_or_create("second")
    assert store.get_or_create("first") is first  # touch makes it newer
    store.get_or_create("third")  # evicts second, not active first
    assert store.get_or_create("first") is first
    assert first.theme == "dark"
    assert store.get_or_create("second") is not second
    assert len(store._states) == 2


def test_restricted_remote_client_is_denied_by_default():
    app = create_app({"TESTING": True, "SECRET_KEY": "test-secret", "QUIZFORGE_ALLOW_REMOTE": False})
    remote = app.test_client()
    response = remote.get("/", environ_base={"REMOTE_ADDR": "192.168.1.7"})
    assert response.status_code == 403


def test_sessions_are_isolated_between_browsers(app):
    first = app.test_client()
    second = app.test_client()
    first.post("/api/settings", json={"theme": "dark"})
    assert first.get("/api/session").json["settings"]["theme"] == "dark"
    assert second.get("/api/session").json["settings"]["theme"] == "light"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_theme_persists_after_option_selection(client, theme):
    client.post("/api/settings", json={"theme": theme})
    selected = client.post("/api/answer", json={"option": 0})
    assert selected.status_code == 200
    assert selected.json["settings"]["theme"] == theme
    assert selected.json["question_state"]["current_question"] == 0
    assert selected.json["question_state"]["selected_option"] == 0


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_theme_persists_after_next(client, theme):
    client.post("/api/settings", json={"theme": theme})
    client.post("/api/answer", json={"option": 0})
    next_state = client.post("/api/next").json
    assert next_state["screen"]["question"]["index"] == 1
    assert next_state["settings"]["theme"] == theme
    assert next_state["navigation_state"] == "NOT_READY"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_theme_persists_after_previous(client, theme):
    client.post("/api/settings", json={"theme": theme})
    first = client.post("/api/answer", json={"option": 0}).json
    client.post("/api/next")
    client.post("/api/answer", json={"option": 1})
    previous = client.post("/api/previous").json
    assert previous["screen"]["question"]["index"] == 0
    assert previous["settings"]["theme"] == theme
    assert previous["selected"] == first["selected"]
    assert previous["answered"] is True
    assert previous["feedback"] is not None


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_theme_persists_after_restart(client, theme):
    client.post("/api/settings", json={"theme": theme, "zoom": 1.5, "variant": "tile_grid_3x2", "chaos": ["reorder"]})
    client.post("/api/answer", json={"option": 0})
    restarted = client.post("/api/restart").json
    assert restarted["screen"]["question"]["index"] == 0
    assert restarted["selected"] is None and restarted["answered"] is False
    assert restarted["feedback"] is None and restarted["correct_count"] == 0
    assert restarted["settings"]["theme"] == theme
    assert restarted["settings"]["zoom"] == 1.5
    assert restarted["settings"]["layout"] == "tile_grid_3x2"
    assert restarted["settings"]["chaos"] == ["reorder"]


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("zoom", ZOOM_LEVELS)
def test_theme_persists_after_zoom(client, theme, zoom):
    client.post("/api/settings", json={"theme": theme})
    changed = client.post("/api/settings", json={"zoom": zoom}).json
    assert changed["settings"]["theme"] == theme
    assert changed["settings"]["zoom"] == zoom
    assert changed["screen"]["zoom"] == zoom


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_theme_persists_after_layout_change(client, theme):
    client.post("/api/settings", json={"theme": theme})
    changed = client.post("/api/settings", json={"variant": "two_options"}).json
    assert changed["settings"]["theme"] == theme
    assert changed["settings"]["layout"] == "two_options"
    assert len(changed["screen"]["options"]) == 2
    assert changed["screen"]["variant"] == "two_options"


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_theme_persists_after_chaos_toggle(client, theme):
    client.post("/api/settings", json={"theme": theme})
    client.post("/api/answer", json={"option": 3})
    changed = client.post("/api/settings", json={"chaos": ["reorder", "toast"]}).json
    assert changed["settings"]["theme"] == theme
    assert changed["settings"]["chaos"] == ["reorder", "toast"]
    assert changed["answered"] is True
    assert changed["selected"] is not None
    assert changed["feedback"] is not None


@pytest.mark.parametrize("variant", [key for key in VARIANTS if key != "results_screen"])
def test_every_layout_supports_each_option_and_verified_next(client, variant):
    # Selecting option 0..N-1 in each layout exercises radio, checkbox, cards,
    # text-only and variable option-count render paths with theme/preferences on.
    client.post("/api/settings", json={"theme": "dark", "zoom": 1.25, "chaos": ["reorder"]})
    changed = client.post("/api/settings", json={"variant": variant}).json
    assert changed["screen"]["variant"] == variant
    option_count = len(changed["screen"]["options"])
    expected_counts = {"two_options": 2, "six_options_long": 6}
    assert option_count == expected_counts.get(variant, 4)
    first_id = changed["question_state"]["question_id"]
    for chosen in range(option_count):
        state = client.get("/api/session").json
        if any(row["kind"] == "modal" for row in state["screen"]["overlays"]):
            state = client.post("/api/overlay/dismiss").json
        answered = client.post("/api/answer", json={"option": chosen})
        assert answered.status_code == 200, (variant, chosen, answered.json)
        assert answered.json["selected"] == chosen
        assert answered.json["settings"]["theme"] == "dark"
        assert answered.json["settings"]["zoom"] == 1.25
        assert answered.json["question_state"]["question_id"] == first_id
        moved = client.post("/api/next")
        assert moved.status_code == 200, (variant, chosen, moved.json)
        after = moved.json
        assert after["screen"]["question"]["index"] == 1
        assert after["question_state"]["question_id"] != first_id
        assert after["selected"] is None and after["answered"] is False
        assert after["feedback"] is None
        assert after["navigation_state"] == "NOT_READY"
        assert after["settings"]["theme"] == "dark" and after["settings"]["zoom"] == 1.25
        assert client.post("/api/restart").status_code == 200


@pytest.mark.parametrize("option", [0, 1, 2])
def test_three_option_question_accepts_every_choice_and_advances(option):
    store = StateStore()
    test_app = create_app({"TESTING": True, "SECRET_KEY": "three-option"}, state_store=store)
    test_client = test_app.test_client()
    test_client.get("/")
    with test_client.session_transaction() as flask_session:
        key = flask_session["quizforge_session"]
    store._states[key] = QuizState(
        questions=[("Pick the third choice", ("Alpha", "Beta", "Gamma"), 2)],
        variants=["radio_vertical"],
    )
    store._last_seen[key] = time.monotonic()
    shown = test_client.get("/api/session").json
    assert len(shown["screen"]["options"]) == 3
    result = test_client.post("/api/answer", json={"option": option})
    assert result.status_code == 200
    assert result.json["feedback"]["correct"] is (option == 2)
    completed = test_client.post("/api/next").json
    assert completed["done"] is True
    assert completed["screen"]["screen_role"] == "end_state"


def test_next_changes_question_id(client):
    before = client.get("/api/session").json
    client.post("/api/answer", json={"option": 0})
    after = client.post("/api/next").json
    assert after["question_id"] != before["question_state"]["question_id"]
    assert after["screen"]["question"]["id"] == after["question_id"]


def test_next_changes_question_content(client):
    before = client.get("/api/session").json["screen"]["question"]["text"]
    client.post("/api/answer", json={"option": 0})
    after = client.post("/api/next").json
    assert after["question"] != before
    assert after["screen"]["question"]["text"] == after["question"]


def test_next_increments_question_number(client):
    client.post("/api/answer", json={"option": 0})
    after = client.post("/api/next").json
    assert after["question_number"] == 2
    assert after["question_state"]["current_question"] == 1


def test_next_from_q1_to_q2(client):
    before = client.get("/api/session").json
    client.post("/api/answer", json={"option": 0})
    after = client.post("/api/next").json
    assert (before["screen"]["question"]["text"], after["question"]) == (
        "Which city is the capital of France?", "What is 12 * 8?"
    )


def test_next_from_q2_to_q3(client):
    client.post("/api/answer", json={"option": 0})
    client.post("/api/next")
    before = client.get("/api/session").json
    client.post("/api/answer", json={"option": 0})
    after = client.post("/api/next").json
    assert before["question_state"]["current_question"] == 1
    assert after["question_number"] == 3
    assert after["question"] == "Which planet is closest to the Sun?"


def test_next_through_entire_quiz(client):
    seen_ids = []
    seen_text = []
    while True:
        current = client.get("/api/session").json
        if current["done"]:
            break
        seen_ids.append(current["question_state"]["question_id"])
        seen_text.append(current["screen"]["question"]["text"])
        client.post("/api/answer", json={"option": 0})
        after = client.post("/api/next").json
        assert after["success"] is True
        if not after["done"]:
            assert after["question_number"] == len(seen_ids) + 1
            assert after["question_id"] != seen_ids[-1]
            assert after["question"] != seen_text[-1]
            assert after["screen"]["question"]["id"] == after["question_id"]
    assert len(seen_ids) == 12
    assert len(set(seen_ids)) == 12
    assert len(set(seen_text)) == 12
    assert after["navigation_state"] == "COMPLETED"
    assert after["question_id"] is None


def test_next_after_each_option(client):
    for variant, count in (("two_options", 2), ("radio_vertical", 4), ("six_options_long", 6)):
        for selected in range(count):
            client.post("/api/restart")
            client.post("/api/settings", json={"variant": variant})
            answered = client.post("/api/answer", json={"option": selected})
            assert answered.status_code == 200, (variant, selected)
            moved = client.post("/api/next").json
            assert moved["question_number"] == 2
            assert moved["question_id"] != answered.json["question_state"]["question_id"]
            assert moved["selected_index"] is None


def test_next_does_not_render_stale_question(client):
    before = client.get("/api/session").json
    client.post("/api/answer", json={"option": 1})
    moved = client.post("/api/next").json
    assert moved["from_question_id"] == before["question_state"]["question_id"]
    assert moved["question_id"] == moved["screen"]["question"]["id"]
    assert moved["question"] == moved["screen"]["question"]["text"]
    assert moved["question_id"] != moved["from_question_id"]


def test_repeated_next_previous_sequence_keeps_identity_and_content(client):
    def next_and_check(expected_index):
        old = client.get("/api/session").json
        client.post("/api/answer", json={"option": 0})
        moved = client.post("/api/next").json
        assert moved["question_state"]["current_question"] == expected_index
        assert moved["question_id"] != old["question_state"]["question_id"]
        assert moved["question"] != old["screen"]["question"]["text"]
        return moved

    q2 = next_and_check(1)
    q3 = next_and_check(2)
    q2_back = client.post("/api/previous").json
    assert q2_back["question_id"] == q2["question_id"]
    assert q2_back["selected_index"] == 0 and q2_back["answered"] is True
    q3_again = client.post("/api/next").json
    assert q3_again["question_id"] == q3["question_id"]
    q4 = next_and_check(3)
    assert q4["question_number"] == 4


def test_auto_advance_layout_does_not_double_navigate(client):
    configured = client.post("/api/settings", json={"variant": "auto_advance"}).json
    assert configured["screen"]["navigation"]["auto_advance"] is True
    client.post("/api/answer", json={"option": 0})
    moved = client.post("/api/next").json
    assert moved["question_number"] == 2
    assert moved["question"] == "What is 12 * 8?"
    duplicate = client.post("/api/next")
    assert duplicate.status_code == 409
    assert client.get("/api/session").json["screen"]["question"]["index"] == 1


def test_final_question_next_transitions_to_completion(client):
    client.post("/api/settings", json={"full_matrix": True})
    for index in range(18):
        before = client.get("/api/session").json
        if any(row["kind"] == "modal" for row in before["screen"]["overlays"]):
            client.post("/api/overlay/dismiss")
            before = client.get("/api/session").json
        client.post("/api/answer", json={"option": 0})
        result = client.post("/api/next").json
        assert result["success"] is True
        if index < 17:
            assert result["question_number"] == index + 2
            assert result["question_id"] != before["question_state"]["question_id"]
        else:
            assert result["done"] is True
            assert result["navigation_state"] == "COMPLETED"
            assert result["question_id"] is None
            assert result["screen"]["screen_role"] == "end_state"


def test_evicted_session_reproduces_same_question_on_old_resync_path():
    store = StateStore()
    test_app = create_app({"TESTING": True, "SECRET_KEY": "loss-repro"}, state_store=store)
    test_client = test_app.test_client()
    before = test_client.get("/api/session").json
    test_client.post("/api/answer", json={"option": 1})
    with test_client.session_transaction() as flask_session:
        key = flask_session["quizforge_session"]
    store._states.pop(key)
    failed_next = test_client.post("/api/next")
    assert failed_next.status_code == 409
    resynced = test_client.get("/api/session").json
    # This is the exact old behavior: Next fails against the empty volatile
    # store, then generic resync silently installs a fresh first question.
    assert resynced["question_state"]["question_id"] == before["question_state"]["question_id"]
    assert resynced["screen"]["question"]["text"] == before["screen"]["question"]["text"]


def test_session_snapshot_recovers_navigation_after_server_restart():
    first_app = create_app({"TESTING": True, "SECRET_KEY": "persistent-test-secret"})
    first_client = first_app.test_client()
    first = first_client.get("/api/session").json
    first_client.post("/api/answer", json={"option": 1})
    answered = first_client.get("/api/session").json
    selected_text = answered["screen"]["options"][answered["selected"]]["text"]
    snapshot = {
        "session_epoch": answered["session_epoch"],
        "state": answered,
        "answers": [{"question_index": 0, "selected_text": selected_text}],
        "dismissed_questions": [],
        "settled_chaos": [],
    }
    cookie = first_client.get_cookie("session")
    restarted_app = create_app({"TESTING": True, "SECRET_KEY": "persistent-test-secret"})
    restarted_client = restarted_app.test_client()
    restarted_client.set_cookie("session", cookie.value)
    # This is the browser's first Next after the server lost its volatile state.
    moved = restarted_client.post("/api/next", json={
        "expected_question_id": first["question_state"]["question_id"],
        "client_snapshot": snapshot,
    }).json
    assert moved["success"] is True
    assert moved["question_number"] == 2
    assert moved["question"] == "What is 12 * 8?"
    assert moved["from_question_id"] == first["question_state"]["question_id"]
    assert moved["question_id"] != first["question_state"]["question_id"]
    assert moved["settings"]["theme"] == "light"


def test_session_snapshot_recovers_after_in_memory_session_eviction():
    store = StateStore()
    test_app = create_app({"TESTING": True, "SECRET_KEY": "eviction-test"}, state_store=store)
    test_client = test_app.test_client()
    test_client.get("/api/session")
    with test_client.session_transaction() as flask_session:
        key = flask_session["quizforge_session"]
    before = test_client.get("/api/session").json
    test_client.post("/api/answer", json={"option": 1})
    answered = test_client.get("/api/session").json
    selected_text = answered["screen"]["options"][answered["selected"]]["text"]
    snapshot = {
        "session_epoch": answered["session_epoch"], "state": answered,
        "answers": [{"question_index": 0, "selected_text": selected_text}],
        "dismissed_questions": [], "settled_chaos": [],
    }
    store._states.pop(key)
    moved = test_client.post("/api/next", json={
        "expected_question_id": before["question_state"]["question_id"],
        "client_snapshot": snapshot,
    })
    assert moved.status_code == 200
    assert moved.json["question_number"] == 2
    assert moved.json["question"] == "What is 12 * 8?"
    back = test_client.post("/api/previous").json
    assert back["screen"]["question"]["text"] == "Which city is the capital of France?"
    assert back["selected"] == 1 and back["answered"] is True
    assert back["feedback"]["correct"] is True


def test_stale_tab_question_identity_is_rejected(client):
    client.get("/api/session")
    stale = client.post("/api/next", json={"expected_question_id": "q-9999-not-the-current-question"})
    assert stale.status_code == 409
    assert stale.json["error"] == "question state changed in another tab; resync before continuing"


def test_next_preserves_zoom(client):
    client.post("/api/settings", json={"zoom": 1.5})
    client.post("/api/answer", json={"option": 0})
    after = client.post("/api/next").json
    assert after["settings"]["zoom"] == 1.5
    assert after["screen"]["zoom"] == 1.5


def test_next_preserves_layout(client):
    client.post("/api/settings", json={"variant": "tile_grid_3x2"})
    client.post("/api/answer", json={"option": 0})
    after = client.post("/api/next").json
    assert after["settings"]["layout"] == "tile_grid_3x2"
    assert after["screen"]["variant"] == "tile_grid_3x2"


def test_next_clears_unvisited_question_selection(client):
    client.post("/api/answer", json={"option": 2})
    after = client.post("/api/next").json
    assert after["selected_index"] is None
    assert after["answered"] is False
    assert after["feedback"] is None
    assert after["question_state"]["selected_option"] is None


def test_next_after_correct_answer(client):
    question = client.get("/api/session").json["screen"]["question"]["text"]
    correct = client.get("/api/answer-key?format=json").json["answer_key"][question]
    answered = client.post("/api/answer", json={"option": correct}).json
    assert answered["feedback"]["correct"] is True
    assert answered["question_state"]["can_navigate"] is True
    assert client.post("/api/next").json["screen"]["question"]["index"] == 1


def test_next_after_incorrect_answer(client):
    question = client.get("/api/session").json["screen"]["question"]["text"]
    correct = client.get("/api/answer-key?format=json").json["answer_key"][question]
    wrong = (correct + 1) % len(client.get("/api/session").json["screen"]["options"])
    answered = client.post("/api/answer", json={"option": wrong}).json
    assert answered["feedback"]["correct"] is False
    assert answered["question_state"]["can_navigate"] is True
    assert client.post("/api/next").json["screen"]["question"]["index"] == 1


def test_next_does_not_duplicate_request(client, app):
    import concurrent.futures
    import threading

    client.get("/api/session")
    client.post("/api/answer", json={"option": 0})
    cookie = client.get_cookie("session")
    barrier = threading.Barrier(2)

    def advance():
        other = app.test_client()
        other.set_cookie("session", cookie.value)
        barrier.wait()
        return other.post("/api/next")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _n: advance(), range(2)))
    assert sorted(response.status_code for response in responses) == [200, 409]
    current = client.get("/api/session").json
    assert current["question_state"]["current_question"] == 1
    assert current["screen"]["question"]["text"] == "What is 12 * 8?"


def test_next_rejects_duplicate_transition_and_does_not_skip_question(client):
    assert client.post("/api/answer", json={"option": 1}).status_code == 200
    assert client.post("/api/next").status_code == 200
    duplicate = client.post("/api/next")
    assert duplicate.status_code == 409
    state = client.get("/api/session").json
    assert state["screen"]["question"]["index"] == 1
    assert state["navigation_state"] == "NOT_READY"


def test_previous_restores_answer_and_forward_restores_same_question_state(client):
    q1 = client.post("/api/answer", json={"option": 2}).json
    client.post("/api/next")
    client.post("/api/answer", json={"option": 1})
    client.post("/api/next")
    q3_id = client.get("/api/session").json["question_state"]["question_id"]
    q2 = client.post("/api/previous").json
    assert q2["screen"]["question"]["index"] == 1
    assert q2["answered"] is True and q2["selected"] == 1 and q2["feedback"] is not None
    q1_back = client.post("/api/previous").json
    assert q1_back["selected"] == 2 and q1_back["answered"] is True
    q2_forward = client.post("/api/next").json
    assert q2_forward["screen"]["question"]["index"] == 1
    assert q2_forward["selected"] == 1 and q2_forward["answered"] is True
    q3_forward = client.post("/api/next").json
    assert q3_forward["question_state"]["question_id"] == q3_id
    assert q3_forward["selected"] is None and q3_forward["answered"] is False
    assert q1["question_state"]["question_id"] != q3_id


def test_restart_zoom_layout_theme_and_chaos_end_to_end(client):
    settings = client.post("/api/settings", json={
        "theme": "dark", "zoom": 2, "variant": "two_options", "chaos": ["reorder", "toast"]
    }).json
    assert len(settings["screen"]["options"]) == 2
    first = client.post("/api/answer", json={"option": 1}).json
    assert first["selected"] == 1
    next_state = client.post("/api/next").json
    assert next_state["screen"]["question"]["index"] == 1
    assert next_state["selected"] is None and next_state["feedback"] is None
    assert next_state["settings"]["theme"] == "dark"
    assert next_state["settings"]["zoom"] == 2
    assert next_state["screen"]["variant"] == "two_options"
    restarted = client.post("/api/restart").json
    assert restarted["screen"]["question"]["index"] == 0
    assert restarted["selected"] is None and restarted["feedback"] is None
    assert restarted["settings"]["theme"] == "dark"
    assert restarted["settings"]["zoom"] == 2
    assert restarted["settings"]["layout"] == "two_options"
    assert restarted["settings"]["chaos"] == ["reorder", "toast"]


def test_real_screen_target_without_attestation_halts_before_display_probe(tmp_path):
    runtime = build_runtime(target="screen", runs_dir=tmp_path)
    assert runtime.config.run.attestation is None
    report = runtime.run()
    assert report.outcome.value == "halted"
    assert report.halted_code.value == "ATTESTATION_MISSING"
    assert not report.questions
