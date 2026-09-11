import time
from threading import Event

from fastapi.testclient import TestClient

from agentic_research_rag.serving.app import create_app
from uuid import UUID


class FakeGraph:
    def __init__(self) -> None:
        self.input = None
        self.config = None

    def invoke(self, input, config):
        self.input = input
        self.config = config

        return {
            "final_answer": "Test answer.",
            "final_documents": [],
            "cited_source_numbers": [],
            "tools_used": ["paper_retrieval"],
            "flags": [],
            "sufficient": True,
            "citation_valid": True,
        }

def test_health_endpoint() -> None:
    app = create_app(application_builder = lambda: object())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }


def test_ready_endpoint_returns_ready_after_initialization() -> None:
    graph = object()
    app = create_app(application_builder = lambda: graph)

    with TestClient(app) as client:
        for _ in range(100):
            response = client.get("/ready")

            if response.status_code == 200:
                break

            time.sleep(0.01)

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "detail": None,
    }
    assert app.state.graph is graph


def test_ready_endpoint_returns_not_ready_during_initialization() -> None:
    release_initialization = Event()

    def slow_builder():
        release_initialization.wait()
        return object()

    app = create_app(application_builder = slow_builder)

    with TestClient(app) as client:
        response = client.get("/ready")
        release_initialization.set()

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "detail": "Application initialization is still in progress.",
    }


def test_ready_endpoint_returns_not_ready_when_initialization_fails() -> None:
    def failing_builder():
        raise RuntimeError("Initialization failed.")

    app = create_app(application_builder = failing_builder)

    with TestClient(app) as client:
        for _ in range(100):
            response = client.get("/ready")

            if app.state.initialization_failed:
                break

            time.sleep(0.01)

        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "detail": "Application initialization failed.",
    }
    
def test_research_endpoint() -> None:
    graph = FakeGraph()
    app = create_app(application_builder = lambda: graph)

    with TestClient(app) as client:
        for _ in range(100):
            if app.state.graph is not None:
                break

            time.sleep(0.01)

        response = client.post(
            "/research",
            json = {
                "question": "What does the document say?",
                "thread_id": "conversation-123",
            },
        )

    assert response.status_code == 200

    assert response.json() == {
        "answer": "Test answer.",
        "sources": [],
        "cited_source_numbers": [],
        "tools_used": ["paper_retrieval"],
        "flags": [],
        "paper_evidence_sufficient": True,
        "citation_valid": True,
    }

    assert graph.input == {
        "question": "What does the document say?",
        "messages": [],
    }

    assert graph.config["configurable"]["thread_id"] == "conversation-123"
    
def test_research_endpoint_rejects_empty_question() -> None:
    app = create_app(application_builder = lambda: FakeGraph())

    with TestClient(app) as client:
        response = client.post(
            "/research",
            json = {
                "question": "    ",
                "thread_id": "conversation-123",
            },
        )

    assert response.status_code == 422
    
def test_research_endpoint_returns_503_while_initializing() -> None:
    release_initialization = Event()

    def slow_builder():
        release_initialization.wait()
        return FakeGraph()

    app = create_app(application_builder = slow_builder)

    with TestClient(app) as client:
        response = client.post(
            "/research",
            json = {
                "question": "Valid question",
                "thread_id": "conversation-123",
            },
        )

        release_initialization.set()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Application initialization is still in progress.",
    }
def test_health_endpoint_returns_request_id() -> None:
    app = create_app(application_builder = lambda: object())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    request_id = response.headers["X-Request-ID"]

    UUID(request_id)
    
def test_health_endpoint_preserves_request_id() -> None:
    app = create_app(application_builder = lambda: object())

    with TestClient(app) as client:
        response = client.get("/health", headers = {"X-Request-ID": "test-request-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-123"
    
class FailingGraph:
    def invoke(self, input, config):
        raise RuntimeError("Sensitive internal error.")
    
def test_unhandled_exception_returns_generic_500() -> None:
    app = create_app(application_builder = lambda: FailingGraph())

    with TestClient(app, raise_server_exceptions = False) as client:
        for _ in range(100):
            if app.state.graph is not None:
                break

            time.sleep(0.01)

        response = client.post(
            "/research",
            headers = {"X-Request-ID": "failed-request-123"},
            json = {
                "question": "Valid question",
                "thread_id": "conversation-123",
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Internal server error.",
    }

    assert response.headers["X-Request-ID"] == "failed-request-123"
    assert "Sensitive internal error" not in response.text