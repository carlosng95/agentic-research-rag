import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from langgraph.graph.state import CompiledStateGraph

from ..artifacts.materialize import materialize_index
from ..assistant import ResearchAssistant, ResearchResponse
from ..bootstrap import build_application
from ..config import Settings
from .schemas import HealthResponse, ReadyResponse, ResearchRequest


logger = logging.getLogger("uvicorn.error")

ApplicationBuilder = Callable[[], CompiledStateGraph]


def build_runtime(
    index_dir: str | Path,
    settings: Settings,
) -> tuple[CompiledStateGraph, str | None]:
    version = materialize_index(
        index_dir = index_dir,
        settings = settings,
    )

    graph = build_application(
        index_dir = index_dir,
        settings = settings,
    )

    return graph, version


async def initialize_application(
    app: FastAPI,
    application_builder: ApplicationBuilder | None = None,
    settings: Settings | None = None,
) -> None:
    try:
        if application_builder is not None:
            graph = await asyncio.to_thread(
                application_builder,
            )

            version = None

        else:
            if settings is None:
                settings = Settings.from_env()

            graph, version = await asyncio.to_thread(
                build_runtime,
                settings.index_dir,
                settings,
            )

        app.state.graph = graph
        app.state.index_version = version
        app.state.initialization_failed = False

        if version is None:
            logger.info(
                "Research application initialized successfully."
            )
        else:
            logger.info(
                "Research application initialized successfully using index version %s.",
                version,
            )

    except Exception:
        app.state.graph = None
        app.state.index_version = None
        app.state.initialization_failed = True

        logger.exception(
            "Research application initialization failed."
        )


def create_app(
    application_builder: ApplicationBuilder | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.graph = None
        app.state.index_version = None
        app.state.initialization_failed = False

        app.state.initialization_task = asyncio.create_task(
            initialize_application(
                app = app,
                application_builder = application_builder,
            )
        )

        try:
            yield

        finally:
            if not app.state.initialization_task.done():
                app.state.initialization_task.cancel()

                try:
                    await app.state.initialization_task
                except asyncio.CancelledError:
                    pass

    app = FastAPI(
        title = "Agentic Research RAG API",
        description = "HTTP serving layer for the Agentic Research RAG application.",
        version = "0.2.0",
        lifespan = lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        started_at = perf_counter()

        try:
            response = await call_next(request)

        except Exception:
            duration_ms = (perf_counter() - started_at) * 1000

            logger.exception(
                "HTTP request failed method=%s path=%s duration_ms=%.2f request_id=%s",
                request.method,
                request.url.path,
                duration_ms,
                request_id,
            )

            raise

        duration_ms = (perf_counter() - started_at) * 1000
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "HTTP request method=%s path=%s status_code=%d duration_ms=%.2f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )

        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        request_id = getattr(
            request.state,
            "request_id",
            str(uuid4()),
        )

        return JSONResponse(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            content = {
                "detail": "Internal server error.",
            },
            headers = {
                "X-Request-ID": request_id,
            },
        )

    @app.get("/health", response_model = HealthResponse, tags = ["system"])
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model = ReadyResponse, tags = ["system"])
    def ready(response: Response) -> ReadyResponse:
        if app.state.initialization_failed:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

            return ReadyResponse(
                status = "not_ready",
                detail = "Application initialization failed.",
            )

        if app.state.graph is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

            return ReadyResponse(
                status = "not_ready",
                detail = "Application initialization is still in progress.",
            )

        return ReadyResponse(
            status = "ready",
        )

    @app.post("/research", response_model = ResearchResponse, tags = ["research"])
    def research(request: ResearchRequest) -> ResearchResponse:
        if app.state.initialization_failed:
            raise HTTPException(
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
                detail = "Application initialization failed.",
            )

        if app.state.graph is None:
            raise HTTPException(
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
                detail = "Application initialization is still in progress.",
            )

        assistant = ResearchAssistant(
            graph = app.state.graph,
            thread_id = request.thread_id,
        )

        return assistant.ask(
            request.question,
        )

    return app


app = create_app()