"""Response headers that keep server-sent events streaming through proxies."""

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# FastAPI sends event streams with Cache-Control: no-cache. no-transform also tells every proxy on
# the way not to transform the body, compression included. The Next.js server that proxies /api
# to this backend gzips any compressible response, text/event-stream included, unless it says
# no-transform, and its gzip stream holds the events back until the answer is complete.
EVENT_STREAM_CACHE_CONTROL = "no-cache, no-transform"


class EventStreamHeaders:
    """ASGI middleware: every text/event-stream response gets EVENT_STREAM_CACHE_CONTROL.

    Pure ASGI, not BaseHTTPMiddleware: it passes `receive` through untouched, so a client
    disconnect still reaches the streaming response and stops generation.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if headers.get("content-type", "").startswith("text/event-stream"):
                    headers["Cache-Control"] = EVENT_STREAM_CACHE_CONTROL  # replaces no-cache
            await send(message)

        await self.app(scope, receive, send_with_headers)
