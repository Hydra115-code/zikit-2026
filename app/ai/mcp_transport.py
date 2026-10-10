from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable


ASGIApp = Callable[[dict, Callable[[], Awaitable[dict]], Callable[[dict], Awaitable[None]]], Awaitable[None]]


def bearer_token_valid(supplied: str | None, expected_token: str | None) -> bool:
    if expected_token is None:
        return True
    return hmac.compare_digest(supplied or "", f"Bearer {expected_token}")


class MCPBearerTokenMiddleware:
    """Protect the MCP HTTP surface without changing the FastMCP transport."""

    def __init__(self, app: ASGIApp, token: str | None) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http" or self.token is None:
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        supplied = headers.get(b"authorization", b"").decode("latin-1")
        if not bearer_token_valid(supplied, self.token):
            body = b'{"detail":"valid MCP bearer token required"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                        (b"www-authenticate", b"Bearer"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
