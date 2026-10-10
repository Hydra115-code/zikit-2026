from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["simulator"])


@router.get("/internal/providers/{provider}", include_in_schema=False)
async def provider(provider: str, request: Request) -> JSONResponse:
    if provider not in {"primary", "backup"}:
        return JSONResponse({"error": "unknown provider"}, status_code=404)
    status, latency = await request.app.state.simulator.response(provider)
    return JSONResponse({"provider": provider, "latency_ms": latency}, status_code=status)


@router.get("/demo/gateway")
async def gateway(request: Request) -> JSONResponse:
    provider_name = request.app.state.simulator.active_provider
    status, latency = await request.app.state.simulator.response(provider_name)
    return JSONResponse(
        {"provider": provider_name, "latency_ms": latency, "status": "ok" if status == 200 else "error"},
        status_code=status,
    )

