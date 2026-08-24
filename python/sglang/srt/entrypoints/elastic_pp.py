"""Transactional elastic pipeline-parallel reconfiguration endpoint."""

import json
from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from sglang.srt.utils.auth import AuthLevel, auth_level

router = APIRouter()


@router.post("/reconfigure_elastic_pp")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def reconfigure_elastic_pp(raw_request: Request):
    """Move model-layer and live-KV ownership inside the resident PP group."""

    try:
        body = await raw_request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ORJSONResponse(
            {"error": f"Invalid JSON: {exc}"},
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if not isinstance(body, dict):
        return ORJSONResponse(
            {"error": "request body must be a JSON object"},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    layer_counts = body.get("layer_counts")
    if (
        not isinstance(layer_counts, list)
        or not layer_counts
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in layer_counts
        )
    ):
        return ORJSONResponse(
            {"error": "layer_counts must be a non-empty list of non-negative integers"},
            status_code=HTTPStatus.BAD_REQUEST,
        )
    generation = body.get("generation")
    if generation is not None and (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 0
    ):
        return ORJSONResponse(
            {"error": "generation must be a non-negative integer when provided"},
            status_code=HTTPStatus.BAD_REQUEST,
        )

    from sglang.srt.entrypoints.http_server import _global_state
    from sglang.srt.managers.io_struct import ReconfigureElasticPPReqInput

    manager = _global_state.tokenizer_manager
    if not manager.server_args.enable_elastic_pp:
        return ORJSONResponse(
            {"error": "elastic PP is not enabled (set --enable-elastic-pp)"},
            status_code=HTTPStatus.NOT_FOUND,
        )

    result = await manager.reconfigure_elastic_pp(
        ReconfigureElasticPPReqInput(
            layer_counts=layer_counts,
            generation=generation,
        )
    )
    payload = {
        "success": result.success,
        "message": result.message,
        "old_generation": result.old_generation,
        "new_generation": result.new_generation,
        "old_layer_counts": result.old_layer_counts,
        "new_layer_counts": result.new_layer_counts,
        "migration_phase": result.migration_phase,
        "source_usable": result.source_usable,
        "fence_microstep": result.fence_microstep,
        "prepare_seconds": result.prepare_seconds,
        "fence_seconds": result.fence_seconds,
        "total_seconds": result.total_seconds,
    }
    if not result.success:
        active_phases = {"preparing", "bulk_ready", "fencing", "cleaning"}
        return ORJSONResponse(
            {**payload, "error": result.message},
            status_code=(
                HTTPStatus.CONFLICT
                if result.migration_phase in active_phases
                else HTTPStatus.BAD_REQUEST
            ),
        )
    return ORJSONResponse(payload)
