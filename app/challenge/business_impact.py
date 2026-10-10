from __future__ import annotations

from typing import Any


def kaiku_potential_impact(suspected_incident_type: str | None) -> dict[str, Any]:
    descriptions = {
        "checkout_degradation": (
            "Podría ralentizar o interrumpir carrito y checkout, reduciendo la "
            "conversión de ventas en línea."
        ),
        "database_connection_pressure": (
            "Podría dejar al checkout sin conexiones para consultar pedidos e "
            "inventario, ralentizando compras y poniendo ventas en riesgo."
        ),
        "database_contention": (
            "Podría retrasar pedidos, inventario y checkout mientras las "
            "transacciones esperan recursos de base de datos."
        ),
        "cpu_saturation": (
            "Podría aumentar la latencia de catálogo, carrito y checkout y reducir "
            "la conversión de ventas."
        ),
        "memory_pressure": (
            "Podría causar lentitud o reinicios que afecten catálogo, carrito y "
            "checkout."
        ),
        "storage_capacity_risk": (
            "Podría impedir escrituras de pedidos, inventario o logs si se agota "
            "el margen de almacenamiento."
        ),
        "application_errors": (
            "Podría interrumpir checkout, pagos o inventario y provocar compras "
            "incompletas."
        ),
    }
    detail = descriptions.get(
        suspected_incident_type,
        "Podría afectar catálogo, carrito, checkout, pedidos o inventario si la "
        "degradación progresa.",
    )
    return {
        "status": "potential_not_observed",
        "business_processes": [
            "catálogo",
            "carrito",
            "checkout",
            "pedidos",
            "inventario",
        ],
        "detail": (
            f"{detail} El dataset no demuestra ventas perdidas ni permite "
            "cuantificar impacto económico."
        ),
        "sales_loss_observed": False,
        "economic_amount_estimated": False,
    }
