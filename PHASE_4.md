# Fase 4 — Remediación autónoma controlada

La Fase 4 conecta las alertas críticas del reto con una remediación autónoma segura. No cambia el modelo, el score ni la política de detección. Toda acción se limita a la infraestructura controlada de demostración y se etiqueta como tal; no representa una ejecución sobre producción real de Kaiku.

## Flujo implementado

1. Una alerta `incident_risk` se persiste con su evidencia causal.
2. Si continúa abierta después de `CHALLENGE_REMEDIATION_WAIT_SECONDS`, el coordinador crea un registro idempotente de remediación.
3. La política local obtiene únicamente runbooks `preauthorized=true` y `reversible=true` del catálogo versionado.
4. Gemini recibe la evidencia causal y esa lista cerrada. Solo puede devolver un `runbook_id` existente; no puede generar comandos ni modificar el score.
5. Un servidor MCP separado ejecuta el runbook en `controlled_infrastructure_demo`. Las herramientas de investigación de solo lectura nunca reciben permisos de ejecución.
6. Sentinel observa CPU, RAM, disco, latencia y errores durante `CHALLENGE_REMEDIATION_VERIFICATION_MINUTES` (10 por defecto) en el reloj acelerado de la infraestructura controlada.
7. Si todas las reglas se sostienen, registra recuperación. Si una verificación falla, MCP revierte la acción y Sentinel escala al operador.
8. Si Gemini o MCP no están disponibles, la detección continúa y la remediación queda escalada; nunca se sustituye la decisión con una acción inventada.

El catálogo está en `config/challenge_remediation_runbooks.json`. Los tiempos, endpoints, tokens, umbral crítico y rutas se configuran mediante variables de entorno de `Settings`; no existen selecciones por corrida ni timestamps especiales en el código.

## Evidencia reproducible

Ejecutar:

```bash
.venv/bin/python -m app.challenge.remediation_audit \
  --output data/challenge-phase-4 \
  --fail-on-gates
```

El gate selecciona corridas oficiales de forma ordenada según el umbral configurado y probó cuatro casos distintos:

| Corrida | Riesgo | Caso | Resultado observado |
|---|---:|---|---|
| `corrida_01` | 0.653438 | camino autónomo | acción MCP y recuperación verificadas |
| `corrida_02` | 0.641718 | operador responde | no se creó ni ejecutó remediación |
| `corrida_03` | 0.655750 | Gemini no disponible | escalamiento seguro, sin acción |
| `corrida_04` | 0.604312 | verificación falla | rollback MCP y escalamiento |

Los valores completos, hashes SHA-256 de features/modelo/política/catálogo, timestamps, muestras, decisiones y acciones están en `data/challenge-phase-4/remediation-audit.json`.

## Límites explícitos

- El dataset oficial no expone infraestructura real de Kaiku. Las acciones y la recuperación se verifican en infraestructura controlada, no en producción.
- El gate automatizado usa un doble contractual de Gemini para ser reproducible y prueba por separado su indisponibilidad. Una llamada real solo ocurre cuando `GEMINI_API_KEY` está configurada.
- El gate invoca exactamente las herramientas del registro MCP mediante el cliente MCP en proceso; el despliegue normal usa Streamable HTTP y un endpoint/token de ejecución separado.
- No se usa ground truth oficial, no se leen datos futuros y no se toca ninguna carpeta reservada del evaluador.

## Variables principales

- `CHALLENGE_CRITICAL_CONFIDENCE_THRESHOLD` (por defecto `0.60`)
- `CHALLENGE_REMEDIATION_ENABLED` (por defecto `true`)
- `CHALLENGE_REMEDIATION_WAIT_SECONDS` (por defecto `300`)
- `CHALLENGE_REMEDIATION_VERIFICATION_MINUTES` (por defecto `10`)
- `CHALLENGE_REMEDIATION_MINUTE_SECONDS` (por defecto `1` en la demo acelerada)
- `CHALLENGE_REMEDIATION_CATALOG_PATH`
- `MCP_EXECUTION_URL`
- `MCP_EXECUTION_TOKEN`
- `GEMINI_API_KEY` y `GEMINI_MODEL`

Endpoints de evidencia:

- `GET /api/challenge/remediations`
- `POST /api/challenge/remediations/process` (token administrativo)
- `GET /api/challenge/remediations/audit.csv`
