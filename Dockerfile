FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MONITOR_ENABLED=false \
    TRAFFIC_ENABLED=false \
    RESOURCE_MONITOR_ENABLED=false \
    AUTO_MITIGATION_ENABLED=false \
    MCP_HTTP_ENABLED=false \
    EVIDENCE_API_ENABLED=false \
    CHALLENGE_FEATURES_PATH=/app/data/challenge-phase-3-logs \
    CHALLENGE_MODEL_PATH=/app/runtime/models/challenge_early_warning.joblib \
    CHALLENGE_POLICY_PATH=/app/runtime/models/challenge_alert_policy.json \
    CHALLENGE_SHOWCASE_PATH=/app/runtime/evidence/showcase.json \
    CHALLENGE_REHEARSAL_PATH=/app/runtime/evidence/rehearsal-report.json \
    CHALLENGE_RUNTIME_PATH=/app/runtime

WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN python -c "import subprocess, sys, tomllib; data = tomllib.load(open('pyproject.toml', 'rb')); dependencies = data['project']['dependencies'] + data['project']['optional-dependencies']['ai']; subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', *dependencies])"
COPY app ./app
COPY models/isolation_forest.joblib models/isolation_forest.joblib
COPY models/isolation_forest.joblib.metadata.json models/isolation_forest.joblib.metadata.json
COPY models/isolation_forest.joblib.sha256 models/isolation_forest.joblib.sha256
COPY deploy/challenge-runtime ./runtime
COPY data/challenge-phase-3-logs ./data/challenge-phase-3-logs
RUN pip install --no-cache-dir --no-deps .
RUN mkdir -p /app/data /app/models && useradd --create-home sentinel && chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health/ready', timeout=3)"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
