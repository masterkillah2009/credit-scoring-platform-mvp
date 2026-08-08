# Retail Lending Scoring Platform - demonstration image.
#
# Two stages: the first trains the scorecard (the only step needing numpy),
# the second carries just the runtime, which has no third-party dependencies
# at all. The result is a small image with no build toolchain in it.

FROM python:3.12-slim AS model
WORKDIR /build
RUN pip install --no-cache-dir numpy==2.* 
COPY core/ core/
COPY model/ model/
COPY tools/ tools/
RUN python -m model.train_scorecard && python -m model.calibrate_cutoffs

FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="Retail Lending Scoring Platform (demo)"
LABEL org.opencontainers.image.description="Demonstration deployment - synthetic data, prototype model, not for production"

# Run as an unprivileged user; the ledger lives on a writable volume.
RUN useradd --create-home --uid 10001 platform \
 && mkdir -p /data && chown platform:platform /data
WORKDIR /app

COPY --chown=platform:platform core/ core/
COPY --chown=platform:platform api/ api/
COPY --chown=platform:platform partners/ partners/
COPY --chown=platform:platform demo/ demo/
COPY --chown=platform:platform ui/ ui/
COPY --chown=platform:platform docs/ docs/
COPY --chown=platform:platform tools/ tools/
COPY --chown=platform:platform model/ model/
COPY --chown=platform:platform --from=model /build/artefacts/ artefacts/
COPY --chown=platform:platform entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Seed the demonstration ledger at build time rather than at boot.
#
# Two reasons. First, seeding several hundred decisions takes the better part
# of a minute; doing it on start means the platform's health check can fail
# before the service ever answers, which on a host like Render aborts the
# deploy. Second, baking it makes every cold start open on the identical
# known state - the same four named applicants, the same portfolio - so a
# demonstration is repeatable rather than merely likely.
#
# The entrypoint copies this into place if the ledger is missing, so a mounted
# empty volume still gets a populated starting point.
RUN DEMO_LEDGER=/app/seed-ledger.db python -m demo.seed \
 && chown platform:platform /app/seed-ledger.db

USER platform
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEMO_BIND=0.0.0.0 \
    DEMO_LEDGER=/data/ledger.db \
    PORT=8080
EXPOSE 8080
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=4s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/healthz',timeout=3).status==200 else 1)"

ENTRYPOINT ["/app/entrypoint.sh"]
