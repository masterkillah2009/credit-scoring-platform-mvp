#!/bin/sh
# Put the demonstration ledger in place, then run the service.
#
# The ledger is seeded at build time (see the Dockerfile), so this is a file
# copy rather than several hundred scoring runs. That matters on a host that
# health-checks a new instance quickly: seeding on boot risks the service
# being killed before it answers.
set -e

LEDGER="${DEMO_LEDGER:-/data/ledger.db}"

if [ ! -f "$LEDGER" ]; then
  if [ -f /app/seed-ledger.db ]; then
    echo "No ledger at $LEDGER - installing the pre-seeded demonstration data."
    mkdir -p "$(dirname "$LEDGER")"
    cp /app/seed-ledger.db "$LEDGER"
  else
    echo "No ledger and no pre-seeded data - seeding now (this is slow)..."
    python -m demo.seed || echo "Seeding failed; starting with an empty ledger."
  fi
fi

exec python -m api.server
