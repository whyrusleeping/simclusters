#!/bin/bash
# Export follow graph from postgres for simmy input
# Only includes follows created in the past 9 months
# Usage: DATABASE_URL=postgres://user:pass@host:port/db ./export-follows.sh [output_file]

set -euo pipefail

: "${DATABASE_URL:?set DATABASE_URL to the postgres connection string}"
OUTPUT=${1:-follows.csv}

psql "$DATABASE_URL" -c "\COPY (
  SELECT f.id, f.author, f.subject
  FROM follows f
  WHERE f.created >= NOW() - INTERVAL '9 months'
) TO STDOUT WITH CSV" > "$OUTPUT"

echo "Exported follows graph (past 9 months) to $OUTPUT"
echo "Row count: $(wc -l < "$OUTPUT")"
echo ""
echo "Usage: ./simmy run $OUTPUT"
