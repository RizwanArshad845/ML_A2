"""Task 5: print the raw prediction records straight from the database.

Uses the same DATABASE_URL as the app, so it works for SQLite locally and Postgres in Compose.

    python -m app.show_history            # last 20 records
    python -m app.show_history 50
"""
import sys

from app.config import Settings
from app.db import Database


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    db = Database(Settings.from_env().database_url)
    print(f"Database: {db.engine.url.render_as_string(hide_password=True)}  ({db.count()} records)\n")
    print(f"{'id':>3}  {'created_at (UTC)':19}  {'filename':32}  {'label':12}  {'P(AI)':>6}  "
          f"{'conf':>6}  {'thr':>4}  {'model':14}  {'ms':>5}")
    for r in db.recent(limit):
        print(f"{r.id:>3}  {r.created_at:%Y-%m-%d %H:%M:%S}  {r.filename[:32]:32}  {r.label:12}  "
              f"{r.prob_ai:6.3f}  {r.confidence:6.1%}  {r.threshold:4.2f}  "
              f"{r.model_name + ' v' + r.model_version:14}  {r.latency_ms:5.0f}")


if __name__ == "__main__":
    main()
