"""Regression: the committed leaderboard table matches what render_table.py emits.

`leaderboard/README.md` embeds a generated table between BEGIN/END markers and
claims it reproduces `python leaderboard/render_table.py leaderboard` exactly.
This test enforces that, so a refactor of GROUPS / formatting / count logic can't
silently drift the published table from the committed records.
"""

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LB = ROOT / "leaderboard"


def _generated_block(readme: str) -> str:
    m = re.search(
        r"<!-- BEGIN GENERATED TABLE.*?-->\n(.*)\n<!-- END GENERATED TABLE -->",
        readme,
        re.DOTALL,
    )
    assert m, "no BEGIN/END GENERATED TABLE markers in leaderboard/README.md"
    return m.group(1).rstrip("\n")


def test_render_table_matches_committed_readme():
    rendered = subprocess.run(
        [sys.executable, str(LB / "render_table.py"), "leaderboard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.rstrip("\n")
    committed = _generated_block((LB / "README.md").read_text(encoding="utf-8"))
    assert rendered == committed, "render_table.py output has drifted from leaderboard/README.md"
