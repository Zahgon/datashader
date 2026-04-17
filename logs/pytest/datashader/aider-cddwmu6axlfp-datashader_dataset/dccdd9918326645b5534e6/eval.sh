#!/bin/bash
set -uxo pipefail
cd /testbed
git reset --hard 3471746e0853cbb16b96f79ea33941dca75285c1
git apply --allow-empty -v /patch.diff
git status
pytest --json-report --json-report-file=report.json --continue-on-collection-errors datashader/tests/test_composite.py > test_output.txt 2>&1
echo $? > pytest_exit_code.txt
