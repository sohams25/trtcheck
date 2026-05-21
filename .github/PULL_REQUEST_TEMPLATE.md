## Summary

<!-- 1-3 sentences describing the change and motivation. -->

## Checklist

- [ ] Tests added or updated (new checker or fix path)
- [ ] `pytest` passes locally (`./scripts/run-tests.sh`)
- [ ] `mypy trtcheck/` is clean
- [ ] `black --check . && isort --check-only .` is clean
- [ ] If editing `trtcheck/data/*.json`, ran `python tools/build_operator_matrix.py` and updated `tests/test_data_files.py` if needed
- [ ] CHANGELOG.md updated (under "Unreleased")
