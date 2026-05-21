"""Tests for the trtcheck.plugins module (v1.0 surface)."""

from __future__ import annotations

from typing import Protocol


class TestPluginsModuleSurface:
    def test_plugins_module_exposes_three_protocols(self) -> None:
        from trtcheck.plugins import Checker, Fixer, Reporter

        # Each Protocol must declare a `name` annotation. Protocol classes
        # carry annotations on __annotations__ rather than as concrete
        # class attributes, so check that.
        for cls in (Checker, Fixer, Reporter):
            assert "name" in cls.__annotations__, f"{cls.__name__} missing 'name'"
            # And one callable method (check / fix / render)
            methods = [n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n))]
            assert methods, f"{cls.__name__} declares no methods"

    def test_existing_paths_re_export_same_protocol_objects(self) -> None:
        """Importing Checker from the old path must yield the same object as
        importing from trtcheck.plugins. Anything else would create a
        protocol-identity split and break callers that rely on
        isinstance/issubclass checks."""
        from trtcheck.checkers import Checker as legacy_checker
        from trtcheck.fixers import Fixer as legacy_fixer
        from trtcheck.plugins import Checker, Fixer, Reporter
        from trtcheck.reporters import Reporter as legacy_reporter

        assert legacy_checker is Checker
        assert legacy_fixer is Fixer
        assert legacy_reporter is Reporter

    def test_runtime_checkable_isinstance_works(self) -> None:
        """Plugins discovered at runtime must isinstance-check against the
        Protocol so the loader can validate them. This requires
        @runtime_checkable on the Protocol."""
        from trtcheck.plugins import Checker

        class Fake:
            name = "fake"

            def check(self, model):  # type: ignore[no-untyped-def]
                return []

        assert isinstance(Fake(), Checker)

    def test_in_tree_checkers_still_satisfy_protocol(self) -> None:
        from trtcheck.checkers.graph_structure import GraphStructureChecker
        from trtcheck.plugins import Checker

        assert isinstance(GraphStructureChecker(), Checker)

    def test_in_tree_fixers_still_satisfy_protocol(self) -> None:
        from trtcheck.fixers.int64_to_int32 import Int64ToInt32Fixer
        from trtcheck.plugins import Fixer

        assert isinstance(Int64ToInt32Fixer(), Fixer)

    def test_in_tree_reporters_still_satisfy_protocol(self) -> None:
        from trtcheck.plugins import Reporter
        from trtcheck.reporters.json import JSONReporter

        assert isinstance(JSONReporter(), Reporter)
