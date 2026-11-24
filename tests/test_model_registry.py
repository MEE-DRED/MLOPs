import json
import tempfile
import unittest
from pathlib import Path

from src.model_registry import ModelRegistry, RegistryError, build_metadata


def meta(**kw):
    return build_metadata(architecture="test", training_dataset={"n": 1}, training_config={},
                          metrics={"validation": {"f1_macro": 0.5}}, **kw)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.reg = ModelRegistry(self.dir / "models")
        self.f = self.dir / "cand.keras"; self.f.write_bytes(b"model-bytes-1")

    def test_empty_registry(self):
        self.assertIsNone(self.reg.current_version()); self.assertEqual(self.reg.list_versions(), [])
        self.assertEqual(self.reg.next_version(), 1)

    def test_register_creates_files_and_metadata(self):
        m = self.reg.register_version(self.f, meta(), "accepted", make_current=True, reason="initial")
        self.assertEqual(m["version"], 1)
        self.assertTrue(self.reg.model_path(1).is_file()); self.assertTrue(self.reg.metadata_path(1).is_file())
        self.assertEqual(self.reg.current_version(), 1)
        self.assertEqual(self.reg.get_current_metadata()["status"], "accepted")
        self.assertIn("created_at", m); self.assertIn("disclaimer", m)

    def test_versions_increment_and_old_files_kept(self):
        self.reg.register_version(self.f, meta(), "accepted", make_current=True)
        self.f.write_bytes(b"model-bytes-2")
        self.reg.register_version(self.f, meta(parent_version=1), "accepted", make_current=True)
        self.assertEqual(self.reg.current_version(), 2)
        self.assertEqual(self.reg.model_path(1).read_bytes(), b"model-bytes-1")   # untouched
        self.assertEqual(len(self.reg.list_versions()), 2)

    def test_rejected_never_current(self):
        self.reg.register_version(self.f, meta(), "accepted", make_current=True)
        with self.assertRaises(RegistryError):
            self.reg.register_version(self.f, meta(), "rejected", make_current=True)
        self.reg.register_version(self.f, meta(), "rejected")
        self.assertEqual(self.reg.current_version(), 1)                           # unchanged
        with self.assertRaises(RegistryError):
            self.reg.set_current(2)
        self.assertEqual(self.reg.get_metadata(2)["status"], "rejected")

    def test_invalid_inputs(self):
        with self.assertRaises(RegistryError):
            self.reg.register_version(self.dir / "missing.keras", meta(), "accepted")
        empty = self.dir / "empty.keras"; empty.write_bytes(b"")
        with self.assertRaises(RegistryError):
            self.reg.register_version(empty, meta(), "accepted")
        with self.assertRaises(RegistryError):
            self.reg.register_version(self.f, meta(), "maybe")

    def test_rollback(self):
        for i in range(3):
            self.f.write_bytes(f"m{i}".encode())
            self.reg.register_version(self.f, meta(), "accepted", make_current=True)
        self.assertEqual(self.reg.rollback(), 2); self.assertEqual(self.reg.current_version(), 2)
        self.assertEqual(self.reg.rollback(), 1)
        with self.assertRaises(RegistryError):
            self.reg.rollback()
        self.assertTrue(self.reg.model_path(3).is_file())                         # nothing deleted

    def test_rollback_skips_rejected_and_invalid(self):
        self.reg.register_version(self.f, meta(), "accepted", make_current=True)     # v1
        self.reg.register_version(self.f, meta(), "accepted", make_current=True)     # v2
        self.reg.register_version(self.f, meta(), "accepted", make_current=True)     # v3
        self.assertEqual(self.reg.rollback(validator=lambda p: (p.name != "model_v2.keras", "")), 1)

    def test_validator_blocks_set_current(self):
        self.reg.register_version(self.f, meta(), "accepted", make_current=True)
        self.reg.register_version(self.f, meta(), "accepted")
        with self.assertRaises(RegistryError):
            self.reg.set_current(2, validator=lambda p: (False, "broken"))
        self.assertEqual(self.reg.current_version(), 1)

    def test_corrupt_pointer_detected(self):
        self.reg.register_version(self.f, meta(), "accepted", make_current=True)
        self.reg.pointer_path.write_text("{not json")
        with self.assertRaises(RegistryError):
            self.reg.current_version()

    def test_state_survives_restart(self):
        self.reg.register_version(self.f, meta(), "accepted", make_current=True)
        again = ModelRegistry(self.dir / "models")
        self.assertEqual(again.current_version(), 1); self.assertEqual(again.next_version(), 2)

    def test_event_log(self):
        self.reg.register_version(self.f, meta(), "accepted", make_current=True, reason="initial")
        ptr = json.loads(self.reg.pointer_path.read_text())
        self.assertEqual(ptr["events"][0]["reason"], "initial")

    def test_no_temp_files_left(self):
        self.reg.register_version(self.f, meta(), "accepted", make_current=True)
        leftovers = list((self.dir / "models").rglob("*.tmp"))
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
