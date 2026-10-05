import os
import json
import shutil
from datetime import datetime


class KBVersionManager:
    def __init__(self, storage_root="kb_storage"):
        self.storage_root = storage_root
        self.versions_dir = os.path.join(self.storage_root, "kb_versions")
        self.pointer_file = os.path.join(self.storage_root, "current_version.json")

        # Make sure the base folders exist.
        os.makedirs(self.versions_dir, exist_ok=True)
        if not os.path.exists(self.pointer_file):
            self._write_pointer(active_version=None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_pointer(self, active_version):
        """Overwrite the pointer file with the given active version name."""
        data = {
            "active_version": active_version,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.pointer_file, "w") as f:
            json.dump(data, f, indent=2)

    def _read_pointer(self):
        with open(self.pointer_file, "r") as f:
            return json.load(f)

    def _version_path(self, version_name):
        return os.path.join(self.versions_dir, version_name)

    def version_exists(self, version_name):
        return os.path.isdir(self._version_path(version_name))

    def _lock_version_directory(self, version_path):
        """
        Mark a finished version folder as read-only, so nothing (including
        this module) can accidentally write into, overwrite, or delete
        files inside it afterward. This is a lightweight, standard-library
        safeguard - not a full permission system.

        Directories are set to read+execute (needed to list/read contents,
        e.g. 0o555) and files are set to read-only (0o444). Any accidental
        write attempt against a locked version will raise a PermissionError
        instead of silently succeeding.
        """
        for dirpath, dirnames, filenames in os.walk(version_path):
            for filename in filenames:
                os.chmod(os.path.join(dirpath, filename), 0o444)
            os.chmod(dirpath, 0o555)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_version(self, source_folder, notes=""):
        """
        Save a new KB version.

        source_folder: path to a folder containing the freshly built
                        KB / vector index files (output of your earlier
                        pipeline steps, already validated in Step 1).
        notes:          optional human-readable note about this version.

        Returns the new version name, e.g. "v_20260828_101500".
        This NEVER touches or deletes any existing version.
        """
        if not os.path.isdir(source_folder):
            raise FileNotFoundError(f"Source folder not found: {source_folder}")

        base_name = "v_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = base_name
        version_path = self._version_path(version_name)

        # If two versions get created within the same second, add a
        # small counter suffix instead of failing (e.g. v_..._2).
        counter = 2
        while os.path.exists(version_path):
            version_name = f"{base_name}_{counter}"
            version_path = self._version_path(version_name)
            counter += 1

        os.makedirs(version_path)

        # Copy the KB files into kb_data/ inside the new version folder.
        kb_data_path = os.path.join(version_path, "kb_data")
        shutil.copytree(source_folder, kb_data_path)

        # Save some basic metadata about this version.
        metadata = {
            "version": version_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_folder": os.path.abspath(source_folder),
            "notes": notes,
        }
        with open(os.path.join(version_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Lock the finished version folder so it can no longer be
        # modified, overwritten, or deleted in-place through normal
        # file operations (see _lock_version_directory for details).
        self._lock_version_directory(version_path)

        print(f"[created] New KB version saved: {version_name}")
        return version_name

    def set_active_version(self, version_name):
        """
        Point the system at a specific existing version.
        Used both for normal 'promote a new version' and for rollback.
        """
        if not self.version_exists(version_name):
            raise ValueError(f"Cannot activate '{version_name}': it does not exist.")

        self._write_pointer(active_version=version_name)
        print(f"[active] Now serving KB version: {version_name}")

    def rollback_to(self, version_name):
        """
        Roll back to a previous version. This is really just
        set_active_version() with a clearer name for intent -
        it's safe because old versions are never deleted or modified.
        """
        if not self.version_exists(version_name):
            raise ValueError(
                f"Cannot roll back to '{version_name}': it does not exist."
            )
        print(f"[rollback] Rolling back to KB version: {version_name}")
        self.set_active_version(version_name)

    def get_active_version(self):
        """Return the name of the currently active version (or None)."""
        return self._read_pointer().get("active_version")

    def get_active_kb_path(self):
        """Return the folder path of the currently active KB's data."""
        active = self.get_active_version()
        if active is None:
            return None
        return os.path.join(self._version_path(active), "kb_data")

    def list_versions(self):
        """Return a list of version names, oldest first."""
        versions = sorted(
            v for v in os.listdir(self.versions_dir)
            if os.path.isdir(self._version_path(v))
        )
        return versions

    def get_version_metadata(self, version_name):
        meta_path = os.path.join(self._version_path(version_name), "metadata.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r") as f:
            return json.load(f)


# ----------------------------------------------------------------------
# Simple demo / manual test when running this file directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print("=== Step 2: Versioning & Storage — demo ===\n")

    manager = KBVersionManager(storage_root="kb_storage_demo")

    # Simulate two KB builds coming out of your Step 1 pipeline.
    with tempfile.TemporaryDirectory() as tmp1:
        with open(os.path.join(tmp1, "index.json"), "w") as f:
            json.dump({"docs": ["doc1", "doc2"]}, f)
        v1 = manager.create_version(tmp1, notes="Initial KB build")

    manager.set_active_version(v1)

    with tempfile.TemporaryDirectory() as tmp2:
        with open(os.path.join(tmp2, "index.json"), "w") as f:
            json.dump({"docs": ["doc1", "doc2", "doc3"]}, f)
        v2 = manager.create_version(tmp2, notes="Added doc3")

    manager.set_active_version(v2)

    print("\nAll versions:", manager.list_versions())
    print("Active version:", manager.get_active_version())
    print("Active KB path:", manager.get_active_kb_path())

    # Simulate discovering v2 is bad -> roll back to v1
    print("\n-- Simulating rollback --")
    manager.rollback_to(v1)
    print("Active version after rollback:", manager.get_active_version())