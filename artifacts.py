"""
Content-addressable artifact store for Session 6 agent.

Raw bytes are stored outside Memory under state/artifacts/.
Memory should keep only the returned artifact handle.
"""
import hashlib
import json
from pathlib import Path

from schemas import Artifact

STATE_DIR = Path(__file__).parent / "state"
ARTIFACTS_DIR = STATE_DIR / "artifacts"
SHA_PREFIX_LENGTH = 16
ARTIFACT_ID_PREFIX = "art:"


class ArtifactStore:
    """Persistent content-addressable byte store."""

    def __init__(self) -> None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def _normalize_artifact_id(self, artifact_id: str) -> str:
        if not artifact_id.startswith(ARTIFACT_ID_PREFIX):
            raise ValueError(f"Invalid artifact id: {artifact_id}")
        prefix = artifact_id[len(ARTIFACT_ID_PREFIX):]
        if not prefix:
            raise ValueError(f"Invalid artifact id: {artifact_id}")
        return prefix

    def _artifact_paths(self, artifact_id: str) -> tuple[Path, Path]:
        prefix = self._normalize_artifact_id(artifact_id)
        return (
            ARTIFACTS_DIR / f"{prefix}.bin",
            ARTIFACTS_DIR / f"{prefix}.json",
        )

    def put(self, blob: bytes, *, content_type: str, source: str, descriptor: str) -> str:
        """
        Persist bytes and metadata, deduplicated by content hash.
        Returns an artifact handle of the form art:<sha256-prefix>.
        """
        sha256 = hashlib.sha256(blob).hexdigest()
        prefix = sha256[:SHA_PREFIX_LENGTH]
        artifact_id = f"{ARTIFACT_ID_PREFIX}{prefix}"
        bin_path, meta_path = self._artifact_paths(artifact_id)

        if not bin_path.exists():
            bin_path.write_bytes(blob)

        if not meta_path.exists():
            artifact = Artifact(
                id=artifact_id,
                content_type=content_type,
                size_bytes=len(blob),
                source=source,
                descriptor=descriptor,
            )
            meta_path.write_text(
                json.dumps(artifact.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )

        return artifact_id

    def get_bytes(self, artifact_id: str) -> bytes:
        """Load raw bytes for an artifact handle."""
        bin_path, _ = self._artifact_paths(artifact_id)
        if not bin_path.exists():
            raise FileNotFoundError(f"Artifact bytes not found for {artifact_id}")
        return bin_path.read_bytes()

    def get_meta(self, artifact_id: str) -> Artifact:
        """Load artifact metadata for an artifact handle."""
        _, meta_path = self._artifact_paths(artifact_id)
        if not meta_path.exists():
            raise FileNotFoundError(f"Artifact metadata not found for {artifact_id}")
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return Artifact(**data)

    def exists(self, artifact_id: str) -> bool:
        """Check whether both payload and metadata exist for an artifact handle."""
        bin_path, meta_path = self._artifact_paths(artifact_id)
        return bin_path.exists() and meta_path.exists()


_artifact_store = None


def get_artifact_store() -> ArtifactStore:
    """Get or create global artifact store instance."""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store

# Made with Bob
