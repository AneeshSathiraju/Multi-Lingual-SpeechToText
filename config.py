"""Settings, read from environment variables so you can change them without editing code."""
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    # default_factory reads the environment when Settings() is created, not at import time.
    model_size: str = field(default_factory=lambda: os.getenv("STT_MODEL", "small"))
    device: str = field(default_factory=lambda: os.getenv("STT_DEVICE", "auto"))  # auto | cuda | cpu
    compute_type: str = field(default_factory=lambda: os.getenv("STT_COMPUTE_TYPE", "auto"))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("STT_MAX_UPLOAD_MB", "50")))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
    live_threshold: float = field(default_factory=lambda: float(os.getenv("STT_LIVE_THRESHOLD", "0.01")))