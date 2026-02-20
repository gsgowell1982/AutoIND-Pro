from dataclasses import dataclass


@dataclass(slots=True)
class VersionRecord:
    regulation_version: str
    rule_version: str


class VersionManager:
    """Track aligned regulation and rule versions used in each run."""

    def __init__(self, regulation_version: str, rule_version: str) -> None:
        self._record = VersionRecord(
            regulation_version=regulation_version,
            rule_version=rule_version,
        )

    def snapshot(self) -> VersionRecord:
        return self._record
