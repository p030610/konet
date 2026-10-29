from __future__ import annotations

from enum import Enum
from typing_extensions import assert_never


class VisualPolicy(str, Enum):
    NATIVE = "native"
    NORMALIZE_1280X720 = "normalize-1280x720"


class SourceEligibility(str, Enum):
    UNRESTRICTED = "unrestricted"
    ACTUAL_720P_16_9 = "actual-width>=1280;height>=720;dar=16:9"


def source_eligibility(policy: VisualPolicy) -> SourceEligibility:
    match policy:
        case VisualPolicy.NATIVE:
            return SourceEligibility.UNRESTRICTED
        case VisualPolicy.NORMALIZE_1280X720:
            return SourceEligibility.ACTUAL_720P_16_9
        case unreachable:
            assert_never(unreachable)


__all__ = ["SourceEligibility", "VisualPolicy", "source_eligibility"]
