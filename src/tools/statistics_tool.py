
import math
from statistics import fmean


def summarize_measurements(
    values: list[float],
) -> dict[str, int | float]:
    """Summarize a nonempty list of finite numerical measurements."""

    if not isinstance(values, list):
        raise TypeError("Measurements must be a list.")

    if not values:
        raise ValueError("Measurements cannot be empty.")

    if len(values) > 10000:
        raise ValueError("Too many measurements.")

    measurements = []

    for value in values:
        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise TypeError("Measurements must contain only numbers.")

        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("Measurement is too large.") from exc

        if not math.isfinite(number):
            raise ValueError("Measurements must be finite.")

        measurements.append(number)

    return {
        "count": len(measurements),
        "mean": fmean(measurements),
        "minimum": min(measurements),
        "maximum": max(measurements),
    }
