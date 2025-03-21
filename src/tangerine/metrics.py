from prometheus_client import REGISTRY, Counter, Gauge
from prometheus_flask_exporter import RESTfulPrometheusMetrics

import tangerine.config as cfg

metrics = RESTfulPrometheusMetrics.for_app_factory(
    registry=REGISTRY, defaults_prefix=cfg.METRICS_PREFIX
)


def get_counter(name: str, description: str, labels: list[str] = None) -> Counter:
    metric_name = f"{cfg.METRICS_PREFIX}_{name}"

    if labels:
        return Counter(metric_name, description, labels)

    return Counter(metric_name, description)


def get_gauge(name: str, description: str, labels: list[str] = None) -> Gauge:
    metric_name = f"{cfg.METRICS_PREFIX}_{name}"

    if labels:
        return Gauge(metric_name, description, labels)

    return Gauge(metric_name, description)
