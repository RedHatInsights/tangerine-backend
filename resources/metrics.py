from prometheus_client import REGISTRY, Counter, Gauge
from prometheus_flask_exporter import RESTfulPrometheusMetrics

import connectors.config as cfg

metrics = RESTfulPrometheusMetrics.for_app_factory(
    registry=REGISTRY, defaults_prefix=cfg.METRICS_PREFIX
)


def get_counter(name: str, description: str) -> Counter:
    return Counter(f"{cfg.METRICS_PREFIX}_{name}", description)


def get_gauge(name: str, description: str) -> Gauge:
    return Gauge(f"{cfg.METRICS_PREFIX}_{name}", description)
