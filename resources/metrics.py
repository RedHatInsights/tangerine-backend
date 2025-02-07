from prometheus_client import REGISTRY, Counter, Gauge
from prometheus_flask_exporter import RESTfulPrometheusMetrics

metrics = RESTfulPrometheusMetrics.for_app_factory(registry=REGISTRY)


def get_counter(name: str, description: str) -> Counter:
    return Counter(name, description)


def get_gauge(name: str, description: str) -> Gauge:
    return Gauge(name, description)
