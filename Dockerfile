# Use ARG to allow switching between development and production base images
# Default: quay.io/sclorg/postgresql-16-c10s:latest (for local development)
# Production: registry.redhat.io/rhel9/postgresql-16 (see ci-build-args.txt)
ARG BASE_IMAGE=quay.io/sclorg/postgresql-16-c10s:latest
FROM ${BASE_IMAGE} AS builder

# builder runs 'pipenv install' using postgres image to properly compile psycopg2

ENV PIP_NO_CACHE_DIR=1
ENV APP_ROOT=/opt/app-root/src

USER root

WORKDIR $APP_ROOT

RUN dnf -y upgrade && \
    dnf -y install --setopt=install_weak_deps=0 --setopt=tsflags=nodocs \
        gcc \
        make \
        python3-devel \
        python3-pip \
        which \
        libpq-devel && \
    dnf clean all && \
    rm -rf /var/cache/dnf/*

COPY Pipfile .
COPY Pipfile.lock .

RUN python3 -m venv .venv && \
    source .venv/bin/activate && \
    python3 -m pip install --upgrade pip setuptools wheel pipenv && \
    pipenv install --system --deploy --verbose

ENV PATH="$APP_ROOT/.venv/bin:$PATH"

FROM registry.access.redhat.com/ubi10/ubi-minimal:latest

ENV APP_ROOT=/opt/app-root/src
ENV LC_ALL=C.utf8
ENV LANG=C.utf8
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV PATH="$APP_ROOT/.venv/bin:$PATH"

USER root

RUN microdnf -y upgrade && \
    microdnf install -y --setopt=install_weak_deps=0 --setopt=tsflags=nodocs python3 libpq && \
    microdnf clean all && \
    rm -rf /var/cache/dnf/*

WORKDIR $APP_ROOT
COPY --from=builder $APP_ROOT/.venv .venv
COPY pyproject.toml .
COPY src ./src
COPY migrations ./migrations
COPY .flaskenv .
RUN pip install .

RUN mkdir /nltk_data && chown -R 1001:0 /nltk_data && chmod -R g=u /nltk_data
ENV NLTK_DATA_DIR=/nltk_data

USER 1001

EXPOSE 8000

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
