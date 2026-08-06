FROM registry.access.redhat.com/ubi9/python-312:9.8-1785963992 AS builder

USER root

ENV PIPENV_VERBOSITY=-1

COPY Pipfile .
COPY Pipfile.lock .
COPY pyproject.toml .

RUN pip install pipenv
RUN python3 -m venv .venv
RUN source .venv/bin/activate
RUN pipenv sync

COPY migrations .
COPY src .
COPY .flaskenv .

FROM registry.access.redhat.com/ubi10/ubi-minimal:10.2-1779722607

ENV APP_ROOT=/opt/app-root/src
ENV LC_ALL=C.utf8
ENV LANG=C.utf8
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV NLTK_DATA_DIR=/nltk_data
ENV PATH="/opt/app-root/src/.venv/bin:$PATH"

USER root

RUN microdnf install -y --setopt=install_weak_deps=0 --setopt=tsflags=nodocs python3 && \
  microdnf clean all && \
  rm -rf /var/cache/dnf/* && \
  mkdir /nltk_data && \
  chown -R 1001:0 /nltk_data && \
  chmod -R g=u /nltk_data

WORKDIR $APP_ROOT

COPY --from=builder $APP_ROOT .

USER 1001

EXPOSE 8000

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
