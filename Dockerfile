FROM registry.access.redhat.com/hi/python:3.12.14-builder-1790221360 AS builder

USER root

ENV PIPENV_VERBOSITY=-1

COPY pyproject.toml .
COPY Pipfile* .

RUN pip install pipenv
RUN python3 -m venv /opt/venv
RUN source /opt/venv/bin/activate && \
	pipenv sync

COPY src /opt/app
COPY migrations /opt/app/migrations
COPY .flaskenv /opt/app/.flaskenv

USER ${CONTAINER_DEFAULT_USER}

FROM registry.access.redhat.com/hi/python:3.12.14-1790221360

ENV LC_ALL=C.utf8
ENV LANG=C.utf8
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV NLTK_DATA_DIR=/tmp/nltk_data
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/app /opt/app

EXPOSE 8000

WORKDIR /opt/app

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
