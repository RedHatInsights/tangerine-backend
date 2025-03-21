import io
import json
import logging
from typing import Optional

import httpx
from httpx_retries import Retry, RetryTransport
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

import tangerine.config as cfg

from .metrics import get_counter

log = logging.getLogger("tangerine.embeddings")

embed_prompt_tokens_metric = get_counter(
    "embed_prompt_tokens", "Embedding model prompt tokens usage"
)


# because we currently cannot access usage_metadata for embedding calls nor use
# get_openai_callback() in the same way we can for chat model calls...
# (see https://github.com/langchain-ai/langchain/issues/945)
#
# we use a work-around inspired by https://github.com/encode/httpx/discussions/3073
class CustomResponse(httpx.Response):
    def iter_bytes(self, *args, **kwargs):
        content = io.BytesIO()

        # copy the chunk into our own buffer but yield same chunk to caller
        for chunk in super().iter_bytes(*args, **kwargs):
            content.write(chunk)
            yield chunk

        # check to see if content can be loaded as json and look for 'usage' key
        content.seek(0)
        try:
            usage = json.load(content).get("usage")
        except json.JSONDecodeError:
            usage = {}

        if not usage:
            log.debug("no 'usage' in embedding response")
            return

        try:
            prompt_tokens = int(usage.get("prompt_tokens", 0))
        except ValueError:
            log.debug("invalid 'usage' content in embedding response")
            return

        log.debug("embedding prompt tokens: %d", prompt_tokens)
        embed_prompt_tokens_metric.inc(prompt_tokens)


# base this on top of RetryTransport so that we can configure http backoff timers
class CustomTransport(RetryTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = super().handle_request(request)

        return CustomResponse(
            status_code=response.status_code,
            headers=response.headers,
            stream=response.stream,
            extensions=response.extensions,
        )


embeddings = OpenAIEmbeddings(
    http_client=httpx.Client(
        transport=CustomTransport(
            retry=Retry(
                total=3,
                backoff_factor=0.5,
                max_backoff_wait=30,
                status_forcelist=[429, 502, 503],  # intentionally omitting 504
            )
        )
    ),
    max_retries=0,  # disable openai client's built-in retry mechanism
    model=cfg.EMBED_MODEL_NAME,
    openai_api_base=cfg.EMBED_BASE_URL,
    openai_api_key=cfg.EMBED_API_KEY,
    check_embedding_ctx_length=False,
)


def embed_query(query: str) -> Optional[Embeddings]:
    if cfg.EMBED_QUERY_PREFIX:
        query = f"{cfg.EMBED_QUERY_PREFIX}: {query}"
    return embeddings.embed_query(query)
