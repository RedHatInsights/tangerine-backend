import logging
import time
from typing import TYPE_CHECKING, Generator

from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    # TODO: if llm.rerank() is taken out of search.py and circular import goes away, come back to edit this
    from tangerine.search import SearchResult

import tangerine.config as cfg
from tangerine.agents.jira_agent import JiraAgent
from tangerine.agents.webrca_agent import WebRCAAgent
from tangerine.metrics import get_counter, get_gauge
from tangerine.models.assistant import Assistant

log = logging.getLogger("tangerine.llm")

assistant_response_counter = get_counter(
    "assistant_response_counter",
    "Total number of responses for an assistant",
    ["assistant_id", "assistant_name"],
)
llm_completion_tokens_metric = get_counter("llm_completion_tokens", "LLM completion tokens usage")
llm_prompt_tokens_metric = get_counter("llm_prompt_tokens", "LLM prompt tokens usage")
llm_completion_rate = get_gauge(
    "llm_completion_rate", "Observed tokens per sec from most recent LLM chat completion"
)
llm_processing_rate = get_gauge(
    "llm_processing_rate", "Observed tokens per sec for most recent LLM processing after prompted"
)
llm_no_answer = get_counter(
    "llm_no_answer", "No search results found", ["assistant_id", "assistant_name"]
)


def _record_metrics(
    cb: OpenAICallbackHandler,
    processing_start: float,
    completion_start: float,
    completion_end: float,
) -> None:
    if not completion_start:
        log.error("no content in llm response stream")
        return

    processing_time = completion_start - processing_start
    completion_time = completion_end - completion_start

    try:
        processing_rate = cb.prompt_tokens / processing_time
        completion_rate = cb.completion_tokens / completion_time
    except ZeroDivisionError:
        log.error("unexpected time diff of 0")
        completion_rate = 0

    log.info(
        (
            "AUDIT: prompt tokens: %s, completion tokens: %s, "
            "processing time: %fsec (%f tokens/sec), completion time: %fsec (%f tokens/sec)"
        ),
        cb.prompt_tokens,
        cb.completion_tokens,
        processing_time,
        processing_rate,
        completion_time,
        completion_rate,
    )
    llm_completion_tokens_metric.inc(cb.completion_tokens)
    llm_prompt_tokens_metric.inc(cb.prompt_tokens)
    llm_processing_rate.set(processing_rate)
    llm_completion_rate.set(completion_rate)


def _build_context(search_results: list["SearchResult"], content_char_limit: int = 0):
    search_metadata = []
    context = ""
    log.info("AUDIT: given %d search results as context", len(search_results))
    for i, result in enumerate(search_results):
        page_content = result.document.page_content
        metadata = result.document.metadata
        search_metadata.append(
            {
                "metadata": metadata,
                "page_content": page_content,
            }
        )

        context += f"\n<<Search result {i + 1}"
        if "title" in metadata:
            title = metadata["title"]
            context += f", document title: '{title}'"
        limit = content_char_limit if content_char_limit else len(page_content)
        search_result = page_content[0:limit]
        context += f">>\n\n{search_result}\n\n<<Search result {i + 1} END>>\n"

    return context, search_metadata


def get_response(
    prompt: ChatPromptTemplate,
    prompt_params: dict,
    model_name: str | None = None,
) -> Generator[str, None, None]:
    # AUDIT LOG: get_response entry
    log.info("AUDIT: get_response() called with model_name=%s", model_name)
    
    model_config = cfg.get_model_config(model_name)
    
    # AUDIT LOG: Model config retrieved
    log.info("AUDIT: Retrieved model_config for model_name=%s: %s", model_name, model_config)

    # AUDIT LOG: Creating ChatOpenAI instance
    log.info("AUDIT: Creating ChatOpenAI with config: %s", model_config)
    
    chat = ChatOpenAI(
        **model_config,
        stream_usage=True,
    )
    
    # AUDIT LOG: ChatOpenAI instance created
    log.info("AUDIT: ChatOpenAI instance created successfully")

    chain = prompt | chat

    completion_start = 0.0
    processing_start = time.time()

    with get_openai_callback() as cb:
        for chunk in chain.stream(prompt_params):
            if not completion_start:
                # this is the first output token received
                completion_start = time.time()
            if len(chunk.content):
                yield chunk.content

            # end for
            completion_end = time.time()

        # end with
    _record_metrics(cb, processing_start, completion_start, completion_end)


def rerank(query, search_results: list["SearchResult"]):
    log.info("AUDIT: llm 'rerank' request")
    if len(search_results) <= 1:
        return search_results  # No need to rank if there's only one result

    context, _ = _build_context(search_results, 300)

    prompt = ChatPromptTemplate(
        [("system", cfg.RERANK_SYSTEM_PROMPT), ("user", cfg.RERANK_PROMPT_TEMPLATE)]
    )
    prompt_params = {"query": query, "context": context}

    llm_response = get_response(prompt, prompt_params)
    return "".join(llm_response)


def identify_agent(query):
    log.info("AUDIT: llm 'identify_agent' request")
    prompt = ChatPromptTemplate(
        [("system", cfg.AGENTIC_ROUTER_PROMPT), ("user", cfg.AGENTIC_ROUTER_USER_PROMPT)]
    )
    prompt_params = {"query": query}

    llm_response = get_response(prompt, prompt_params)
    return "".join(llm_response)


def ask(
    assistants: list[Assistant],
    previous_messages,
    question,
    search_results: list["SearchResult"],
    interaction_id=None,
    prompt: str | None = None,
    model: str | None = None,
    disable_agentic: bool = False,
    user_prompt: str | None = None,
) -> tuple[Generator[str, None, None], list[dict]]:
    log.info("AUDIT: llm 'ask' request")
    
    # AUDIT LOG: Function entry
    log.info("AUDIT: llm.ask() called with model=%s, disable_agentic=%s", model, disable_agentic)
    log.info("AUDIT: Assistant details: %s", [{"name": a.name, "model": a.model} for a in assistants])
    search_context = ""
    search_metadata = []

    # Skip agentic workflow if disabled
    log.info("AUDIT: disable_agentic=%s, checking if agentic workflow should run", disable_agentic)
    if not disable_agentic:
        log.info("AUDIT: Running agentic workflow - calling identify_agent()")
        agent = identify_agent(question)
        log.info("AUDIT: identified agent: %s", agent)
        match agent.strip():
            case "JiraAgent":
                if cfg.ENABLE_JIRA_AGENT:
                    log.info("AUDIT: Routing to JiraAgent")
                    return JiraAgent().fetch(question), search_metadata
            case "WebRCAAgent":
                if cfg.ENABLE_WEB_RCA_AGENT:
                    log.info("AUDIT: Routing to WebRCAAgent")
                    return WebRCAAgent().fetch(question), search_metadata
    else:
        log.info("AUDIT: Skipping agentic workflow due to disable_agentic=True")

    if len(search_results) == 0:
        log.info("AUDIT: given 0 search results")
        search_context = "No matching search results found"
        for assistant in assistants:
            # Increment no answer counter for each assistant
            llm_no_answer.labels(assistant_id=assistant.id, assistant_name=assistant.name).inc()
    else:
        search_context, search_metadata = _build_context(search_results)
        for assistant in assistants:
            # Increment response counter for each assistant
            assistant_response_counter.labels(
                assistant_id=assistant.id, assistant_name=assistant.name
            ).inc()

    if not search_metadata:
        search_metadata = [{}]
    for m in search_metadata:
        m["interactionId"] = interaction_id

    # Determine system prompt: use provided prompt, then first assistant's system_prompt, then default
    # TODO: handle the case where assistants are configured with different system prompts
    system_prompt = prompt or assistants[0].system_prompt or cfg.DEFAULT_SYSTEM_PROMPT

    msg_list = [("system", system_prompt)]
    if previous_messages:
        for msg in previous_messages:
            if msg["sender"] == "human":
                msg_list.append(("human", f"[INST] {msg['text']} [/INST]"))
            if msg["sender"] == "ai":
                msg_list.append(("ai", f"{msg['text']}</s>"))
    # Use provided user prompt or default template
    final_user_prompt = user_prompt or cfg.USER_PROMPT_TEMPLATE
    msg_list.append(("human", final_user_prompt))

    # Determine model: use provided model, then first assistant's model
    # TODO: handle the case where assistants are configured with different models
    selected_model = model or assistants[0].model
    
    # AUDIT LOG: Model selection decision
    log.info("AUDIT: Model selection - API model=%s, assistant[0].model=%s, selected_model=%s", 
             model, assistants[0].model, selected_model)

    prompt_params = {"context": search_context, "question": question}
    
    # AUDIT LOG: About to call get_response
    log.info("AUDIT: Calling get_response() with selected_model=%s", selected_model)
    llm_response = get_response(ChatPromptTemplate(msg_list), prompt_params, selected_model)

    return llm_response, search_metadata


def generate_conversation_title(user_queries: list[str]) -> str:
    """
    Generate a conversation title based on a user query using LLM.
    Expects a list with one query (kept as list for consistency).
    """
    log.info("AUDIT: llm 'generate_conversation_title' request")

    # Validate input: ensure at least one non-empty query is provided
    if not user_queries or not user_queries[0].strip():
        raise ValueError("The 'user_queries' list must contain at least one non-empty query.")

    # Take the first (and typically only) query
    query = user_queries[0]

    # Create a simple prompt for title generation
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "You are an AI assistant that creates very short, high-level conversation titles. "
                "Generate a concise title (maximum 5-7 words) that captures the main topic or theme "
                "of the user's query. The title should be professional and descriptive.",
            ),
            ("user", "Based on this user query, generate a short conversation title: {query}"),
        ]
    )

    prompt_params = {"query": query}

    llm_response = get_response(prompt, prompt_params, "default")
    title = "".join(llm_response).strip()

    # Ensure the title isn't too long and remove any quotes
    title = title.replace('"', "").replace("'", "")
    if len(title) > 60:
        title = title[:57] + "..."

    return title
