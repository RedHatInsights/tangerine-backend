import logging

import requests
from langchain_core.prompts import ChatPromptTemplate

import tangerine.config as cfg
import tangerine.llm as llm
from tangerine.nltk import get_words

log = logging.getLogger("tangerine.models.interactions")


class JiraAgent:
    def __init__(self):
        self.url = cfg.JIRA_AGENT_URL

    def fetch(self, query: str):
        user_list = self._find_usernames(query)
        user_count = len(user_list)
        users = ",".join(user_list)
        query_url = f"{self.url}/?users={users}"
        # Perform the GET request
        try:
            response = requests.get(query_url, timeout=120)
            response.raise_for_status()
            summaries = response.json()
        except Exception:
            log.exception("Error fetching info from Jira")
            return "I tried getting info from Jira, but something went wrong."

        # data is a list of strings. we need to convert it to a string
        summaries = "\n".join(summaries)
        clean_summaries = summaries.replace("<br>", "\n")
        return_value = clean_summaries
        if user_count > 1:
            return_value = self._higher_order_summary(summaries)
        return return_value

    def _higher_order_summary(self, summaries: str) -> str:
        # This function takes a list of summaries and returns a higher order summary
        prompt = ChatPromptTemplate(
            [
                ("system", cfg.JIRA_SUMMARIZER_SYSTEM_PROMPT),
                ("user", cfg.JIRA_SUMMARIZER_USER_PROMPT),
            ]
        )
        prompt_params = {"query": summaries}

        llm_response = llm.get_response(prompt, prompt_params)
        return "".join(llm_response)

    def _find_usernames(self, query: str):
        # The query is a string of human entered prose text
        # We need to find all the usernames in the text
        # I think the way to do this is
        tokens = [token.lower().strip() for token in query.split(" ")]
        usernames = [
            token
            for token in tokens
            if token
            and token not in get_words()
            and token not in ["jira", "issue", "issues", "project", "projects"]
        ]
        # strip every username of any special characters
        usernames = [username.strip("!@#$%^&*()[]{};:'\"\\|,.<>?/`~") for username in usernames]
        # remove duplicates
        usernames = list(set(usernames))
        return usernames
