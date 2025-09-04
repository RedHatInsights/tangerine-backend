import logging
import re
import urllib

import requests

import tangerine.config as cfg

log = logging.getLogger("tangerine.agents.webrca_agent")


class WebRCAAgent:
    def __init__(self):
        self.url = cfg.WEB_RCA_AGENT_URL

    def fetch(self, query: str):
        incidents = self._find_incidents(query)
        query_url = f"{self.url}/incidents?public_id={incidents}"
        token = self._get_token()
        try:
            log.info("AUDIT: WebRCAAgent making HTTP request to: %s", query_url)
            # Perform the GET request
            response = requests.get(
                query_url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            log.info("AUDIT: WebRCAAgent HTTP request successful")
        except Exception:
            log.error("AUDIT: WebRCAAgent HTTP request FAILED - returning error message")
            log.exception("Error fetching incidents from Web RCA")
            return "I tried getting info from Web RCA, but something went wrong."

        # response is an object with a list of incients in the items key
        ai_summaries = [incident.get("ai_summary", "") for incident in data.get("items", [])]
        return "\n".join(ai_summaries)

    def _find_incidents(self, query: str) -> str:
        # Matches patterns like ITN-2024-12345, optionally followed by punctuation
        matches = re.findall(r"\bITN-\d{4}-\d+\b", query, re.IGNORECASE)
        # Normalize and deduplicate
        unique_ids = sorted(set(match.upper() for match in matches))
        return ", ".join(unique_ids)

    def _get_token(self):
        token_url = f"{cfg.SSO_URL}/auth/realms/redhat-external/protocol/openid-connect/token"

        payload = {
            "grant_type": "client_credentials",
            "client_id": cfg.WEB_RCA_AGENT_CLIENT_ID,
            "client_secret": cfg.WEB_RCA_AGENT_CLIENT_SECRET,
            "scope": "openid api.ocm",
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }

        form_body = urllib.parse.urlencode(payload)

        try:
            response = requests.post(token_url, data=form_body, headers=headers, timeout=120)
            response.raise_for_status()
            return response.json().get("access_token", None)
        except requests.RequestException as exc:
            return {"error": str(exc)}
