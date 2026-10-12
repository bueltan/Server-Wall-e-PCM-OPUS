"""Narrow, provider-specific recovery rules; never accept a generic rejection."""
import re

class ProviderOperationError(RuntimeError):
    def __init__(self, payload, error):
        self.payload, self.error = payload, error
        super().__init__(f"{payload.get('type')} rejected: {error.get('code')}: {error.get('message', '')}")

def already_absent(error):
    code=error.get('code', '')
    message=str(error.get('message', '')).lower()
    return code in ('item_not_found', 'conversation_item_not_found') or bool(re.search(
        r"\bitem\b.{0,100}\b(not found|does not exist|already deleted)\b", message))

def no_active_response(error):
    code=error.get('code', '')
    message=str(error.get('message', '')).lower()
    return code in ('response_cancel_not_active', 'response_not_active') or bool(re.search(
        r"\b(no active response|no response to cancel)\b|\bresponse\b.{0,100}\balready (cancelled|canceled|completed)\b", message))
