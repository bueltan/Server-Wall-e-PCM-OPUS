"""Fixed MCP diagnostic codes. Never return exception text or argument values."""
from functools import wraps
from zoneinfo import ZoneInfoNotFoundError


VALIDATION_CODES = frozenset('''invalid_search_arguments invalid_message_id
invalid_contact_search_arguments invalid_contact_resource_name invalid_contact_arguments
invalid_contact_email invalid_contact_phone invalid_send_arguments invalid_recipient
invalid_calendar_or_event_id timestamp_requires_rfc3339_offset end_must_follow_start
invalid_calendar_list_arguments invalid_event_list_arguments invalid_event_content
all_day_requires_dates offset_does_not_match_time_zone invalid_event_id invalid_event_datetime'''.split())

SAFE_ERROR_CODES = VALIDATION_CODES | frozenset('''invalid_time_zone invalid_tool_arguments
mcp_tool_error tool_timeout tool_unavailable unsupported_tool tool_busy tool_cancelled
calendar_auth_required calendar_rate_limited calendar_api_disabled invalid_calendar_request
calendar_access_denied calendar_not_found event_id_conflict calendar_api_error
calendar_write_not_confirmed calendar_unavailable availability_unknown
contacts_write_auth_required contacts_auth_required people_api_disabled invalid_contact_data
contacts_access_denied contact_not_found contacts_rate_limited contacts_unavailable
contact_create_not_confirmed send_not_confirmed'''.split()) | frozenset(
    prefix + suffix for prefix in ('gmail', 'calendar') for suffix in
    ('_invalid_request', '_auth_required', '_access_denied', '_not_found', '_rate_limited',
     '_api_error', '_api_disabled', '_unavailable'))


def google_reason(exc):
    """Read Google's reason identifiers without returning any remote text."""
    import json
    try:
        error = json.loads(exc.content).get('error', {})
        reasons = {item.get('reason') for key in ('errors', 'details') for item in error.get(key, [])}
    except (AttributeError, ValueError, TypeError):
        return None
    if reasons & {'rateLimitExceeded', 'userRateLimitExceeded', 'quotaExceeded'}:
        return 'rate_limited'
    if reasons & {'accessNotConfigured', 'SERVICE_DISABLED'}:
        return 'api_disabled'
    return None


def http_status(exc):
    status = getattr(getattr(exc, 'resp', None), 'status', None)
    return status if type(status) is int and 100 <= status <= 599 else None


def exception_type(exc):
    # Custom exception class names are also untrusted strings.
    return next((kind.__name__ for kind in (TimeoutError, ConnectionError, OSError,
                 ValueError, TypeError, KeyError, RuntimeError) if isinstance(exc, kind)), 'Exception')


def failure_details(exc):
    """Metadata safe to add to existing domain-specific failure responses."""
    status = http_status(exc)
    return ({'error_kind': 'api', 'http_status': status} if status else
            {'error_kind': 'exception', 'exception_type': exception_type(exc)})


def diagnostic_tool(function, integration):
    """Catch domain exceptions before FastMCP formats them with private data.

    Preserve the original signature/docstring for MCP tool discovery. Schema
    validation is handled by the manager before requests reach FastMCP.
    """
    @wraps(function)
    def invoke(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ZoneInfoNotFoundError:
            return {'status': 'failed', 'error': 'invalid_time_zone',
                    'error_kind': 'validation', 'retry_safe': False}
        except Exception as exc:
            code = str(exc) if type(exc) is ValueError else ''
            if code in VALIDATION_CODES:
                kind = 'validation'
            elif isinstance(exc, RuntimeError) and str(exc).startswith(integration + '_auth_required:'):
                code, kind = integration + '_auth_required', 'auth'
            else:
                status = http_status(exc)
                reason = google_reason(exc)
                code = (integration + '_' + reason) if reason else {
                        400: integration + '_invalid_request', 401: integration + '_auth_required',
                        403: integration + '_access_denied', 404: integration + '_not_found',
                        429: integration + '_rate_limited'}.get(status, integration + '_unavailable')
                kind = 'api' if status else 'exception'
            return {'status': 'failed', 'error': code, 'error_kind': kind,
                    'retry_safe': False, 'exception_type': exception_type(exc),
                    **({'http_status': http_status(exc)} if http_status(exc) else {})}
    return invoke
