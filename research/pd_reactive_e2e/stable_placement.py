"""Process-scoped placement header; preserve official payload and correlation IDs."""
import hashlib

HEADER = 'X-AgentX-Placement-Key'


def placement_key(request_info):
    conversation = request_info.conversation_id
    if not isinstance(conversation, str) or not conversation:
        raise ValueError('Stable placement requires original conversation_id')
    correlation = request_info.x_correlation_id
    if not isinstance(correlation, str) or not correlation:
        raise ValueError('AgentX conversation clone needs its stable correlation ID')
    return hashlib.sha256(('agentx-session-conversation-v2:' + correlation + ':' + conversation).encode()).hexdigest()


def install():
    from aiperf.transports.base_transports import BaseTransport
    if getattr(BaseTransport, '_agentx_stable_placement', False):
        return
    original = BaseTransport.build_headers

    def build_headers(self, request_info):
        headers = original(self, request_info)
        headers = {key: value for key, value in headers.items()
                   if key.lower() != HEADER.lower()}
        headers[HEADER] = placement_key(request_info)
        return headers

    BaseTransport.build_headers = build_headers
    BaseTransport._agentx_stable_placement = True
