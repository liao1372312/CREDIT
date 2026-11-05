"""SDK v3 clarifies response-relative pointers and repairs invalid sources."""

import json

from .emp_binding import leaves, pointer
from .emp_binding_tools_v2 import SYSTEM as SYSTEM_V2, assemble, retrieve_compatible
from .emp_binding_tools_v2 import messages as messages_v2

SYSTEM = SYSTEM_V2 + '''
来源 pointer 相对于 producer 响应的根对象，必须逐字使用 observable_response_fields 的键。fields_to_bind 是目标列表，不是接口响应，禁止作为来源路径。不要引用消息结构的路径（例如 /fields_to_bind/0 或 /observable_response_fields/...）。目标字段的写法和来源字段的写法可以不同；请检查当前响应中的实际字段。'''


def messages(env, memories):
    chats = messages_v2(env, memories)
    chats[0]['content'] = SYSTEM
    # Re-expose every public source, without filtering by reference bindings.
    if env.history[1:] and (env.history[-1].get('response') or {}).get('sdk_errors'):
        feedback = json.loads(chats[-1]['content'])
        feedback['source_pointer_rule'] = '来源必须是 producer 响应叶字段的指针；不是目标列表或消息结构的路径。'
        feedback['allowed_producer_response_pointers'] = [pointer(path) for path, _ in leaves(env.responses['producer'])]
        chats[-1]['content'] = json.dumps(feedback, ensure_ascii=False)
    return chats
