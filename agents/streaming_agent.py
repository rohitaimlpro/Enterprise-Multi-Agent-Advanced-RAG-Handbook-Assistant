from agents.state import RAGState


def log_step(state: RAGState, message: str) -> RAGState:
    if "stream_log" not in state:
        state["stream_log"] = []
    state["stream_log"].append(message)
    return state


def streaming_node(message: str):
    def _node(state: RAGState) -> RAGState:
        return log_step(state, message)
    return _node