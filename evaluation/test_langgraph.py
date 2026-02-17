from agents.langgraph_supervisor import build_graph


def main():
    app = build_graph()

    query = "What is the notice period for resignation?"
    initial_state = {
        "user_query": query,
        "retry_count": 0,
        "max_retries": 1
    }

    # IMPORTANT: required for checkpointing
    config = {
        "configurable": {
            "thread_id": "demo_session_1"
        }
    }

    result = app.invoke(initial_state, config=config)

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result["answer"])

    print("\nVerification:", result["verification"])

    if result.get("action_output"):
        print("\nAction Output:\n", result["action_output"])


if __name__ == "__main__":
    main()