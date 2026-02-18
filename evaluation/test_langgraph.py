from agents.langgraph_supervisor import build_graph


def pretty_print_sources(docs):
    if not docs:
        print("No sources found.")
        return

    for i, d in enumerate(docs, start=1):
        hb = d.metadata.get("handbook_name", "unknown")
        page = d.metadata.get("page", "N/A")
        chunk = d.metadata.get("chunk_id", "N/A")
        print(f"[{i}] {hb} (page {page}, chunk {chunk})")


def main():
    app = build_graph()

    queries = [
        "Explain probation period policy",
        "What is the notice period and what happens if I don't serve it fully?",
        "How many sick leaves are allowed?",
        "Write an email to HR requesting work from home for 2 days due to illness",
        "What is the stock bonus policy?"
    ]

    for q in queries:
        print("\n" + "=" * 100)
        print(f"QUERY: {q}")
        print("=" * 100)

        initial_state = {
            "user_query": q,
            "retry_count": 0,
            "max_retries": 1
        }

        config = {"configurable": {"thread_id": "test-thread"}}

        try:
            result = app.invoke(initial_state, config=config)
        except Exception as e:
            print("\n❌ ERROR DURING GRAPH RUN")
            print(str(e))
            continue

        # --- core outputs ---
        answer = result.get("answer", "")
        verification = result.get("verification", {})
        reranked_docs = result.get("reranked_docs", [])

        # --- handbook info ---
        primary_handbook = result.get("primary_handbook", "unknown")
        dist = result.get("handbook_distribution", {})

        # --- logs ---
        logs = result.get("stream_log", [])

        print("\n--- FINAL ANSWER ---")
        print(answer if answer else "(empty answer)")

        print("\n--- VERIFICATION ---")
        print(verification if verification else "(no verification returned)")

        print("\n--- PRIMARY HANDBOOK ---")
        print(primary_handbook)

        print("\n--- HANDBOOK DISTRIBUTION ---")
        if dist:
            for k, v in dist.items():
                print(f"- {k}: {v}")
        else:
            print("(no distribution found)")

        print("\n--- SOURCES (RERANKED DOCS) ---")
        pretty_print_sources(reranked_docs)

        print("\n--- INTERNAL AGENT LOGS ---")
        if logs:
            for l in logs:
                print(f"- {l}")
        else:
            print("(no logs)")

    print("\n✅ Done testing.")


if __name__ == "__main__":
    main()