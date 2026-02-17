from agents.supervisor import run_supervisor

query = "What is the notice period for resignation?"
result = run_supervisor(query)

print("\nIntent:", result["intent"])
print("\nRewritten:", result["rewritten_query"])
print("\nAnswer:\n", result["answer"])
print("\nVerification:\n", result["verification"])