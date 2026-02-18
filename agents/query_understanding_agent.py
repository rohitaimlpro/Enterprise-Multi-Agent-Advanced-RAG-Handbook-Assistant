from typing import Dict, Any
from sentence_transformers import SentenceTransformer, util

# Local CPU model
_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

INTENTS = {
    "leave_policy": "questions about leave, holidays, sick leave, casual leave, earned leave",
    "benefits": "questions about employee benefits, insurance, allowances, perks",
    "payroll": "questions about salary, payroll, payslip, deductions, PF, taxes",
    "resignation": "questions about resignation process, exit, handover, final settlement",
    "notice_period": "questions about notice period, serving notice, buyout",
    "probation": "questions about probation period, confirmation, performance review",
    "wfh_policy": "questions about work from home, remote work, hybrid policy",
    "code_of_conduct": "questions about employee behavior, discipline, ethics, harassment",
    "termination": "questions about termination, dismissal, misconduct, termination rules",
    "grievance": "questions about grievance process, complaints, reporting issues",
    "travel_policy": "questions about travel reimbursement, travel policy, expenses, claims",
    "general_policy": "general handbook questions"
}

ACTION_HINTS = [
    "write email", "draft email", "generate email",
    "create checklist", "make checklist",
    "summarize", "summary",
    "draft", "prepare", "template"
]


def _classify_intent(query: str) -> str:
    labels = list(INTENTS.keys())
    label_texts = [INTENTS[k] for k in labels]

    q_emb = _embedder.encode(query, convert_to_tensor=True)
    l_emb = _embedder.encode(label_texts, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, l_emb)[0].cpu().tolist()
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return labels[best_idx]


def _detect_multihop(query: str) -> str:
    q = query.lower()
    multi_hop_triggers = [
        "and", "also", "along with", "plus",
        "documents required", "eligibility", "process",
        "steps", "how to", "approval", "exception"
    ]
    if any(t in q for t in multi_hop_triggers):
        return "multi_hop"
    return "single_hop"


def _needs_action(query: str) -> bool:
    q = query.lower()
    return any(h in q for h in ACTION_HINTS)


def query_understanding_agent(user_query: str) -> Dict[str, Any]:
    intent = _classify_intent(user_query)
    retrieval_strategy = _detect_multihop(user_query)
    needs_action = _needs_action(user_query)

    # Entities can be added later (NER). Keep simple for now.
    entities = {}

    return {
        "intent": intent,
        "entities": entities,
        "retrieval_strategy": retrieval_strategy,
        "needs_action": needs_action
    }