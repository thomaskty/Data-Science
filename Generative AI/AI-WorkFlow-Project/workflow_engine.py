
from typing import TypedDict, List, Optional, Literal
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pathlib import Path
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
MAX_CLARIFICATION_ROUNDS = 2  # Maximum number of clarification cycles before forcing escalation
RULES_CACHE_FILE = Path("guidelines/cached_rules.json")

class Rule(BaseModel):
    rule_id: str
    description: str
    category: str
    severity: Literal['VERY_LOW','LOW','MEDIUM','HIGH','VERY_HIGH']
    rule_type: str
    target_sections: List[str]

class RuleSet(BaseModel):
    rules: List[Rule]

class DocumentSection(BaseModel):
    section_name: str
    content: str

class SectionCollection(BaseModel):
    sections: List[DocumentSection]

class RuleAssessment(BaseModel):
    rule_id: str
    evidence: str
    reasoning: str
    status: Literal[
        'SATISFIED',
        'PARTIALLY_SATISFIED',
        'NOT_SATISFIED',
        'NOT_APPLICABLE',
        'INSUFFICIENT_EVIDENCE',
        'CONTRADICTORY_EVIDENCE',
        'CLARIFICATION_REQUIRED'
    ]
    confidence: float = Field(ge=0, le=1)

class FinalDecision(BaseModel):
    determination: Literal[
        "MODEL",
        "NON_MODEL",
        "NEEDS_CLARIFICATION",
        "ESCALATE_TO_HUMAN"
    ]
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    critical_rules: List[str]
    blocking_rules: List[str]
    clarification_questions: List[str]

class AnsweredItem(BaseModel):
    question: str
    extracted_answer: str
    sufficiently_answered: bool
    related_rules: List[str]

class ClarificationResponse(BaseModel):
    answered_items: List[AnsweredItem]
    unresolved_items: List[str]
    additional_information: str
    confidence: float = Field(ge=0, le=1)


class MRMState(TypedDict):
    case_id: str
    owner_email: str
    document: str
    guidelines: str
    sections: List[DocumentSection]
    rules: List[Rule]
    rule_results: List[RuleAssessment]
    clarification_history: List[ClarificationResponse]
    prior_decisions: List[FinalDecision]
    affected_rules: List[str]
    final_decision: Optional[FinalDecision]
    review_iteration: int
    clarification_count: int

# STRUCTURED OUTPUTS
section_llm = llm.with_structured_output(SectionCollection)
rule_llm = llm.with_structured_output(RuleSet)

assessment_llm = llm.with_structured_output(RuleAssessment)
decision_llm = llm.with_structured_output(FinalDecision)
clarification_llm = llm.with_structured_output(ClarificationResponse)


# SECTION EXTRACTION
def extract_sections(state):
    prompt = f"""
    You are an MRM document parser.
    Split the MIT document into meaningful governance sections.
    Possible sections:
    - purpose
    - methodology
    - assumptions
    - limitations
    - outputs
    - business usage
    - uncertainty
    - ai usage
    - vendor model information
    - estimation methodology
    
    DOCUMENT:
    {state["document"]}
    """
    result = section_llm.invoke(prompt)
    state["sections"] = result.model_dump()["sections"]
    return state

# RULE GENERATION
def generate_rules(state):

    if RULES_CACHE_FILE.exists():
        print("✓ Using cached rules (skipping generation)")
        with open(RULES_CACHE_FILE, 'r') as f:
            cached = json.load(f)
            state["rules"] = cached["rules"]
            return state

    prompt = f"""
    You are a senior MRM governance expert.
    Using the MRM guidelines below, generate governance rules for model identification.
    MRM GUIDELINES:
    {state["guidelines"]}
    
    Generate rules covering:
    - predictive/statistical logic
    - AI/ML usage
    - computational estimation
    - assumptions
    - uncertainty handling
    - business impact
    - personalization systems
    - vendor models
    
    Return:
    - rule_id
    - description
    - category
    - severity
    - rule_type
    - target_sections
    """
    result = rule_llm.invoke(prompt)
    rules_data = result.model_dump()['rules']

    # Save to cache file
    with open(RULES_CACHE_FILE, 'w') as f:
        json.dump(rules_data, f, indent=2)
    print(f"✓ Rules cached to {RULES_CACHE_FILE}")

    state["rules"] = rules_data

    return state

# HELPERS
def get_relevant_sections(sections, target_sections):
    relevant = []
    for sec in sections:
        if sec["section_name"].lower() in [s.lower() for s in target_sections]:
            relevant.append(sec)
    return relevant


# RULE ASSESSMENT
def assess_single_rule(rule, sections, clarification_history):
    relevant_sections = get_relevant_sections(sections,rule["target_sections"])
    prompt = f"""
    You are an MRM evaluator.
    
    RULE:
    {rule}
    
    RELEVANT SECTIONS:
    {relevant_sections}
    
    CLARIFICATION HISTORY:
    {clarification_history}
    
    Evaluate whether the rule is satisfied.
    Return:
    - status
    - confidence
    - evidence
    - reasoning
    """
    result = assessment_llm.invoke(prompt)
    return result.model_dump()

# RULE EVALUATION
def evaluate_rules(state):
    if state["review_iteration"] == 1:
        updated_results = []
        for rule in state["rules"]:
            result = assess_single_rule(
                rule,
                state["sections"],
                state["clarification_history"]
            )
            updated_results.append(result)
    else:
        existing_results = {
            r["rule_id"]: r for r in state["rule_results"]
        }
        for rule in state["rules"]:
            if rule["rule_id"] in state["affected_rules"]:
                result = assess_single_rule(
                    rule,
                    state["sections"],
                    state["clarification_history"]
                )
                existing_results[rule["rule_id"]] = result

        updated_results = list(existing_results.values())
    state["rule_results"] = updated_results
    return state

# FINAL DECISION
def synthesize_decision(state):
    prompt = f"""
    You are a senior MRM reviewer.
    
    Determine whether the submission is:
    
    - MODEL
    - NON_MODEL
    - NEEDS_CLARIFICATION
    - ESCALATE_TO_HUMAN
    
    Important:
    - clarification is required ONLY if determination cannot be made confidently
    - decisive rules may override others
    - predictive/statistical estimation matters
    - uncertainty handling matters
    - business usage matters
    
    RULE EVALUATIONS:
    {state["rule_results"]}
    
    CLARIFICATION HISTORY:
    {state["clarification_history"]}
    
    PRIOR DECISIONS:
    {state["prior_decisions"]}
    
    Return:
    - determination (MODEL,NON_MODEL,NEEDS_CLARIFICATION,NEEDS_REVIEW)
    - confidence (0-1) 
    - reasoning (description of the reasining) 
    - critical_rules (list of rule_id values only)
    - blocking_rules (list of rule_id values only)
    - clarification_questions (list of clarification questions) 
    """
    result = decision_llm.invoke(prompt)
    decision = result.model_dump()
    state["final_decision"] = decision
    state["prior_decisions"].append(decision)
    return state

# ROUTING
def route_after_decision(state):
    """
    Route after decision with loop limit protection
    """
    determination = state["final_decision"]["determination"]

    if determination == "NEEDS_CLARIFICATION":
        # Check if we've exceeded maximum clarification rounds
        clarification_count = state.get("clarification_count", 0)

        if clarification_count >= MAX_CLARIFICATION_ROUNDS:
            print(
                f"Case {state['case_id']}: Max clarification rounds ({MAX_CLARIFICATION_ROUNDS}) reached. Escalating to human.")
            # Override determination to escalate
            state["final_decision"]["determination"] = "ESCALATE_TO_HUMAN"
            state["final_decision"]["reasoning"] = (
                f"Maximum clarification rounds ({MAX_CLARIFICATION_ROUNDS}) reached. "
                f"Unable to reach a final determination after multiple clarification attempts."
            )
            return "end"

        # Increment counter and continue to clarification
        state["clarification_count"] = clarification_count + 1
        print(
            f"Case {state['case_id']}: Clarification round {state['clarification_count']} of {MAX_CLARIFICATION_ROUNDS}")
        return "clarification"

    return "end"

# CLARIFICATION PLACEHOLDER
def clarification_pause(state):
    print(f"Case {state['case_id']} waiting for clarification.")
    return state

# BUILD GRAPH
graph = StateGraph(MRMState)
graph.add_node("extract_sections", extract_sections)
graph.add_node("generate_rules", generate_rules)
graph.add_node("evaluate_rules", evaluate_rules)
graph.add_node("decision", synthesize_decision)
graph.add_node("clarification_pause", clarification_pause)

graph.set_entry_point("extract_sections")
graph.add_edge("extract_sections", "generate_rules")
graph.add_edge("generate_rules", "evaluate_rules")
graph.add_edge("evaluate_rules", "decision")
graph.add_conditional_edges(
    "decision",
    route_after_decision,
    {"clarification": "clarification_pause","end": END}
)
app = graph.compile()


def create_initial_state(case_id, owner_email, document, guidelines):
    """
    Create initial state with loop counter
    """
    return {
        "case_id": case_id,
        "owner_email": owner_email,
        "document": document,
        "guidelines": guidelines,
        "sections": [],
        "rules": [],
        "rule_results": [],
        "clarification_history": [],
        "prior_decisions": [],
        "affected_rules": [],
        "final_decision": None,
        "review_iteration": 1,
        "clarification_count": 0  # Initialize counter
    }

def run_new_case(case_id, owner_email, document, guidelines):
    state = create_initial_state(case_id,owner_email,document,guidelines)
    result = app.invoke(state)
    return result


def parse_clarification_email(email_body, questions, blocking_rules):
    """
    Enhanced parsing that explicitly maps answers to specific questions
    and infers which rules are affected by each answer
    """
    import json

    # Create a numbered list of questions for easier reference
    questions_list = []
    for i, q in enumerate(questions, 1):
        questions_list.append(f"{i}. {q}")

    prompt = f"""
    You are processing an MRM clarification email.

    ORIGINAL QUESTIONS ASKED (numbered for reference):
    {chr(10).join(questions_list)}

    BLOCKING RULES (these rules were preventing a decision):
    {json.dumps(blocking_rules, indent=2)}

    EMAIL BODY FROM PROCESS OWNER:
    {email_body}

    IMPORTANT: For EACH question in the ORIGINAL QUESTIONS ASKED list, determine:
    1. Was this question answered in the email?
    2. If answered, what is the extracted answer?
    3. Which rules (from blocking_rules) does this answer help satisfy?

    Return:
    - answered_items: List of items for questions that were answered
      Each item must have:
      - question: The EXACT original question text
      - extracted_answer: The relevant answer from the email
      - sufficiently_answered: true if answer provides clear information, false if vague/partial
      - related_rules: List of rule_id values from blocking_rules that this answer affects

    - unresolved_items: List of original question texts that were NOT answered

    - additional_information: Any extra information provided that wasn't directly answering a question

    - confidence: Overall confidence in the parsing (0-1)

    IMPORTANT: Be precise. If an answer partially addresses a question, mark sufficiently_answered as false
    but still include it with the partial answer.
    """
    result = clarification_llm.invoke(prompt)
    return result.model_dump()


def resume_case(saved_state, parsed_clarification):
    """
    Resume workflow with enhanced tracking of which rules are affected
    """
    # Add clarification to history
    saved_state["clarification_history"].append(parsed_clarification)

    # Collect affected rules from answered items
    related_rules = []
    for item in parsed_clarification["answered_items"]:
        if item.get("related_rules"):
            related_rules.extend(item["related_rules"])
        # Also consider questions that were sufficiently answered as potentially affecting more rules
        if item.get("sufficiently_answered"):
            # Mark for comprehensive re-evaluation
            pass

    # Also check unresolved items - these might require escalation
    if parsed_clarification.get("unresolved_items") and len(parsed_clarification["unresolved_items"]) > 0:
        print(f"Warning: {len(parsed_clarification['unresolved_items'])} questions remain unanswered")

    saved_state["affected_rules"] = list(set(related_rules))

    # If no rules were explicitly identified as affected, mark all blocking rules for re-evaluation
    if not saved_state["affected_rules"] and saved_state.get("final_decision", {}).get("blocking_rules"):
        saved_state["affected_rules"] = saved_state["final_decision"]["blocking_rules"]
        print(f"No explicit rule mapping found, re-evaluating all {len(saved_state['affected_rules'])} blocking rules")

    saved_state["review_iteration"] += 1

    # Re-invoke the workflow with updated state
    result = app.invoke(saved_state)
    return result
