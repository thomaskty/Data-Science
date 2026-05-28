
# MRM AI POC
This Proof of Concept (POC) demonstrates an AI-powered Model Risk Management (MRM) workflow that automatically identifies 
whether a quantitative method qualifies as a "model" based on regulatory guidelines (OCC 2011-12/SR 11-7).
## Start PostgreSQL

docker run --name mrm-postgres \
  -e POSTGRES_USER=mrm_user \
  -e POSTGRES_PASSWORD=mrm_pass \
  -e POSTGRES_DB=mrm_db \
  -p 5432:5432 \
  -d postgres:16

# Pull PostgreSQL image
docker pull postgres:14

# Run PostgreSQL container
docker run -d \
  --name mrm_postgres \
  -e POSTGRES_USER=mrm_user \
  -e POSTGRES_PASSWORD=mrm_pass \
  -e POSTGRES_DB=mrm_db \
  -p 5432:5432 \
  postgres:14

# Verify container is running
docker ps | grep mrm_postgres

# Connect to PostgreSQL inside container
docker exec -it mrm_post gres psql -U mrm_user -d mrm_db


## AI Techniques & LLM Stack

### LLM & Framework Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **LLM Model** | OpenAI GPT-4o-mini | gpt-4o-mini-2024-07-18 | Core reasoning, rule assessment, decision synthesis |
| **Framework** | LangGraph | latest | Stateful multi-step workflow with conditional routing |
| **Output Parsing** | LangChain + Pydantic | latest | Structured, type-safe LLM responses |
| **API Client** | OpenAI Python SDK | >=1.0.0 | LLM API communication |

### AI Techniques Used

| Technique | Implementation | Benefit |
|-----------|---------------|---------|
| **Structured Prompt Engineering** | Custom prompts with explicit output schemas (SectionCollection, RuleSet, RuleAssessment, FinalDecision) | Ensures LLM returns predictable, validated JSON |
| **State Machine Workflow** | LangGraph StateGraph with conditional edges | Manages complex decision flow with clear branching logic |
| **Multi-step Chain-of-Thought** | Document → Sections → Rules → Assessment → Decision | Breaks complex classification into manageable steps |
| **Clarification Loop with Guardrails** | Max 2 rounds, counter in state, automatic escalation | Prevents infinite loops, ensures bounded execution |
| **Context Window Optimization** | Only relevant sections passed per rule assessment | Reduces token usage, improves accuracy |
| **Few-shot Learning** | Guidelines contain examples of GOOD/BAD descriptions | Improves classification accuracy without fine-tuning |
| **Selective Re-evaluation** | Only affected rules re-assessed after clarification | Reduces API calls by 60-70% in clarification rounds |
| **File-based Caching** | Generated rules cached to `cached_rules.json` | Eliminates redundant LLM calls (80% cost reduction) |
| **Structured Output Validation** | Pydantic models with field validation (confidence 0-1, literals) | Prevents hallucinated values, ensures data quality |


## Points 
* Designed and developed an AI-driven Model Risk Management (MRM) workflow automation solution to streamline business procedure review and model identification processes, reducing manual dependency in governance assessments.

* Automated the end-to-end evaluation lifecycle for business process submissions, where process owners upload procedural documents through a portal and the system determines whether the process qualifies as a “Model” or “Non-Model” based on enterprise MRM guidelines.

* Built a multi-stage orchestration engine using LangGraph and LangChain with LLM-powered reasoning workflows to manage document parsing, rule evaluation, clarification handling, and human escalation paths.

* Engineered a structured document intelligence pipeline using OpenAI LLMs to extract governance-relevant sections (methodology, assumptions, uncertainty handling, AI usage, vendor model information, estimation logic, etc.) from unstructured business documents into standardized schema-based outputs. 

* Developed a dynamic rule-generation framework that converts MRM guideline documents into reusable governance rules with severity, category, target sections, and rule types, enabling scalable policy-driven evaluations instead of hardcoded logic. 

* Implemented rule-level compliance assessment logic to evaluate each extracted document section against generated governance rules and classify outcomes into statuses such as SATISFIED, NOT_SATISFIED, NOT_APPLICABLE, INSUFFICIENT_EVIDENCE, and CLARIFICATION_REQUIRED with confidence scoring. 

* Designed an iterative decision-making workflow that synthesizes rule evaluation results to determine final classifications: MODEL, NON_MODEL, NEEDS_CLARIFICATION, or ESCALATE_TO_HUMAN, improving governance consistency and reducing subjective interpretation across reviewers. 

* Built a human-in-the-loop clarification mechanism where the AI system automatically generates clarification questions for process owners when evidence is insufficient, sends email requests, resumes workflow execution upon receiving responses, and re-evaluates only impacted rules using incremental state updates.  

* Implemented persistent workflow state management using PostgreSQL JSONB storage to maintain case history, clarification iterations, prior decisions, workflow states, and auditability for governance tracking. 

* Developed scheduler and orchestration services supporting sequential and parallel-safe processing with case-level locking mechanisms to prevent race conditions and ensure reliable concurrent execution of new submissions and clarification responses. 

* Optimized processing efficiency through rule caching and selective re-evaluation of only affected rules during clarification cycles, significantly reducing repeated LLM inference costs and workflow latency.  

* Reduced manual review effort from approximately 30 minutes per document to near real-time AI-assisted assessment for ~50+ monthly submissions, resulting in an estimated savings of 25+ reviewer hours per month while improving review consistency, traceability, and scalability.

* Improved operational governance by introducing automated escalation controls, clarification loop limits, confidence-based determinations, and audit-ready decision histories for regulatory and compliance review scenarios.


## Resume Points 

* Developed an AI-powered MRM workflow automation solution using LangGraph, LangChain, and LLMs to classify business processes as “Model” or “Non-Model” based on governance guidelines.

* Built an end-to-end document intelligence pipeline for structured extraction, dynamic rule generation, rule-based compliance evaluation, clarification handling, and workflow state management using PostgreSQL.

* Automated a manual review process (~30 mins/document for 50+ monthly cases), saving 25+ hours/month while improving governance consistency, auditability, and escalation handling.


