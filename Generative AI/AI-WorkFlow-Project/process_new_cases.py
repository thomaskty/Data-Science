
from pathlib import Path
from workflow_engine import run_new_case
from db import create_tables, save_case
from mock_portal import fetch_new_cases
from mock_email_service import send_clarification_email

def process_new_cases():
    create_tables()

    guidelines = Path("guidelines/model_identification_guidelines.txt").read_text()
    cases = fetch_new_cases()

    for case in cases:
        print(f"\nProcessing Case: {case['case_id']}")

        result = run_new_case(
            case["case_id"],
            case["owner_email"],
            case["document"],
            guidelines
        )

        decision = result["final_decision"]
        print(case['case_id'],decision['determination'])

        if decision["determination"] == "NEEDS_CLARIFICATION":
            print(f'Clarification is required for this case :{case['case_id']}')
            status = "WAITING_FOR_CLARIFICATION"
            send_clarification_email(
                case["case_id"],
                case["owner_email"],
                decision["clarification_questions"]
            )
        elif decision['determination']=='ESCALATE_TO_HUMAN':
            status = 'ESCALATED'
        else: # MODEL/NON-MODEL - > determinism is completed and process is completed
            status = "COMPLETED"

        save_case(
            case["case_id"],
            status,
            case["owner_email"],
            result,
            decision,
            result["review_iteration"]
        )

        print(f"Saved Case: {case['case_id']}")
