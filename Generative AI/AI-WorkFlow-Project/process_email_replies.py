# process_email_replies.py (updated)
from db import load_case, save_case
from workflow_engine import parse_clarification_email, resume_case
from mock_email_service import collect_email_replies


def process_email_replies():
    """Process email replies with proper state loading"""
    print("\nCollecting email replies...")
    replies = collect_email_replies()
    print(f"Found {len(replies)} email replies")

    for reply in replies:
        case_id = reply["case_id"]
        print(f"\nProcessing Email Reply: {case_id}")

        # Fetch the complete saved data for this case
        case_data = load_case(case_id)

        if not case_data:
            print(f"Warning: Case {case_id} not found in database, skipping")
            continue

        if case_data['status'] != 'WAITING_FOR_CLARIFICATION':
            print(f"Case {case_id} status is '{case_data['status']}', not waiting for clarification. Skipping.")
            continue

        # Extract the workflow state (full MRMState object)
        saved_state = case_data["workflow_state"]

        # Get questions from the latest final decision
        final_decision = case_data.get("final_decision", {})
        if isinstance(final_decision, dict):
            questions = final_decision.get("clarification_questions", [])
            blocking_rules = final_decision.get("blocking_rules", [])
        else:
            # Handle if final_decision is stored as JSON string or different format
            questions = []
            blocking_rules = []

        if not questions:
            print(f"Warning: No clarification questions found for case {case_id}")
            # Still try to process with empty questions
            questions = []
            blocking_rules = []

        print(f"Questions asked: {len(questions)}")
        print(f"Blocking rules: {len(blocking_rules)}")

        # Parse the clarification email
        parsed_clarification = parse_clarification_email(
            reply["email_body"],
            questions,
            blocking_rules
        )

        # Check if any answers were provided
        answered_count = len(parsed_clarification.get("answered_items", []))
        unresolved_count = len(parsed_clarification.get("unresolved_items", []))
        print(f"Answered {answered_count} of {len(questions)} questions")
        print(f"Unresolved questions: {unresolved_count}")

        # Resume the decision process for this case
        updated_result = resume_case(saved_state, parsed_clarification)
        decision = updated_result["final_decision"]

        # Determine new status with loop limit protection
        if decision["determination"] == "NEEDS_CLARIFICATION":
            # This should not happen if loop limit is working, but handle it
            status = "WAITING_FOR_CLARIFICATION"  # Stay in waiting state
            print(f"Case {case_id} still needs clarification (further rounds may be limited)")
        elif decision['determination'] == 'ESCALATE_TO_HUMAN':
            status = "ESCALATED"
            print(f"Case {case_id} escalated to human review")
        else:
            status = "COMPLETED"
            print(f"Case {case_id} completed with determination: {decision['determination']}")

        # Save updated case
        save_case(
            case_id,
            status,
            case_data.get("owner_email", "unknown@test.com"),
            updated_result,
            decision,
            updated_result.get("review_iteration", case_data.get("review_iteration", 0) + 1)
        )

        print(f"Case {case_id} saved with status: {status}")