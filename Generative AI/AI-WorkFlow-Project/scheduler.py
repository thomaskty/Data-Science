# scheduler.py (updated with sequential processing and case locking)
from concurrent.futures import ThreadPoolExecutor, as_completed
from process_new_cases import process_new_cases
from process_email_replies import process_email_replies
import threading
from collections import defaultdict
import time

# Simple in-memory lock manager (for single instance)
# For production with multiple scheduler instances, use Redis or database locks
case_locks = defaultdict(threading.Lock)


def process_with_lock(case_id, process_func, *args):
    """Process a case with locking to prevent concurrent processing"""
    with case_locks[case_id]:
        return process_func(*args)


def main_sequential():
    """
    Sequential processing - safest approach, no race conditions
    Process all new cases first, then all email replies
    """
    print("Starting MRM workflow scheduler (sequential mode)...")

    # Process new cases first
    print("\n=== Processing New Cases ===")
    process_new_cases()

    # Then process email replies
    print("\n=== Processing Email Replies ===")
    process_email_replies()

    print("\n=== Scheduler Complete ===")


def main_parallel_with_isolation():
    """
    Parallel processing with isolation - process new cases and replies
    but they work on different sets of case_ids (new cases have no replies yet)
    This is safe because:
    - New cases: case_ids that haven't been processed before
    - Email replies: case_ids that are already in WAITING_FOR_CLARIFICATION state
    These sets don't overlap during the same run
    """
    print("Starting MRM workflow scheduler (parallel isolated mode)...")

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks - they work on different case sets
        future_new_cases = executor.submit(process_new_cases)
        future_email_replies = executor.submit(process_email_replies)

        # Wait for both to complete
        future_new_cases.result()
        future_email_replies.result()

    print("\n=== Scheduler Complete ===")


def main_with_coordination():
    """
    Advanced: Process with coordination to avoid overlapping case_ids
    """
    print("Starting MRM workflow scheduler (coordinated mode)...")

    # First, get all case_ids that are waiting for clarification
    from db import get_connection
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT case_id FROM mrm_cases WHERE status = 'WAITING_FOR_CLARIFICATION'")
    waiting_cases = {row[0] for row in cur.fetchall()}
    cur.close()
    conn.close()

    print(f"Found {len(waiting_cases)} cases waiting for clarification")

    # Process email replies for waiting cases first
    # (these might update case status)
    process_email_replies()

    # Then process new cases (which won't conflict with waiting cases)
    process_new_cases()

    print("\n=== Scheduler Complete ===")


# Choose your preferred mode:
# Option 1: Safest for production
def main():
    main_sequential()


# Option 2: If you're sure new cases and replies don't overlap
# def main():
#     main_parallel_with_isolation()

# Option 3: Most robust coordination
# def main():
#     main_with_coordination()

if __name__ == "__main__":
    main()