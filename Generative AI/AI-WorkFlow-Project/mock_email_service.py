from pathlib import Path
import json
SENT_DIR = Path("data/mailbox/sent_items")
RECEIVED_DIR = Path("data/mailbox/inbox")

def send_clarification_email(case_id, owner_email, questions):
    payload = {
        "case_id": case_id,
        "owner_email": owner_email,
        "questions": questions
    }
    file_path = SENT_DIR / f"{case_id}.json"
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Clarification email created for {case_id}")

# collecting the mails : assuming that the output is json values
def collect_email_replies():
    replies = []

    for file in RECEIVED_DIR.glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
        replies.append(data)
    return replies
