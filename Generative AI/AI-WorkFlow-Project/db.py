# db.py (updated)
import psycopg2
from psycopg2.extras import Json

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mrm_db",
    "user": "mrm_user",
    "password": "mrm_pass"
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def create_tables():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS mrm_cases (
        case_id TEXT PRIMARY KEY,
        status TEXT,
        owner_email TEXT,
        workflow_state JSONB,
        final_decision JSONB,
        review_iteration INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    cur.close()
    conn.close()


def save_case(case_id, status, owner_email, workflow_state, final_decision, review_iteration):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO mrm_cases (
        case_id,
        status,
        owner_email,
        workflow_state,
        final_decision,
        review_iteration
    )
    VALUES (%s,%s,%s,%s,%s,%s)
    ON CONFLICT (case_id)
    DO UPDATE SET
        status = EXCLUDED.status,
        workflow_state = EXCLUDED.workflow_state,
        final_decision = EXCLUDED.final_decision,
        review_iteration = EXCLUDED.review_iteration,
        updated_at = CURRENT_TIMESTAMP
    """,
                (
                    case_id,
                    status,
                    owner_email,
                    Json(workflow_state),
                    Json(final_decision),
                    review_iteration
                ))

    conn.commit()
    cur.close()
    conn.close()


def load_case(case_id):
    """
    Load complete case data including workflow_state, status, and final_decision
    Returns None if case not found
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT workflow_state, status, final_decision, review_iteration, owner_email
    FROM mrm_cases
    WHERE case_id = %s
    """, (case_id,))

    row = cur.fetchone()

    cur.close()
    conn.close()

    if row:
        return {
            "workflow_state": row[0],  # This contains the full MRMState
            "status": row[1],
            "final_decision": row[2],
            "review_iteration": row[3],
            "owner_email": row[4]
        }

    return None