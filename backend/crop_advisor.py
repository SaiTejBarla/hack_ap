import sqlite3
import logging

from config import DB_PATH

def normalize_disease_label(label):
    """
    Normalize model label to match database entries.
    Converts to lowercase, replaces " with " with "_",
    replaces spaces with underscores, and capitalizes
    the first letter of each word.
    """
    label = label.lower().strip()
    if "healthy" in label:
        return "Healthy"

    # Remove extra words and standardize label
    label = label.replace(" with ", "_").replace(" ", "_").replace("__", "_")
    return label.capitalize()


def get_advice(disease_label):
    """
    Retrieves advice from database for a given disease label.
    Returns a default message if no advice is found.
    """
    try:
        disease_label = normalize_disease_label(disease_label)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT advice FROM advisories WHERE disease_label=?", (disease_label,)
        )
        result = cursor.fetchone()

        conn.close()

        if result and result[0]:
            return result[0]
        else:
            logging.warning(f"No advice found for disease label: {disease_label}")
            return "Advice not available yet. Please consult an expert."

    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return "Error retrieving advice."
