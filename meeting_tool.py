from typing import List
from salesforce_api import SalesforceAPI
import logging

class MeetingTool:
    def __init__(self, salesforce_api: SalesforceAPI):
        self.logger = logging.getLogger("meeting_tool")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.salesforce = salesforce_api
        self.available_slots = []

    def get_slots(self) -> List[str]:
        self.logger.info("Fetching available meeting slots from Salesforce...")
        self.available_slots = self.salesforce.show_availableMeeting() or []
        self.logger.info(f"Available slots: {self.available_slots}")
        return self.available_slots

    def schedule(self, lead_id: str, slot: str) -> bool:
        self.logger.info(f"Scheduling meeting for lead_id={lead_id} at slot={slot}")
        result = self.salesforce.create_meeting(lead_id, slot)
        if result:
            self.logger.info("Meeting scheduled successfully.")
        else:
            self.logger.error("Failed to schedule meeting.")
        return result

    def format_slots(self, slots: List[str], columns: int = 3) -> str:
        self.logger.info(f"Formatting slots for display: {slots}")
        if not slots:
            return "No available time slots."
        max_length = max(len(slot) for slot in slots)
        col_width = max_length + 5
        rows = []
        for i in range(0, len(slots), columns):
            row = []
            for j in range(columns):
                idx = i + j
                if idx < len(slots):
                    row.append(f"{slots[idx]:>{col_width}}")
            rows.append("".join(row))
        return (
             "Available meeting times:\n\n" +
            "\n".join(rows) +  # only 1 newline between rows
            "\n\nPlease pick one."
        )

