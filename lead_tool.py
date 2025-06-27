import json
from typing import Dict, Any, Optional, List
from lead_state import LeadState
from salesforce_api import SalesforceAPI
import logging

class LeadTool:
    def __init__(self):
        self.logger = logging.getLogger("lead_tool")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.salesforce = SalesforceAPI()
        self.partial_lead_info = {}
        self.state = LeadState.NO_INTEREST
        self.current_lead_id = None

    def extract_lead_info(self, message: str, llm) -> Optional[Dict[str, str]]:
        self.logger.info(f"Extracting lead info from message: {message}")
        prompt = (
            "Extract contact information from the following message. "
            "Return ONLY a minified JSON object (no markdown, no code block, no comments) with these exact fields (always include all keys, even if missing): "
            "Name, Company, Email, Phone. If a field is not found, return its value as null. "
            "The value for Company must always be 'Iquee Tech'. "
            f"Message: {message}\n"
            "Return ONLY the JSON object, nothing else."
        )
        self.logger.info(f"Prompt sent to LLM: {prompt}")
        response = llm.invoke(prompt)
        self.logger.info(f"LLM raw response: {response.content}")
        try:
            # Remove code block markers if present
            content = response.content.strip()
            if content.startswith('```'):
                content = content.strip('`')
                if content.startswith('json'):
                    content = content[4:].strip()
            lead_data = json.loads(content)
            self.logger.info(f"Parsed lead_data: {lead_data}")
            if not lead_data:
                self.logger.info("No lead data extracted from message.")
                return None
            normalized = dict(self.partial_lead_info)
            for k in ['Name', 'Email', 'Phone', 'Company']:
                v = lead_data.get(k)
                self.logger.info(f"Field {k}: {v}")
                if v is not None and v != "N/A" and v != "":
                    normalized[k] = v.strip() if isinstance(v, str) else v
            normalized['Company'] = 'Iqueestbee Technology'
            self.logger.info(f"Extracted/merged lead info: {normalized}")
            return normalized
        except Exception as e:
            self.logger.error(f"Failed to extract lead info: {e}")
        return None

    def update_state(self, message: str, llm) -> None:
        self.logger.info(f"Updating lead state. Current state: {self.state}, message: {message}")
        self.logger.info(f"Current partial_lead_info before update: {self.partial_lead_info}")
        interest_indicators = ["schedule", "meeting", "interested", "pricing", "cost", "interest", "sign up", "enroll", "register", "buy", "purchase", "want", "desire"]
        if self.state == LeadState.NO_INTEREST:
            if any(ind in message.lower() for ind in interest_indicators):
                self.logger.info("Interest detected in message.")
                self.state = LeadState.INTEREST_DETECTED
        if self.state in [LeadState.INTEREST_DETECTED, LeadState.COLLECTING_INFO]:
            lead_info = self.extract_lead_info(message, llm)
            self.logger.info(f"Lead info returned from extract_lead_info: {lead_info}")
            if lead_info:
                self.partial_lead_info.update(lead_info)
                self.logger.info(f"Updated partial_lead_info: {self.partial_lead_info}")
                self.state = LeadState.COLLECTING_INFO
                if all(
                    k in self.partial_lead_info and self.partial_lead_info[k] not in [None, "N/A", ""]
                    for k in ['Name', 'Email', 'Phone']
                ):
                    self.state = LeadState.INFO_COMPLETE
                    self.logger.info("Lead info complete.")
        self.logger.info(f"Current partial_lead_info after update: {self.partial_lead_info}")
        self.logger.info(f"Current state after update: {self.state}")

    def get_missing_fields(self) -> List[str]:
        missing = [f for f in ['Name', 'Email', 'Phone'] if f not in self.partial_lead_info or self.partial_lead_info[f] == "N/A"]
        self.logger.info(f"Missing lead fields: {missing}")
        return missing

    def create_lead(self) -> Optional[str]:
        self.logger.info(f"Creating lead in Salesforce with info: {self.partial_lead_info}")
        lead_created, lead_id = self.salesforce.create_lead(self.partial_lead_info)
        if lead_created:
            self.current_lead_id = lead_id
            self.state = LeadState.AWAITING_MEETING_CONFIRMATION
            self.logger.info(f"Lead created with ID: {lead_id}")
            return lead_id
        self.logger.error("Failed to create lead in Salesforce.")
        return None
