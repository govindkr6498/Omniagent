import os
import logging
import requests
from datetime import datetime, timedelta
import pytz

class SalesforceAPI:
    def __init__(self):
        self.logger = logging.getLogger("salesforce_api")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.auth_url = "https://iqb4-dev-ed.develop.my.salesforce.com/services/oauth2/token"
        self.client_id = os.getenv("SF_CLIENT_ID")
        self.client_secret = os.getenv("SF_CLIENT_SECRET")
        self.access_token = None
        self.instance_url = None
        self._authenticate()

    def _authenticate(self):
        self.logger.info("Authenticating with Salesforce...")
        try:
            auth_data = {"grant_type": "client_credentials", "client_id": self.client_id, "client_secret": self.client_secret}
            response = requests.post(self.auth_url, data=auth_data)
            response.raise_for_status()
            data = response.json()
            self.access_token = data.get("access_token")
            self.instance_url = data.get("instance_url")
            self.logger.info("Salesforce authentication successful.")
        except Exception as e:
            self.logger.error(f"Salesforce authentication failed: {str(e)}")
            raise

    def create_lead(self, lead_info):
        self.logger.info(f"Creating lead with info: {lead_info}")
        try:
            if not self.access_token or not self.instance_url:
                self._authenticate()
            if any(value == "N/A" for value in lead_info.values()):
                self.logger.warning("Lead info contains 'N/A', aborting lead creation.")
                return False, None
            lead_url = f"{self.instance_url}/services/data/v60.0/sobjects/Lead/"
            headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
            sf_lead_payload = {
                "LastName": lead_info["Name"],
                "Company": lead_info["Company"],
                "Email": lead_info["Email"],
                "Phone": lead_info["Phone"]
            }
            response = requests.post(lead_url, headers=headers, json=sf_lead_payload)
            if response.status_code == 201:
                self.logger.info("Lead created successfully.")
                return True, response.json().get("id")
            elif response.status_code == 400 and "DUPLICATES_DETECTED" in response.text:
                error_data = response.json()
                match_records = (error_data[0].get("duplicateResult", {}).get("matchResults", [])[0].get("matchRecords", []))
                if match_records:
                    self.logger.info("Duplicate lead detected, returning existing lead ID.")
                    return True, match_records[0]["record"]["Id"]
            self.logger.error(f"Failed to create lead: {response.text}")
            return False, None
        except Exception as e:
            self.logger.error(f"Failed to create lead: {str(e)}")
            return False, None

    def create_meeting(self, lead_id, start_time_str):
        self.logger.info(f"Creating meeting for lead_id={lead_id} at {start_time_str}")
        try:
            if not self.access_token or not self.instance_url:
                self._authenticate()
            event_url = f"{self.instance_url}/services/data/v60.0/sobjects/Event/"
            headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
            start_dt = datetime.strptime(start_time_str, "%H:%M")
            ist = pytz.timezone('Asia/Kolkata')
            today_local = datetime.now(ist).date()
            start_local_dt = ist.localize(datetime.combine(today_local, start_dt.time()))
            start_utc_dt = start_local_dt.astimezone(pytz.utc) + timedelta(hours=5) + timedelta(minutes=30)
            end_utc_dt = start_utc_dt + timedelta(minutes=30)
            event_payload = {
                "Subject": "Call with Sales Advisor",
                "StartDateTime": start_utc_dt.isoformat(),
                "EndDateTime": end_utc_dt.isoformat(),
                "OwnerId": "0055j00000BYNIBAA5",
                "WhoId": lead_id,
                "Location": "Virtual Call",
                "Description": "Scheduled via Agentic Bot"
            }
            response = requests.post(event_url, headers=headers, json=event_payload)
            if response.status_code == 201:
                self.logger.info("Meeting created successfully.")
                return True
            else:
                self.logger.error(f"Failed to create meeting: {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Exception while creating meeting: {str(e)}")
            return False

    def show_availableMeeting(self):
        self.logger.info("Fetching available meeting slots...")
        start_times = set()
        try:
            if not self.access_token or not self.instance_url:
                self._authenticate()
            event_url = f"{self.instance_url}/services/data/v60.0/query?q=SELECT+StartDateTime,+EndDateTime+FROM+Event+WHERE+StartDateTime+=+TODAY"
            headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
            response = requests.get(event_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                records = data.get("records", [])
                fmt = "%H:%M"
                start_time = datetime.strptime("08:00", fmt)
                end_time = datetime.strptime("17:00", fmt)
                all_slots = set()
                current = start_time
                while current < end_time:
                    all_slots.add(current.strftime(fmt))
                    current += timedelta(minutes=30)
                for event in records:
                    start = event.get("StartDateTime")
                    if start:
                        try:
                            dt = datetime.strptime(start, "%Y-%m-%dT%H:%M:%S.%f%z")
                            time_only = dt.strftime("%H:%M")
                            start_times.add(time_only)
                        except Exception:
                            self.logger.warning(f"Could not parse event start time: {start}")
                available_slots = sorted(all_slots - start_times)
                self.logger.info(f"Available slots: {available_slots}")
                return available_slots
            self.logger.error(f"Failed to fetch meeting slots: {response.text}")
            return []
        except Exception as e:
            self.logger.error(f"Exception while showing meeting: {str(e)}")
            return []
