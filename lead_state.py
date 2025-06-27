from enum import Enum

class LeadState(Enum):
    NO_INTEREST = "no_interest"
    INTEREST_DETECTED = "interest_detected"
    COLLECTING_INFO = "collecting_info"
    INFO_COMPLETE = "info_complete"
    AWAITING_MEETING_CONFIRMATION = "awaiting_meeting_confirmation"
    WAITING_MEETING_SLOT_SELECTION = "waiting_meeting_slot_selection"
