from pydantic import BaseModel, Field, EmailStr

class LeadInfo(BaseModel):
    Name: str = Field(description="Full name of the lead")
    Company: str = Field(description="Company name")
    Email: EmailStr = Field(description="Email address of the lead")
    Phone: str = Field(description="Phone number of the lead")
