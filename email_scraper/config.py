"""Configuration module for email summarizer application."""

from typing import List, Literal, Optional
from dataclasses import dataclass
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class EmailConfig:
    """Configuration for email summarization process.
    
    Attributes:
        email_address: Gmail address for authentication
        app_password: Gmail app password for IMAP/SMTP access
        lookback_days: Number of days to look back for emails
        include_senders: List of senders to include (None = all senders)
        exclude_senders: List of senders to exclude
        subject_keywords: Keywords to filter subjects (None = no filtering)
        summary_level: Grouping level for summaries
        max_emails_per_summary: Maximum emails to include in one summary
        llm_provider: LLM service to use for summarization
        api_key: API key for LLM provider
        summary_style: Style instructions for LLM summaries
        send_summary_to: Email address to send summary (None = use email_address)
        summary_subject_prefix: Prefix for summary email subject
    """
    # Gmail credentials
    email_address: str
    app_password: str
    
    # Selection criteria
    lookback_days: int = 7
    include_senders: Optional[List[str]] = None
    exclude_senders: Optional[List[str]] = None
    subject_keywords: Optional[List[str]] = None
    
    # Summary configuration
    summary_level: Literal['sender', 'day', 'week'] = 'sender'
    max_emails_per_summary: int = 50
    
    # LLM configuration
    llm_provider: Literal['gemini', 'claude'] = 'gemini'
    api_key: str = None
    summary_style: str = "concise bullet points"
    
    # Output configuration
    send_summary_to: Optional[str] = None
    summary_subject_prefix: str = "Email Summary"


def load_config_from_env() -> EmailConfig:
    """Load configuration from environment variables.
    
    Returns:
        EmailConfig: Configuration object with values from environment
        
    Raises:
        ValueError: If required environment variables are missing
    """
    email_address = os.getenv('EMAIL_ADDRESS')
    app_password = os.getenv('EMAIL_APP_PASSWORD')
    
    if not email_address:
        raise ValueError("EMAIL_ADDRESS environment variable is required")
    if not app_password:
        raise ValueError("EMAIL_APP_PASSWORD environment variable is required")
    
    # Determine LLM provider and API key
    gemini_key = os.getenv('GEMINI_API_KEY')
    claude_key = os.getenv('CLAUDE_API_KEY')
    
    if gemini_key:
        llm_provider = 'gemini'
        api_key = gemini_key
    elif claude_key:
        llm_provider = 'claude'
        api_key = claude_key
    else:
        raise ValueError("Either GEMINI_API_KEY or CLAUDE_API_KEY environment variable is required")
    
    return EmailConfig(
        email_address=email_address,
        app_password=app_password,
        api_key=api_key,
        llm_provider=llm_provider
    )


# Example configurations for common use cases
WEEKLY_NEWSLETTER_CONFIG = {
    'lookback_days': 7,
    'summary_level': 'sender',
    'summary_style': 'concise bullet points highlighting key topics and actionable items',
    'summary_subject_prefix': 'Weekly Newsletter Summary'
}

MONTHLY_REPORT_CONFIG = {
    'lookback_days': 30,
    'summary_level': 'week',
    'summary_style': 'weekly digest format with key themes and important updates',
    'summary_subject_prefix': 'Monthly Email Report'
}