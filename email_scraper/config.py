"""Configuration module for email summarizer application."""

from typing import List, Literal, Optional
from dataclasses import dataclass, field
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import control

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
    
    # Selection criteria (loaded from control.py)
    lookback_days: int = control.LOOKBACK_DAYS
    include_senders: Optional[List[str]] = field(default_factory=lambda: control.INCLUDE_SENDERS)
    exclude_senders: Optional[List[str]] = field(default_factory=lambda: control.EXCLUDE_SENDERS)
    subject_keywords: Optional[List[str]] = field(default_factory=lambda: control.SUBJECT_KEYWORDS)
    
    # Summary configuration (loaded from control.py)
    summary_level: Literal['sender', 'day', 'week'] = control.SUMMARY_LEVEL
    max_emails_per_summary: int = control.MAX_EMAILS_PER_SUMMARY
    
    # LLM configuration
    llm_provider: Literal['gemini', 'claude'] = 'gemini'
    api_key: str = None
    summary_style: str = control.SUMMARY_STYLE
    
    # Output configuration (loaded from control.py)
    send_summary_to: Optional[str] = control.SEND_SUMMARY_TO
    summary_subject_prefix: str = control.SUMMARY_SUBJECT_PREFIX


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
    
    # Determine LLM provider and API key based on control.py preference
    gemini_key = os.getenv('GEMINI_API_KEY')
    claude_key = os.getenv('CLAUDE_API_KEY')
    
    # Use control.py preference to determine provider
    if control.PREFERRED_AI_PROVIDER == 'claude' and claude_key:
        llm_provider = 'claude'
        api_key = claude_key
    elif control.PREFERRED_AI_PROVIDER == 'gemini' and gemini_key:
        llm_provider = 'gemini'
        api_key = gemini_key
    elif control.PREFERRED_AI_PROVIDER == 'auto':
        # Auto mode: prefer Claude if available, fallback to Gemini
        if claude_key:
            llm_provider = 'claude'
            api_key = claude_key
        elif gemini_key:
            llm_provider = 'gemini'
            api_key = gemini_key
        else:
            raise ValueError("Either GEMINI_API_KEY or CLAUDE_API_KEY environment variable is required")
    else:
        raise ValueError(f"Invalid AI provider preference: {control.PREFERRED_AI_PROVIDER} or missing API key")
    
    return EmailConfig(
        email_address=email_address,
        app_password=app_password,
        api_key=api_key,
        llm_provider=llm_provider
    )


# Preset configurations (loaded from control.py)
WEEKLY_NEWSLETTER_CONFIG = control.WEEKLY_NEWSLETTER_PRESET
MONTHLY_REPORT_CONFIG = control.MONTHLY_REPORT_PRESET
DAILY_DIGEST_CONFIG = control.DAILY_DIGEST_PRESET