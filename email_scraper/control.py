"""
CONTROL PANEL - Easy Configuration for Email Summarizer
=======================================================

This file contains all the main settings you might want to change.
Edit these values instead of hunting through code files.

🔧 QUICK SETTINGS
"""

# Email Filtering Settings
# ========================
# Which emails to include - set to None to include ALL emails
INCLUDE_SENDERS = [
    'email@stratechery.com',
    # 'newsletter@example.com',
    # 'updates@company.com',
]

# Which emails to exclude (even if they match include list above)
EXCLUDE_SENDERS = [
    # 'spam@example.com',
    # 'notifications@social.com',
]

# How many days back to look for emails
LOOKBACK_DAYS = 7

# Keywords that must appear in email subjects (None = no filtering)
SUBJECT_KEYWORDS = None
# Example: ['newsletter', 'update', 'digest']


# AI Model Settings
# =================
# Claude model to use - update this when new models are released
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Gemini model to use  
GEMINI_MODEL = "gemini-pro"

# Which AI provider to prefer if both API keys are available
# Options: 'claude', 'gemini', 'auto' (auto = prefer Claude)
PREFERRED_AI_PROVIDER = 'auto'


# Summary Configuration
# =====================
# How to group emails for summary
# Options: 'sender', 'day', 'week'
SUMMARY_LEVEL = 'sender'

# Maximum emails to include in one summary (prevents token overflow)
MAX_EMAILS_PER_SUMMARY = 30

# Style instructions for the AI
SUMMARY_STYLE = "concise bullet points highlighting key topics and actionable items"

# Subject prefix for summary emails
SUMMARY_SUBJECT_PREFIX = "📧 Email Summary"


# LLM Prompts
# ===========
# Base prompt template for sender-based summaries
SENDER_SUMMARY_PROMPT = """Please create a summary of these emails from {sender} in {style}. 
Focus on key topics, important updates, and actionable items.
Take extra care that formatting is clear and concise and looks good in email clients.

Emails:
{email_content}

Summary:"""

# Base prompt template for daily summaries
DAY_SUMMARY_PROMPT = """Please create a summary of these emails from {date} in {style}.
Group by sender and highlight important information.
Take extra care that formatting is clear and concise and looks good in email clients.

Emails:
{email_content}

Summary:"""

# Base prompt template for weekly summaries  
WEEK_SUMMARY_PROMPT = """Please create a weekly summary of these emails from {date_range} in {style}.
Organize by key themes and important developments.
Take extra care that formatting is clear and concise and looks good in email clients.

Emails:
{email_content}

Weekly Summary:"""


# Advanced Settings
# =================
# Email body preview length (characters) before truncation
EMAIL_BODY_PREVIEW_LENGTH = 1000

# Maximum emails to include per sender in summaries
MAX_EMAILS_PER_SENDER = 10

# Recipient for summary email (None = send to yourself)
SEND_SUMMARY_TO = None


# Preset Configurations
# =====================
# You can uncomment one of these in main.py to apply preset configurations

WEEKLY_NEWSLETTER_PRESET = {
    'lookback_days': 7,
    'summary_level': 'sender', 
    'summary_style': 'concise bullet points highlighting key topics and actionable items',
    'summary_subject_prefix': '📰 Weekly Newsletter Summary'
}

MONTHLY_REPORT_PRESET = {
    'lookback_days': 30,
    'summary_level': 'week',
    'summary_style': 'weekly digest format with key themes and important updates', 
    'summary_subject_prefix': '📊 Monthly Email Report'
}

DAILY_DIGEST_PRESET = {
    'lookback_days': 1,
    'summary_level': 'day',
    'summary_style': 'brief daily digest with key highlights',
    'summary_subject_prefix': '🌅 Daily Email Digest'
}