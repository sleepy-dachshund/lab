Instructions for Claude Code - Gmail Email Summarizer Project
Project Overview
Create a lightweight Python application that:

Connects to Gmail to fetch specific emails
Summarizes them using an LLM API
Sends the summary back via email
Supports flexible configuration for different use cases (weekly newsletters, monthly reports)

Initial Setup Instructions
Please create a new Python project for summarizing Gmail emails. Follow the CLAUDE.md guidelines throughout.

## Phase 1: Project Planning

Create a projectplan.md with:

**Problem Statement**: 
- Need automated email summarization for newsletters and other recurring emails
- Should support multiple summary cadences (weekly, monthly) with minimal config changes
- Must be lightweight and easy to extend

**Approach**:
- Use IMAP for Gmail access (with app password)
- Implement configurable email selection criteria
- Use LLM API for summarization (Gemini or Claude)
- Send summaries via SMTP

**Project Structure**:
email_summarizer/
├── config.py           # Configuration and hyperparameters
├── gmail_client.py     # Gmail IMAP/SMTP handling
├── summarizer.py       # LLM integration for summaries
├── main.py            # Main orchestration logic
├── requirements.txt    # Dependencies
├── .env.example       # Environment variable template
└── projectplan.md     # Project documentation

**Todo List**:
- [ ] Set up project structure and dependencies
- [ ] Create config.py with hyperparameters
- [ ] Implement Gmail connection (IMAP/SMTP)
- [ ] Create email fetching logic with filters
- [ ] Integrate LLM API for summarization
- [ ] Build summary formatting (by sender/day/week)
- [ ] Implement email sending functionality
- [ ] Add error handling and logging
- [ ] Create main orchestration script
- [ ] Write usage documentation

## Phase 2: Configuration Design

Create config.py with these parameters:

```python
from typing import List, Literal
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class EmailConfig:
    # Gmail credentials (from environment)
    email_address: str
    app_password: str
    
    # Selection criteria
    lookback_days: int = 7
    include_senders: List[str] = None  # If None, include all
    exclude_senders: List[str] = None
    subject_keywords: List[str] = None  # Optional filtering
    
    # Summary configuration
    summary_level: Literal['sender', 'day', 'week'] = 'sender'
    max_emails_per_summary: int = 50
    
    # LLM configuration
    llm_provider: Literal['gemini', 'claude'] = 'gemini'
    api_key: str  # From environment
    summary_style: str = "concise bullet points"
    
    # Output configuration
    send_summary_to: str = None  # If None, use email_address
    summary_subject_prefix: str = "Email Summary"
Phase 3: Core Implementation
gmail_client.py requirements:

Connect via IMAP using app password
Fetch emails with criteria:

Date range (lookback_days)
Sender filtering (include/exclude lists)
Optional subject keyword matching


Return structured email data (sender, subject, date, body)
Send emails via SMTP

summarizer.py requirements:

Abstract interface for LLM providers
Implement Gemini integration (using google-generativeai)
Consider Claude integration if API key works for chat
Generate summaries based on summary_level:

'sender': Group by sender, summarize each group
'day': Group by date, daily digests
'week': Single weekly summary


Format output as markdown for email

main.py requirements:

Load configuration from config.py and environment
Orchestrate the workflow:

Connect to Gmail
Fetch relevant emails
Group according to summary_level
Generate summaries via LLM
Format and send summary email


Include proper logging
Handle errors gracefully

Phase 4: Environment Setup
Create .env.example:
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_APP_PASSWORD=your_app_password
GEMINI_API_KEY=your_gemini_key
# Optional
CLAUDE_API_KEY=your_claude_key
Implementation Notes:

Use these libraries:

imaplib and email (built-in) for IMAP
smtplib (built-in) for sending
google-generativeai for Gemini
python-dotenv for environment variables
Optional: anthropic for Claude API


Error handling priorities:

Gmail connection failures
API rate limits
Empty email results
LLM API failures


Keep it simple:

No database needed
No complex scheduling (use cron/Task Scheduler)
Minimal dependencies
Clear separation of concerns


Testing approach:

Start with fetching 1-2 emails
Test summarization separately
Verify email sending with test recipient



Please implement this step-by-step, following the todo list and updating projectplan.md as you progress. Use type hints and numpy docstrings throughout. Focus on making it work for the weekly newsletter use case first, then ensure it's flexible enough for other scenarios.

### Additional Context for Claude Code:
Available Secrets:

EMAIL_APP_PASSWORD (GitHub secret) - for Gmail authentication
GEMINI_API_KEY (GitHub secret) - for Gemini API
I also have a Claude API key from Claude Code subscription

Key Requirements:

The same code should work for both weekly newsletter summaries and monthly email summaries by just changing config parameters
Keep dependencies minimal
Prefer built-in Python libraries where possible
Make the summary format readable and actionable

Example Use Cases:

Weekly newsletter digest: lookback_days=7, summary_level='sender', include_senders=[list of newsletter addresses]
Monthly activity summary: lookback_days=30, summary_level='week', include_senders=None

Start by creating the project structure and projectplan.md, then proceed with implementation.

These instructions should give Claude Code a clear roadmap to build your email summarizer. The key points are:

1. **Structured approach** following your CLAUDE.md workflow
2. **Clear configuration** that supports your different use cases
3. **Simple architecture** with separated concerns
4. **Flexibility** through configuration rather than code changes