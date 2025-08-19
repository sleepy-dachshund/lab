# Gmail Email Summarizer - Project Plan

## Problem Statement
Need automated email summarization for newsletters and other recurring emails. The solution should:
- Support multiple summary cadences (weekly, monthly) with minimal config changes
- Be lightweight and easy to extend
- Handle different email selection criteria flexibly
- Provide readable, actionable summaries via email

## Approach
- Use IMAP for Gmail access (with app password)
- Implement configurable email selection criteria (date range, senders, keywords)
- Use LLM API for summarization (Gemini primary, Claude optional)
- Send summaries via SMTP
- Configuration-driven design for flexibility

## Project Structure
```
email_summarizer/
├── config.py           # Configuration and hyperparameters
├── gmail_client.py     # Gmail IMAP/SMTP handling
├── summarizer.py       # LLM integration for summaries
├── main.py            # Main orchestration logic
├── requirements.txt    # Dependencies
├── .env.example       # Environment variable template
└── projectplan.md     # Project documentation
```

## Todo List
- [ ] Set up project structure and dependencies
- [ ] Create config.py with hyperparameters
- [ ] Implement Gmail connection (IMAP/SMTP)
- [ ] Create email fetching logic with filters
- [ ] Integrate LLM API for summarization
- [ ] Build summary formatting (by sender/day/week)
- [ ] Implement email sending functionality
- [ ] Add error handling and logging
- [ ] Create main orchestration script
- [ ] Create .env.example template
- [ ] Write usage documentation

## Success Criteria
- Successfully connects to Gmail via IMAP
- Fetches emails based on configurable criteria
- Generates coherent summaries using LLM
- Sends formatted summary emails
- Supports both weekly newsletter and monthly report use cases
- Handles errors gracefully with appropriate logging

## Example Use Cases
1. **Weekly newsletter digest**: `lookback_days=7`, `summary_level='sender'`, specific newsletter senders
2. **Monthly activity summary**: `lookback_days=30`, `summary_level='week'`, all senders

## Dependencies
- imaplib, email (built-in) for IMAP
- smtplib (built-in) for sending emails  
- google-generativeai for Gemini API
- python-dotenv for environment variables
- Optional: anthropic for Claude API

## Key Design Decisions
- No database required - stateless operation
- Configuration over code changes
- Clear separation of concerns across modules
- Minimal external dependencies
- Type hints and numpy docstrings throughout