# Gmail Email Summarizer

A lightweight Python application that automatically summarizes Gmail emails using LLM APIs and sends the summary back via email.

## How It Works

Here's what happens when you run this: 
- First, it loads your settings from environment variables (load_config_from_env() reads your .env file). 
- Then it connects to Gmail using IMAP (connect_imap() logs in with your app password). 
- Next, it searches for emails in your date range and applies any filters you set (fetch_emails() returns a list of EmailData objects). 
   - Gmail's IMAP server gives us RFC822 message objects that contain all the email metadata and content
   - We filter these using IMAP search commands (SINCE/BEFORE for dates, FROM for senders) before downloading
   - For each message, we extract the plain text body (_extract_body() handles multipart emails and encoding), plus sender/subject/date
   - Additional filtering happens locally (_should_include_email() checks exclude lists and subject keywords)
- The emails get grouped by sender, day, or week depending on your config. 
- Each group gets sent to an LLM - either Gemini or Claude (generate_summary() takes email text and returns markdown). 
- Finally, it converts the markdown to HTML (markdown_to_html() does basic formatting) and emails it back to you (send_email() handles SMTP). 
- If anything breaks, it logs the error and optionally prints the summary to console instead.


## Features

- **Flexible Email Selection**: Filter by date range, senders, and subject keywords
- **Multiple Summary Levels**: Group by sender, day, or week
- **LLM Integration**: Support for both Gemini and Claude APIs
- **Configurable Output**: Customize summary style and recipients
- **Built-in Presets**: Ready-to-use configurations for weekly newsletters and monthly reports

## Quick Start

### 1. Setup

```bash
# Clone and install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env
# Edit .env with your credentials
```

### 2. Configure Gmail

1. Enable 2-factor authentication on your Gmail account
2. Generate an App Password: [Google Account Settings](https://myaccount.google.com/apppasswords)
3. Add your email and app password to `.env`

### 3. Get API Keys

**For Gemini (recommended):**
- Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Add to `.env` as `GEMINI_API_KEY`

**For Claude (optional):**
- Get API key from [Anthropic Console](https://console.anthropic.com/)
- Add to `.env` as `CLAUDE_API_KEY`

### 4. Run

```bash
python main.py
```

## Configuration

### Basic Usage

The application uses environment variables and built-in defaults. For custom configurations, modify the values in `main.py`:

```python
# Weekly newsletter summary
update_config_for_preset(config, 'weekly')

# Monthly report summary  
update_config_for_preset(config, 'monthly')
```

### Advanced Configuration

For custom setups, modify the `EmailConfig` object in `main.py`:

```python
# Example: Custom newsletter configuration
config.lookback_days = 7
config.summary_level = 'sender'
config.include_senders = ['newsletter@example.com', 'updates@company.com']
config.summary_style = 'concise bullet points with actionable items'
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lookback_days` | Days to look back for emails | 7 |
| `include_senders` | List of senders to include (None = all) | None |
| `exclude_senders` | List of senders to exclude | None |
| `subject_keywords` | Keywords to filter subjects | None |
| `summary_level` | Grouping: 'sender', 'day', 'week' | 'sender' |
| `max_emails_per_summary` | Max emails to process | 50 |
| `llm_provider` | LLM service: 'gemini' or 'claude' | 'gemini' |
| `summary_style` | Instructions for summary generation | 'concise bullet points' |
| `send_summary_to` | Recipient email (None = sender) | None |

## Use Cases

### Weekly Newsletter Digest
```bash
# Summarizes newsletters by sender over the past week
# Perfect for staying up-to-date with subscriptions
python main.py  # with weekly preset enabled
```

### Monthly Activity Summary  
```bash
# Creates weekly summaries over the past month
# Great for periodic email reviews
python main.py  # with monthly preset enabled
```

### Custom Filtering
Modify configuration for specific needs:
- Sales team updates: Filter by sender domains
- Project notifications: Filter by subject keywords
- Client communications: Exclude internal senders

## Output

The application generates:
- **Markdown-formatted summaries** with clear structure
- **HTML emails** sent to specified recipients  
- **Console output** with progress and status
- **Log files** for debugging (`email_summarizer.log`)

## Troubleshooting

### Gmail Connection Issues
- Verify 2FA is enabled
- Ensure App Password is 16 characters
- Check firewall/network restrictions

### API Errors
- Verify API keys are valid and have quota
- Check rate limits for your API plan
- Ensure proper network connectivity

### No Emails Found
- Verify date range covers expected emails
- Check sender filters aren't too restrictive  
- Confirm emails exist in Gmail inbox

### Summary Quality
- Adjust `summary_style` for different output formats
- Modify `max_emails_per_summary` for performance vs completeness
- Try different `summary_level` groupings

## File Structure

```
email_summarizer/
├── main.py              # Main orchestration script
├── config.py            # Configuration management
├── gmail_client.py      # Gmail IMAP/SMTP handling  
├── summarizer.py        # LLM integration
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── projectplan.md       # Development documentation
└── README.md           # Usage documentation
```

## Scheduling

For automated execution, use system schedulers:

**Linux/macOS (cron):**
```bash
# Weekly summary every Monday at 9 AM
0 9 * * 1 /usr/bin/python3 /path/to/main.py
```

**Windows (Task Scheduler):**
- Create a basic task to run `python main.py`  
- Set trigger for desired frequency

## Security Notes

- App passwords are more secure than account passwords
- API keys should be kept confidential  
- Consider running in isolated environments for production use
- Log files may contain email metadata

## Contributing

This project follows the development workflow documented in `CLAUDE.md`. Key principles:
- Simplicity first with minimal dependencies
- Clear separation of concerns across modules
- Comprehensive error handling and logging
- Type hints and docstring documentation