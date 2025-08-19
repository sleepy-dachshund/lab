"""Main orchestration script for email summarization."""

import logging
import sys
from datetime import datetime, timedelta
from typing import Optional

from config import load_config_from_env, WEEKLY_NEWSLETTER_CONFIG, MONTHLY_REPORT_CONFIG
from gmail_client import GmailClient
from summarizer import EmailSummarizer


def setup_logging(level: str = 'INFO') -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('email_summarizer.log')
        ]
    )


def update_config_for_preset(config, preset_name: Optional[str] = None):
    """Update configuration with preset values.
    
    Args:
        config: EmailConfig object to update
        preset_name: Name of preset ('weekly', 'monthly', or None)
    """
    if preset_name == 'weekly':
        for key, value in WEEKLY_NEWSLETTER_CONFIG.items():
            setattr(config, key, value)
    elif preset_name == 'monthly':
        for key, value in MONTHLY_REPORT_CONFIG.items():
            setattr(config, key, value)


def main():
    """Main function to orchestrate email summarization workflow."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration from environment
        logger.info("Loading configuration from environment...")
        config = load_config_from_env()

        config.include_senders = ['email@stratechery.com']
        
        # Optional: Apply preset configuration
        # Uncomment and modify one of these lines for preset configs:
        # update_config_for_preset(config, 'weekly')
        # update_config_for_preset(config, 'monthly')
        
        logger.info(f"Configuration loaded: {config.lookback_days} days, {config.summary_level} level, {config.llm_provider} provider")
        
        # Initialize clients
        logger.info("Initializing Gmail client...")
        gmail_client = GmailClient(config)
        
        logger.info("Initializing email summarizer...")
        summarizer = EmailSummarizer(config)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=config.lookback_days)
        
        logger.info(f"Fetching emails from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Fetch emails
        emails = gmail_client.fetch_emails(start_date, end_date)
        
        if not emails:
            logger.info("No emails found matching criteria")
            return
        
        logger.info(f"Found {len(emails)} emails to summarize")
        
        # Generate summary
        logger.info("Generating summary...")
        summary = summarizer.summarize_emails(emails)
        
        # Determine recipient
        recipient = config.send_summary_to or config.email_address
        
        # Create subject
        date_range = f"{start_date.strftime('%m/%d')} - {end_date.strftime('%m/%d/%Y')}"
        subject = f"{config.summary_subject_prefix} ({date_range})"
        
        # Convert markdown to simple HTML for email
        html_summary = markdown_to_html(summary)
        
        # Send summary email
        logger.info(f"Sending summary to {recipient}...")
        success = gmail_client.send_email(recipient, subject, html_summary, is_html=True)
        
        if success:
            logger.info("Email summary sent successfully!")
            print(f"✅ Summary sent to {recipient}")
            print(f"📧 Subject: {subject}")
            print(f"📊 {len(emails)} emails summarized")
        else:
            logger.error("Failed to send summary email")
            print("❌ Failed to send summary email")
            
            # Print summary to console as fallback
            print("\n" + "="*50)
            print("SUMMARY (fallback to console):")
            print("="*50)
            print(summary)
    
    except Exception as e:
        logger.error(f"Email summarization failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


def markdown_to_html(markdown_text: str) -> str:
    """Convert simple markdown to HTML for email.
    
    Args:
        markdown_text: Markdown formatted text
        
    Returns:
        str: HTML formatted text
    """
    html = markdown_text
    
    # Convert headers
    html = html.replace('# ', '<h1>').replace('\n# ', '</h1>\n<h1>')
    html = html.replace('## ', '<h2>').replace('\n## ', '</h2>\n<h2>')
    html = html.replace('### ', '<h3>').replace('\n### ', '</h3>\n<h3>')
    
    # Close unclosed headers at end of line
    lines = html.split('\n')
    processed_lines = []
    for line in lines:
        if line.startswith('<h1>') and not line.endswith('</h1>'):
            line += '</h1>'
        elif line.startswith('<h2>') and not line.endswith('</h2>'):
            line += '</h2>'
        elif line.startswith('<h3>') and not line.endswith('</h3>'):
            line += '</h3>'
        processed_lines.append(line)
    
    html = '\n'.join(processed_lines)
    
    # Convert bullet points
    html = html.replace('- ', '• ')
    html = html.replace('* ', '• ')
    
    # Convert bold text
    html = html.replace('**', '<strong>', 1).replace('**', '</strong>', 1)
    
    # Convert line breaks to HTML
    html = html.replace('\n\n', '<br><br>')
    html = html.replace('\n', '<br>')
    
    # Wrap in basic HTML structure
    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    {html}
    </body>
    </html>
    """
    
    return html


if __name__ == "__main__":
    main()