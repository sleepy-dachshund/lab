"""LLM integration for email summarization."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from datetime import datetime
from collections import defaultdict, OrderedDict

from config import EmailConfig
from gmail_client import EmailData
import control

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from anthropic import Anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_summary(self, prompt: str) -> str:
        """Generate summary from prompt.
        
        Args:
            prompt: Input prompt for summarization
            
        Returns:
            str: Generated summary
        """
        pass


class GeminiProvider(BaseLLMProvider):
    """Gemini LLM provider for summarization."""
    
    def __init__(self, api_key: str):
        """Initialize Gemini provider.
        
        Args:
            api_key: Gemini API key
            
        Raises:
            ImportError: If google-generativeai not installed
            Exception: If API key configuration fails
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(control.GEMINI_MODEL)
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            raise Exception(f"Failed to configure Gemini: {e}")
    
    def generate_summary(self, prompt: str) -> str:
        """Generate summary using Gemini.
        
        Args:
            prompt: Input prompt for summarization
            
        Returns:
            str: Generated summary
            
        Raises:
            Exception: If API call fails
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Gemini API error: {e}")
            raise


class ClaudeProvider(BaseLLMProvider):
    """Claude LLM provider for summarization."""
    
    def __init__(self, api_key: str):
        """Initialize Claude provider.
        
        Args:
            api_key: Claude API key
            
        Raises:
            ImportError: If anthropic package not installed
        """
        if not CLAUDE_AVAILABLE:
            raise ImportError("anthropic package not installed")
        
        self.client = Anthropic(api_key=api_key)
        self.logger = logging.getLogger(__name__)
    
    def generate_summary(self, prompt: str) -> str:
        """Generate summary using Claude.
        
        Args:
            prompt: Input prompt for summarization
            
        Returns:
            str: Generated summary
            
        Raises:
            Exception: If API call fails
        """
        try:
            response = self.client.messages.create(
                model=control.CLAUDE_MODEL,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            raise


class EmailSummarizer:
    """Email summarization service using LLM providers."""
    
    def __init__(self, config: EmailConfig):
        """Initialize email summarizer.
        
        Args:
            config: EmailConfig with LLM provider and settings
            
        Raises:
            ValueError: If unsupported LLM provider specified
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM provider
        if config.llm_provider == 'gemini':
            self.llm = GeminiProvider(config.api_key)
        elif config.llm_provider == 'claude':
            self.llm = ClaudeProvider(config.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
    
    def summarize_emails(self, emails: List[EmailData]) -> str:
        """Summarize emails based on configuration.
        
        Args:
            emails: List of EmailData objects to summarize
            
        Returns:
            str: Formatted summary in markdown
        """
        if not emails:
            return "No emails found matching the specified criteria."
        
        self.logger.info(f"Summarizing {len(emails)} emails with level '{self.config.summary_level}'")
        
        if self.config.summary_level == 'sender':
            return self._summarize_by_sender(emails)
        elif self.config.summary_level == 'day':
            return self._summarize_by_day(emails)
        elif self.config.summary_level == 'week':
            return self._summarize_by_week(emails)
        else:
            raise ValueError(f"Unsupported summary level: {self.config.summary_level}")
    
    def _summarize_by_sender(self, emails: List[EmailData]) -> str:
        """Group emails by sender and summarize each group.
        
        Args:
            emails: List of EmailData objects
            
        Returns:
            str: Markdown-formatted summary grouped by sender
        """
        # Group emails by sender
        sender_groups = defaultdict(list)
        for email in emails:
            # Extract just the email address from sender
            sender_key = self._extract_email_address(email.sender)
            sender_groups[sender_key].append(email)
        
        summary_parts = [f"# Email Summary by Sender ({len(emails)} total emails)\n"]
        
        for sender, sender_emails in sender_groups.items():
            sender_emails.sort(key=lambda e: e.date, reverse=True)  # Most recent first
            
            # Create prompt for this sender's emails
            prompt = self._create_sender_prompt(sender, sender_emails)
            
            try:
                sender_summary = self.llm.generate_summary(prompt)
                summary_parts.append(f"## {sender} ({len(sender_emails)} emails)\n")
                summary_parts.append(f"{sender_summary}\n")
            except Exception as e:
                self.logger.error(f"Failed to summarize emails from {sender}: {e}")
                summary_parts.append(f"## {sender} ({len(sender_emails)} emails)\n")
                summary_parts.append("*Summary generation failed*\n")
        
        return "\n".join(summary_parts)
    
    def _summarize_by_day(self, emails: List[EmailData]) -> str:
        """Group emails by day and summarize each day.
        
        Args:
            emails: List of EmailData objects
            
        Returns:
            str: Markdown-formatted summary grouped by day
        """
        # Group emails by date
        day_groups = defaultdict(list)
        for email in emails:
            day_key = email.date.date()
            day_groups[day_key].append(email)
        
        # Sort days
        sorted_days = OrderedDict(sorted(day_groups.items(), reverse=True))
        
        summary_parts = [f"# Daily Email Summary ({len(emails)} total emails)\n"]
        
        for day, day_emails in sorted_days.items():
            day_emails.sort(key=lambda e: e.date, reverse=True)
            
            prompt = self._create_day_prompt(day, day_emails)
            
            try:
                day_summary = self.llm.generate_summary(prompt)
                summary_parts.append(f"## {day.strftime('%A, %B %d, %Y')} ({len(day_emails)} emails)\n")
                summary_parts.append(f"{day_summary}\n")
            except Exception as e:
                self.logger.error(f"Failed to summarize emails for {day}: {e}")
                summary_parts.append(f"## {day.strftime('%A, %B %d, %Y')} ({len(day_emails)} emails)\n")
                summary_parts.append("*Summary generation failed*\n")
        
        return "\n".join(summary_parts)
    
    def _summarize_by_week(self, emails: List[EmailData]) -> str:
        """Create a single weekly summary of all emails.
        
        Args:
            emails: List of EmailData objects
            
        Returns:
            str: Markdown-formatted weekly summary
        """
        emails.sort(key=lambda e: e.date, reverse=True)
        
        date_range = self._get_date_range(emails)
        prompt = self._create_week_prompt(emails, date_range)
        
        try:
            weekly_summary = self.llm.generate_summary(prompt)
            summary_parts = [
                f"# Weekly Email Summary ({len(emails)} emails)",
                f"**Period:** {date_range}\n",
                weekly_summary
            ]
            return "\n".join(summary_parts)
        except Exception as e:
            self.logger.error(f"Failed to generate weekly summary: {e}")
            return f"# Weekly Email Summary ({len(emails)} emails)\n\n*Summary generation failed*"
    
    def _create_sender_prompt(self, sender: str, emails: List[EmailData]) -> str:
        """Create prompt for sender-based summarization.
        
        Args:
            sender: Email sender
            emails: List of emails from this sender
            
        Returns:
            str: Formatted prompt for LLM
        """
        email_content = []
        for email in emails[:control.MAX_EMAILS_PER_SENDER]:  # Limit to prevent token overflow
            email_content.append(f"Subject: {email.subject}")
            email_content.append(f"Date: {email.date.strftime('%Y-%m-%d %H:%M')}")
            # Truncate very long bodies
            body_preview = email.body[:control.EMAIL_BODY_PREVIEW_LENGTH] + "..." if len(email.body) > control.EMAIL_BODY_PREVIEW_LENGTH else email.body
            email_content.append(f"Body: {body_preview}")
            email_content.append("---")
        
        return control.SENDER_SUMMARY_PROMPT.format(
            sender=sender,
            style=self.config.summary_style,
            email_content=chr(10).join(email_content)
        )
    
    def _create_day_prompt(self, day, emails: List[EmailData]) -> str:
        """Create prompt for day-based summarization.
        
        Args:
            day: Date object
            emails: List of emails from this day
            
        Returns:
            str: Formatted prompt for LLM
        """
        email_content = []
        for email in emails:
            email_content.append(f"From: {self._extract_email_address(email.sender)}")
            email_content.append(f"Subject: {email.subject}")
            # Truncate very long bodies
            body_preview = email.body[:control.EMAIL_BODY_PREVIEW_LENGTH] + "..." if len(email.body) > control.EMAIL_BODY_PREVIEW_LENGTH else email.body
            email_content.append(f"Body: {body_preview}")
            email_content.append("---")
        
        return control.DAY_SUMMARY_PROMPT.format(
            date=day.strftime('%B %d, %Y'),
            style=self.config.summary_style,
            email_content=chr(10).join(email_content)
        )
    
    def _create_week_prompt(self, emails: List[EmailData], date_range: str) -> str:
        """Create prompt for weekly summarization.
        
        Args:
            emails: List of all emails
            date_range: String representation of date range
            
        Returns:
            str: Formatted prompt for LLM
        """
        email_content = []
        for email in emails[:control.MAX_EMAILS_PER_SUMMARY]:  # Limit to prevent token overflow
            email_content.append(f"From: {self._extract_email_address(email.sender)}")
            email_content.append(f"Subject: {email.subject}")
            email_content.append(f"Date: {email.date.strftime('%Y-%m-%d')}")
            # Truncate very long bodies
            body_preview = email.body[:control.EMAIL_BODY_PREVIEW_LENGTH] + "..." if len(email.body) > control.EMAIL_BODY_PREVIEW_LENGTH else email.body
            email_content.append(f"Body: {body_preview}")
            email_content.append("---")
        
        return control.WEEK_SUMMARY_PROMPT.format(
            date_range=date_range,
            style=self.config.summary_style,
            email_content=chr(10).join(email_content)
        )
    
    def _extract_email_address(self, sender: str) -> str:
        """Extract email address from sender field.
        
        Args:
            sender: Full sender string (may include name)
            
        Returns:
            str: Clean email address
        """
        # Extract email from "Name <email@domain.com>" format
        if '<' in sender and '>' in sender:
            start = sender.find('<') + 1
            end = sender.find('>')
            return sender[start:end]
        return sender
    
    def _get_date_range(self, emails: List[EmailData]) -> str:
        """Get date range string from emails.
        
        Args:
            emails: List of EmailData objects
            
        Returns:
            str: Formatted date range
        """
        if not emails:
            return "No emails"
        
        dates = [email.date.date() for email in emails]
        min_date = min(dates)
        max_date = max(dates)
        
        if min_date == max_date:
            return min_date.strftime('%B %d, %Y')
        else:
            return f"{min_date.strftime('%B %d, %Y')} - {max_date.strftime('%B %d, %Y')}"