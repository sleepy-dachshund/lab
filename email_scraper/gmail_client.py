"""Gmail client for IMAP and SMTP operations."""

import imaplib
import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from config import EmailConfig


class EmailData:
    """Container for email data."""
    
    def __init__(self, sender: str, subject: str, date: datetime, body: str, message_id: str):
        self.sender = sender
        self.subject = subject
        self.date = date
        self.body = body
        self.message_id = message_id
    
    def __str__(self) -> str:
        return f"Email from {self.sender}: {self.subject[:50]}..."


class GmailClient:
    """Gmail client for fetching and sending emails via IMAP/SMTP."""
    
    def __init__(self, config: EmailConfig):
        """Initialize Gmail client with configuration.
        
        Args:
            config: EmailConfig object with credentials and settings
        """
        self.config = config
        self.imap_server = 'imap.gmail.com'
        self.smtp_server = 'smtp.gmail.com'
        self.imap_port = 993
        self.smtp_port = 587
        self.logger = logging.getLogger(__name__)
    
    def connect_imap(self) -> imaplib.IMAP4_SSL:
        """Connect to Gmail IMAP server.
        
        Returns:
            IMAP4_SSL: Connected IMAP client
            
        Raises:
            Exception: If connection fails
        """
        try:
            mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            mail.login(self.config.email_address, self.config.app_password)
            self.logger.info(f"Connected to IMAP for {self.config.email_address}")
            return mail
        except Exception as e:
            self.logger.error(f"IMAP connection failed: {e}")
            raise
    
    def fetch_emails(self, start_date: datetime, end_date: datetime = None) -> List[EmailData]:
        """Fetch emails from Gmail based on date range and filters.
        
        Args:
            start_date: Start date for email search
            end_date: End date for email search (defaults to now)
            
        Returns:
            List[EmailData]: List of email data objects matching criteria
            
        Raises:
            Exception: If fetching fails
        """
        if end_date is None:
            end_date = datetime.now()
        
        mail = self.connect_imap()
        emails = []
        
        try:
            mail.select('inbox')
            
            # Format dates for IMAP search
            since_date = start_date.strftime('%d-%b-%Y')
            before_date = (end_date + timedelta(days=1)).strftime('%d-%b-%Y')
            
            # Build search criteria
            search_criteria = [f'SINCE {since_date}', f'BEFORE {before_date}']
            
            # Add sender filters if specified
            if self.config.include_senders:
                sender_criteria = []
                for sender in self.config.include_senders:
                    sender_criteria.append(f'FROM "{sender}"')
                search_criteria.append(f'({" OR ".join(sender_criteria)})')
            
            search_string = ' '.join(search_criteria)
            self.logger.info(f"IMAP search: {search_string}")
            
            # Search for emails
            _, message_numbers = mail.search(None, search_string)
            
            if not message_numbers[0]:
                self.logger.info("No emails found matching criteria")
                return emails
            
            message_nums = message_numbers[0].split()
            self.logger.info(f"Found {len(message_nums)} emails")
            
            # Limit number of emails processed
            if len(message_nums) > self.config.max_emails_per_summary:
                message_nums = message_nums[-self.config.max_emails_per_summary:]
                self.logger.info(f"Limited to {len(message_nums)} most recent emails")
            
            # Fetch email data
            for num in message_nums:
                try:
                    _, msg_data = mail.fetch(num, '(RFC822)')
                    email_body = msg_data[0][1]
                    email_message = email.message_from_bytes(email_body)
                    
                    # Extract email components
                    sender = self._decode_header(email_message['From'])
                    subject = self._decode_header(email_message['Subject'])
                    date_str = email_message['Date']
                    message_id = email_message['Message-ID']
                    
                    # Parse date
                    email_date = email.utils.parsedate_to_datetime(date_str)
                    
                    # Extract body
                    body = self._extract_body(email_message)
                    
                    # Apply filters
                    if self._should_include_email(sender, subject):
                        email_data = EmailData(sender, subject, email_date, body, message_id)
                        emails.append(email_data)
                
                except Exception as e:
                    self.logger.warning(f"Error processing email {num}: {e}")
                    continue
            
            self.logger.info(f"Successfully fetched {len(emails)} emails")
            return emails
        
        finally:
            mail.close()
            mail.logout()
    
    def send_email(self, to_address: str, subject: str, body: str, is_html: bool = True) -> bool:
        """Send email via SMTP.
        
        Args:
            to_address: Recipient email address
            subject: Email subject
            body: Email body content
            is_html: Whether body contains HTML
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.config.email_address
            msg['To'] = to_address
            msg['Subject'] = subject
            
            # Create message part
            if is_html:
                part = MIMEText(body, 'html')
            else:
                part = MIMEText(body, 'plain')
            
            msg.attach(part)
            
            # Connect and send
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.config.email_address, self.config.app_password)
            
            server.sendmail(self.config.email_address, to_address, msg.as_string())
            server.quit()
            
            self.logger.info(f"Email sent successfully to {to_address}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def _decode_header(self, header: str) -> str:
        """Decode email header handling various encodings.
        
        Args:
            header: Raw header string
            
        Returns:
            str: Decoded header string
        """
        if not header:
            return ""
        
        decoded_header = decode_header(header)[0]
        if isinstance(decoded_header[0], bytes):
            return decoded_header[0].decode(decoded_header[1] or 'utf-8')
        return decoded_header[0]
    
    def _extract_body(self, email_message) -> str:
        """Extract text body from email message.
        
        Args:
            email_message: Email message object
            
        Returns:
            str: Email body text
        """
        body = ""
        
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition'))
                
                if content_type == 'text/plain' and 'attachment' not in content_disposition:
                    try:
                        body = part.get_payload(decode=True).decode('utf-8')
                        break
                    except:
                        continue
                elif content_type == 'text/html' and not body and 'attachment' not in content_disposition:
                    try:
                        # Fallback to HTML if no plain text
                        body = part.get_payload(decode=True).decode('utf-8')
                    except:
                        continue
        else:
            try:
                body = email_message.get_payload(decode=True).decode('utf-8')
            except:
                body = str(email_message.get_payload())
        
        return body
    
    def _should_include_email(self, sender: str, subject: str) -> bool:
        """Check if email should be included based on filters.
        
        Args:
            sender: Email sender
            subject: Email subject
            
        Returns:
            bool: True if email should be included
        """
        # Check exclude senders
        if self.config.exclude_senders:
            for excluded in self.config.exclude_senders:
                if excluded.lower() in sender.lower():
                    return False
        
        # Check subject keywords if specified
        if self.config.subject_keywords:
            subject_lower = subject.lower()
            for keyword in self.config.subject_keywords:
                if keyword.lower() in subject_lower:
                    return True
            return False  # If keywords specified but none found
        
        return True