import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config.settings import SETTINGS


class EmailSender:
    def __init__(self):
        self.settings = SETTINGS['email']

    def send_report(self, html_content, subject: str = 'Daily Heads Up Report'):
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.settings['sender_email']
        msg['To'] = self.settings['recipient_email']

        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)

        try:
            with smtplib.SMTP(self.settings['smtp_server'], self.settings['smtp_port']) as server:
                server.starttls()
                server.login(
                    self.settings['sender_email'],
                    self.settings['sender_password']
                )
                server.send_message(msg)
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
