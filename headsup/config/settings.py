import os
from dotenv import load_dotenv

load_dotenv()

SETTINGS = {
    'api_key': os.getenv('VANTAGE_API_KEY'),
    'api_requests_per_min': os.getenv('VANTAGE_RPM'),
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': os.getenv('SENDER_EMAIL'),
        'sender_password': os.getenv('EMAIL_APP_PASSWORD'),
        'recipient_email': os.getenv('RECIPIENT_EMAIL')
    }
}