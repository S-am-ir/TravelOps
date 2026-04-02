"""
Email service — send emails via SMTP (Gmail app password or any SMTP provider).
No Google Cloud / OAuth required — just an app password.
"""

import smtplib
from email.message import EmailMessage
from typing import Optional


def send_email(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    from_email: str,
    to_email: str,
    subject: str,
    body: str,
) -> dict:
    """
    Send an email via SMTP.
    Returns {"status": "sent", ...} or {"status": "error", "error": "..."}
    """
    try:
        message = EmailMessage()
        message.set_content(body)
        message["To"] = to_email
        message["From"] = from_email
        message["Subject"] = subject

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(smtp_user, smtp_password)
            server.send_message(message)

        return {"status": "sent"}

    except Exception as e:
        return {"status": "error", "error": str(e)}


def send_email_gmail(
    gmail_email: str,
    app_password: str,
    to_email: str,
    subject: str,
    body: str,
) -> dict:
    """
    Send via Gmail SMTP with app password.
    The user generates an app password at:
    myaccount.google.com → Security → 2-Step Verification → App passwords
    """
    return send_email(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        smtp_user=gmail_email,
        smtp_password=app_password,
        from_email=gmail_email,
        to_email=to_email,
        subject=subject,
        body=body,
    )
