import requests

BOT_TOKEN = "8424792466:AAFlQBOQdkHA_MdBL14lmOKao-zQoQaB1Jo"
CHAT_ID = "5377934639"

message = "🚨 IMPORTANT ALERT: Crowd density has exceeded safe limits at the monitored location. Immediate attention required."

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

payload = {
    "chat_id": CHAT_ID,
    "text": message
}

requests.post(url, data=payload)
print("Telegram alert sent")
