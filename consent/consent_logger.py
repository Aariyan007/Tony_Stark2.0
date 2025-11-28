import datetime
import json
from advanced_consent import get_consent

def log_consent(user_info, permissions):
    consent_record = {
        "timestamp": str(datetime.datetime.now()),
        "user_name": user_info.get("name", "anonymous"),
        "permissions_granted": permissions,
        "consent_version": "1.0"
    }
    
    with open('consent_log.json', 'a') as f:
        f.write(json.dumps(consent_record) + '\n')
    
    print("Consent logged successfully!")

def get_user_info():
    print("\nPlease provide basic info:")
    name = input("Your name (optional): ")
    email = input("Your email (optional): ")
    
    return {"name": name, "email": email}

# Main flow
print("OSINT TOOL - CONSENT MANAGEMENT")
user_info = get_user_info()
permissions = get_consent()

if permissions:
    log_consent(user_info, permissions)
    print(" Ready to start OSINT analysis!")
else:
    print("Consent not provided.")