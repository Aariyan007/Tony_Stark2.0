print("=" * 50)
print("OSINT TOOL - CONSENT REQUIRED")
print("=" * 50)

with open('consent_form.txt', 'r') as f:
    print(f.read())

user_input = input("Do you agree? Type 'I AGREE': ")

if user_input.upper() == "I AGREE":
    print("CONSENT GRANTED - Starting OSINT analysis...")
else:
    print("CONSENT DENIED - Exiting tool.")
    exit()