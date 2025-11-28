def get_consent():
    print("\n" + "="*60)
    print("CUSTOMIZE YOUR OSINT SEARCH")
    print("="*60)
    
    permissions = {
        "social_media": False,
        "face_recognition": False, 
        "public_records": False,
        "professional_data": False
    }
    
    print("What would you like to search?")
    print("[1] Social Media Profiles")
    print("[2] Facial Recognition") 
    print("[3] Public Records")
    print("[4] Professional Information")
    print("[5] ALL OF THE ABOVE")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == "1":
        permissions["social_media"] = True
    elif choice == "2":
        permissions["face_recognition"] = True
    elif choice == "3":
        permissions["public_records"] = True  
    elif choice == "4":
        permissions["professional_data"] = True
    elif choice == "5":
        for key in permissions:
            permissions[key] = True
    else:
        print("Invalid choice!")
        return None
    
    print("\nPERMISSIONS GRANTED:")
    for perm, granted in permissions.items():
        if granted:
            print(f"   - {perm.replace('_', ' ').title()}")
    
    confirm = input("\nType 'CONFIRM' to proceed: ")
    if confirm.upper() == "CONFIRM":
        return permissions
    else:
        print("Consent cancelled.")
        return None

# Usage
consent = get_consent()
if consent:
    print("Starting your OSINT search...")
else:
    print("Goodbye!")