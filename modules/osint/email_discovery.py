# modules/osint/email_discovery.py

import re
from typing import List, Dict, Optional, Any
import requests
from urllib.parse import urlparse


class EmailDiscovery:
    """Discover and verify email addresses for a person"""
    
    # Common email domains
    COMMON_DOMAINS = [
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "protonmail.com",
        "icloud.com",
    ]
    
    def __init__(self):
        self.discovered_emails = []
    
    def discover_emails(
        self,
        full_name: str,
        usernames: List[str],
        accounts: List[Dict[str, Any]],
        organizations: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover potential email addresses for a person.
        
        Args:
            full_name: Person's full name
            usernames: List of known usernames
            accounts: List of social media accounts
            organizations: List of organizations (for work emails)
        
        Returns:
            List of potential emails with confidence scores
        """
        emails = []
        
        # 1. Extract emails from bios/profiles
        emails.extend(self._extract_from_profiles(accounts))
        
        # 2. Generate pattern-based emails
        emails.extend(self._generate_pattern_emails(full_name, usernames, organizations))
        
        # 3. Extract from GitHub commits (if GitHub account exists)
        for account in accounts:
            if account.get("platform") == "github" and account.get("username"):
                github_emails = self._extract_from_github(account["username"])
                emails.extend(github_emails)
        
        # Remove duplicates
        unique_emails = self._deduplicate_emails(emails)
        
        # Sort by confidence
        unique_emails.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_emails
    
    def _extract_from_profiles(self, accounts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract emails from social media bios/descriptions"""
        emails = []
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        for account in accounts:
            bio = account.get("bio", "") or ""
            
            # Find all email addresses in bio
            found_emails = re.findall(email_pattern, bio)
            
            for email in found_emails:
                emails.append({
                    "address": email.lower(),
                    "source": f"{account.get('platform')}_bio",
                    "confidence": 90.0,  # High confidence - directly found
                    "verified": False,
                    "type": "personal"
                })
        
        return emails
    
    def _generate_pattern_emails(
        self,
        full_name: str,
        usernames: List[str],
        organizations: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Generate potential email addresses based on patterns"""
        emails = []
        
        # Parse name
        name_parts = full_name.lower().strip().split()
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = name_parts[-1]
        else:
            first_name = name_parts[0] if name_parts else ""
            last_name = ""
        
        # Common email patterns
        patterns = []
        
        if first_name and last_name:
            patterns.extend([
                f"{first_name}.{last_name}",
                f"{first_name}{last_name}",
                f"{first_name}_{last_name}",
                f"{first_name[0]}{last_name}",
                f"{first_name}{last_name[0]}",
            ])
        elif first_name:
            patterns.append(first_name)
        
        # Add usernames as patterns
        patterns.extend([u.lower() for u in usernames])
        
        # Generate emails for common domains
        for pattern in patterns:
            for domain in self.COMMON_DOMAINS:
                emails.append({
                    "address": f"{pattern}@{domain}",
                    "source": "pattern_generation",
                    "confidence": 30.0,  # Lower confidence - generated
                    "verified": False,
                    "type": "personal"
                })
        
        # Generate work emails if organizations are known
        if organizations:
            for org in organizations:
                # Try to extract domain from organization name
                domain = self._org_to_domain(org)
                
                if domain:
                    for pattern in patterns[:3]:  # Only top 3 patterns for work emails
                        emails.append({
                            "address": f"{pattern}@{domain}",
                            "source": "work_pattern_generation",
                            "confidence": 40.0,
                            "verified": False,
                            "type": "work"
                        })
        
        return emails
    
    def _extract_from_github(self, username: str) -> List[Dict[str, Any]]:
        """Extract email from GitHub commits"""
        emails = []
        
        try:
            # Get user's recent events (includes commit info)
            url = f"https://api.github.com/users/{username}/events/public"
            headers = {"Accept": "application/vnd.github.v3+json"}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                events = response.json()
                
                # Look for PushEvents (commits)
                for event in events[:20]:  # Check last 20 events
                    if event.get("type") == "PushEvent":
                        commits = event.get("payload", {}).get("commits", [])
                        
                        for commit in commits:
                            author_email = commit.get("author", {}).get("email")
                            
                            if author_email and not author_email.endswith("@users.noreply.github.com"):
                                emails.append({
                                    "address": author_email.lower(),
                                    "source": "github_commits",
                                    "confidence": 85.0,  # High confidence
                                    "verified": True,  # Verified through GitHub
                                    "type": "personal"
                                })
            
        except Exception as e:
            print(f"[Email] Failed to extract from GitHub: {e}")
        
        return emails
    
    def _org_to_domain(self, org_name: str) -> Optional[str]:
        """Convert organization name to likely domain"""
        # Clean organization name
        org_clean = org_name.lower().strip()
        
        # Remove common suffixes
        suffixes = [
            " inc", " incorporated", " llc", " ltd", " limited",
            " corporation", " corp", " company", " co",
            " pvt", " private", " institute", " university"
        ]
        
        for suffix in suffixes:
            org_clean = org_clean.replace(suffix, "")
        
        # Remove special characters
        org_clean = re.sub(r'[^a-z0-9\s]', '', org_clean)
        
        # Convert spaces to nothing or dash
        org_clean = org_clean.replace(" ", "")
        
        # Common domain mappings
        domain_map = {
            "google": "google.com",
            "microsoft": "microsoft.com",
            "apple": "apple.com",
            "amazon": "amazon.com",
            "facebook": "fb.com",
            "meta": "meta.com",
            "twitter": "twitter.com",
        }
        
        if org_clean in domain_map:
            return domain_map[org_clean]
        
        # Otherwise, assume .com
        if org_clean:
            return f"{org_clean}.com"
        
        return None
    
    def _deduplicate_emails(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate emails, keeping highest confidence"""
        email_dict = {}
        
        for email in emails:
            address = email["address"]
            
            if address not in email_dict:
                email_dict[address] = email
            else:
                # Keep email with higher confidence
                if email["confidence"] > email_dict[address]["confidence"]:
                    email_dict[address] = email
                # If same confidence but one is verified, keep verified
                elif email["confidence"] == email_dict[address]["confidence"] and email["verified"]:
                    email_dict[address] = email
        
        return list(email_dict.values())
    
    def validate_email_format(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$'
        return bool(re.match(pattern, email))
    
    def check_email_deliverability(self, email: str) -> Dict[str, Any]:
        """
        Check if email is likely to be deliverable (basic checks).
        
        For production, integrate with:
        - Hunter.io Email Verifier API
        - EmailRep.io
        - ZeroBounce
        """
        result = {
            "email": email,
            "valid_format": self.validate_email_format(email),
            "deliverable": "unknown",
            "disposable": False,
            "free_provider": False,
        }
        
        if not result["valid_format"]:
            result["deliverable"] = "invalid"
            return result
        
        # Check if free provider
        domain = email.split("@")[1].lower()
        free_providers = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com"]
        result["free_provider"] = domain in free_providers
        
        # Check for disposable email domains
        disposable_domains = ["tempmail.com", "10minutemail.com", "guerrillamail.com", "throwaway.email"]
        result["disposable"] = domain in disposable_domains
        
        if result["disposable"]:
            result["deliverable"] = "invalid"
        
        return result


# Example usage
if __name__ == "__main__":
    discovery = EmailDiscovery()
    
    # Sample data
    full_name = "Aariyan S"
    usernames = ["aariyan007", "aariyan_07", "aariyan"]
    accounts = [
        {
            "platform": "github",
            "username": "aariyan007",
            "bio": "CSE student | Contact: aariyan@example.com"
        },
        {
            "platform": "instagram",
            "username": "aariyan_07",
            "bio": "Just a guy living like 8GB RAM running 58 Chrome tabs"
        }
    ]
    organizations = ["Muthoot Institute of Technology and Science"]
    
    # Discover emails
    emails = discovery.discover_emails(full_name, usernames, accounts, organizations)
    
    print("\nDiscovered Email Addresses:")
    print("=" * 70)
    for email in emails[:10]:  # Show top 10
        print(f"{email['address']}")
        print(f"  Source: {email['source']}")
        print(f"  Confidence: {email['confidence']:.1f}%")
        print(f"  Type: {email['type']}")
        print()