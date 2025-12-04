# main.py

import json
from datetime import datetime
from pathlib import Path

from core.pipeline import process_image


def save_results(results: dict, person_id: str):
    """Save pipeline results to JSON file"""
    output_dir = Path("data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"{person_id}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[MAIN] Results saved to: {filename}")


def print_summary(results: dict):
    """Print a nice summary of the results"""
    print("\n" + "="*70)
    print("OSINT PIPELINE RESULTS SUMMARY")
    print("="*70)
    
    # Face Analysis
    if results.get("analysis"):
        analysis = results["analysis"][0]
        print("\n📸 FACE ANALYSIS:")
        print(f"  Age:     {analysis.get('age', 'N/A')}")
        print(f"  Gender:  {analysis.get('dominant_gender', 'N/A')}")
        print(f"  Emotion: {analysis.get('dominant_emotion', 'N/A')}")
        print(f"  Race:    {analysis.get('dominant_race', 'N/A')}")
    
    # Instagram
    if results.get("instagram"):
        insta = results["instagram"]
        face_match = insta.get("face_match_result", {})
        
        print("\n📱 INSTAGRAM:")
        print(f"  Username: {insta.get('username')}")
        print(f"  URL:      {insta.get('url')}")
        
        if face_match:
            verified = face_match.get("verified", False)
            confidence = face_match.get("confidence", 0)
            status = "✓ VERIFIED" if verified else "✗ Not verified"
            print(f"  Match:    {status} ({confidence:.2f}% confidence)")
        else:
            print(f"  Match:    Failed to perform face matching")
    
    # GitHub
    if results.get("github"):
        github = results["github"]
        print("\n💻 GITHUB:")
        print(f"  Username:  {github.get('username')}")
        print(f"  Name:      {github.get('name')}")
        print(f"  Bio:       {github.get('bio')}")
        print(f"  Followers: {github.get('followers')}")
        print(f"  Repos:     {github.get('public_repos')}")
    
    # LinkedIn
    linkedin_data = results.get("linkedin", {})
    
    if linkedin_data.get("inferred"):
        inferred = linkedin_data["inferred"]
        print("\n💼 LINKEDIN (Search Result):")
        print(f"  URL:   {inferred.get('link')}")
        print(f"  Title: {inferred.get('title')}")
        print(f"  Score: {inferred.get('score')}/100")
    
    if linkedin_data.get("scraped"):
        scraped = linkedin_data["scraped"]
        print("\n💼 LINKEDIN (Scraped Profile):")
        print(f"  Name:     {scraped.get('name')}")
        print(f"  Headline: {scraped.get('headline')}")
        print(f"  Location: {scraped.get('location')}")
        
        if scraped.get("experiences"):
            print(f"  Experience: {len(scraped['experiences'])} positions")
            for exp in scraped["experiences"][:3]:
                print(f"    - {exp.get('title')} at {exp.get('company')}")
    
    # Search Results
    if results.get("search"):
        print(f"\n🔍 GOOGLE SEARCH:")
        print(f"  Found {len(results['search'])} results")
        
        # Count social media links
        social_count = 0
        for result in results["search"]:
            link = result.get("link", "").lower()
            if any(platform in link for platform in ["linkedin", "github", "instagram", "twitter", "facebook"]):
                social_count += 1
        
        print(f"  Social media links: {social_count}")
    
    # Username Scan
    if results.get("usernames"):
        print(f"\n👤 USERNAME SCAN:")
        print(f"  Found {len(results['usernames'])} potential accounts")
        
        # Group by platform
        platforms = {}
        for user in results["usernames"]:
            platform = user.get("platform")
            if platform not in platforms:
                platforms[platform] = 0
            platforms[platform] += 1
        
        for platform, count in sorted(platforms.items()):
            print(f"  - {platform.upper()}: {count}")
    
    print("\n" + "="*70)


def main():
    """Main execution function"""
    print("="*70)
    print("FACE-BASED OSINT RECONNAISSANCE PIPELINE")
    print("="*70)
    
    # Image to process
    image_path = "test.jpg"
    
    print(f"\n[MAIN] Processing image: {image_path}")
    print(f"[MAIN] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run pipeline
    try:
        results = process_image(image_path)
        
        # Print summary
        print_summary(results)
        
        # Save results
        save_results(results, person_id="user_test")
        
        # Embedding info
        if results.get("embedding"):
            embedding_len = len(results["embedding"][0]["embedding"])
            print(f"\n[MAIN] ✓ Face embedding created: {embedding_len} dimensions")
        
        print(f"\n[MAIN] ✓ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n[MAIN] ✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n[MAIN] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()