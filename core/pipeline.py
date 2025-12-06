# core/pipeline.py - PRODUCTION VERSION

from deepface import DeepFace
from typing import Dict, Any, Optional, List

from core.storage import save_embedding
from modules.social_media.google_search import google_search_name
from modules.social_media.username_scan import scan_usernames
from modules.social_media.instagram import scrape_instagram
from modules.social_media.github import scrape_github
from modules.social_media.linkedin_inference import find_linkedin_profile
from modules.social_media.linkedin import scrape_linkedin_profile

from modules.graph.neo4j_client import Neo4jClient
from modules.osint.email_discovery import EmailDiscovery


def process_image(image_path: str, person_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    PRODUCTION OSINT Pipeline with:
    - Face analysis & embedding
    - Social media discovery
    - Email discovery
    - Location & organization extraction
    - Face similarity search
    - Network inference
    
    Args:
        image_path: Path to image file
        person_context: Known information about the person (optional)
    
    Returns:
        Complete OSINT profile
    """
    print("\n" + "="*70)
    print("PRODUCTION OSINT PIPELINE")
    print("="*70)
    
    if person_context is None:
        person_context = {
            "full_name": "Aariyan S",
            "known_usernames": ["aariyan007", "aariyan_07", "aariyan"],
            "locations": ["Kerala", "India", "Ernakulam"],
            "keywords": [
                "computer science",
                "developer",
                "CSE",
                "Muthoot Institute of Technology and Science"
            ]
        }

    faces = _extract_face_data(image_path)

    search_results = _perform_osint_search(person_context)

    usernames = _scan_social_platforms(person_context["known_usernames"])

    # Step 4: Scrape Instagram with face matching
    instagram_result = scrape_instagram("aariyan_07", image_path)

    # Step 5: Scrape GitHub profile
    github_info = scrape_github("aariyan007")

    # Step 6: Find and scrape LinkedIn profile
    linkedin_best, linkedin_profile_data = _discover_linkedin(person_context)
    
    # Step 7: Extract locations and organizations
    locations, organizations = _extract_entities(
        linkedin_profile_data,
        github_info,
        person_context
    )
    
    # Step 8: Discover email addresses
    emails = _discover_emails(
        person_context["full_name"],
        person_context["known_usernames"],
        [instagram_result, github_info, linkedin_profile_data],
        [org["name"] for org in organizations]
    )

    # Step 9: Push everything to Neo4j
    person_id = _push_to_neo4j(
        person_context=person_context,
        faces=faces,
        instagram_result=instagram_result,
        github_info=github_info,
        linkedin_inferred=linkedin_best,
        linkedin_scraped=linkedin_profile_data,
        locations=locations,
        organizations=organizations,
        emails=emails,
    )
    
    
    neo = Neo4jClient()
    try:

        print("\n[SIMILARITY] Finding similar faces...")
        similar_faces = neo.find_similar_faces(person_id, similarity_threshold=0.7, limit=5)
        if similar_faces:
            print(f"[SIMILARITY] ✓ Found {len(similar_faces)} similar faces")
            for face in similar_faces:
                print(f"  - {face['name']}: {face['similarity']:.2%} similar")
        else:
            print("[SIMILARITY] No similar faces found")
        

        print("\n[NETWORK] Inferring connections...")
        connections = neo.infer_connections(person_id, min_connection_strength=2)
        if connections:
            print(f"[NETWORK] Found {len(connections)} potential connections")
            for conn in connections:
                print(f"  - {conn['name']}: connection strength {conn['connection_strength']}")
        else:
            print("[NETWORK] No connections found")
        

        print("\n[QUALITY] Calculating data completeness...")
        completeness = neo.update_data_completeness(person_id)
        
        stats = neo.get_statistics()
        print("\n[DATABASE] Statistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
    finally:
        neo.close()

    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)

    return {
        "person_id": person_id,
        "faces": faces["faces"],
        "analysis": faces["analysis"],
        "embedding": faces["embedding"],
        "search": search_results,
        "usernames": usernames,
        "instagram": instagram_result,
        "github": github_info,
        "linkedin": {
            "inferred": linkedin_best,
            "scraped": linkedin_profile_data,
        },
        "locations": locations,
        "organizations": organizations,
        "emails": emails,
        "similar_faces": similar_faces if 'similar_faces' in locals() else [],
        "connections": connections if 'connections' in locals() else [],
        "data_completeness": completeness if 'completeness' in locals() else 0.0,
    }


def _extract_face_data(image_path: str) -> Dict[str, Any]:
    """Extract face features with enhanced error handling"""
    print("\n[FACE] Extracting facial features...")

    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend="retinaface",
            enforce_detection=False,
        )
        print(f"[FACE] Detected {len(faces)} face(s)")

        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=["age", "gender", "emotion", "race"],
            detector_backend="retinaface",
            enforce_detection=False,
        )
        
        if analysis:
            print(f"[FACE] ✓ Analysis complete:")
            print(f"  - Age: {analysis[0].get('age')}")
            print(f"  - Gender: {analysis[0].get('dominant_gender')}")
            print(f"  - Emotion: {analysis[0].get('dominant_emotion')}")

        embedding = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512",
            detector_backend="retinaface",
            enforce_detection=False,
        )
        print(f"[FACE] Embedding created: {len(embedding[0]['embedding'])} dimensions")

        save_embedding("user_test", embedding)

        return {
            "faces": faces,
            "analysis": analysis,
            "embedding": embedding,
        }
    
    except Exception as e:
        print(f"[FACE] Face extraction failed: {e}")
        raise


def _perform_osint_search(context: Dict[str, Any]) -> List[Dict]:
    """Perform Google search with context"""
    print("\n[OSINT] Performing Google search...")

    query_parts = [context["full_name"]]
    query_parts.extend(context.get("locations", [])[:2])
    query_parts.extend(context.get("keywords", [])[:3])
    query_parts.append("LinkedIn Github Instagram")
    
    query = " ".join(query_parts)
    
    try:
        search_results = google_search_name(query)
        print(f"[OSINT] ✓ Found {len(search_results)} results")

        SOCIAL_KEYWORDS = ["linkedin", "instagram", "github", "facebook", "twitter", "youtube"]
        social_results = []
        
        print("\n[OSINT] Social media results:")
        for r in search_results:
            link = (r.get("link") or "").lower()
            if any(key in link for key in SOCIAL_KEYWORDS):
                social_results.append(r)
                print(f"  - {r.get('title', 'N/A')[:60]}")
        
        print(f"[OSINT] Found {len(social_results)} social media links")

        return search_results
    
    except Exception as e:
        print(f"[OSINT] Search failed: {e}")
        return []


def _scan_social_platforms(usernames: List[str]) -> List[Dict]:
    """Scan usernames across platforms"""
    print("\n[SCAN] Scanning usernames across platforms...")

    all_results = []
    
    for username in usernames:
        try:
            results = scan_usernames(username)
            all_results.extend(results)
            print(f"[SCAN] '{username}': found {len(results)} accounts")
        
        except Exception as e:
            print(f"[SCAN] Failed to scan '{username}': {e}")

    return all_results


def _discover_linkedin(context: Dict[str, Any]) -> tuple:
    """Find and scrape LinkedIn profile"""
    print("\n[LINKEDIN] Searching for profile...")

    try:
        best = find_linkedin_profile(
            full_name=context["full_name"],
            usernames=context.get("known_usernames", []),
            locations=context.get("locations", []),
            keywords=context.get("keywords", []),
        )

        if not best:
            print("[LINKEDIN] ✗ No reliable match found")
            return None, None

        print(f"[LINKEDIN] Best candidate found (score: {best.get('score')}/100)")
        

        linkedin_url = best.get("link")
        if linkedin_url:
            profile_data = scrape_linkedin_profile(linkedin_url)
            return best, profile_data
        
        return best, None
    
    except Exception as e:
        print(f"[LINKEDIN] ✗ Search failed: {e}")
        return None, None


def _extract_entities(
    linkedin_data: Optional[Dict],
    github_data: Optional[Dict],
    context: Dict[str, Any]
) -> tuple:
    """Extract locations and organizations from profiles"""
    print("\n[ENTITIES] Extracting locations and organizations...")
    
    locations = []
    organizations = []
    
    for loc in context.get("locations", []):
        locations.append({
            "name": loc,
            "source": "context",
            "confidence": 80.0
        })
    
    if linkedin_data:

        if linkedin_data.get("location"):
            locations.append({
                "name": linkedin_data["location"],
                "source": "linkedin",
                "confidence": 90.0
            })
        
        if linkedin_data.get("experiences"):
            for exp in linkedin_data["experiences"]:
                if exp.get("company"):
                    organizations.append({
                        "name": exp["company"],
                        "type": "company",
                        "relationship": "WORKS_AT",
                        "role": exp.get("title"),
                        "source": "linkedin_experience"
                    })
        
        if linkedin_data.get("education"):
            for edu in linkedin_data["education"]:
                if edu.get("school"):
                    organizations.append({
                        "name": edu["school"],
                        "type": "educational",
                        "relationship": "STUDIES_AT",
                        "role": edu.get("degree"),
                        "source": "linkedin_education"
                    })
    

    if github_data and github_data.get("location"):
        locations.append({
            "name": github_data["location"],
            "source": "github",
            "confidence": 85.0
        })
    
    unique_locations = {loc["name"]: loc for loc in locations}
    locations = list(unique_locations.values())
    
    unique_orgs = {org["name"]: org for org in organizations}
    organizations = list(unique_orgs.values())
    
    print(f"[ENTITIES] Found {len(locations)} locations, {len(organizations)} organizations")
    
    return locations, organizations


def _discover_emails(
    full_name: str,
    usernames: List[str],
    accounts: List[Optional[Dict]],
    organizations: List[str]
) -> List[Dict[str, Any]]:
    """Discover potential email addresses"""
    print("\n[EMAIL] Discovering email addresses...")
    
    valid_accounts = [acc for acc in accounts if acc is not None]
    
    discovery = EmailDiscovery()
    
    emails = discovery.discover_emails(
        full_name=full_name,
        usernames=usernames,
        accounts=valid_accounts,
        organizations=organizations
    )
    
    high_confidence_emails = [e for e in emails if e["confidence"] >= 50.0]
    
    print(f"[EMAIL] Discovered {len(emails)} potential emails")
    print(f"[EMAIL] {len(high_confidence_emails)} high-confidence emails")
    

    for email in high_confidence_emails[:5]:
        print(f"  - {email['address']} ({email['confidence']:.1f}% confidence)")
    
    return emails


def _push_to_neo4j(
    person_context: Dict[str, Any],
    faces: Dict[str, Any],
    instagram_result: Optional[Dict] = None,
    github_info: Optional[Dict] = None,
    linkedin_inferred: Optional[Dict] = None,
    linkedin_scraped: Optional[Dict] = None,
    locations: Optional[List[Dict]] = None,
    organizations: Optional[List[Dict]] = None,
    emails: Optional[List[Dict]] = None,
) -> str:
    """Push all data to Neo4j graph database"""
    print("\n[NEO4J] Pushing data to graph database...")
    
    person_id = person_context["full_name"].lower().replace(" ", "_")
    person_name = person_context["full_name"]

    embedding_list = faces.get("embedding") or []
    if not embedding_list:
        print("[NEO4J] No embedding found, skipping")
        return person_id

    emb_vec = embedding_list[0]["embedding"]
    analysis = faces.get("analysis", [{}])[0]
    age = analysis.get("age")
    gender = analysis.get("dominant_gender")

    neo = Neo4jClient()

    try:
        neo.upsert_person_with_embedding(
            person_id=person_id,
            name=person_name,
            embedding_vec=emb_vec,
            model="Facenet512",
            age=age,
            gender=gender,
        )
        print("[NEO4J] Person node created")

        if instagram_result:
            face_match = instagram_result.get("face_match_result") or {}
            confidence = face_match.get("confidence", 0.0)
            verified = face_match.get("verified", False)

            neo.link_account(
                person_id=person_id,
                platform="instagram",
                username=instagram_result.get("username"),
                url=instagram_result.get("url"),
                score=confidence,
                verified=verified,
                source="instagram_face_match",
                metadata={"face_match": face_match}
            )
            print(f"[NEO4J] Instagram linked (confidence: {confidence:.1f}%)")

        if github_info:
            neo.link_account(
                person_id=person_id,
                platform="github",
                username=github_info.get("username"),
                url=github_info.get("url"),
                display_name=github_info.get("name"),
                bio=github_info.get("bio"),
                followers=github_info.get("followers"),
                score=60.0,
                source="github_scrape",
                metadata={"public_repos": github_info.get("public_repos")}
            )
            print("[NEO4J] GitHub linked")

        if linkedin_scraped:
            linkedin_score = linkedin_inferred.get("score", 50) if linkedin_inferred else 50
            neo.link_account(
                person_id=person_id,
                platform="linkedin",
                url=linkedin_scraped.get("url"),
                display_name=linkedin_scraped.get("name"),
                bio=linkedin_scraped.get("headline"),
                score=linkedin_score,
                source="linkedin_scrape",
                metadata={
                    "location": linkedin_scraped.get("location"),
                    "about": linkedin_scraped.get("about"),
                }
            )
            print(f"[NEO4J] ✓ LinkedIn linked (score: {linkedin_score})")
        
        if locations:
            for loc in locations:
                neo.link_location(
                    person_id=person_id,
                    location=loc["name"],
                    location_type="residence",
                    confidence=loc.get("confidence", 0.0)
                )
            print(f"[NEO4J] ✓ {len(locations)} locations linked")
        
        if organizations:
            for org in organizations:
                neo.link_organization(
                    person_id=person_id,
                    org_name=org["name"],
                    relationship_type=org.get("relationship", "ASSOCIATED_WITH"),
                    role=org.get("role"),
                    org_type=org.get("type")
                )
            print(f"[NEO4J] {len(organizations)} organizations linked")
        
        if emails:
            high_conf_emails = [e for e in emails if e["confidence"] >= 60.0]
            for email in high_conf_emails:
                neo.link_email(
                    person_id=person_id,
                    email_address=email["address"],
                    email_type=email.get("type", "personal"),
                    verified=email.get("verified", False),
                    source=email["source"],
                    confidence=email["confidence"]
                )
            print(f"[NEO4J] {len(high_conf_emails)} emails linked")

        print("[NEO4J] All data pushed successfully")
        
        return person_id

    except Exception as e:
        print(f"[NEO4J] Failed to push data: {e}")
        return person_id
    
    finally:
        neo.close()