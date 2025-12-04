# modules/graph/neo4j_client.py - PRODUCTION VERSION

from neo4j import GraphDatabase
from datetime import datetime
import json
import math
from typing import List, Dict, Optional, Any, Tuple

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class Neo4jClient:
    """Production-ready Neo4j client with advanced OSINT features"""
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints()

    def close(self):
        self.driver.close()

    def _create_constraints(self):
        """Create unique constraints and indexes"""
        with self.driver.session() as session:
            try:
                # Unique constraints
                session.run("CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.person_id IS UNIQUE")
                session.run("CREATE CONSTRAINT account_url_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.url IS UNIQUE")
                session.run("CREATE CONSTRAINT embedding_id_unique IF NOT EXISTS FOR (e:FaceEmbedding) REQUIRE e.embedding_id IS UNIQUE")
                session.run("CREATE CONSTRAINT location_name_unique IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE")
                session.run("CREATE CONSTRAINT org_name_unique IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE")
                session.run("CREATE CONSTRAINT email_address_unique IF NOT EXISTS FOR (e:Email) REQUIRE e.address IS UNIQUE")
                
                # Indexes for search
                session.run("CREATE INDEX person_name_idx IF NOT EXISTS FOR (p:Person) ON (p.name)")
                session.run("CREATE INDEX account_username_idx IF NOT EXISTS FOR (a:Account) ON (a.username)")
                session.run("CREATE INDEX account_platform_idx IF NOT EXISTS FOR (a:Account) ON (a.platform)")
                session.run("CREATE INDEX email_address_idx IF NOT EXISTS FOR (e:Email) ON (e.address)")
            except Exception as e:
                print(f"[Neo4j] Warning: Could not create constraints/indexes: {e}")

    # ============================================================================
    # PERSON + EMBEDDING OPERATIONS
    # ============================================================================

    def upsert_person_with_embedding(
        self, 
        person_id: str, 
        name: Optional[str], 
        embedding_vec: List[float], 
        model: str = "Facenet512",
        age: Optional[int] = None,
        gender: Optional[str] = None,
        **additional_props
    ):
        """Create or update Person node with face embedding"""
        if hasattr(embedding_vec, "tolist"):
            embedding_vec = embedding_vec.tolist()

        vector_str = json.dumps(embedding_vec)

        with self.driver.session() as session:
            session.execute_write(
                self._upsert_person_with_embedding_tx,
                person_id, 
                name, 
                vector_str, 
                model,
                age,
                gender,
                additional_props
            )

    @staticmethod
    def _upsert_person_with_embedding_tx(tx, person_id, name, vector_str, model, age, gender, additional_props):
        props_query = ""
        params = {
            "person_id": person_id,
            "name": name,
            "vector": vector_str,
            "model": model,
            "age": age,
            "gender": gender,
        }
        
        for key, value in additional_props.items():
            params[f"prop_{key}"] = value
            props_query += f", p.{key} = ${f'prop_{key}'}"
        
        query = f"""
        MERGE (p:Person {{person_id: $person_id}})
        ON CREATE SET 
            p.name = $name,
            p.age = $age,
            p.gender = $gender,
            p.created_at = datetime(),
            p.last_seen = datetime(),
            p.data_completeness = 0.0,
            p.verification_level = 'unverified'
            {props_query}
        ON MATCH SET 
            p.name = coalesce($name, p.name),
            p.age = coalesce($age, p.age),
            p.gender = coalesce($gender, p.gender),
            p.last_seen = datetime()
            {props_query}
        
        MERGE (e:FaceEmbedding {{embedding_id: $person_id + '_' + $model}})
        ON CREATE SET 
            e.vector = $vector,
            e.model = $model,
            e.dimension = 512,
            e.created_at = datetime()
        ON MATCH SET 
            e.vector = $vector,
            e.last_used = datetime()
        
        MERGE (p)-[:HAS_FACE]->(e)
        """
        
        tx.run(query, **params)

    # ============================================================================
    # ACCOUNT OPERATIONS
    # ============================================================================

    def link_account(
        self,
        person_id: str,
        platform: str,
        url: str,
        username: Optional[str] = None,
        display_name: Optional[str] = None,
        bio: Optional[str] = None,
        followers: Optional[int] = None,
        score: float = 0.0,
        verified: bool = False,
        source: str = "scan",
        metadata: Optional[Dict] = None,
    ):
        """Link a social media account to a person"""
        with self.driver.session() as session:
            session.execute_write(
                self._link_account_tx,
                person_id,
                platform,
                url,
                username,
                display_name,
                bio,
                followers,
                score,
                verified,
                source,
                metadata or {},
            )

    @staticmethod
    def _link_account_tx(
        tx,
        person_id,
        platform,
        url,
        username,
        display_name,
        bio,
        followers,
        score,
        verified,
        source,
        metadata,
    ):
        query = """
        MATCH (p:Person {person_id: $person_id})
        
        MERGE (a:Account {url: $url})
        ON CREATE SET 
            a.platform = $platform,
            a.username = $username,
            a.display_name = $display_name,
            a.bio = $bio,
            a.followers = $followers,
            a.verified = $verified,
            a.metadata = $metadata,
            a.created_at = datetime(),
            a.last_seen = datetime()
        ON MATCH SET 
            a.username = coalesce($username, a.username),
            a.display_name = coalesce($display_name, a.display_name),
            a.bio = coalesce($bio, a.bio),
            a.followers = coalesce($followers, a.followers),
            a.verified = CASE WHEN $verified THEN true ELSE a.verified END,
            a.metadata = $metadata,
            a.last_seen = datetime()
        
        MERGE (p)-[r:HAS_ACCOUNT]->(a)
        ON CREATE SET 
            r.confidence_score = $score,
            r.source = $source,
            r.verified = $verified,
            r.created_at = datetime(),
            r.last_updated = datetime()
        ON MATCH SET 
            r.confidence_score = CASE 
                WHEN $score > r.confidence_score THEN $score 
                ELSE r.confidence_score 
            END,
            r.source = coalesce($source, r.source),
            r.verified = CASE WHEN $verified THEN true ELSE r.verified END,
            r.last_updated = datetime()
        
        RETURN a, r
        """
        
        tx.run(
            query,
            person_id=person_id,
            platform=platform,
            url=url,
            username=username,
            display_name=display_name,
            bio=bio,
            followers=followers,
            score=score,
            verified=verified,
            source=source,
            metadata=json.dumps(metadata) if metadata else "{}",
        )

    # ============================================================================
    # LOCATION OPERATIONS (NEW)
    # ============================================================================

    def link_location(
        self, 
        person_id: str, 
        location: str, 
        location_type: str = "residence",
        country: Optional[str] = None,
        confidence: float = 0.0
    ):
        """Link person to a location"""
        with self.driver.session() as session:
            query = """
            MATCH (p:Person {person_id: $person_id})
            MERGE (l:Location {name: $location})
            ON CREATE SET 
                l.country = $country,
                l.created_at = datetime()
            MERGE (p)-[r:LOCATED_IN]->(l)
            ON CREATE SET 
                r.type = $location_type,
                r.confidence = $confidence,
                r.created_at = datetime()
            ON MATCH SET 
                r.last_seen = datetime()
            RETURN l
            """
            session.run(
                query, 
                person_id=person_id, 
                location=location, 
                location_type=location_type,
                country=country,
                confidence=confidence
            )

    # ============================================================================
    # ORGANIZATION OPERATIONS (NEW)
    # ============================================================================

    def link_organization(
        self,
        person_id: str,
        org_name: str,
        relationship_type: str = "WORKS_AT",
        role: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        org_type: Optional[str] = None,
    ):
        """Link person to organization (company, school, etc.)"""
        with self.driver.session() as session:
            query = f"""
            MATCH (p:Person {{person_id: $person_id}})
            MERGE (o:Organization {{name: $org_name}})
            ON CREATE SET 
                o.type = $org_type,
                o.created_at = datetime()
            MERGE (p)-[r:{relationship_type}]->(o)
            ON CREATE SET 
                r.role = $role,
                r.start_date = $start_date,
                r.end_date = $end_date,
                r.current = CASE WHEN $end_date IS NULL THEN true ELSE false END,
                r.created_at = datetime()
            ON MATCH SET 
                r.role = coalesce($role, r.role),
                r.end_date = coalesce($end_date, r.end_date),
                r.current = CASE WHEN $end_date IS NULL THEN true ELSE false END,
                r.last_seen = datetime()
            RETURN o, r
            """
            
            session.run(
                query,
                person_id=person_id,
                org_name=org_name,
                role=role,
                start_date=start_date,
                end_date=end_date,
                org_type=org_type,
            )

    # ============================================================================
    # EMAIL OPERATIONS (NEW)
    # ============================================================================

    def link_email(
        self,
        person_id: str,
        email_address: str,
        email_type: str = "personal",
        verified: bool = False,
        source: str = "inferred",
        confidence: float = 0.0
    ):
        """Link an email address to a person"""
        with self.driver.session() as session:
            query = """
            MATCH (p:Person {person_id: $person_id})
            MERGE (e:Email {address: $email_address})
            ON CREATE SET 
                e.type = $email_type,
                e.verified = $verified,
                e.created_at = datetime()
            MERGE (p)-[r:HAS_EMAIL]->(e)
            ON CREATE SET 
                r.source = $source,
                r.confidence = $confidence,
                r.verified = $verified,
                r.created_at = datetime()
            ON MATCH SET
                r.verified = CASE WHEN $verified THEN true ELSE r.verified END,
                r.confidence = CASE 
                    WHEN $confidence > r.confidence THEN $confidence 
                    ELSE r.confidence 
                END,
                r.last_seen = datetime()
            RETURN e, r
            """
            session.run(
                query,
                person_id=person_id,
                email_address=email_address,
                email_type=email_type,
                verified=verified,
                source=source,
                confidence=confidence
            )

    # ============================================================================
    # FACE SIMILARITY SEARCH (NEW)
    # ============================================================================

    def find_similar_faces(
        self, 
        person_id: str, 
        similarity_threshold: float = 0.7, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find people with similar face embeddings using cosine similarity.
        
        Args:
            person_id: Person to find similar faces for
            similarity_threshold: Minimum cosine similarity (0-1, higher is more similar)
            limit: Maximum number of results
        
        Returns:
            List of similar people with similarity scores
        """
        with self.driver.session() as session:
            # Get the target person's embedding
            query_embedding = """
            MATCH (p:Person {person_id: $person_id})-[:HAS_FACE]->(e:FaceEmbedding)
            RETURN e.vector AS vector
            """
            result = session.run(query_embedding, person_id=person_id)
            record = result.single()
            
            if not record:
                print(f"[Neo4j] No embedding found for person {person_id}")
                return []
            
            target_vector = json.loads(record["vector"])
            
            # Get all other embeddings and calculate similarity
            query_all = """
            MATCH (p:Person)-[:HAS_FACE]->(e:FaceEmbedding)
            WHERE p.person_id <> $person_id
            RETURN p.person_id AS person_id, 
                   p.name AS name,
                   e.vector AS vector
            """
            results = session.run(query_all, person_id=person_id)
            
            similar_people = []
            
            for rec in results:
                other_vector = json.loads(rec["vector"])
                similarity = self._cosine_similarity(target_vector, other_vector)
                
                if similarity >= similarity_threshold:
                    similar_people.append({
                        "person_id": rec["person_id"],
                        "name": rec["name"],
                        "similarity": similarity
                    })
            
            # Sort by similarity (descending)
            similar_people.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Create SIMILAR_TO relationships for top matches
            for person in similar_people[:limit]:
                self._create_similarity_relationship(
                    session, 
                    person_id, 
                    person["person_id"], 
                    person["similarity"]
                )
            
            return similar_people[:limit]

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

    @staticmethod
    def _create_similarity_relationship(session, person1_id: str, person2_id: str, similarity: float):
        """Create SIMILAR_TO relationship between two people"""
        query = """
        MATCH (p1:Person {person_id: $person1_id})
        MATCH (p2:Person {person_id: $person2_id})
        MERGE (p1)-[r:SIMILAR_TO]-(p2)
        ON CREATE SET 
            r.similarity = $similarity,
            r.created_at = datetime()
        ON MATCH SET 
            r.similarity = $similarity,
            r.last_updated = datetime()
        """
        session.run(query, person1_id=person1_id, person2_id=person2_id, similarity=similarity)

    # ============================================================================
    # NETWORK INFERENCE (NEW)
    # ============================================================================

    def infer_connections(self, person_id: str, min_connection_strength: int = 2) -> List[Dict[str, Any]]:
        """
        Infer connections between people based on shared entities.
        
        Creates KNOWS relationships for people who share:
        - Same organizations
        - Same locations
        - Mutual account connections
        
        Args:
            person_id: Person to find connections for
            min_connection_strength: Minimum number of shared entities
        
        Returns:
            List of connected people with connection strength
        """
        with self.driver.session() as session:
            query = """
            MATCH (p1:Person {person_id: $person_id})
            
            // Shared organizations (work/study)
            OPTIONAL MATCH (p1)-[:WORKS_AT|STUDIES_AT]->(o:Organization)<-[:WORKS_AT|STUDIES_AT]-(p2:Person)
            WHERE p1 <> p2
            
            // Shared locations
            OPTIONAL MATCH (p1)-[:LOCATED_IN]->(l:Location)<-[:LOCATED_IN]-(p3:Person)
            WHERE p1 <> p3
            
            // People with similar accounts (following same people, etc.)
            OPTIONAL MATCH (p1)-[:HAS_ACCOUNT]->(a:Account)<-[:HAS_ACCOUNT]-(p4:Person)
            WHERE p1 <> p4
            
            WITH p1, 
                 collect(DISTINCT p2) + collect(DISTINCT p3) + collect(DISTINCT p4) AS connections
            
            UNWIND connections AS connected_person
            WITH p1, connected_person
            WHERE connected_person IS NOT NULL
            
            WITH p1, connected_person, count(connected_person) AS connection_strength
            WHERE connection_strength >= $min_strength
            
            MERGE (p1)-[r:KNOWS]-(connected_person)
            ON CREATE SET 
                r.strength = connection_strength,
                r.inferred = true,
                r.created_at = datetime()
            ON MATCH SET 
                r.strength = connection_strength,
                r.last_updated = datetime()
            
            RETURN connected_person.person_id AS person_id,
                   connected_person.name AS name,
                   connection_strength
            ORDER BY connection_strength DESC
            """
            
            result = session.run(query, person_id=person_id, min_strength=min_connection_strength)
            return [dict(record) for record in result]

    # ============================================================================
    # DATA COMPLETENESS CALCULATION (NEW)
    # ============================================================================

    def update_data_completeness(self, person_id: str):
        """Calculate and update data completeness score for a person"""
        with self.driver.session() as session:
            query = """
            MATCH (p:Person {person_id: $person_id})
            
            OPTIONAL MATCH (p)-[:HAS_FACE]->(e:FaceEmbedding)
            OPTIONAL MATCH (p)-[:HAS_ACCOUNT]->(a:Account)
            OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
            OPTIONAL MATCH (p)-[:WORKS_AT|STUDIES_AT]->(o:Organization)
            OPTIONAL MATCH (p)-[:HAS_EMAIL]->(em:Email)
            
            WITH p,
                 CASE WHEN p.name IS NOT NULL THEN 10 ELSE 0 END AS name_score,
                 CASE WHEN p.age IS NOT NULL THEN 5 ELSE 0 END AS age_score,
                 CASE WHEN p.gender IS NOT NULL THEN 5 ELSE 0 END AS gender_score,
                 CASE WHEN e IS NOT NULL THEN 20 ELSE 0 END AS embedding_score,
                 count(DISTINCT a) * 10 AS accounts_score,
                 count(DISTINCT l) * 10 AS location_score,
                 count(DISTINCT o) * 15 AS org_score,
                 count(DISTINCT em) * 15 AS email_score
            
            WITH p,
                 name_score + age_score + gender_score + embedding_score + 
                 CASE WHEN accounts_score > 30 THEN 30 ELSE accounts_score END +
                 CASE WHEN location_score > 10 THEN 10 ELSE location_score END +
                 CASE WHEN org_score > 15 THEN 15 ELSE org_score END +
                 CASE WHEN email_score > 15 THEN 15 ELSE email_score END AS completeness
            
            SET p.data_completeness = toFloat(completeness)
            
            // Update verification level based on completeness
            SET p.verification_level = CASE
                WHEN completeness >= 80 THEN 'highly_verified'
                WHEN completeness >= 60 THEN 'verified'
                WHEN completeness >= 40 THEN 'partial'
                ELSE 'unverified'
            END
            
            RETURN p.data_completeness AS completeness, p.verification_level AS level
            """
            
            result = session.run(query, person_id=person_id)
            record = result.single()
            
            if record:
                print(f"[Neo4j] Data completeness: {record['completeness']:.1f}% ({record['level']})")
                return record["completeness"]
            
            return 0.0

    # ============================================================================
    # QUERY OPERATIONS
    # ============================================================================

    def get_person_profile(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get complete profile for a person"""
        with self.driver.session() as session:
            query = """
            MATCH (p:Person {person_id: $person_id})
            
            OPTIONAL MATCH (p)-[r:HAS_ACCOUNT]->(a:Account)
            OPTIONAL MATCH (p)-[:LOCATED_IN]->(l:Location)
            OPTIONAL MATCH (p)-[rel:WORKS_AT|STUDIES_AT]->(o:Organization)
            OPTIONAL MATCH (p)-[:HAS_EMAIL]->(em:Email)
            OPTIONAL MATCH (p)-[:HAS_FACE]->(e:FaceEmbedding)
            OPTIONAL MATCH (p)-[sim:SIMILAR_TO]-(similar:Person)
            OPTIONAL MATCH (p)-[knows:KNOWS]-(connection:Person)
            
            RETURN p,
                   collect(DISTINCT {
                       platform: a.platform,
                       username: a.username,
                       url: a.url,
                       display_name: a.display_name,
                       followers: a.followers,
                       confidence_score: r.confidence_score,
                       verified: r.verified,
                       source: r.source
                   }) AS accounts,
                   collect(DISTINCT l.name) AS locations,
                   collect(DISTINCT {
                       name: o.name,
                       type: type(rel),
                       role: rel.role,
                       current: rel.current
                   }) AS organizations,
                   collect(DISTINCT em.address) AS emails,
                   collect(DISTINCT {
                       person_id: similar.person_id,
                       name: similar.name,
                       similarity: sim.similarity
                   }) AS similar_people,
                   collect(DISTINCT {
                       person_id: connection.person_id,
                       name: connection.name,
                       strength: knows.strength
                   }) AS connections,
                   e.vector AS embedding
            """
            
            result = session.run(query, person_id=person_id)
            record = result.single()
            
            if not record:
                return None
            
            person = dict(record["p"])
            person["accounts"] = [acc for acc in record["accounts"] if acc["platform"]]
            person["locations"] = [loc for loc in record["locations"] if loc]
            person["organizations"] = [org for org in record["organizations"] if org["name"]]
            person["emails"] = [email for email in record["emails"] if email]
            person["similar_people"] = [p for p in record["similar_people"] if p["person_id"]]
            person["connections"] = [c for c in record["connections"] if c["person_id"]]
            person["embedding"] = record["embedding"]
            
            return person

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.driver.session() as session:
            query = """
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:HAS_ACCOUNT]->(a:Account)
            OPTIONAL MATCH (p)-[:HAS_FACE]->(e:FaceEmbedding)
            OPTIONAL MATCH (l:Location)
            OPTIONAL MATCH (o:Organization)
            OPTIONAL MATCH (em:Email)
            
            RETURN count(DISTINCT p) AS total_people,
                   count(DISTINCT a) AS total_accounts,
                   count(DISTINCT e) AS total_embeddings,
                   count(DISTINCT l) AS total_locations,
                   count(DISTINCT o) AS total_organizations,
                   count(DISTINCT em) AS total_emails
            """
            
            result = session.run(query)
            record = result.single()
            
            return dict(record) if record else {}

    def search_by_username(self, username: str, platform: Optional[str] = None) -> List[Dict]:
        """Search for people by account username"""
        with self.driver.session() as session:
            if platform:
                query = """
                MATCH (p:Person)-[:HAS_ACCOUNT]->(a:Account)
                WHERE a.username = $username AND a.platform = $platform
                RETURN p.person_id AS person_id, 
                       p.name AS name,
                       a.platform AS platform,
                       a.username AS username,
                       a.url AS url
                """
                result = session.run(query, username=username, platform=platform)
            else:
                query = """
                MATCH (p:Person)-[:HAS_ACCOUNT]->(a:Account)
                WHERE a.username = $username
                RETURN p.person_id AS person_id, 
                       p.name AS name,
                       a.platform AS platform,
                       a.username AS username,
                       a.url AS url
                """
                result = session.run(query, username=username)
            
            return [dict(record) for record in result]

    def delete_person(self, person_id: str):
        """Delete a person and all their relationships"""
        with self.driver.session() as session:
            query = """
            MATCH (p:Person {person_id: $person_id})
            OPTIONAL MATCH (p)-[r]-()
            DELETE r, p
            """
            session.run(query, person_id=person_id)
            print(f"[Neo4j] Deleted person: {person_id}")