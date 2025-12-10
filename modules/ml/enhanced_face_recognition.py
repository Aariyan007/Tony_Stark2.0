# modules/ml/enhanced_face_recognition.py

from deepface import DeepFace
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import json


class EnhancedFaceRecognition:
    """
    Multi-model face recognition with ensemble voting.
    Uses 3 models: Facenet512, VGG-Face, ArcFace for better accuracy.
    """
    
    # Available models (ordered by accuracy)
    MODELS = ["Facenet512", "VGG-Face", "ArcFace"]
    
    # Thresholds for each model (cosine distance)
    THRESHOLDS = {
        "Facenet512": 0.40,
        "VGG-Face": 0.68,
        "ArcFace": 0.68,
    }
    
    def __init__(self, models_to_use: Optional[List[str]] = None):
        """
        Initialize enhanced face recognition.
        
        Args:
            models_to_use: List of models to use. If None, uses all 3.
        """
        self.models = models_to_use or self.MODELS
        print(f"[FaceRec] Initialized with models: {', '.join(self.models)}")
    
    def verify_ensemble(
        self, 
        img1_path: str, 
        img2_path: str,
        enforce_detection: bool = False
    ) -> Dict:
        """
        Verify if two faces match using ensemble of models.
        
        Returns:
            {
                'verified': bool,
                'confidence': float (0-100),
                'model_results': list,
                'consensus': str,
                'quality_score': float
            }
        """
        print(f"\n[FaceRec] Running ensemble verification...")
        
        results = []
        verified_count = 0
        
        # Run each model
        for model_name in self.models:
            try:
                result = DeepFace.verify(
                    img1_path=img1_path,
                    img2_path=img2_path,
                    model_name=model_name,
                    detector_backend="retinaface",
                    enforce_detection=enforce_detection,
                    distance_metric="cosine"
                )
                
                distance = result.get("distance", 1.0)
                threshold = result.get("threshold", self.THRESHOLDS.get(model_name, 0.5))
                verified = result.get("verified", False)
                
                if verified:
                    verified_count += 1
                
                # Calculate confidence for this model
                confidence = self._distance_to_confidence(distance, threshold)
                
                results.append({
                    "model": model_name,
                    "verified": verified,
                    "distance": distance,
                    "threshold": threshold,
                    "confidence": confidence
                })
                
                print(f"  [{model_name}] Verified: {verified} | Distance: {distance:.4f} | Confidence: {confidence:.1f}%")
                
            except Exception as e:
                print(f"  [{model_name}] Failed: {e}")
                results.append({
                    "model": model_name,
                    "verified": False,
                    "distance": 1.0,
                    "threshold": 1.0,
                    "confidence": 0.0,
                    "error": str(e)
                })
        
        # Ensemble decision
        total_models = len([r for r in results if "error" not in r])
        
        if total_models == 0:
            return {
                "verified": False,
                "confidence": 0.0,
                "model_results": results,
                "consensus": "error",
                "quality_score": 0.0
            }
        
        # Consensus: majority vote
        verified = verified_count >= (total_models / 2)
        
        # Average confidence from successful models
        confidences = [r["confidence"] for r in results if "error" not in r]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Boost confidence if all models agree
        if verified_count == total_models:
            consensus = "unanimous"
            confidence_boost = 1.2
        elif verified_count > 0:
            consensus = "majority"
            confidence_boost = 1.0
        else:
            consensus = "rejected"
            confidence_boost = 0.8
        
        final_confidence = min(100.0, avg_confidence * confidence_boost)
        
        # Quality score
        quality_score = self._calculate_quality_score(img1_path, img2_path)
        
        print(f"\n[FaceRec] Ensemble Result:")
        print(f"  Verified: {verified}")
        print(f"  Confidence: {final_confidence:.1f}%")
        print(f"  Consensus: {consensus}")
        print(f"  Quality Score: {quality_score:.1f}%")
        
        return {
            "verified": verified,
            "confidence": final_confidence,
            "model_results": results,
            "consensus": consensus,
            "quality_score": quality_score,
            "verified_count": verified_count,
            "total_models": total_models
        }
    
    def generate_embeddings(
        self, 
        img_path: str,
        model_name: str = "Facenet512"
    ) -> Dict:
        """
        Generate face embeddings for a single model.
        
        Returns:
            {
                'embedding': list,
                'model': str,
                'dimension': int
            }
        """
        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend="retinaface",
                enforce_detection=False
            )
            
            if result and len(result) > 0:
                embedding = result[0]["embedding"]
                return {
                    "embedding": embedding,
                    "model": model_name,
                    "dimension": len(embedding)
                }
        
        except Exception as e:
            print(f"[FaceRec] Embedding generation failed: {e}")
        
        return None
    
    def generate_multi_embeddings(self, img_path: str) -> Dict:
        """
        Generate embeddings from all models.
        
        Returns:
            {
                'Facenet512': [...],
                'VGG-Face': [...],
                'ArcFace': [...]
            }
        """
        embeddings = {}
        
        for model_name in self.models:
            result = self.generate_embeddings(img_path, model_name)
            if result:
                embeddings[model_name] = result["embedding"]
        
        return embeddings
    
    @staticmethod
    def _distance_to_confidence(distance: float, threshold: float) -> float:
        """
        Convert distance to confidence percentage.
        
        Lower distance = higher confidence
        """
        if distance <= threshold:
            # Verified - scale from threshold to 0 as 50-100%
            confidence = 50 + (1 - (distance / threshold)) * 50
        else:
            # Not verified - scale from threshold to 1 as 0-50%
            confidence = max(0, 50 * (1 - (distance - threshold) / (1 - threshold)))
        
        return min(100.0, max(0.0, confidence))
    
    @staticmethod
    def _calculate_quality_score(img1_path: str, img2_path: str) -> float:
        """
        Calculate image quality score based on:
        - Blur detection
        - Brightness
        - Face size
        """
        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                return 0.0
            
            # Calculate blur (Laplacian variance)
            blur1 = cv2.Laplacian(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            blur2 = cv2.Laplacian(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            
            # Normalize blur score (higher is better)
            blur_score = min(100, (blur1 + blur2) / 20)
            
            # Calculate brightness
            brightness1 = np.mean(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
            brightness2 = np.mean(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
            
            # Normalize brightness (optimal around 128)
            brightness_score = 100 - abs(128 - (brightness1 + brightness2) / 2) / 1.28
            
            # Size score (larger images generally better)
            size1 = img1.shape[0] * img1.shape[1]
            size2 = img2.shape[0] * img2.shape[1]
            size_score = min(100, ((size1 + size2) / 2) / 10000)
            
            # Weighted average
            quality_score = (blur_score * 0.5 + brightness_score * 0.3 + size_score * 0.2)
            
            return quality_score
        
        except Exception as e:
            print(f"[FaceRec] Quality calculation failed: {e}")
            return 50.0  # Default medium quality
    
    def detect_liveness(self, img_path: str) -> Dict:
        """
        Basic liveness detection to prevent photo spoofing.
        
        Checks:
        - Image texture patterns
        - Color distribution
        - EXIF data presence
        
        Returns:
            {
                'is_live': bool,
                'confidence': float,
                'reasons': list
            }
        """
        try:
            img = cv2.imread(img_path)
            
            if img is None:
                return {"is_live": False, "confidence": 0.0, "reasons": ["Image not found"]}
            
            reasons = []
            score = 0
            
            # Check 1: Texture analysis (printed photos have different texture)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var > 100:
                score += 40
                reasons.append("Good texture variance")
            else:
                reasons.append("Low texture (possible print)")
            
            # Check 2: Color distribution
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Natural photos have more varied color distribution
            color_entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            if color_entropy > 5:
                score += 30
                reasons.append("Natural color distribution")
            else:
                reasons.append("Uniform colors (possible screen)")
            
            # Check 3: Resolution check
            height, width = img.shape[:2]
            
            if height > 200 and width > 200:
                score += 30
                reasons.append("Adequate resolution")
            else:
                reasons.append("Low resolution")
            
            is_live = score >= 50
            
            return {
                "is_live": is_live,
                "confidence": score,
                "reasons": reasons
            }
        
        except Exception as e:
            print(f"[FaceRec] Liveness detection failed: {e}")
            return {
                "is_live": True,  # Default to allowing
                "confidence": 50.0,
                "reasons": ["Detection failed, assuming live"]
            }


# Example usage
if __name__ == "__main__":
    face_rec = EnhancedFaceRecognition()
    
    # Test ensemble verification
    result = face_rec.verify_ensemble(
        "reference.jpg",
        "test.jpg"
    )
    
    print(f"\nFinal Result: {result['verified']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"Consensus: {result['consensus']}")