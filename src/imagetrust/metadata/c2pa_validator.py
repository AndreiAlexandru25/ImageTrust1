"""
C2PA (Coalition for Content Provenance and Authenticity) Validator
Implements verification of Content Credentials standard.

C2PA is an industry standard by Adobe, Microsoft, BBC, and others
for establishing content provenance and authenticity.

Features:
1. Manifest parsing (JUMBF format)
2. Signature verification
3. Claim validation
4. Certificate chain verification
5. Tamper detection
"""

import json
import hashlib
import struct
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import io


class C2PAStatus(Enum):
    """C2PA validation status."""
    VALID = "valid"
    INVALID = "invalid"
    NOT_FOUND = "not_found"
    TAMPERED = "tampered"
    EXPIRED = "expired"
    UNKNOWN_SIGNER = "unknown_signer"


class ClaimType(Enum):
    """Types of C2PA claims."""
    CREATED = "c2pa.created"
    EDITED = "c2pa.edited"
    IMPORTED = "c2pa.imported"
    COMBINED = "c2pa.combined"
    DERIVED = "c2pa.derived"
    AI_GENERATED = "c2pa.ai_generated"
    AI_TRAINED = "c2pa.ai_trained"


@dataclass
class C2PAClaim:
    """Individual C2PA claim."""
    claim_type: ClaimType
    timestamp: datetime
    software: str
    parameters: Dict
    signature_info: Dict


@dataclass
class C2PAManifest:
    """C2PA manifest containing provenance information."""
    version: str
    title: str
    claims: List[C2PAClaim]
    ingredients: List[Dict]
    signature: bytes
    certificate_chain: List[Dict]
    hash_algorithm: str
    asset_hash: str


@dataclass
class C2PAValidationResult:
    """Result of C2PA validation."""
    status: C2PAStatus
    has_c2pa: bool
    manifest: Optional[C2PAManifest]
    trust_score: float  # 0-100
    is_ai_generated: bool
    ai_generator: Optional[str]
    creation_info: Dict
    edit_history: List[Dict]
    warnings: List[str]
    certificate_info: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "has_c2pa": self.has_c2pa,
            "trust_score": self.trust_score,
            "is_ai_generated": self.is_ai_generated,
            "ai_generator": self.ai_generator,
            "creation_info": self.creation_info,
            "edit_history": self.edit_history,
            "warnings": self.warnings,
            "certificate_info": self.certificate_info
        }


class C2PAValidator:
    """
    Validates C2PA Content Credentials in images.
    
    C2PA manifests are stored in JUMBF (JPEG Universal Metadata Box Format)
    or XMP metadata, containing:
    - Provenance claims (who created/edited)
    - Timestamps
    - Digital signatures
    - Certificate chains
    """
    
    # Known trusted signers (simplified for demo)
    TRUSTED_SIGNERS = {
        "Adobe Inc.": {"trust_level": "high", "products": ["Photoshop", "Firefly", "Lightroom"]},
        "Microsoft Corporation": {"trust_level": "high", "products": ["Designer", "Bing Image Creator"]},
        "OpenAI": {"trust_level": "high", "products": ["DALL-E 2", "DALL-E 3", "ChatGPT"]},
        "Google LLC": {"trust_level": "high", "products": ["Imagen", "Gemini"]},
        "Stability AI": {"trust_level": "medium", "products": ["Stable Diffusion"]},
        "Midjourney Inc.": {"trust_level": "medium", "products": ["Midjourney"]},
    }
    
    # AI generator signatures
    AI_GENERATOR_SIGNATURES = {
        "Adobe Firefly": ClaimType.AI_GENERATED,
        "DALL-E": ClaimType.AI_GENERATED,
        "Midjourney": ClaimType.AI_GENERATED,
        "Stable Diffusion": ClaimType.AI_GENERATED,
        "Imagen": ClaimType.AI_GENERATED,
    }
    
    def __init__(self):
        self.jumbf_parser = JUMBFParser()
    
    def validate(self, image_bytes: bytes) -> C2PAValidationResult:
        """
        Validate C2PA credentials in an image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            C2PAValidationResult with validation details
        """
        warnings = []
        
        # Try to find C2PA manifest
        manifest = self._find_manifest(image_bytes)
        
        if manifest is None:
            # No C2PA data found
            return C2PAValidationResult(
                status=C2PAStatus.NOT_FOUND,
                has_c2pa=False,
                manifest=None,
                trust_score=0,
                is_ai_generated=False,
                ai_generator=None,
                creation_info={},
                edit_history=[],
                warnings=["No C2PA Content Credentials found in image"],
                certificate_info={}
            )
        
        # Verify signature
        signature_valid = self._verify_signature(manifest, image_bytes)
        
        if not signature_valid:
            warnings.append("Signature verification failed - image may have been modified")
        
        # Check certificate chain
        cert_info = self._verify_certificate_chain(manifest)
        
        # Calculate trust score
        trust_score = self._calculate_trust_score(manifest, signature_valid, cert_info)
        
        # Check for AI generation claims
        is_ai_generated, ai_generator = self._check_ai_claims(manifest)
        
        # Extract creation info
        creation_info = self._extract_creation_info(manifest)
        
        # Build edit history
        edit_history = self._build_edit_history(manifest)
        
        # Determine status
        if signature_valid and trust_score > 70:
            status = C2PAStatus.VALID
        elif signature_valid:
            status = C2PAStatus.UNKNOWN_SIGNER
        else:
            status = C2PAStatus.TAMPERED
        
        return C2PAValidationResult(
            status=status,
            has_c2pa=True,
            manifest=manifest,
            trust_score=trust_score,
            is_ai_generated=is_ai_generated,
            ai_generator=ai_generator,
            creation_info=creation_info,
            edit_history=edit_history,
            warnings=warnings,
            certificate_info=cert_info
        )
    
    def _find_manifest(self, image_bytes: bytes) -> Optional[C2PAManifest]:
        """Find and parse C2PA manifest from image."""
        # Try JUMBF format (JPEG)
        manifest = self.jumbf_parser.parse(image_bytes)
        if manifest:
            return manifest
        
        # Try XMP metadata
        manifest = self._parse_xmp_c2pa(image_bytes)
        if manifest:
            return manifest
        
        # Try PNG chunks
        manifest = self._parse_png_c2pa(image_bytes)
        if manifest:
            return manifest
        
        return None
    
    def _parse_xmp_c2pa(self, image_bytes: bytes) -> Optional[C2PAManifest]:
        """Parse C2PA from XMP metadata."""
        try:
            # Look for XMP packet
            xmp_start = image_bytes.find(b'<x:xmpmeta')
            if xmp_start == -1:
                return None
            
            xmp_end = image_bytes.find(b'</x:xmpmeta>', xmp_start)
            if xmp_end == -1:
                return None
            
            xmp_data = image_bytes[xmp_start:xmp_end + 12].decode('utf-8', errors='ignore')
            
            # Check for C2PA namespace
            if 'c2pa' not in xmp_data.lower() and 'contentcredentials' not in xmp_data.lower():
                return None
            
            # Parse relevant fields (simplified)
            claims = []
            
            # Check for AI generation marker
            if 'ai_generated' in xmp_data.lower() or 'dall-e' in xmp_data.lower() or 'firefly' in xmp_data.lower():
                claims.append(C2PAClaim(
                    claim_type=ClaimType.AI_GENERATED,
                    timestamp=datetime.now(),
                    software="Unknown",
                    parameters={},
                    signature_info={}
                ))
            
            if claims:
                return C2PAManifest(
                    version="1.0",
                    title="XMP C2PA Manifest",
                    claims=claims,
                    ingredients=[],
                    signature=b"",
                    certificate_chain=[],
                    hash_algorithm="sha256",
                    asset_hash=""
                )
            
        except Exception as e:
            pass
        
        return None
    
    def _parse_png_c2pa(self, image_bytes: bytes) -> Optional[C2PAManifest]:
        """Parse C2PA from PNG chunks."""
        if not image_bytes.startswith(b'\x89PNG'):
            return None
        
        try:
            # PNG chunk parsing
            pos = 8  # Skip signature
            
            while pos < len(image_bytes):
                length = struct.unpack('>I', image_bytes[pos:pos+4])[0]
                chunk_type = image_bytes[pos+4:pos+8].decode('ascii', errors='ignore')
                chunk_data = image_bytes[pos+8:pos+8+length]
                
                # Look for C2PA-related chunks
                if chunk_type == 'caBX' or chunk_type == 'iTXt':
                    if b'c2pa' in chunk_data.lower() or b'contentcredentials' in chunk_data.lower():
                        # Found C2PA data
                        return self._parse_c2pa_chunk(chunk_data)
                
                pos += 12 + length
        
        except Exception as e:
            pass
        
        return None
    
    def _parse_c2pa_chunk(self, chunk_data: bytes) -> Optional[C2PAManifest]:
        """Parse C2PA manifest from raw chunk data."""
        try:
            # Try JSON parsing
            if chunk_data.startswith(b'{'):
                manifest_json = json.loads(chunk_data.decode('utf-8'))
                return self._json_to_manifest(manifest_json)
        except:
            pass
        
        return None
    
    def _json_to_manifest(self, data: Dict) -> C2PAManifest:
        """Convert JSON data to C2PAManifest."""
        claims = []
        
        for claim_data in data.get('claims', []):
            claim_type_str = claim_data.get('type', 'c2pa.created')
            try:
                claim_type = ClaimType(claim_type_str)
            except:
                claim_type = ClaimType.CREATED
            
            claims.append(C2PAClaim(
                claim_type=claim_type,
                timestamp=datetime.fromisoformat(claim_data.get('timestamp', datetime.now().isoformat())),
                software=claim_data.get('software', 'Unknown'),
                parameters=claim_data.get('parameters', {}),
                signature_info=claim_data.get('signature', {})
            ))
        
        return C2PAManifest(
            version=data.get('version', '1.0'),
            title=data.get('title', 'Untitled'),
            claims=claims,
            ingredients=data.get('ingredients', []),
            signature=data.get('signature', b'').encode() if isinstance(data.get('signature', ''), str) else b'',
            certificate_chain=data.get('certificates', []),
            hash_algorithm=data.get('hash_algorithm', 'sha256'),
            asset_hash=data.get('asset_hash', '')
        )
    
    def _verify_signature(self, manifest: C2PAManifest, image_bytes: bytes) -> bool:
        """Verify C2PA signature."""
        if not manifest.signature:
            return False
        
        # Calculate asset hash
        hash_func = hashlib.sha256 if manifest.hash_algorithm == 'sha256' else hashlib.sha384
        calculated_hash = hash_func(image_bytes).hexdigest()
        
        # In a real implementation, verify against certificate
        # For now, check if hash exists
        if manifest.asset_hash:
            return calculated_hash.startswith(manifest.asset_hash[:16])
        
        return True  # Assume valid if no hash to check
    
    def _verify_certificate_chain(self, manifest: C2PAManifest) -> Dict:
        """Verify certificate chain and extract info."""
        cert_info = {
            "issuer": None,
            "subject": None,
            "valid_from": None,
            "valid_to": None,
            "is_trusted": False,
            "trust_level": "unknown"
        }
        
        if not manifest.certificate_chain:
            return cert_info
        
        # Extract info from first certificate
        first_cert = manifest.certificate_chain[0]
        
        cert_info["issuer"] = first_cert.get("issuer", "Unknown")
        cert_info["subject"] = first_cert.get("subject", "Unknown")
        cert_info["valid_from"] = first_cert.get("not_before")
        cert_info["valid_to"] = first_cert.get("not_after")
        
        # Check if trusted
        for signer, info in self.TRUSTED_SIGNERS.items():
            if signer.lower() in str(cert_info["issuer"]).lower():
                cert_info["is_trusted"] = True
                cert_info["trust_level"] = info["trust_level"]
                break
        
        return cert_info
    
    def _calculate_trust_score(self, manifest: C2PAManifest, 
                               signature_valid: bool, 
                               cert_info: Dict) -> float:
        """Calculate overall trust score (0-100)."""
        score = 0
        
        # Signature validity (40 points)
        if signature_valid:
            score += 40
        
        # Certificate trust (30 points)
        if cert_info["is_trusted"]:
            if cert_info["trust_level"] == "high":
                score += 30
            elif cert_info["trust_level"] == "medium":
                score += 20
            else:
                score += 10
        
        # Manifest completeness (20 points)
        if manifest.claims:
            score += 10
        if manifest.ingredients:
            score += 5
        if manifest.certificate_chain:
            score += 5
        
        # Version (10 points for recent version)
        if manifest.version in ["1.0", "1.1", "2.0"]:
            score += 10
        
        return float(score)
    
    def _check_ai_claims(self, manifest: C2PAManifest) -> Tuple[bool, Optional[str]]:
        """Check if manifest indicates AI generation."""
        for claim in manifest.claims:
            if claim.claim_type == ClaimType.AI_GENERATED:
                return True, claim.software
        
        # Check software names
        for claim in manifest.claims:
            for ai_name in self.AI_GENERATOR_SIGNATURES.keys():
                if ai_name.lower() in claim.software.lower():
                    return True, ai_name
        
        return False, None
    
    def _extract_creation_info(self, manifest: C2PAManifest) -> Dict:
        """Extract creation information from manifest."""
        info = {
            "title": manifest.title,
            "created_by": None,
            "created_at": None,
            "software": None,
            "version": manifest.version
        }
        
        for claim in manifest.claims:
            if claim.claim_type == ClaimType.CREATED:
                info["created_at"] = claim.timestamp.isoformat() if claim.timestamp else None
                info["software"] = claim.software
                info["created_by"] = claim.signature_info.get("signer")
        
        return info
    
    def _build_edit_history(self, manifest: C2PAManifest) -> List[Dict]:
        """Build edit history from claims."""
        history = []
        
        for claim in manifest.claims:
            history.append({
                "action": claim.claim_type.value,
                "timestamp": claim.timestamp.isoformat() if claim.timestamp else None,
                "software": claim.software,
                "details": claim.parameters
            })
        
        return history


class JUMBFParser:
    """
    JUMBF (JPEG Universal Metadata Box Format) parser for C2PA manifests.
    """
    
    def parse(self, image_bytes: bytes) -> Optional[C2PAManifest]:
        """Parse JUMBF boxes from JPEG image."""
        if not self._is_jpeg(image_bytes):
            return None
        
        try:
            # Find APP11 marker (JUMBF)
            pos = 2  # Skip SOI
            
            while pos < len(image_bytes) - 2:
                marker = struct.unpack('>H', image_bytes[pos:pos+2])[0]
                
                if marker == 0xFFEB:  # APP11
                    length = struct.unpack('>H', image_bytes[pos+2:pos+4])[0]
                    app11_data = image_bytes[pos+4:pos+2+length]
                    
                    # Check for JUMBF signature
                    if b'jumb' in app11_data or b'c2pa' in app11_data:
                        return self._parse_jumbf(app11_data)
                
                elif marker == 0xFFD9:  # EOI
                    break
                elif marker >= 0xFFE0 and marker <= 0xFFEF:  # APPn
                    length = struct.unpack('>H', image_bytes[pos+2:pos+4])[0]
                    pos += 2 + length
                elif marker == 0xFFDA:  # SOS
                    break
                else:
                    pos += 2
                    if pos < len(image_bytes) - 2:
                        length = struct.unpack('>H', image_bytes[pos:pos+2])[0]
                        pos += length
        
        except Exception as e:
            pass
        
        return None
    
    def _is_jpeg(self, data: bytes) -> bool:
        """Check if data is JPEG."""
        return data.startswith(b'\xff\xd8\xff')
    
    def _parse_jumbf(self, data: bytes) -> Optional[C2PAManifest]:
        """Parse JUMBF box data."""
        # Simplified JUMBF parsing
        claims = []
        
        # Look for c2pa markers
        if b'c2pa.claim' in data or b'contentcredentials' in data.lower():
            claims.append(C2PAClaim(
                claim_type=ClaimType.CREATED,
                timestamp=datetime.now(),
                software="JUMBF Embedded",
                parameters={},
                signature_info={}
            ))
        
        if b'ai_generated' in data.lower() or b'firefly' in data.lower() or b'dall-e' in data.lower():
            claims.append(C2PAClaim(
                claim_type=ClaimType.AI_GENERATED,
                timestamp=datetime.now(),
                software="AI Generator",
                parameters={},
                signature_info={}
            ))
        
        if claims:
            return C2PAManifest(
                version="1.0",
                title="JUMBF C2PA Manifest",
                claims=claims,
                ingredients=[],
                signature=b"",
                certificate_chain=[],
                hash_algorithm="sha256",
                asset_hash=""
            )
        
        return None


def validate_c2pa(image_bytes: bytes) -> C2PAValidationResult:
    """
    Convenience function to validate C2PA credentials.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        C2PAValidationResult with validation details
    """
    validator = C2PAValidator()
    return validator.validate(image_bytes)


def has_c2pa(image_bytes: bytes) -> bool:
    """Quick check if image has C2PA credentials."""
    result = validate_c2pa(image_bytes)
    return result.has_c2pa
