# ImageTrust Threat Model

## Overview

This document describes the threat model for ImageTrust, identifying potential attack vectors, system assumptions, and security boundaries.

## System Context

ImageTrust is a forensic tool designed to detect AI-generated and manipulated images. It operates in adversarial environments where attackers may attempt to:

1. Evade detection (make AI images appear real)
2. Cause false positives (make real images appear AI-generated)
3. Compromise the system itself

## Assets

### Primary Assets
- **Detection Accuracy**: The core value proposition
- **User Trust**: Reliability of the verdicts
- **Model Weights**: Trained model parameters

### Secondary Assets
- **User Data**: Uploaded images (privacy concern)
- **Analysis Results**: Forensic reports
- **System Availability**: Service uptime

## Threat Actors

### 1. Evasion Attackers (Sophistication: Medium-High)
**Goal**: Make AI-generated images pass as authentic

**Capabilities**:
- Access to AI image generators
- Knowledge of detection methods
- Ability to post-process images

**Attack Vectors**:
- Adversarial perturbations
- Post-processing (JPEG, blur, noise)
- Generator fine-tuning to avoid artifacts
- Metadata injection

### 2. False Positive Attackers (Sophistication: Medium)
**Goal**: Make authentic images appear AI-generated

**Capabilities**:
- Access to authentic images
- Knowledge of what triggers detectors

**Attack Vectors**:
- Adding AI-like artifacts
- Metadata manipulation
- Specific image processing

### 3. System Attackers (Sophistication: High)
**Goal**: Compromise the ImageTrust system

**Capabilities**:
- Technical expertise
- Persistent access attempts

**Attack Vectors**:
- API abuse
- Malicious file uploads
- Model extraction attacks

## Security Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRUST BOUNDARY 1                            │
│                    (External → API)                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ • Input validation                                         │  │
│  │ • Rate limiting                                            │  │
│  │ • File type verification                                   │  │
│  │ • Size limits                                              │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRUST BOUNDARY 2                            │
│                   (API → Processing)                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ • Image parsing in sandbox                                 │  │
│  │ • Memory limits                                            │  │
│  │ • Timeout enforcement                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRUST BOUNDARY 3                            │
│                (Processing → ML Models)                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ • Model integrity verification                             │  │
│  │ • Input normalization                                      │  │
│  │ • Output validation                                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Attack Scenarios & Mitigations

### Scenario 1: Adversarial Evasion

**Attack**: Attacker adds imperceptible perturbations to AI-generated image to fool detector.

**Mitigations**:
- ✅ Ensemble of diverse models (harder to fool all)
- ✅ Input preprocessing (JPEG, resize) disrupts perturbations
- ✅ Frequency domain analysis (orthogonal signal)
- ✅ Metadata analysis (complementary evidence)
- ⚠️ Adversarial training (future enhancement)

**Residual Risk**: Medium - Sophisticated attacks may still succeed

### Scenario 2: Post-Processing Evasion

**Attack**: Attacker applies heavy JPEG compression, blur, or noise to hide artifacts.

**Mitigations**:
- ✅ Training with degraded images
- ✅ Degradation-specific evaluation
- ✅ Confidence calibration (lower confidence for degraded images)
- ✅ Warning indicators for heavily processed images

**Residual Risk**: Medium - Heavy degradation reduces detectability

### Scenario 3: Metadata Spoofing

**Attack**: Attacker adds fake EXIF data to AI-generated image.

**Mitigations**:
- ✅ Cross-validation of metadata fields
- ✅ Detection of common spoofing patterns
- ✅ Primary reliance on pixel analysis, not metadata
- ⚠️ Camera fingerprinting (future enhancement)

**Residual Risk**: Low - Metadata is supplementary evidence

### Scenario 4: C2PA Forgery

**Attack**: Attacker creates fake C2PA credentials.

**Mitigations**:
- ✅ Cryptographic signature verification
- ✅ Certificate chain validation
- ✅ Timestamp verification
- ✅ Cross-reference with trusted issuers

**Residual Risk**: Low - Requires breaking cryptography

### Scenario 5: Model Extraction

**Attack**: Attacker queries API to reconstruct model.

**Mitigations**:
- ✅ Rate limiting
- ✅ Output perturbation (slight noise)
- ⚠️ Query monitoring for suspicious patterns
- ⚠️ Watermarking model outputs

**Residual Risk**: Medium - Determined attacker with many queries

### Scenario 6: Denial of Service

**Attack**: Attacker floods API with requests or malformed images.

**Mitigations**:
- ✅ Rate limiting per IP/user
- ✅ Request timeouts
- ✅ File size limits
- ✅ Input validation before processing

**Residual Risk**: Low - Standard DoS protections

## Confidence Communication

### What We Communicate

| Confidence Level | Meaning | Appropriate Use |
|------------------|---------|-----------------|
| High (≥85%) | Very likely correct | Can act on this with minimal additional verification |
| Medium (65-84%) | Probably correct | Recommend additional verification |
| Low (50-64%) | Uncertain | Significant additional verification needed |
| Very Low (<50%) | Likely incorrect | Do not rely on this result |

### What We Don't Promise

1. **100% Accuracy**: No detection system is perfect
2. **Future-Proof**: New generators may evade detection
3. **Adversarial Robustness**: Targeted attacks may succeed
4. **Legal Evidence**: Results should be corroborated

## Failure Modes

### Graceful Failures
- Unknown image format → Clear error message
- Processing timeout → Partial results with warning
- Model load failure → Fallback to simpler model

### Ungraceful Failures (Monitored)
- Out of memory → Service restart
- GPU error → CPU fallback
- Corrupted model → Alert and service halt

## Security Recommendations for Users

1. **Don't rely solely on ImageTrust** - Use multiple verification methods
2. **Consider context** - Source, metadata, and content together
3. **High-stakes decisions** - Require human expert review
4. **Report false positives/negatives** - Help improve the system

## Future Security Enhancements

1. [ ] Adversarial training against known attacks
2. [ ] Real-time attack detection
3. [ ] Federated learning for privacy
4. [ ] Hardware security module for model protection
5. [ ] Differential privacy for user data
