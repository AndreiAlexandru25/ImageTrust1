# ImageTrust User Study Protocol

## Overview

This document outlines the protocol for conducting a user study to evaluate the usability, trustworthiness, and effectiveness of the ImageTrust forensic application.

## Study Objectives

### Primary Objectives
1. **Usability Assessment**: Evaluate how easily users can understand and use ImageTrust
2. **Trust Calibration**: Assess whether users appropriately trust/distrust the system's outputs
3. **Accuracy Perception**: Measure users' understanding of the confidence scores
4. **Explainability Effectiveness**: Evaluate if Grad-CAM heatmaps help users understand decisions

### Secondary Objectives
1. Identify UI/UX improvements
2. Gather qualitative feedback on features
3. Compare user performance with/without AI assistance
4. Measure task completion time and accuracy

## Study Design

### Participants

**Target Sample Size**: 30-50 participants

**Demographics**:
- 10-15 Technical users (developers, data scientists)
- 10-15 Semi-technical users (graphic designers, photographers)
- 10-15 Non-technical users (general public)

**Recruitment**:
- University students and staff
- Online communities (Reddit r/photography, design forums)
- Social media recruitment
- Friends and family network

**Exclusion Criteria**:
- Prior experience with AI image detection tools
- Involvement in the project development
- Visual impairments affecting image analysis

### Study Conditions

1. **Control**: Participants classify images without any AI assistance
2. **AI-Assisted**: Participants use ImageTrust with full features
3. **AI-Limited**: Participants see only probability scores (no explainability)

### Image Dataset

Prepare a balanced test set of 40 images:
- 20 Real photographs (diverse sources, subjects)
- 10 AI-generated images (Midjourney, DALL-E 3, Stable Diffusion)
- 10 Manipulated images (face swaps, object removal, compositing)

Image categories should include:
- Portraits/faces
- Landscapes/nature
- Objects/products
- News/documentary style
- Art/creative

---

## Study Procedure

### Phase 1: Pre-Study Questionnaire (5 min)

```markdown
## Participant Information

1. Age: ___
2. Gender: [ ] Male [ ] Female [ ] Non-binary [ ] Prefer not to say
3. Occupation: ___
4. Education Level: [ ] High School [ ] Bachelor's [ ] Master's [ ] PhD

## Technical Background

5. How would you rate your technical expertise? (1-5)
   [ ] 1-Beginner [ ] 2 [ ] 3-Intermediate [ ] 4 [ ] 5-Expert

6. Have you heard of AI-generated images (deepfakes, DALL-E, Midjourney)?
   [ ] Yes [ ] No

7. How confident are you in identifying AI-generated images? (1-5)
   [ ] 1-Not confident [ ] 2 [ ] 3 [ ] 4 [ ] 5-Very confident

8. How often do you encounter images of uncertain authenticity online?
   [ ] Daily [ ] Weekly [ ] Monthly [ ] Rarely [ ] Never
```

### Phase 2: Baseline Test (10 min)

Participants classify 10 images WITHOUT AI assistance:
- 5 Real images
- 3 AI-generated images
- 2 Manipulated images

For each image, record:
- Classification decision (Real/AI/Manipulated/Unsure)
- Confidence level (1-5)
- Time to decision
- Reasoning (optional free text)

### Phase 3: System Introduction (5 min)

Brief tutorial covering:
1. System overview and purpose
2. How to upload/analyze images
3. Understanding probability scores (0-100%)
4. Interpreting confidence levels (Low/Medium/High)
5. Reading Grad-CAM heatmaps
6. Using the forensic report

Provide printed quick-reference card.

### Phase 4: Assisted Classification (15 min)

Participants classify 20 images WITH ImageTrust:
- 10 Real images
- 6 AI-generated images
- 4 Manipulated images

For each image, record:
- Initial guess (before seeing AI result)
- Final decision (after AI assistance)
- Whether they agreed with the AI
- Confidence in final decision (1-5)
- Time to decision
- Usefulness of heatmap (1-5)

### Phase 5: Trust Calibration Tasks (10 min)

Special scenarios to test appropriate trust:

1. **Correct High-Confidence**: AI correctly identifies with 95% confidence
2. **Correct Low-Confidence**: AI correctly identifies with 65% confidence
3. **Incorrect High-Confidence**: AI incorrectly identifies with 90% confidence
4. **Edge Cases**: Deliberately challenging images

Record whether participants:
- Follow AI recommendations blindly
- Question AI when confidence is low
- Identify when AI might be wrong
- Appropriately adjust their confidence

### Phase 6: Post-Study Questionnaire (10 min)

```markdown
## System Usability Scale (SUS)

Rate 1-5 (Strongly Disagree to Strongly Agree):

1. I think I would like to use this system frequently
2. I found the system unnecessarily complex
3. I thought the system was easy to use
4. I think I would need technical support to use this system
5. I found the various functions in this system were well integrated
6. I thought there was too much inconsistency in this system
7. I would imagine that most people would learn to use this system very quickly
8. I found the system very cumbersome to use
9. I felt very confident using the system
10. I needed to learn a lot of things before I could get going with this system

## Trust and Accuracy

11. How much do you trust the AI's detection accuracy? (1-5)
12. How understandable were the probability scores? (1-5)
13. How helpful were the heatmap explanations? (1-5)
14. How reliable did the system feel overall? (1-5)

## Feature Evaluation

15. Rate the usefulness of each feature (1-5):
    - [ ] Probability score
    - [ ] Confidence level
    - [ ] Grad-CAM heatmap
    - [ ] Metadata analysis
    - [ ] C2PA verification
    - [ ] Forensic report

## Open Questions

16. What did you like most about ImageTrust?
    ____________________________________________

17. What would you improve?
    ____________________________________________

18. Would you use this tool in your daily life? Why/why not?
    ____________________________________________

19. In what situations would you find this tool most useful?
    ____________________________________________
```

### Phase 7: Semi-Structured Interview (5-10 min, optional)

For selected participants, conduct brief interviews:

1. Walk through their decision-making process for specific images
2. Discuss moments of agreement/disagreement with the AI
3. Explore understanding of the explainability features
4. Gather suggestions for improvements
5. Discuss potential use cases they envision

---

## Data Collection

### Quantitative Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| Accuracy (Baseline) | Human accuracy without AI | Correct / Total |
| Accuracy (Assisted) | Human accuracy with AI | Correct / Total |
| Accuracy Improvement | Gain from AI assistance | Assisted - Baseline |
| Agreement Rate | How often humans agree with AI | Agreed / Total |
| Appropriate Trust | Correct trust calibration | (Trust when correct + Distrust when wrong) / Total |
| Over-reliance | Following incorrect AI | Agreed with wrong AI / Wrong AI total |
| Under-reliance | Ignoring correct AI | Disagreed with correct AI / Correct AI total |
| SUS Score | System usability | See SUS formula |
| Task Completion Time | Average time per image | Sum(times) / Total |

### Qualitative Data

- Open-ended questionnaire responses
- Interview transcripts
- Think-aloud protocols (if recorded)
- Observation notes

---

## Analysis Plan

### Quantitative Analysis

1. **Descriptive Statistics**
   - Mean, median, SD for all metrics
   - Frequency distributions
   - Demographic breakdowns

2. **Comparative Analysis**
   - Paired t-test: Baseline vs. Assisted accuracy
   - ANOVA: Performance across user groups
   - Chi-square: Agreement patterns

3. **Correlation Analysis**
   - Trust vs. Accuracy
   - Confidence vs. Correctness
   - Technical background vs. Performance

4. **Regression Analysis**
   - Predictors of appropriate trust
   - Factors affecting accuracy improvement

### Qualitative Analysis

1. **Thematic Analysis**
   - Code open-ended responses
   - Identify recurring themes
   - Group by sentiment (positive/negative/neutral)

2. **Pattern Analysis**
   - Decision-making strategies
   - Trust patterns
   - Feature usage patterns

---

## Ethical Considerations

### Informed Consent

All participants must:
- Be 18+ years old
- Sign informed consent form
- Understand data usage and privacy
- Know they can withdraw at any time

### Data Privacy

- All data anonymized (participant IDs only)
- No personally identifiable information stored
- Data stored securely (encrypted)
- Data retained for 5 years post-publication

### IRB Approval

Submit for Institutional Review Board approval:
- Study protocol
- Consent form
- Questionnaires
- Data management plan

---

## Materials Checklist

### Before Study
- [ ] IRB approval obtained
- [ ] Consent forms printed
- [ ] Image test set prepared and validated
- [ ] ImageTrust deployed and tested
- [ ] Questionnaires finalized
- [ ] Recording equipment (if used) ready
- [ ] Participant schedule confirmed
- [ ] Quick-reference cards printed

### During Study
- [ ] Quiet, distraction-free environment
- [ ] Consistent hardware (same computer/display)
- [ ] Standardized instructions
- [ ] Timer for each phase
- [ ] Note-taking materials

### After Study
- [ ] All data backed up
- [ ] Participant compensated (if applicable)
- [ ] Follow-up email sent (thank you + summary option)

---

## Timeline

| Week | Activity |
|------|----------|
| 1-2 | IRB submission and approval |
| 3 | Pilot study (3-5 participants) |
| 4 | Revisions based on pilot |
| 5-7 | Main study data collection |
| 8-9 | Data analysis |
| 10 | Report writing |

---

## Expected Outcomes

### For Thesis

1. Quantitative evidence of system effectiveness
2. User trust and accuracy metrics
3. Usability scores (SUS)
4. Qualitative insights for discussion

### For Publication

1. Novel user study methodology
2. Trust calibration findings
3. Explainability effectiveness evidence
4. Design recommendations

---

## Appendices

### A. Consent Form Template
### B. Full Questionnaires
### C. Interview Guide
### D. Image Dataset Metadata
### E. Statistical Analysis Scripts
### F. Data Recording Sheets
