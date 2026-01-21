# ImageTrust Architecture

## Overview

ImageTrust is a modular forensic application for detecting AI-generated and manipulated images. The architecture follows a layered design with clear separation of concerns.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Presentation Layer                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │   REST API   │  │   Web UI     │  │   CLI        │  │   Mobile App     │ │
│  │   (FastAPI)  │  │   (React)    │  │   (Typer)    │  │   (Future)       │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Application Layer                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        ImageTrust Analyzer                            │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐ │   │
│  │  │  Detection  │ │  Metadata   │ │ Explainability │ │   Reporting   │ │   │
│  │  │  Pipeline   │ │  Pipeline   │ │    Pipeline    │ │   Pipeline    │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               Domain Layer                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────────┐ │
│  │    Detection    │  │    Metadata    │  │       Explainability           │ │
│  │     Module      │  │     Module     │  │          Module                │ │
│  ├────────────────┤  ├────────────────┤  ├────────────────────────────────┤ │
│  │ • CNN Detector │  │ • EXIF Parser  │  │ • Grad-CAM                     │ │
│  │ • ViT Detector │  │ • XMP Parser   │  │ • Patch Analysis               │ │
│  │ • CLIP Detector│  │ • C2PA Valid.  │  │ • Frequency Analysis           │ │
│  │ • Ensemble     │  │ • Provenance   │  │ • Visualizations               │ │
│  │ • Calibration  │  │                │  │                                │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────────┐ │
│  │   Evaluation    │  │   Reporting    │  │          Core                  │ │
│  │     Module      │  │     Module     │  │         Module                 │ │
│  ├────────────────┤  ├────────────────┤  ├────────────────────────────────┤ │
│  │ • Metrics      │  │ • PDF Export   │  │ • Configuration                │ │
│  │ • Benchmark    │  │ • JSON Export  │  │ • Types & Models               │ │
│  │ • Cross-Gen    │  │ • HTML Export  │  │ • Exceptions                   │ │
│  │ • Degradation  │  │ • Templates    │  │ • Utilities                    │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Infrastructure Layer                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────────┐ │
│  │   PyTorch      │  │   PIL/OpenCV   │  │      External APIs             │ │
│  │   Models       │  │   Image I/O    │  │      (C2PA, HuggingFace)       │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### Core Module (`imagetrust.core`)
- **Configuration**: Environment-based settings using pydantic-settings
- **Types**: Pydantic models for type-safe data handling
- **Exceptions**: Custom exception hierarchy for precise error handling

### Detection Module (`imagetrust.detection`)
- **Models**: CNN, ViT, CLIP-based detectors with common interface
- **Ensemble**: Multiple combination strategies (weighted, voting, stacking)
- **Calibration**: Probability calibration for honest confidence scores
- **Preprocessing**: Standardized image preprocessing pipelines

### Metadata Module (`imagetrust.metadata`)
- **EXIF Parser**: Camera and capture metadata extraction
- **XMP Parser**: Edit history and software metadata
- **C2PA Validator**: Content Credentials verification
- **Provenance Analyzer**: Combined provenance assessment

### Explainability Module (`imagetrust.explainability`)
- **Grad-CAM**: Gradient-based attention visualization
- **Patch Analysis**: Region-level detection scores
- **Frequency Analysis**: Spectral artifacts detection
- **Visualizations**: Overlay and comparison generation

### Evaluation Module (`imagetrust.evaluation`)
- **Metrics**: Standard and calibration metrics
- **Benchmark**: Standardized evaluation framework
- **Cross-Generator**: Generator generalization testing
- **Degradation**: Robustness to image degradations

### Reporting Module (`imagetrust.reporting`)
- **Forensic Reports**: Comprehensive analysis reports
- **Exporters**: PDF, JSON, HTML output formats
- **Templates**: Customizable report templates

### API Module (`imagetrust.api`)
- **FastAPI Application**: REST API endpoints
- **Schemas**: Request/response validation
- **Middleware**: CORS, rate limiting, error handling

## Data Flow

```
Image Input
     │
     ▼
┌─────────────────┐
│  Preprocessing  │
│  (resize, norm) │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│  ML   │ │ Meta  │
│Detect │ │ data  │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│  Decision       │
│  Fusion         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Explainability │
│  Generation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Report         │
│  Generation     │
└─────────────────┘
```

## Key Design Principles

1. **Modularity**: Each component can be used independently
2. **Type Safety**: Pydantic models throughout for validation
3. **Extensibility**: Easy to add new detectors, exporters, etc.
4. **Configurability**: Environment-based configuration
5. **Testability**: Dependency injection for easy mocking
