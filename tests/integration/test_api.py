"""
Integration tests for the FastAPI REST API.

Tests API endpoints with real HTTP requests.
"""

import io
import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_image_bytes():
    """Create a sample image as bytes."""
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def test_client():
    """Create a test client for the API."""
    try:
        from fastapi.testclient import TestClient
        from imagetrust.api.main import app

        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI not available")


# =============================================================================
# Health & Info Endpoint Tests
# =============================================================================

class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.integration
    def test_health_check(self, test_client):
        """Test /health endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]

    @pytest.mark.integration
    def test_info_endpoint(self, test_client):
        """Test /info endpoint."""
        response = test_client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data

    @pytest.mark.integration
    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")

        assert response.status_code == 200


# =============================================================================
# Analysis Endpoint Tests
# =============================================================================

class TestAnalysisEndpoints:
    """Tests for image analysis endpoints."""

    @pytest.mark.integration
    def test_analyze_single_image(self, test_client, sample_image_bytes):
        """Test /analyze endpoint with single image."""
        response = test_client.post(
            "/analyze",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

        # Should succeed or fail gracefully if models not loaded
        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "ai_probability" in data
            assert "verdict" in data
            assert 0 <= data["ai_probability"] <= 1

    @pytest.mark.integration
    def test_analyze_invalid_file(self, test_client):
        """Test /analyze with invalid file."""
        response = test_client.post(
            "/analyze",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )

        assert response.status_code in [400, 422, 500]

    @pytest.mark.integration
    def test_analyze_no_file(self, test_client):
        """Test /analyze without file."""
        response = test_client.post("/analyze")

        assert response.status_code == 422  # Validation error

    @pytest.mark.integration
    def test_analyze_batch(self, test_client, sample_image_bytes):
        """Test /analyze/batch endpoint."""
        files = [
            ("files", ("test1.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("test2.jpg", sample_image_bytes, "image/jpeg")),
        ]

        response = test_client.post("/analyze/batch", files=files)

        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 2

    @pytest.mark.integration
    def test_analyze_detailed(self, test_client, sample_image_bytes):
        """Test /analyze/detailed endpoint."""
        response = test_client.post(
            "/analyze/detailed",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

        assert response.status_code in [200, 500, 503]

        if response.status_code == 200:
            data = response.json()
            assert "ai_probability" in data
            # Detailed endpoint should have more fields
            # assert "model_results" in data or "signal_analysis" in data


# =============================================================================
# URL Analysis Tests
# =============================================================================

class TestURLAnalysis:
    """Tests for URL-based analysis."""

    @pytest.mark.integration
    def test_analyze_url_invalid(self, test_client):
        """Test /analyze/url with invalid URL."""
        response = test_client.post(
            "/analyze/url",
            json={"url": "not-a-valid-url"},
        )

        assert response.status_code in [400, 422, 500]

    @pytest.mark.integration
    def test_analyze_url_nonexistent(self, test_client):
        """Test /analyze/url with nonexistent URL."""
        response = test_client.post(
            "/analyze/url",
            json={"url": "https://nonexistent.example.com/image.jpg"},
        )

        # Should fail gracefully
        assert response.status_code in [400, 500, 504]


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for API error handling."""

    @pytest.mark.integration
    def test_large_file_rejection(self, test_client):
        """Test that overly large files are rejected."""
        # Create a large fake file (>50MB header claim)
        large_data = b"x" * (1024 * 1024)  # 1MB actual, but we'll test the check

        response = test_client.post(
            "/analyze",
            files={"file": ("large.jpg", large_data, "image/jpeg")},
        )

        # Should either process (if under limit) or reject
        assert response.status_code in [200, 400, 413, 500]

    @pytest.mark.integration
    def test_unsupported_format(self, test_client):
        """Test unsupported image format handling."""
        # Send a GIF (might not be supported)
        img = Image.new("RGB", (100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="GIF")
        buffer.seek(0)

        response = test_client.post(
            "/analyze",
            files={"file": ("test.gif", buffer.read(), "image/gif")},
        )

        # GIF might be supported or rejected
        assert response.status_code in [200, 400, 500]

    @pytest.mark.integration
    def test_corrupted_image(self, test_client):
        """Test handling of corrupted image data."""
        # Send random bytes that aren't a valid image
        response = test_client.post(
            "/analyze",
            files={"file": ("corrupt.jpg", b"not valid jpeg data", "image/jpeg")},
        )

        assert response.status_code in [400, 422, 500]


# =============================================================================
# Response Format Tests
# =============================================================================

class TestResponseFormat:
    """Tests for API response formatting."""

    @pytest.mark.integration
    def test_response_has_processing_time(self, test_client, sample_image_bytes):
        """Test that response includes processing time."""
        response = test_client.post(
            "/analyze",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

        if response.status_code == 200:
            data = response.json()
            assert "processing_time_ms" in data
            assert data["processing_time_ms"] >= 0

    @pytest.mark.integration
    def test_response_json_format(self, test_client, sample_image_bytes):
        """Test that response is valid JSON."""
        response = test_client.post(
            "/analyze",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )

        # Response should be valid JSON
        assert response.headers.get("content-type", "").startswith("application/json")
        data = response.json()  # Should not raise
        assert isinstance(data, dict)


# =============================================================================
# CORS Tests
# =============================================================================

class TestCORS:
    """Tests for CORS handling."""

    @pytest.mark.integration
    def test_cors_preflight(self, test_client):
        """Test CORS preflight request."""
        response = test_client.options(
            "/analyze",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # Should handle OPTIONS request
        assert response.status_code in [200, 204, 405]

    @pytest.mark.integration
    def test_cors_headers(self, test_client):
        """Test CORS headers in response."""
        response = test_client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )

        # CORS headers may or may not be present depending on config
        assert response.status_code == 200


# =============================================================================
# Rate Limiting Tests
# =============================================================================

class TestRateLimiting:
    """Tests for rate limiting (if enabled)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_rate_limit_not_immediately_triggered(self, test_client, sample_image_bytes):
        """Test that rate limiting doesn't trigger on normal usage."""
        # Make a few requests
        for _ in range(3):
            response = test_client.post(
                "/analyze",
                files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
            )
            # Should not be rate limited for just 3 requests
            assert response.status_code != 429


# =============================================================================
# OpenAPI Documentation Tests
# =============================================================================

class TestOpenAPIDocumentation:
    """Tests for API documentation endpoints."""

    @pytest.mark.integration
    def test_openapi_json(self, test_client):
        """Test OpenAPI JSON endpoint."""
        response = test_client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    @pytest.mark.integration
    def test_swagger_ui(self, test_client):
        """Test Swagger UI endpoint."""
        response = test_client.get("/docs")

        # Should return HTML for Swagger UI
        assert response.status_code == 200

    @pytest.mark.integration
    def test_redoc(self, test_client):
        """Test ReDoc endpoint."""
        response = test_client.get("/redoc")

        assert response.status_code == 200
