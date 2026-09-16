import json
import mimetypes
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 2
DRIVING_SCENE_REASONING_PROMPT = """You are providing structured scene reasoning for a research autonomous-driving trajectory pipeline. You are not controlling a real vehicle and must not claim unsupported sensor measurements.

Analyze the attached current front-camera image and the context below. Use only the current image, the supplied 2D detections, and historical ego-motion available before this frame.

OBJECT DETECTIONS FROM THE PRETRAINED 2D YOLO DETECTOR FOR THIS SAME IMAGE:
{detections_context}

HISTORICAL EGO-MOTION ONLY:
{historical_motion_context}

PREVIOUS DRIVING INTENT:
{previous_intent}

Reason about:
1. Overall scene and road/environment context.
2. Important detected objects and which are critical to the ego vehicle.
3. Relative interaction and qualitative risk without inventing distance, velocity, depth, or 3D coordinates.
4. The driving situation and likely driving intent.
5. Expected near-future behavior and qualitative implications for the existing trajectory predictor.

Return only one valid JSON object matching this schema:
{{
    "scene_description": "string",
    "road_context": "string",
    "critical_objects": [
        {{"class": "string", "importance": "string", "reason": "string"}}
    ],
    "driving_situation": "string",
    "driving_intent": "string",
    "risk_level": "low | medium | high | unknown",
    "recommended_behavior": "string",
    "expected_near_future_behavior": "string",
    "motion_implications": "string"
}}

Do not include numeric future speeds, curvatures, object distances, object velocities, or 3D coordinates. The existing OpenEMMA motion code will retain responsibility for numeric trajectory integration."""
REQUIRED_FIELDS = (
    "scene_description",
    "road_context",
    "critical_objects",
    "driving_situation",
    "driving_intent",
    "risk_level",
    "recommended_behavior",
    "expected_near_future_behavior",
    "motion_implications",
)


class GeminiIntegrationError(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def _load_api_key() -> str:
    load_dotenv(Path(__file__).resolve().with_name(".env"))
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiIntegrationError(
            "missing_api_key",
            "GEMINI_API_KEY is missing or empty in the project environment.",
        )
    return api_key


def _get_configuration() -> tuple[str, int, int]:
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    try:
        timeout_seconds = max(
            1, int(os.getenv("GEMINI_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
        )
        max_retries = max(
            0, int(os.getenv("GEMINI_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
        )
    except ValueError as exc:
        raise GeminiIntegrationError(
            "configuration",
            "GEMINI_TIMEOUT_SECONDS and GEMINI_MAX_RETRIES must be integers.",
        ) from exc
    return model, timeout_seconds, max_retries


def _json_context(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)


@lru_cache(maxsize=4)
def _get_client(api_key: str, timeout_seconds: int) -> genai.Client:
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
    )


def _build_prompt(
    detections: list[dict[str, Any]],
    historical_motion: dict[str, Any],
    previous_intent: str | None,
) -> str:
    previous_intent_text = previous_intent or "None available"
    return DRIVING_SCENE_REASONING_PROMPT.format(
        detections_context=_json_context(detections),
        historical_motion_context=_json_context(historical_motion),
        previous_intent=previous_intent_text,
    )


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    raise GeminiIntegrationError(
        "unexpected_response",
        "Gemini returned no textual response.",
    )


def _extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    if not fenced:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise GeminiIntegrationError(
                "malformed_response",
                "Gemini response did not contain a JSON object.",
            )
        candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise GeminiIntegrationError(
            "malformed_response",
            f"Gemini response JSON could not be parsed: {exc.msg}.",
        ) from exc
    if not isinstance(parsed, dict):
        raise GeminiIntegrationError(
            "unexpected_response",
            "Gemini response JSON must be an object.",
        )
    return parsed


def _validate_analysis(parsed: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in parsed]
    if missing:
        raise GeminiIntegrationError(
            "missing_fields",
            "Gemini response is missing required fields: " + ", ".join(missing),
        )
    for field in REQUIRED_FIELDS:
        if field == "critical_objects":
            if not isinstance(parsed[field], list):
                raise GeminiIntegrationError(
                    "invalid_fields",
                    "Gemini field critical_objects must be a list.",
                )
            for index, item in enumerate(parsed[field]):
                if not isinstance(item, dict):
                    raise GeminiIntegrationError(
                        "invalid_fields",
                        f"Gemini critical_objects[{index}] must be an object.",
                    )
                missing_object_fields = [
                    key for key in ("class", "importance", "reason") if key not in item
                ]
                if missing_object_fields:
                    raise GeminiIntegrationError(
                        "missing_fields",
                        f"Gemini critical_objects[{index}] is missing: "
                        + ", ".join(missing_object_fields),
                    )
        elif not isinstance(parsed[field], str) or not parsed[field].strip():
            raise GeminiIntegrationError(
                "invalid_fields",
                f"Gemini field {field} must be a non-empty string.",
            )
    return parsed


def _classify_exception(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "auth" in name or "permission" in message or "unauthenticated" in message:
        return "authentication"
    if "rate" in name or "quota" in message or "429" in message:
        return "rate_limit"
    if "timeout" in name or "timeout" in message:
        return "timeout"
    if "connection" in name or "network" in message or "connect" in message:
        return "network"
    return "api_error"


def analyze_driving_scene(
    image_path: str | Path,
    detections: list[dict[str, Any]] | None,
    historical_motion: dict[str, Any],
    previous_intent: str | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise GeminiIntegrationError(
            "image_error",
            f"Driving image does not exist: {image_path}",
        )
    model, timeout_seconds, max_retries = _get_configuration()
    prompt = _build_prompt(detections or [], historical_motion, previous_intent)
    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        raise GeminiIntegrationError(
            "image_error",
            f"Driving image could not be read: {image_path}",
        ) from exc

    try:
        client = _get_client(_load_api_key(), timeout_seconds)
    except GeminiIntegrationError:
        raise
    except Exception as exc:
        raise GeminiIntegrationError(
            "client_error",
            f"Gemini client initialization failed: {type(exc).__name__}.",
        ) from exc

    contents = [
        prompt,
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=mimetypes.guess_type(image_path.name)[0] or "image/jpeg",
        ),
    ]
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
    )
    last_error: GeminiIntegrationError | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return _validate_analysis(_extract_json(_response_text(response)))
        except GeminiIntegrationError:
            raise
        except Exception as exc:
            category = _classify_exception(exc)
            last_error = GeminiIntegrationError(
                category,
                f"Gemini request failed ({category}): {type(exc).__name__}.",
            )
            if attempt < max_retries and category in {"network", "timeout", "rate_limit", "api_error"}:
                time.sleep(2**attempt)
                continue
            raise last_error from exc
    raise last_error or GeminiIntegrationError("api_error", "Gemini request failed.")
