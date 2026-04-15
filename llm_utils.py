"""Utility functions for LLM interactions."""

import base64
import io
from typing import Any


def _pil_to_input_image_part(image) -> dict:
    """Convert a PIL.Image to a responses-API input_image content part."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{b64}",
    }


def build_llm_input(prompt: str, images: list | None = None) -> list[Any]:
    """Build LLM input with text prompt and optional images.

    Args:
        prompt: Text prompt for the LLM
        images: Optional list of PIL.Image objects to attach as input_image parts

    Returns:
        Formatted input list for LiteLLM
    """
    content: list[dict] = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]

    if images:
        for img in images:
            if img is None:
                continue
            content.append(_pil_to_input_image_part(img))

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def extract_llm_response_text(response) -> str:
    """Extract text from LLM response with error handling.

    Args:
        response: LiteLLM response object

    Returns:
        Extracted response text

    Raises:
        RuntimeError: If response format is invalid
    """
    try:
        response_output_text = response.output[-1].content[0].text
        return response_output_text
    except AttributeError as e:
        error_msg = f"Error in response-\n {response}"
        raise RuntimeError(error_msg) from e


def extract_xml_key(data: str, key: str) -> str | None:
    """Extract a single XML key from text.

    Args:
        data: Text containing XML tags
        key: XML tag name to extract

    Returns:
        Extracted text or None if not found
    """
    if f"<{key}>" in data:
        data = data.split(f"<{key}>")[1]
        if f"</{key}>" in data:
            return data.split(f"</{key}>")[0]
    return None


def extract_xml_kv(data: str, keys: list[str]) -> dict[str, Any]:
    """Extract multiple XML keys from text.

    Args:
        data: Text containing XML tags
        keys: List of XML tag names to extract

    Returns:
        Dictionary mapping keys to their extracted values
    """
    extracted = {}
    for k in keys:
        if (v := extract_xml_key(data, k)) is not None:
            extracted[k] = v
    return extracted


def build_llm_input_multiturn(
    history: list[dict],
    user_message: str,
    images: list | None = None,
) -> list[dict]:
    """Build multi-turn LLM input by appending a user message to conversation history.

    Args:
        history: Existing conversation history (list of message dicts)
        user_message: New user message text to append
        images: Optional list of PIL.Image objects to attach to the new user turn

    Returns:
        Updated message list with the new user message appended
    """
    content: list[dict] = [{"type": "input_text", "text": user_message}]
    if images:
        for img in images:
            if img is None:
                continue
            content.append(_pil_to_input_image_part(img))
    return history + [
        {
            "role": "user",
            "content": content,
        }
    ]


def append_assistant_message(history: list[dict], text: str) -> list[dict]:
    """Append an assistant response to conversation history.

    Args:
        history: Existing conversation history
        text: Assistant response text

    Returns:
        Updated message list with the assistant message appended
    """
    return history + [
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }
    ]


def validate_response_fields(response_dict: dict, response_output_text: str, required_fields: list[str]) -> bool:
    """Validate that required fields exist in response.

    Args:
        response_dict: Parsed XML response as dictionary
        response_output_text: Original response text for error messages
        required_fields: List of required field names
    """
    for field in required_fields:
        if field not in response_dict:
            # error_msg = f"{field.capitalize()} missing from response -\n{response_output_text}"
            # raise RuntimeError(error_msg)
            return False
        
    return True
