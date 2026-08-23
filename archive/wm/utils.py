import numpy as np
from typing import Any
from pathlib import Path
import base64
import io

from PIL import Image


def image_to_base64(image_data: Path | str | np.ndarray):
    if isinstance(image_data, (str, Path)):
        image_path = Path(image_data)
        image = Image.open(image_path)
        format = image_path.suffix
    elif isinstance(image_data, np.ndarray):
        image = Image.fromarray(image_data.astype('uint8'), 'RGB')
        format = "png"
    else:
        raise RuntimeError(f"Image data type not supported: {type(image_data)}")

    image_bytes = io.BytesIO()
    image.save(image_bytes, format=format)
    base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return base64_image



def validate_response_fields(response_dict: dict, response_output_text: str, required_fields: list[str]):
    """Validate that required fields exist in response.

    Args:
        response_dict: Parsed XML response as dictionary
        response_output_text: Original response text for error messages
        required_fields: List of required field names

    Raises:
        RuntimeError: If any required field is missing
    """
    for field in required_fields:
        if field not in response_dict:
            error_msg = f"{field.capitalize()} missing from response -\n{response_output_text}"
            raise RuntimeError(error_msg)

def build_llm_input(prompt: str, images: list[np.ndarray] | None = None) -> list[dict]:
    """Build LLM input with text prompt and images.

    Args:
        prompt: Text prompt for the LLM
        images: List of image arrays to include

    Returns:
        Formatted input list for LiteLLM
    """
    content = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]

    if images is not None:
        for image in images:
            img_b64 = image_to_base64(image)
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{img_b64}"
            })

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
    if f"<{key}>" in data:
        data = data.split(f"<{key}>")[1]
        if f"</{key}>" in data:
            return data.split(f"</{key}>")[0]
        

def extract_xml_kv(data: str, keys: list[str]) -> dict[str, Any]:
    extracted = {}
    for k in keys:
        if (v := extract_xml_key(data, k)) is not None:
            extracted[k] = v
    return extracted

