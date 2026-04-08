"""Call an LLM with a prompt from a file, using a YAML config for model settings.

Example config (llm_config.yaml):
    client_name: openrouter
    model_id: google/gemini-2.5-flash
    api_key: sk-...          # optional, falls back to env var
    api_base: null           # optional base URL override
    temperature: 0.0
    max_tokens: 4096
    max_retries: 3
    system_prompt: null      # optional system prompt

Usage:
    uv run scripts/query_llm.py prompt.txt --config llm_config.yaml
    uv run scripts/query_llm.py prompt.txt --config llm_config.yaml --n 3
    uv run scripts/query_llm.py prompt.txt --config llm_config.yaml --n 3 --output out.txt
"""

import concurrent.futures
import sys
from pathlib import Path

import litellm
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model_name(cfg: dict) -> str:
    client_name = cfg.get("client_name", "openai")
    model_id = cfg["model_id"]
    if client_name == "vllm":
        return f"hosted_vllm/{model_id}"
    return f"{client_name}/{model_id}"


def call_once(kwargs: dict) -> str:
    response = litellm.completion(**kwargs)
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        raise RuntimeError(f"Error extracting response: {e}\nRaw response: {response}") from e


def main(
    input_file: str,
    config: str = "llm_config.yaml",
    system: str | None = None,
    n: int = 1,
    output: str | None = None,
):
    """Call an LLM with a prompt read from a file.

    Args:
        input_file: Path to file containing the prompt text.
        config: Path to YAML config file specifying the LLM.
        system: Optional system prompt (overrides config system_prompt).
        n: Number of independent responses to request (run in parallel).
        output: Optional path to write responses to (stdout if omitted).
    """
    prompt = Path(input_file).read_text()
    cfg = load_config(config)

    model_name = build_model_name(cfg)
    api_key = cfg.get("api_key")
    api_base = cfg.get("api_base") or cfg.get("base_url")
    system_prompt = system or cfg.get("system_prompt")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": cfg.get("temperature", 0.0),
        "max_tokens": cfg.get("max_tokens", 4096),
        "num_retries": cfg.get("max_retries", 3),
    }
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    if n == 1:
        try:
            responses = [call_once(kwargs)]
        except RuntimeError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
            futures = [executor.submit(call_once, kwargs) for _ in range(n)]
            responses = []
            for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    responses.append((i, fut.result()))
                except RuntimeError as e:
                    print(f"Response {i} failed: {e}", file=sys.stderr)

        responses = [text for _, text in sorted(responses)]

    separator = "\n" + "=" * 80 + "\n"
    if n == 1:
        result = responses[0]
    else:
        result = separator.join(f"--- Response {i+1} ---\n{r}" for i, r in enumerate(responses))

    if output:
        Path(output).write_text(result)
        print(f"Wrote {len(responses)} response(s) to {output}")
    else:
        print(result)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
