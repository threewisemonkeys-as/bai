import litellm


def main(
    model: str,
    prompt: str,
):
    input = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt,
                }
            ],
        }
    ]

    response = litellm.responses(
        model=model,
        input=input
    )

    try:
        response_output_text = response.output[-1].content[0].text
    except AttributeError as e:
        raise RuntimeError(f"Error in response-\n {response}") from e

    print(response_output_text)


if __name__ == '__main__':
    import fire
    fire.Fire(main)