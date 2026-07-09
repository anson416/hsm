from __future__ import annotations

import base64
import io
import json as _json
import os
from typing import List, Union

from dotenv import load_dotenv
from matplotlib.figure import Figure
from openai import OpenAI
from PIL import Image

from .base_session import BaseVLMSession
from .utils import extract_code, extract_json, extract_program

MODEL: str = "gpt-5.1-2025-11-13"
# MODEL: str = "gpt-4o-2024-08-06"
# MODEL = "o4-mini"
# MODEL = "gpt-4.1-2025-04-14"
# MODEL: str = "gpt-5"
# Optional OpenAI-compatible endpoint override for the audit. Leave unset to use
# the default OpenAI endpoint. When pointing at chatanywhere, also export
# OPENAI_API_KEY=$CHATANYWHERE_API_KEY so the client authenticates correctly.
# All four of base_url / api_key / model / temperature may be overridden via the
# OPENAI_* env vars (set by cli.py from its --base-url/--api-key/--model/
# --temperature flags) so the whole pipeline picks them up without threading
# parameters through every create_session() call site.
CUSTOM_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")

DEFAULT_MODEL: str = "gpt-5.1-2025-11-13"
REASONING_MODELS: list[str] = [
    "o3-mini",
    "o4-mini",
    "gpt-5",
    "gpt-5.1-2025-11-13",
]
RETRY_COUNT: int = 10
MAX_IMAGE_SIZE: int = 2048


def _env_float(name: str, default: float) -> float:
    """Parse a float from an env var, returning `default` on absence/parse error."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


class Session(BaseVLMSession):
    """GPT-based VLM session using OpenAI API."""

    def __init__(
        self,
        prompts_path,
        model=None,
        temperature: float | None = None,
        output_dir: str = "",
        prompt_info: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize a GPT Session.

        Endpoint / API key / model / temperature resolution order:
          explicit arg -> OPENAI_* env var -> module default.
        This lets cli.py set the env vars once and have every Session
        created across the pipeline pick them up.
        """
        load_dotenv()
        # Resolve base_url + api_key from env (explicit None -> OpenAI default).
        _base_url = os.environ.get("OPENAI_BASE_URL")
        _api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "CHATANYWHERE_API_KEY"
        )
        self.client = OpenAI(
            base_url=_base_url if _base_url else None, api_key=_api_key
        )
        # model: explicit arg wins, else env, else module default.
        resolved_model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        # temperature: explicit arg wins, else env, else 0.7.
        if temperature is None:
            temperature = _env_float("OPENAI_TEMPERATURE", 0.7)
        self.model = resolved_model
        super().__init__(prompts_path, self.model, temperature, output_dir, prompt_info)

    def send(
        self,
        task: str,
        prompt_info: dict[str, str] | None = None,
        info_validate: bool = True,
        is_json: bool = False,
        verbose: bool = False,
        images: str | Figure | List[str | Figure] | None = None,
        image_detail: str = "high",
        append_text: str = "",
    ) -> str:
        """
        Send a message of a specific task to the VLM model and return the response.

        Args:
            task: string, the task of the message
            prompt_info: dictionary, the extra information for making the prompt for the task
            info_validate: boolean, whether to validate the input info
            is_json: boolean, whether the response should be in JSON format
            verbose: boolean, whether to print the prompt
            images: string, Figure, or list of them, the image(s) to be sent to the model
            image_detail: string, the detail level of the image
            append_text: string, additional text to append to the prompt

        Returns:
            response: string, the response from the model
        """

        self.logger.debug(f"Sending task: {task}")
        self.past_tasks.append(task)
        prompt = self._make_prompt(task, prompt_info, info_validate)
        if append_text:
            prompt = append_text + "\n\n" + prompt

        if images is not None:
            num_images = len(images) if isinstance(images, list) else 1
        else:
            num_images = 0
        self.logger.debug(
            f"Past messages: {len(self.past_messages)} Prompt length: {len(prompt)} with {num_images} images"
        )
        if verbose:
            self.logger.debug(f"Prompt:\n{prompt}")
        self._send(prompt, is_json, images, image_detail)
        response = self.past_responses[-1]
        if verbose:
            self.logger.debug(f"Response:\n{response}")

        return response

    def _encode_image(self, image_or_path, detail="auto"):
        """Encode image for VLM models"""
        if isinstance(image_or_path, str):
            img = Image.open(image_or_path)
        elif isinstance(image_or_path, Figure):
            buf = io.BytesIO()
            image_or_path.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)
        else:
            self.logger.debug(
                f"Unsupported image type: {type(image_or_path)}, value: {image_or_path}"
            )
            raise ValueError(
                f"Warning: Unsupported image type: {type(image_or_path)}. Please provide a file path or a matplotlib Figure."
            )

        # Optimize image size based on detail level
        if detail == "low":
            target_size = (512, 512)
        elif detail == "high":
            target_size = (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE)
        else:
            width, height = img.size
            if width * height <= 512 * 512:
                target_size = (512, 512)
            else:
                target_size = (MAX_IMAGE_SIZE, MAX_IMAGE_SIZE)

        # Preserve aspect ratio while resizing
        img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # removes alpha channel (Ref: https://www.oranlooney.com/post/gpt-cnn/)
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            img = background

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Strip a surrounding markdown code fence (```...```) from a model response.

        Some non-OpenAI models wrap JSON output in ```json ... ``` fences even when
        json mode is requested; json.loads() then fails at char 0. Strip the fence so
        downstream parsing succeeds.
        """
        stripped = text.strip()
        if not stripped.startswith("```"):
            return text
        # Drop the opening fence (with optional language tag, e.g. ```json).
        first_newline = stripped.find("\n")
        if first_newline == -1:
            return text
        inner = stripped[first_newline + 1 :]
        # Drop a trailing fence if present.
        if inner.rstrip().endswith("```"):
            inner = inner.rstrip()[:-3]
        return inner.strip()

    def _send(
        self,
        new_message: str,
        json: bool = False,
        images: Union[str, Figure, List[Union[str, Figure]], None] = None,
        image_detail="high",
    ) -> None:
        """Send message to VLM models with image."""
        message_content = []

        # Add text content first
        if new_message.strip():
            message_content.append({"type": "text", "text": new_message})

        # Handle multiple images
        if images is not None:
            # Convert single image to list for uniform processing and filter out None values
            image_list_raw = images if isinstance(images, list) else [images]
            image_list = [img for img in image_list_raw if img is not None]

            for image in image_list:
                try:
                    image_base64 = self._encode_image(image, detail=image_detail)
                except ValueError as exc:
                    self.logger.warning(
                        "Skipping unsupported image in _send (type=%s): %s",
                        type(image),
                        exc,
                    )
                    continue
                if image_base64 is None:
                    continue

                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": image_detail,
                        },
                    }
                )

        self.past_messages.append({"role": "user", "content": message_content})

        retries = 0
        max_retries = 10
        while retries < max_retries:
            params = {
                "model": self.model,
                "messages": self.past_messages,
                "response_format": {"type": "json_object"} if json else None,
                "temperature": self.temperature
                if self.model not in REASONING_MODELS
                else 1.0,
            }

            if self.model in REASONING_MODELS:
                params["reasoning_effort"] = "high"

            completion = self.client.chat.completions.create(**params)
            response = completion.choices[0].message.content

            if completion.usage:
                self.total_prompt_tokens += completion.usage.prompt_tokens
                self.total_completion_tokens += completion.usage.completion_tokens
                self.total_tokens_this_session += completion.usage.total_tokens

            # Treat empty/whitespace-only content the same as None: a failed response
            # that should be retried, rather than handing "" to a downstream json.loads
            # ("Expecting value: line 1 column 1 (char 0)").
            if response is not None:
                response = response.strip()
            if response:
                # Some non-OpenAI models wrap JSON in ```json ... ``` fences even
                # under json mode; strip so json.loads() succeeds downstream.
                if json:
                    response = self._strip_code_fences(response)
                    # Validate now so a malformed-but-non-empty response is retried
                    # rather than thrown one frame up by the caller's json.loads().
                    try:
                        _json.loads(response)
                    except _json.JSONDecodeError:
                        self.logger.info(
                            "Received non-JSON response, retrying... (Attempt %d/%d)",
                            retries + 1,
                            max_retries,
                        )
                        retries += 1
                        continue
                self.past_messages.append({"role": "assistant", "content": response})
                self.past_responses.append(response)
                return

            self.logger.info(
                f"Received empty response, retrying... (Attempt {retries + 1}/{max_retries})"
            )
            retries += 1

        raise RuntimeError(
            f"Failed to get a valid response after {max_retries} attempts"
        )

