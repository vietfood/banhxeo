from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel

from banhxeo.utils.logging import DEFAULT_LOGGER


class Transforms(BaseModel):
    # arbitrary dictionary acts as metadata for transforms
    metadata: Dict[str, Any] = {}

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class ComposeTransforms:
    """Composes several transforms together.
    - Source: https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
    - Example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.PILToTensor(),
    >>>     transforms.ConvertImageDtype(torch.float),
    >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, text) -> str:
        for t in self.transforms:
            text = t(text)
        return text

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class RemoveHTMLTag(Transforms):
    def __call__(self, text: str) -> str:
        import re

        # stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
        pattern = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
        return pattern.sub(r"", text)


class RemoveURL(Transforms):
    def __call__(self, text: str) -> str:
        import re

        pattern = re.compile(r"https?://\S+|www\.\S+")
        return pattern.sub(r"", text)


class RemovePunctuation(Transforms):
    def __call__(self, text: str) -> str:
        import string

        return text.translate(str.maketrans("", "", string.punctuation))


class Strip(Transforms):
    def __call__(self, text: str) -> str:
        if self.metadata["lower"]:
            return text.strip().lower()
        else:
            return text.strip()


class SpellingCorrection(Transforms):
    def __call__(self, text: str) -> str:
        try:
            from textblob import TextBlob  # type: ignore

            correct_text = TextBlob(text)
            return correct_text.correct().string
        except ImportError:
            DEFAULT_LOGGER.warning(
                "You need to install `textblob` to use `SpellingCorrection` transform"
            )
            return text
