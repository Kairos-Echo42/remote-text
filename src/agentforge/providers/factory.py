from __future__ import annotations

from agentforge.config import get_settings
from agentforge.models import ModelProfile
from agentforge.providers.base import EmbeddingProvider, ModelProvider
from agentforge.providers.fake import FakeEmbeddingProvider, FakeModelProvider
from agentforge.providers.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleModelProvider,
)
from agentforge.security import decrypt_secret


def create_model_provider(profile: ModelProfile | None = None) -> ModelProvider:
    settings = get_settings()
    if profile is None or profile.provider == "fake":
        return FakeModelProvider()
    api_key = decrypted_profile_secret(profile)
    return OpenAICompatibleModelProvider(
        profile.model,
        api_key=api_key,
        base_url=profile.base_url or settings.openai_base_url,
        default_params=profile.default_params,
    )


def create_embedding_provider(profile: ModelProfile | None = None) -> EmbeddingProvider:
    settings = get_settings()
    if profile is None or profile.provider == "fake":
        return FakeEmbeddingProvider()
    return OpenAICompatibleEmbeddingProvider(
        profile.model,
        api_key=decrypted_profile_secret(profile),
        base_url=profile.base_url or settings.openai_base_url,
    )


def decrypted_profile_secret(profile: ModelProfile) -> str | None:
    if profile.credential is None:
        return get_settings().openai_api_key
    return decrypt_secret(profile.credential.encrypted_value)
