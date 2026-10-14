"""Erros de domínio com mensagens seguras para o terminal."""


class ShopeeShortError(Exception):
    exit_code = 1


class InvalidProductLink(ShopeeShortError):
    exit_code = 2


class SourceBlocked(ShopeeShortError):
    exit_code = 3


class ProductNotFound(ShopeeShortError):
    exit_code = 4


class UserDeclined(ShopeeShortError):
    exit_code = 5


class MediaProcessingError(ShopeeShortError):
    exit_code = 6


class ProviderError(ShopeeShortError):
    exit_code = 7
