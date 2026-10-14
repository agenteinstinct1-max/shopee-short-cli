from shopee_short.pronunciation import prepare_for_tts


def test_brand_decimal_and_code():
    out = prepare_for_tts("Magcubic HY300 com Bluetooth 5.4 e Wi-Fi 6, 1,6 metro.")
    assert "Mag quiúbic" in out and "agá ípsilon trezentos" in out
    assert "cinco vírgula quatro" in out and "um vírgula seis" in out
    assert "uai fai" in out and "blu tuf" in out


def test_user_dictionary_overrides():
    assert prepare_for_tts("Foo", {"foo": "fú"}) == "fú"


def test_plain_integers_untouched():
    assert "135" in prepare_for_tts("até 135 polegadas")
