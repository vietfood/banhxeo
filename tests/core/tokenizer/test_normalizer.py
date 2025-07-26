# NOTE: This test is written by Claude 4. Thank you !

import unicodedata

import pytest

from banhxeo.core.tokenizer.normalizers import (
    LowercaseNormalizer,
    NFCNormalizer,
    NormalizedString,
    SequenceNormalizer,
    StripNormalizer,
)


class TestStripNormalizer:
    @pytest.mark.parametrize(
        "input_str, expected_output",
        [
            # Basic cases
            (" hello world ", "hello world"),
            (" ", ""),
            ("", ""),
            (" one leading", "one leading"),
            ("one trailing ", "one trailing"),
            ("     Hello, World!  \r\n", "Hello, World!"),
            # Edge cases with various whitespace characters
            ("\t\n\r\f\v hello \t\n\r\f\v", "hello"),
            ("\u00a0hello\u00a0", "hello"),  # Non-breaking space
            ("\u2000\u2001\u2002hello\u2003\u2004", "hello"),  # En quad, em quad, etc.
            ("\u3000hello\u3000", "hello"),  # Ideographic space
            # Zero-width characters
            ("\u200b\u200c\u200dhello\u200b\u200c\u200d", "hello"),
            # Mixed whitespace
            (" \t\n\r hello world \t\n\r ", "hello world"),
            # Only whitespace
            ("\t\n\r\f\v", ""),
            ("\u00a0\u2000\u3000", ""),
            # No whitespace to strip
            ("hello", "hello"),
            ("hello world", "hello world"),
            # Unicode content with whitespace
            (" 你好世界 ", "你好世界"),
            (" مرحبا بالعالم ", "مرحبا بالعالم"),
            (" Здравствуй мир ", "Здравствуй мир"),
        ],
    )
    def test_strip_normalizer_comprehensive(self, input_str, expected_output):
        normalizer = StripNormalizer()
        input_normalized = NormalizedString.from_str(input_str)
        output = normalizer.normalize(input_normalized)
        assert output.normalized == expected_output

    def test_strip_normalizer_preserves_internal_whitespace(self):
        normalizer = StripNormalizer()
        input_str = "  hello   world  test  "
        expected = "hello   world  test"

        input_normalized = NormalizedString.from_str(input_str)
        output = normalizer.normalize(input_normalized)
        assert output.normalized == expected


class TestLowercaseNormalizer:
    @pytest.mark.parametrize(
        "input_str, expected_output",
        [
            # Basic cases
            ("HELLO", "hello"),
            ("Hello", "hello"),
            ("hello", "hello"),
            ("", ""),
            (" HELLO ", " hello "),
            # Mixed case
            ("HeLLo WoRLd", "hello world"),
            ("123ABC", "123abc"),
            ("ABC123def", "abc123def"),
            # Unicode cases
            ("ÀÁÂÃÄÅÆÇÈÉÊË", "àáâãäåæçèéêë"),
            ("ÌÍÎÏÐÑÒÓÔÕÖ", "ìíîïðñòóôõö"),
            ("ØÙÚÛÜÝÞ", "øùúûüýþ"),
            # Special Unicode cases
            ("İSTANBUL", "i̇stanbul"),  # Turkish I with dot
            ("МОСКВА", "москва"),  # Cyrillic
            ("ΤΌΚΥΟ", "τόκυο"),  # Greek
            ("ΕΛΛΗΝΙΚΆ", "ελληνικά"),  # Greek
            # German sharp S
            ("STRAßE", "straße"),
            # Numbers and symbols (should remain unchanged)
            ("123!@#$%", "123!@#$%"),
            ("Hello123World!", "hello123world!"),
            # Emoji and symbols
            ("HELLO 😀 WORLD", "hello 😀 world"),
            ("TEST™", "test™"),
            # Edge cases with combining characters
            ("É", "é"),  # E with acute
            ("Ñ", "ñ"),  # N with tilde
        ],
    )
    def test_lowercase_normalizer_comprehensive(self, input_str, expected_output):
        normalizer = LowercaseNormalizer()
        input_normalized = NormalizedString.from_str(input_str)
        output = normalizer.normalize(input_normalized)
        assert output.normalized == expected_output

    def test_lowercase_preserves_numbers_and_symbols(self):
        normalizer = LowercaseNormalizer()
        input_str = "Test123!@#$%^&*()_+-=[]{}|;:,.<>?/~`"
        expected = "test123!@#$%^&*()_+-=[]{}|;:,.<>?/~`"

        input_normalized = NormalizedString.from_str(input_str)
        output = normalizer.normalize(input_normalized)
        assert output.normalized == expected


class TestNFCNormalizer:
    @pytest.mark.parametrize(
        "input_str, expected_output",
        [
            # Basic ASCII (no change expected)
            ("hello", "hello"),
            ("Hello World", "Hello World"),
            ("", ""),
            # Composed vs decomposed characters
            ("é", "é"),  # Already NFC
            ("e\u0301", "é"),  # NFD -> NFC (e + combining acute)
            ("ñ", "ñ"),  # Already NFC
            ("n\u0303", "ñ"),  # NFD -> NFC (n + combining tilde)
            # Complex combining characters
            ("ệ", "ệ"),  # e with circumflex and dot below
            ("e\u0302\u0323", "ệ"),  # Decomposed version
            # Multiple combining characters
            ("à̧", "à̧"),  # a with grave and cedilla
            ("a\u0300\u0327", "à̧"),  # Decomposed
            # Hangul (Korean)
            ("한", "한"),  # Already composed
            ("하ᄂ", "한"),  # Decomposed Hangul
            # Arabic with diacritics
            ("مُحَمَّد", "مُحَمَّد"),
            # Mixed content
            ("café naïve résumé", "café naïve résumé"),
            ("cafe\u0301 nai\u0308ve re\u0301sume\u0301", "café naïve résumé"),
            # Edge cases
            ("\u0041\u0300", "À"),  # A + grave -> À
            ("\u0065\u0301\u0302", "ế"),  # e + acute + circumflex
        ],
    )
    def test_nfc_normalizer_comprehensive(self, input_str, expected_output):
        normalizer = NFCNormalizer()
        input_normalized = NormalizedString.from_str(input_str)
        output = normalizer.normalize(input_normalized)
        assert output.normalized == expected_output

    def test_nfc_normalizer_idempotent(self):
        """Test that applying NFC normalization twice gives the same result"""
        normalizer = NFCNormalizer()
        test_strings = [
            "e\u0301",  # e + combining acute
            "n\u0303",  # n + combining tilde
            "a\u0300\u0327",  # a + grave + cedilla
        ]

        for input_str in test_strings:
            input_normalized = NormalizedString.from_str(input_str)
            first_pass = normalizer.normalize(input_normalized)
            second_pass = normalizer.normalize(first_pass)
            assert first_pass.normalized == second_pass.normalized

    def test_nfc_normalizer_equivalence(self):
        """Test that NFC normalization produces equivalent results to unicodedata.normalize"""
        normalizer = NFCNormalizer()
        test_strings = [
            "e\u0301",
            "n\u0303",
            "a\u0300\u0327",
            "하ᄂ",
        ]

        for input_str in test_strings:
            input_normalized = NormalizedString.from_str(input_str)
            our_result = normalizer.normalize(input_normalized)
            expected = unicodedata.normalize("NFC", input_str)
            assert our_result.normalized == expected


class TestSequenceNormalizer:
    @pytest.mark.parametrize(
        "input_str, expected_output, sequence",
        [
            # Basic sequences
            (" HELLO ", "hello", [LowercaseNormalizer, StripNormalizer]),
            (" HELLO ", "hello", [StripNormalizer, LowercaseNormalizer]),
            # Three normalizers
            (
                " E\u0301XAMPLE ",
                "éxample",
                [NFCNormalizer, LowercaseNormalizer, StripNormalizer],
            ),
            (
                " E\u0301XAMPLE ",
                "éxample",
                [StripNormalizer, NFCNormalizer, LowercaseNormalizer],
            ),
            (
                " E\u0301XAMPLE ",
                "éxample",
                [LowercaseNormalizer, StripNormalizer, NFCNormalizer],
            ),
            # Order matters for some cases
            (
                "  CaFe\u0301  ",
                "café",
                [NFCNormalizer, LowercaseNormalizer, StripNormalizer],
            ),
            (
                "  CaFe\u0301  ",
                "café",
                [StripNormalizer, NFCNormalizer, LowercaseNormalizer],
            ),
            # Empty sequence (should return unchanged)
            ("Hello", "Hello", []),
            # Single normalizer in sequence
            ("HELLO", "hello", [LowercaseNormalizer]),
            (" hello ", "hello", [StripNormalizer]),
            ("e\u0301", "é", [NFCNormalizer]),
            # Complex Unicode cases
            (
                " MO\u0301SCOW ",
                "móscow",
                [NFCNormalizer, LowercaseNormalizer, StripNormalizer],
            ),
            (
                " \u0041\u0300LPHA ",
                "àlpha",
                [NFCNormalizer, LowercaseNormalizer, StripNormalizer],
            ),
        ],
    )
    def test_sequence_normalizer_comprehensive(
        self, input_str, expected_output, sequence
    ):
        normalizer = SequenceNormalizer([cls() for cls in sequence])
        input_normalized = NormalizedString.from_str(input_str)
        output = normalizer.normalize(input_normalized)
        assert output.normalized == expected_output

    def test_sequence_normalizer_order_independence_cases(self):
        """Test cases where order doesn't matter"""
        test_cases = [
            (" HELLO ", "hello", [LowercaseNormalizer, StripNormalizer]),
            (" HELLO ", "hello", [StripNormalizer, LowercaseNormalizer]),
        ]

        for input_str, expected, sequence in test_cases:
            normalizer = SequenceNormalizer([cls() for cls in sequence])
            input_normalized = NormalizedString.from_str(input_str)
            output = normalizer.normalize(input_normalized)
            assert output.normalized == expected

    def test_sequence_normalizer_order_dependence(self):
        """Test that some sequences may have order-dependent behavior"""
        input_str = " E\u0301XAMPLE "

        # Different orders should still produce the same final result for this case
        sequences = [
            [NFCNormalizer, LowercaseNormalizer, StripNormalizer],
            [StripNormalizer, NFCNormalizer, LowercaseNormalizer],
            [LowercaseNormalizer, StripNormalizer, NFCNormalizer],
        ]

        results = []
        for sequence in sequences:
            normalizer = SequenceNormalizer([cls() for cls in sequence])
            input_normalized = NormalizedString.from_str(input_str)
            output = normalizer.normalize(input_normalized)
            results.append(output.normalized)

        # All should produce the same result
        assert all(result == results[0] for result in results)
        assert results[0] == "éxample"

    def test_sequence_normalizer_empty_sequence(self):
        """Test that empty sequence returns input unchanged"""
        normalizer = SequenceNormalizer([])
        input_str = " HELLO WORLD "
        input_normalized = NormalizedString.from_str(input_str)
        output = normalizer.normalize(input_normalized)
        assert output.normalized == input_str

    def test_sequence_normalizer_single_normalizer(self):
        """Test sequence with single normalizer behaves like that normalizer alone"""
        input_str = " HELLO "

        # Test with single normalizer in sequence
        sequence_normalizer = SequenceNormalizer([LowercaseNormalizer()])
        sequence_result = sequence_normalizer.normalize(
            NormalizedString.from_str(input_str)
        )

        # Test with normalizer alone
        single_normalizer = LowercaseNormalizer()
        single_result = single_normalizer.normalize(
            NormalizedString.from_str(input_str)
        )

        assert sequence_result.normalized == single_result.normalized


class TestNormalizedStringEdgeCases:
    def test_normalized_string_empty(self):
        """Test NormalizedString with empty input"""
        ns = NormalizedString.from_str("")
        assert ns.normalized == ""

    def test_normalized_string_unicode_edge_cases(self):
        """Test NormalizedString with various Unicode edge cases"""
        test_cases = [
            "\u0000",  # Null character
            "\ufffe",  # Non-character
            "\U0001f600",  # Emoji
            "\u200b",  # Zero-width space
            "\ufeff",  # Byte order mark
        ]

        for test_str in test_cases:
            ns = NormalizedString.from_str(test_str)
            assert ns.normalized == test_str


class TestNormalizerChaining:
    """Test complex chaining scenarios and edge cases"""

    def test_all_normalizers_chained(self):
        """Test all normalizers chained together"""
        normalizer = SequenceNormalizer(
            [
                StripNormalizer(),
                NFCNormalizer(),
                LowercaseNormalizer(),
            ]
        )

        input_str = (
            "  CA\u0301FE\u0301 EXAMPLE  "  # "  CAFÉ EXAMPLE  " in decomposed form
        )
        expected = "café example"

        input_normalized = NormalizedString.from_str(input_str)
        output = normalizer.normalize(input_normalized)
        assert output.normalized == expected

    def test_normalizer_with_very_long_string(self):
        """Test normalizers with very long strings"""
        long_string = (
            "A" * 10000 + "\u0301" + "B" * 10000
        )  # Long string with combining character

        normalizer = SequenceNormalizer(
            [
                NFCNormalizer(),
                LowercaseNormalizer(),
            ]
        )

        input_normalized = NormalizedString.from_str(long_string)
        output = normalizer.normalize(input_normalized)

        # Should handle long strings without issues
        assert (
            len(output.normalized) == len(long_string) - 1
        )  # Combining character composed
        assert output.normalized.startswith("a" * 10000 + "á")
        assert output.normalized.endswith("b" * 10000)

    def test_normalizer_with_mixed_scripts(self):
        """Test normalizers with mixed writing systems"""
        mixed_text = " HELLO مرحبا МОСКВА 你好 こんにちは 🌍 "

        normalizer = SequenceNormalizer(
            [
                StripNormalizer(),
                LowercaseNormalizer(),
            ]
        )

        input_normalized = NormalizedString.from_str(mixed_text)
        output = normalizer.normalize(input_normalized)

        # Should preserve non-Latin scripts and handle Latin properly
        expected = "hello مرحبا москва 你好 こんにちは 🌍"
        assert output.normalized == expected
