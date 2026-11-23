import unittest

from lynceus.advice_llm import (
    _format_generation_prompt,
    _pad_token_id,
    _strip_think_blocks,
)


class AdviceLlmTest(unittest.TestCase):
    def test_strip_think_blocks_removes_closed_truncated_and_case_variants(self):
        cases = {
            "<think>hidden</think>Visible": "Visible",
            "<THINK>hidden</ThInK>Visible": "Visible",
            "Visible\n<think>truncated reasoning": "Visible\n",
            "A <think>one</think>B <think>two</think>C": "A B C",
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(_strip_think_blocks(text), expected)

    def test_pad_token_id_preserves_zero(self):
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2

        self.assertEqual(_pad_token_id(Tokenizer()), 0)

    def test_pad_token_id_falls_back_only_when_missing(self):
        class Tokenizer:
            pad_token_id = None
            eos_token_id = 2

        self.assertEqual(_pad_token_id(Tokenizer()), 2)

    def test_format_generation_prompt_disables_thinking_when_supported(self):
        class Tokenizer:
            def apply_chat_template(
                self, messages, *, add_generation_prompt, enable_thinking, tokenize
            ):
                self.messages = messages
                self.add_generation_prompt = add_generation_prompt
                self.enable_thinking = enable_thinking
                self.tokenize = tokenize
                return "formatted"

        tokenizer = Tokenizer()

        self.assertEqual(_format_generation_prompt(tokenizer, "prompt"), "formatted")
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "prompt"}])
        self.assertTrue(tokenizer.add_generation_prompt)
        self.assertFalse(tokenizer.enable_thinking)
        self.assertFalse(tokenizer.tokenize)

    def test_format_generation_prompt_supports_templates_without_thinking_flag(self):
        class Tokenizer:
            def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
                self.messages = messages
                self.add_generation_prompt = add_generation_prompt
                self.tokenize = tokenize
                return "formatted"

        tokenizer = Tokenizer()

        self.assertEqual(_format_generation_prompt(tokenizer, "prompt"), "formatted")
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "prompt"}])
        self.assertTrue(tokenizer.add_generation_prompt)
        self.assertFalse(tokenizer.tokenize)


if __name__ == "__main__":
    unittest.main()
