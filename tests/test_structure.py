"""Tests for lynceus.structure -- structural audit (Task F, Phase 3.4)."""

import unittest

from lynceus.structure import audit_structure


class StructureAuditTest(unittest.TestCase):
    def test_t5_suspiciously_even_paragraph_lengths(self):
        doc = (
            "This section describes the approach and the main assumptions it makes.\n\n"
            "This section describes the approach and the main assumptions it makes.\n\n"
            "This section describes the approach and the main assumptions it makes.\n\n"
            "This section describes the approach and the main assumptions it makes.\n\n"
            "This section describes the approach and the main assumptions it makes."
        )
        result = audit_structure(doc)
        names = [f["name"] for f in result["findings"]]
        self.assertIn("suspiciously_even_paragraphs", names)
        t5 = next(f for f in result["findings"]
                  if f["name"] == "suspiciously_even_paragraphs")
        self.assertEqual(t5["kb_ref"], "T5")
        self.assertGreaterEqual(t5["severity"], "low")

    def test_t3_hourglass_closing_fires(self):
        doc = (
            "It provides some introduction about the topic that is broad.\n\n"
            "It then goes into the body of the argument being made.\n\n"
            "In conclusion, a generic closing sentence that wraps it all up."
        )
        result = audit_structure(doc)
        names = [f["name"] for f in result["findings"]]
        self.assertIn("hourglass_closing", names)
        t3 = next(f for f in result["findings"]
                  if f["name"] == "hourglass_closing")
        self.assertEqual(t3["kb_ref"], "T3")

    def test_t3_silent_when_conclusion_is_substantive(self):
        doc = (
            "One paragraph of some meaningful body text goes here."
            "\n\nThe real takeaway from this work is that the slow season revealed "
            "more than the busy one ever did about how we actually work."
        )
        result = audit_structure(doc)
        self.assertNotIn("hourglass_closing",
                         [f["name"] for f in result["findings"]])

    def test_t2_challenges_future_header_fires(self):
        doc = (
            "An introduction paragraph gives the broad framing and context.\n\n"
            "Challenges and Future Directions\n\n"
            "A body paragraph discussing the challenges in more detail follows."
        )
        result = audit_structure(doc)
        names = [f["name"] for f in result["findings"]]
        self.assertIn("challenges_future_section", names)
        t2 = next(f for f in result["findings"]
                  if f["name"] == "challenges_future_section")
        self.assertEqual(t2["kb_ref"], "T2")

    def test_synthetic_ai_document_fires_t3_and_t5(self):
        doc = (
            "This section describes the approach and the main assumptions it makes.\n\n"
            "This section describes the approach and the main assumptions it makes.\n\n"
            "This section describes the approach and the main assumptions it makes.\n\n"
            "This section describes the approach and the main assumptions it makes.\n\n"
            "In conclusion, this paper concludes that the approach worked well enough."
        )
        result = audit_structure(doc)
        names = {f["name"] for f in result["findings"]}
        self.assertIn("suspiciously_even_paragraphs", names)
        self.assertIn("hourglass_closing", names)

    def test_natural_text_is_mostly_clean(self):
        doc = (
            "I grew up near a small house where the river would freeze most winters.\n\n"
            "Every spring the ice broke up in heavy chunks that jammed against the "
            "mill sluice, and my brother and I would argue about which pool held "
            "the bigger fish.\n\n"
            "By autumn the trees shed great heaps of leaves that clogged the drain "
            "until the whole yard smelled of wet bark and old wood, and it took us "
            "days to clear it.\n\n"
            "It was that smell, more than any single memory, that I still recall "
            "whenever the town comes to mind, and I have long since stopped "
            "throwing stones into running water."
        )
        result = audit_structure(doc)
        names = {f["name"] for f in result["findings"]}
        self.assertNotIn("hourglass_closing", names)
        self.assertNotIn("challenges_future_section", names)
        self.assertNotIn("suspiciously_even_paragraphs", names)
        self.assertEqual(result["score"], 0.0)

    def test_score_is_bounded_zero_to_one(self):
        for doc in ("", "A single short paragraph with no newlines at all."):
            result = audit_structure(doc)
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"], 1.0)
            for f in result["findings"]:
                self.assertIn(f["severity"], ("low", "medium", "high"))
                self.assertIn("kb_ref", f)
                self.assertIn("evidence", f)


if __name__ == "__main__":
    unittest.main()
