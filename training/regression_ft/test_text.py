import unittest

from text import normalize_training_text as clean


class LabelFormatting(unittest.TestCase):
    def test_lexical_content_and_annotations(self):
        self.assertEqual(clean(r'Ja,\Kommaat se.\Punktum', 'nst_da_da'), 'Ja, Komma at se. Punktum')
        self.assertEqual(clean('A # uh [laugh] (overlap) word (/overlap).', 'gigaspeechbench_sgp_en'), 'A uh word.')
        self.assertEqual(clean('Un *(Applausi)* test (detto bene).', 'eurospeech_it'), 'Un test detto bene.')
        self.assertEqual(clean('( ... tyst under denna inspelning ...)', 'nst_sv_sv'), '')

    def test_unicode_whitespace_and_idempotence(self):
        self.assertEqual(clean('κόσμος;\n προϋπόθεση', 'lesbos_el'), 'κόσμοσ? προυπόθεση')
        self.assertEqual(clean('text' + ' ' * 500 + 'word', 'nst_sv_sv'), 'text word')
        for split, text in [('nst_da_da', r'Ja.\Punktum'), ('eurospeech_el', 'λέξεις;'), ('eurospeech_it', '*bene*; grazie')]:
            self.assertEqual(clean(clean(text, split), split), clean(text, split))


if __name__ == '__main__':
    unittest.main()
