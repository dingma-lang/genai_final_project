import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

try:
    import spacy
    NLP = spacy.load('en_core_web_lg')
except (ImportError, OSError):
    NLP = None


@unittest.skipIf(NLP is None, 'spaCy model en_core_web_lg is not installed')
class MaskingTests(unittest.TestCase):
    TEXT = ('Arsenal host Leicester on Saturday 12 August 2017 at the Emirates. Arsène Wenger has a fit squad, '
            'and Alexandre Lacazette starts after his move from Lyon, while the Gunners lost to Chelsea at Wembley. '
            'Wenger plays a 4-2-3-1 in the Premier League. Nick Ames')

    def levels(self):
        import mask_previews as mp
        text = mp.normalized(self.TEXT)
        spans = mp.find_spans(text, 'Arsenal', 'Leicester', NLP(text), set(), set(), set())
        numbering, counts = {}, {}
        for _, _, _, key in spans:
            if key.startswith(('PERSON:', 'CLUB:', 'NAME:')) and key not in numbering:
                kind = key.split(':')[0]
                counts[kind] = counts.get(kind, 0) + 1
                numbering[key] = counts[kind]
        return {lvl: mp.render(text, spans, lvl, numbering)[0] for lvl in range(1, 6)}

    def test_each_level_removes_its_category(self):
        out = self.levels()
        self.assertIn('Arsenal', out[1])
        self.assertNotIn('Arsenal', out[2])
        self.assertNotIn('Gunners', out[2])
        self.assertIn('[HOME]', out[2])
        self.assertIn('[AWAY]', out[2])
        self.assertNotIn('Nick Ames', out[3])
        self.assertNotIn('2017', out[3])
        for name in ['Wenger', 'Lacazette', 'Lyon', 'Chelsea', 'Emirates', 'Wembley']:
            self.assertNotIn(name, out[4])
        self.assertNotIn('Saturday', out[5])

    def test_levels_are_cumulative_and_keep_useful_terms(self):
        out = self.levels()
        for lvl in range(2, 6):
            self.assertLessEqual(len(out[lvl].split()), len(out[lvl - 1].split()) + 2)
        self.assertIn('4-2-3-1', out[5])              # formations are not dates
        self.assertIn('Premier League', out[5])       # stakes information is kept
        self.assertEqual(out[4].count('[PERSON_1]'), 2)  # "Arsène Wenger" and "Wenger" share a number


if __name__ == '__main__':
    unittest.main()
