from unittest import TestCase

import pyEHRtool

class TestJoke(TestCase):
    def test_is_string(self):
        s = pyEHRtool.load_data()
        self.assertTrue(isinstance(s, basestring))