# -*- coding: utf-8 -*-

from .context import classifier-refit

import unittest


class AdvancedTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_thoughts(self):
        self.assertIsNone(classifier-refit.hmm())


if __name__ == '__main__':
    unittest.main()
