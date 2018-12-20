import unittest
import re
from functions import read_ncs


class MyTest(unittest.TestCase):
    def test_read_ncs(self):
        self.assertTrue(re.match('\\w+\\s\\w+', read_ncs('test/test_ncs.txt')[0]),
                        msg='Noun compound could not be extracted correctly')
        self.assertTrue(re.match('\\w+\\s\\w+', read_ncs('test/test_ncs.txt')[-1]),
                        msg='Well formatted nc could not be identified')

    def test_read_ncs_len(self):
        self.assertTrue(len(read_ncs('test/test_ncs.txt')) > 0, msg='Length is 0')


if __name__ == '__main__':
    unittest.main()
