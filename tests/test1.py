# tests/test_module1.py
import unittest
from cgisim_sims.module1 import function1

class TestModule1(unittest.TestCase):
    def test_function1(self):
        self.assertEqual(function1(), "This is function1 from module1")

if __name__ == '__main__':
    unittest.main()
