import unittest

import tensorflow as tf


class TestCase(unittest.TestCase):
    def test_tensorflow(self):
        # Test if tensorflow can access the gpu or cpu
        assert len(tf.config.list_physical_devices('GPU')) >= 1 or len(tf.config.list_physical_devices('CPU')) >= 1


if __name__ == '__main__':
    unittest.main()
