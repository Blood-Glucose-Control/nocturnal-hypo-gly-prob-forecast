class TestCicdRun:
    def test_simple_math(self):
        assert 1 + 1 == 2

    def test_list(self):
        test_list = [1, 2, 3]
        assert len(test_list) == 3
