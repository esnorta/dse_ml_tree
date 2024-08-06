from models import Node
from splitter import Splitter


class TestSplitter:
    def test_get_information_gain(self):
        parent = Node(entropy=0.6, example_count=40)
        child_1 = Node(entropy=0.2, example_count=16)
        child_2 = Node(entropy=0.1, example_count=24)

        information_gain = Splitter().get_information_gain(parent, [child_1, child_2])

        assert information_gain == 0.46
