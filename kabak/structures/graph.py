from typing import Any


class ListNode:
    """
    A class of list nodes for linked lists.
    """

    def __init__(
        self, val: Any = None, next: "ListNode" = None, prev: "ListNode" = None
    ):
        self.val = val
        self.next = next
        self.prev = prev

    def __str__(self):
        return f"ListNode({self.__dict__})"


class TreeNode:
    """
    Class of simple tree nodes with a value and parent."""

    def __init__(self, val: Any = None, parent: "TreeNode" = None):
        self.val = val
        self.parent = parent

    def __str__(self):
        return f"TreeNode({self.__dict__})"
