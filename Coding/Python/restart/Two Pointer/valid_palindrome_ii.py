# Problem: Valid Palindrome II
# Given string s, return True if it can become a palindrome after deleting
# at most one character.
#
# Example 1:
# Input: s = "aba"
# Output: True
#
# Example 2:
# Input: s = "abca"
# Output: True
# Explanation: Remove 'c' to get "aba".
#
# Constraints:
# 1 <= len(s) <= 10^5
# s contains lowercase English letters

# Time: O(n)
# Space: O(1)
def valid_palindrome_ii(s):
    left = 0
    right = len(s) - 1

    while left < right:
        if s[left] != s[right]:
            return _is_palindrome_range(s, left + 1, right) or _is_palindrome_range(s, left, right - 1)
        left += 1
        right -= 1

    return True


# Time: O(n)
# Space: O(1)
def _is_palindrome_range(s, left, right):
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
