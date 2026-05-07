# Problem: Backspace String Compare
# Given strings s and t containing '#' as backspace, return True if both are equal
# after processing backspaces.
#
# Example 1:
# Input: s = "ab#c", t = "ad#c"
# Output: True
#
# Example 2:
# Input: s = "a#c", t = "b"
# Output: False
#
# Constraints:
# 1 <= len(s), len(t) <= 200
# s and t contain lowercase letters and '#'

# Time: O(n + m)
# Space: O(1)
def backspace_compare(s, t):
    i = len(s) - 1
    j = len(t) - 1
    skip_s = 0
    skip_t = 0

    while i >= 0 or j >= 0:
        while i >= 0:
            if s[i] == '#':
                skip_s += 1
                i -= 1
            elif skip_s > 0:
                skip_s -= 1
                i -= 1
            else:
                break

        while j >= 0:
            if t[j] == '#':
                skip_t += 1
                j -= 1
            elif skip_t > 0:
                skip_t -= 1
                j -= 1
            else:
                break

        ch_s = s[i] if i >= 0 else None
        ch_t = t[j] if j >= 0 else None

        if ch_s != ch_t:
            return False

        i -= 1
        j -= 1

    return True
