# Problem: Trapping Rain Water
# Given n non-negative integers height where each value is width-1 bar height,
# compute total trapped rain water after raining.
#
# Example 1:
# Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
# Output: 6
#
# Example 2:
# Input: height = [4,2,0,3,2,5]
# Output: 9
#
# Constraints:
# 1 <= len(height) <= 2 * 10^4
# 0 <= height[i] <= 10^5

# Time: O(n)
# Space: O(n)
def trap_prefix_suffix(height):
    n = len(height)
    if n == 0:
        return 0

    left_max = [0] * n
    right_max = [0] * n

    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])

    right_max[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])

    water = 0
    for i in range(n):
        water += min(left_max[i], right_max[i]) - height[i]

    return water


# Time: O(n)
# Space: O(1)
def trap_two_pointers(height):
    left, right = 0, len(height) - 1
    left_max = 0
    right_max = 0
    water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water
