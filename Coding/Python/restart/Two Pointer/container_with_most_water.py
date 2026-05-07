# Problem: Container With Most Water
# You are given an array height where height[i] is the height of a vertical line.
# Find two lines that together with x-axis form a container with the maximum area.
#
# Example 1:
# Input: height = [1,8,6,2,5,4,8,3,7]
# Output: 49
#
# Example 2:
# Input: height = [1,1]
# Output: 1
#
# Constraints:
# 2 <= len(height) <= 10^5
# 0 <= height[i] <= 10^4

# Time: O(n^2)
# Space: O(1)
def max_area_bruteforce(height):
    n = len(height)
    answer = 0

    for i in range(n):
        for j in range(i + 1, n):
            area = min(height[i], height[j]) * (j - i)
            answer = max(answer, area)

    return answer


# Time: O(n)
# Space: O(1)
def max_area_two_pointers(height):
    left, right = 0, len(height) - 1
    answer = 0

    while left < right:
        area = min(height[left], height[right]) * (right - left)
        answer = max(answer, area)

        # Move the limiting side because only that can increase min-height.
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return answer
