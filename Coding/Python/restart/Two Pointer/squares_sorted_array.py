# Problem: Squares of a Sorted Array
# Given a sorted integer array nums (non-decreasing order), return a new array
# of the squares of each number also in non-decreasing order.
#
# Example 1:
# Input: nums = [-4,-1,0,3,10]
# Output: [0,1,9,16,100]
#
# Example 2:
# Input: nums = [-7,-3,2,3,11]
# Output: [4,9,9,49,121]
#
# Constraints:
# 1 <= len(nums) <= 10^4
# -10^4 <= nums[i] <= 10^4

# Time: O(n)
# Space: O(n)
def sorted_squares(nums):
    n = len(nums)
    result = [0] * n

    left, right = 0, n - 1
    pos = n - 1

    while left <= right:
        if abs(nums[left]) > abs(nums[right]):
            result[pos] = nums[left] * nums[left]
            left += 1
        else:
            result[pos] = nums[right] * nums[right]
            right -= 1
        pos -= 1

    return result
