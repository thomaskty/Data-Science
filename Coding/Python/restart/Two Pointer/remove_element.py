# Problem: Remove Element
# Remove all occurrences of val in nums in-place and return the new length.
# The first k elements of nums should contain the kept values.
#
# Example 1:
# Input: nums = [3,2,2,3], val = 3
# Output: 2, nums starts with [2,2]
#
# Example 2:
# Input: nums = [0,1,2,2,3,0,4,2], val = 2
# Output: 5, nums starts with [0,1,3,0,4]
#
# Constraints:
# 0 <= len(nums) <= 100
# 0 <= nums[i], val <= 50

# Time: O(n)
# Space: O(1)
def remove_element_swap_end(nums, val):
    left = 0
    right = len(nums) - 1

    while left <= right:
        if nums[left] == val:
            nums[left] = nums[right]
            right -= 1
        else:
            left += 1

    return left


# Time: O(n)
# Space: O(1)
def remove_element_stable(nums, val):
    write = 0

    for read in range(len(nums)):
        if nums[read] != val:
            nums[write] = nums[read]
            write += 1

    return write
