def find_equilibrium_indices(nums):
    result = []
    for i in range(len(nums)):
        if sum(nums[:i]) == sum(nums[i+1:]):
            result.append(i)
    return result


def find_triplets(nums, target):
    n = len(nums)
    for i in range(n):
        seen = {}
        rem = target - nums[i]
        for j in range(i + 1, n):
            diff = rem - nums[j]
            if diff in seen:
                return (i, seen[diff], j)
            seen[nums[j]] = j
    return None


def merge_two_sorted_arrays(nums1, nums2):
    i = j = 0
    output = []
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            output.append(nums1[i])
            i += 1
        else:
            output.append(nums2[j])
            j += 1

    output.extend(nums1[i:])
    output.extend(nums2[j:])
    return output


def max_subarray(nums):
    max_sum = nums[0]
    best_i = best_j = 0

    for i in range(len(nums)):
        current = 0
        for j in range(i, len(nums)):
            current += nums[j]
            if current > max_sum:
                max_sum = current
                best_i, best_j = i, j

    return max_sum, nums[best_i:best_j + 1]


def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return (seen[target - x], i)
        seen[x] = i
    return None


def union_arrays(nums1, nums2):
    return list(set(nums1).union(set(nums2)))


def intersection_two_arrays(nums1, nums2):
    return list(set(nums1).intersection(set(nums2)))


def find_duplicates(nums):
    counts = {}
    for x in nums:
        counts[x] = counts.get(x, 0) + 1
    return [k for k, v in counts.items() if v > 1]


def max_min(nums):
    mx = mn = nums[0]
    for x in nums:
        mx = max(mx, x)
        mn = min(mn, x)
    return mx, mn


def reverse_inplace(nums):
    nums = nums.copy()
    i, j = 0, len(nums) - 1
    while i < j:
        nums[i], nums[j] = nums[j], nums[i]
        i += 1
        j -= 1
    return nums


def rotate_one_step(nums):
    if not nums:
        return nums
    return [nums[-1]] + nums[:-1]


def rotate_array_k_forward(nums, k):
    if not nums:
        return nums

    k %= len(nums)
    for _ in range(k):
        nums = rotate_one_step(nums)
    return nums


def move_zeros_end(nums):
    non_zero = [x for x in nums if x != 0]
    zeros = [0] * (len(nums) - len(non_zero))
    return non_zero + zeros


def is_sorted(nums, ascending=True):
    for i in range(len(nums) - 1):
        if ascending:
            if nums[i] > nums[i + 1]:
                return False
        else:
            if nums[i] < nums[i + 1]:
                return False
    return True


def remove_duplicates_from_sorted_array(nums):
    if not nums:
        return []

    result = [nums[0]]
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            result.append(nums[i])

    return result


def find_missing_nums_12n(nums):
    missing = []
    for i in range(1, len(nums)):
        missing.extend(range(nums[i - 1] + 1, nums[i]))
    return missing


def get_counts(nums):
    counts = {}
    for x in nums:
        counts[x] = counts.get(x, 0) + 1
    return counts


def peak_element(nums):

    if len(nums) == 1:
        return nums[0]

    if nums[0] > nums[1]:
        return nums[0]

    if nums[-1] > nums[-2]:
        return nums[-1]

    for i in range(1, len(nums) - 1):
        if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
            return nums[i]

    return None


# -------------------------------
# Tests
# -------------------------------

def run_tests():

    # Q1 Equilibrium
    assert find_equilibrium_indices([1,3,5,2,2]) == [2]
    assert find_equilibrium_indices([0]) == [0]
    assert find_equilibrium_indices([]) == []

    # Q2 Triplets
    assert find_triplets([1,4,45,6,10,8],22) is not None
    assert find_triplets([1,2,3],100) is None

    # Q3 Merge
    assert merge_two_sorted_arrays([1,3,5],[2,4,6]) == [1,2,3,4,5,6]
    assert merge_two_sorted_arrays([],[]) == []
    assert merge_two_sorted_arrays([1,2],[]) == [1,2]
    assert merge_two_sorted_arrays([], [1,2]) == [1,2]

    # Q4 Max Subarray
    assert max_subarray([-5,-1,-8])[0] == -1
    assert max_subarray([1,2,3])[0] == 6
    assert max_subarray([5])[0] == 5

    # Q5 Two Sum
    assert two_sum([2,7,11,15],9) == (0,1)
    assert two_sum([1,2,3],100) is None

    # Q6 Union
    assert set(union_arrays([1,2,2],[2,3])) == {1,2,3}

    # Q7 Intersection
    assert set(intersection_two_arrays([1,2,3],[2,3,4])) == {2,3}
    assert intersection_two_arrays([],[]) == []

    # Q8 Duplicates
    assert set(find_duplicates([1,2,2,3,3])) == {2,3}
    assert find_duplicates([1,2,3]) == []

    # Q9 Max Min
    assert max_min([5,1,9,-2]) == (9,-2)
    assert max_min([7]) == (7,7)

    # Q10 Reverse
    assert reverse_inplace([1,2,3]) == [3,2,1]
    assert reverse_inplace([5]) == [5]
    assert reverse_inplace([]) == []

    # Q11 Rotate One Step
    assert rotate_one_step([1,2,3,4]) == [4,1,2,3]
    assert rotate_one_step([5]) == [5]

    # Q12 Rotate K
    assert rotate_array_k_forward([1,2,3,4],2) == [3,4,1,2]
    assert rotate_array_k_forward([1,2,3,4],6) == [3,4,1,2]

    # Q13 Move Zeros
    assert move_zeros_end([0,1,0,2]) == [1,2,0,0]
    assert move_zeros_end([0,0,0]) == [0,0,0]
    assert move_zeros_end([1,2,3]) == [1,2,3]

    # Q14 Sorted
    assert is_sorted([1,2,3])
    assert not is_sorted([3,2,1])
    assert is_sorted([3,2,1], ascending=False)

    # Q15 Remove Duplicates
    assert remove_duplicates_from_sorted_array([1,1,2,2,3]) == [1,2,3]
    assert remove_duplicates_from_sorted_array([]) == []

    # Q16 Missing Numbers
    assert find_missing_nums_12n([1,2,4,7,10]) == [3,5,6,8,9]
    assert find_missing_nums_12n([1,2,3]) == []

    # Q17 Counts
    assert get_counts([1,1,2,3,3]) == {1:2,2:1,3:2}

    # Q18 Peak
    assert peak_element([1,5,3]) == 5
    assert peak_element([5,3,2]) == 5
    assert peak_element([1,2,5]) == 5
    assert peak_element([7]) == 7

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_tests()

