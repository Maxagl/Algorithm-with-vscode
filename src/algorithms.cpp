#include "algorithms.h"

/*--------------------------17--------------------------*/
vector<string> letter_Combination_Of_A_Phone::letterCombinations(string digits)
{
    unordered_map<char, string> hash{{'2', "abc"},
                                     {'3', "def"},
                                     {'4', "ghi"},
                                     {'5', "jkl"},
                                     {'6', "mno"},
                                     {'7', "pqrs"},
                                     {'8', "tuv"},
                                     {'9', "wxyz"}
                                     };
    if(digits.length() == 0) return {};
    string temp{};
    vector<string> ans{};
    backTracking(digits, hash, ans, temp, 0);
    return ans;
}

void letter_Combination_Of_A_Phone::backTracking(string digits, unordered_map<char, string>& hash, vector<string>& ans, string temp, int index)
{
    if(digits.length() == temp.length())
    {
        ans.push_back(temp);
    }
    for(char alp : hash[digits[index]])
    {
        temp.push_back(alp);
        backTracking(digits, hash, ans, temp, index + 1);
        temp.pop_back();
    }
}

/*--------------------------22--------------------------*/
vector<string> GenerateParentheses::generateParenthesis(int n)
{
    vector<string> ans{};
    string temp{};
    backTracking(ans, n, n, temp);
    return ans;
}

void GenerateParentheses::backTracking(vector<string>& ans, int left, int right, string temp)
{
    if(left == 0 && right == 0)
    {
        ans.push_back(temp);
        return;
    } 
    // 先加左边的，然后补充右边。右括号只有在剩余数量大于左边的时候才能补充
    // 类似于for循环，知识这里是两个值，并且条件不同 
    if(left > 0) 
    {
        backTracking(ans, left - 1, right, temp + '(');  
    }
    if(left < right) 
    {
        backTracking(ans, left, right - 1, temp + ')');
    }
}

/*--------------------------39--------------------------*/
vector<vector<int>> CombinationSum::combinationSum(vector<int>& candidates, int target)
{
    vector<vector<int>> ans{};
    vector<int> temp;
    int sum{0};
    int index{0};
    backTracking(candidates, target, index, ans, temp, sum);
    return ans;
}

void CombinationSum::backTracking(vector<int>& candidates, int target, int index, vector<vector<int>>& ans, vector<int>& temp, int sum)
{
    if(sum == target)
    {
        ans.push_back(temp);
        return;
    }
    if(sum > target) return;
    for(int i{index}; i < candidates.size(); ++i)
    {
        temp.push_back(candidates[i]);
        backTracking(candidates, target, i, ans, temp, sum + candidates[i]);
        temp.pop_back();
    }
}

/*--------------------------46--------------------------*/
vector<vector<int>> Permutations::permute(vector<int>& nums)
{
    vector<vector<int>> ans{};
    vector<int> temp{};
    unordered_map<int, bool> visited{};
    for(int num : nums)
    {
        visited[num] = false;
    }
    backTracking(ans, nums, temp, visited);
    return ans;
}
void Permutations::backTracking(vector<vector<int>>& ans, vector<int>& nums, vector<int>& temp, unordered_map<int, bool>& visited)
{
    if(temp.size() == nums.size())
    {
        ans.push_back(temp);
        return;
    }
    for(int i{0}; i < nums.size(); ++i)
    {
        if(!visited[nums[i]])
        {
            temp.push_back(nums[i]);
            visited[nums[i]] = true;
            backTracking(ans, nums, temp, visited);
            temp.pop_back();
            visited[nums[i]] = false;
        }
    }
}
