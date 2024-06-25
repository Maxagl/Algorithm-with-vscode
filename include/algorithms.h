#include <unordered_map>
#include <vector>
#include <string>

using namespace std;

class letter_Combination_Of_A_Phone
{
public:
    vector<string> letterCombinations(string digits);
    void backTracking(string digits, unordered_map<char, string>& hash, vector<string>& ans, string temp, int index);
};

class GenerateParentheses
{
public:
    vector<string> generateParenthesis(int n);
    void backTracking(vector<string>& ans, int left, int right, string temp);
};

class CombinationSum
{
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target);
    void backTracking(vector<int>& candidates, int target, int index, vector<vector<int>>& ans, vector<int>& temp, int sum);
};
