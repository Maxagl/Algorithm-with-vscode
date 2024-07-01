#include <unordered_map>
#include <vector>
#include <string>

using namespace std;
const int DIR4[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

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

class Permutations
{
public:
    vector<vector<int>> permute(vector<int>& nums);
    void backTracking(vector<vector<int>>& ans, vector<int>& nums, vector<int>& temp, unordered_map<int, bool>& visited);
};

class Subsets
{
public:
    vector<vector<int>> subsets(vector<int>& nums);
    void backTracking(vector<vector<int>>& ans, vector<int>& nums, vector<int>& temp, int index);
};

class WordSearch
{
public:
    bool exist(vector<vector<char>>& board, string word);
    void backTracking(vector<vector<char>>& board, const string& word, bool& ans, int index, int col, int row);
};