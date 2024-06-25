#include "algorithms.h"

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