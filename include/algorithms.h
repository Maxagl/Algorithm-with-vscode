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
