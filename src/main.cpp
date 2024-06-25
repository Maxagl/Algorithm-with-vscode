#include "algorithms.h"
#include <iostream>

void showVectorString(vector<string> ans);
void showVectorInt(vector<int> ans);
void showInt(int num);
void showString(string str);

int main()
{
    letter_Combination_Of_A_Phone algorithmInstance{};
    string digits{"23"};
    vector<string> ans = algorithmInstance.letterCombinations(digits);
    showVectorString(ans);
    return 0;
}

void showVectorString(vector<string> ans)
{
    for(string str : ans) std::cout << str <<", " << std::endl;
}
void showVectorInt(vector<int> ans)
{
    for(int num : ans) std::cout << num <<", " << std::endl; 
}
void showInt(int num)
{
    std::cout << num << std::endl; 
} 
void showString(string str)
{
    std::cout << str << std::endl; 
} 

