#include "algorithms.h"
#include <iostream>

void showVectorString(vector<string> ans);
void showVectorInt(vector<int> ans);
void showInt(int num);
void showString(string str);
void showVectorVectorInt(vector<vector<int>> ans);

int main()
{
    Subsets algorithmInstance{};
    vector<int> candidates = {1,2,3};
    vector<vector<int>> ans = algorithmInstance.subsets(candidates);
    showVectorVectorInt(ans);
    return 0;
}

void showVectorString(vector<string> ans)
{
    for(string str : ans) std::cout << str <<", " << std::endl;
}
void showVectorInt(vector<int> ans)
{
    for(int num : ans) std::cout << num <<", "; 
}
void showVectorVectorInt(vector<vector<int>> ans)
{
    for(auto v : ans)
    {
        showVectorInt(v);
        std::cout <<" " << std::endl;
    }
}
void showInt(int num)
{
    std::cout << num << std::endl; 
} 
void showString(string str)
{
    std::cout << str << std::endl; 
} 

