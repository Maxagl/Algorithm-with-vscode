#include "algorithms.h"
#include <iostream>

void showVectorString(vector<string> ans);
void showVectorInt(vector<int> ans);
void showInt(int num);
void showString(string str);
void showVectorVectorInt(vector<vector<int>> ans);

int main()
{
    SearchInRotatedSortedArray algorithmInstance{};
    vector<int> candidates = {4,5,6,7,0,1,2};
    int ans = algorithmInstance.search(candidates, 0);
    showInt(ans);
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

