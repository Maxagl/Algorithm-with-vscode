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

/*--------------------------78--------------------------*/
vector<vector<int>> Subsets::subsets(vector<int>& nums)
{
    vector<vector<int>> ans{};
    vector<int> temp{};
    backTracking(ans, nums, temp, 0);
    return ans;
}
void Subsets::backTracking(vector<vector<int>>& ans, vector<int>& nums, vector<int>& temp, int index)
{
    // 为什么可以不用等到index==nums.size()
    // 因为我不取后，i不会更新
    // 这里的理解是，递归的上一层的总是会单独形成一份子集。而我这一层只需要考虑取哪一个
    ans.push_back(temp);
    for(int i{index}; i < nums.size(); ++i)
    {
        temp.push_back(nums[i]);
        backTracking(ans, nums, temp, i + 1);
        temp.pop_back();
    }
}
/*--------------------------79--------------------------*/
bool WordSearch::exist(vector<vector<char>>& board, string word)
{
    int m = board.size();
    int n = board[0].size();
    // visited矩阵好像可以去掉
    // vector<vector<int>> visited(m, vector<int>(n, 0));
    bool ans{false};
    for(int i{0}; i < m; ++i)
    {
        for(int j{0}; j < n; ++j)
        {
            if(board[i][j] == word[0])
            {
                char ch = board[i][j];
                board[i][j] = '*';
                backTracking(board, word, ans, 1, j, i);
                board[i][j] = ch;
            }
        }
    }
    return ans;  
}
// 以后写函数，记得row在前面，这里就算了别改了
void WordSearch::backTracking(vector<vector<char>>& board, const string& word, bool& ans, int index, int col, int row)
{
    if(index == word.size())
    {
        ans = true;
        return;
    } 
    for(int i{0}; i < 4; ++i)
    {
        int nextRow = row + DIR4[i][0];
        int nextCol = col + DIR4[i][1];
        if(nextRow >= 0 && nextRow < board.size() && nextCol >= 0 && nextCol < board[0].size() && board[nextRow][nextCol] == word[index])
        {
                char ch = board[nextRow][nextCol];
                board[nextRow][nextCol] = '*';
                backTracking(board, word, ans, index + 1, nextCol, nextRow);
                board[nextRow][nextCol] = ch;
        }
    }
}

/*--------------------------131--------------------------*/
vector<vector<string>> PalindromePartition::partition(string s)
{
    n = s.size();
    f.assign(n, vector<int>(n, true));
    // 记录回文串
    for(int i{n - 1}; i >= 0; --i)
    {
        for(int j = i + 1; j < n; ++j)
        {
            f[i][j] = (s[i] == s[j]) && f[i + 1][j - 1];
        }
    }
    backTracking(s, 0);
    return ret;
}
// 真正的分割
void PalindromePartition::backTracking(const string& s, int i)
{
    if(i == n)
    {
        ret.push_back(ans);
        return;
    }
    // 分割完的字符不能在利用了，因为是一次性分割。不是子集的回文串个数
    for(int j{i}; j < n; ++j)
    {
        // 当前j的位置分割完后，看后续还有没有能分割的位置
        if(f[i][j])
        {
            ans.push_back(s.substr(i, j - i + 1));
            backTracking(s, j + 1);
            ans.pop_back();
        }
    }
}

/*--------------------------4--------------------------*/
double MedianOfTwoSortedArray::findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2)
{
    int n1 = nums1.size();
    int n2 = nums2.size();
    if(n1 > n2) return findMedianSortedArrays(nums2, nums1);

    int n = n1 + n2;
    int left = (n1 + n2 + 1) / 2;
    int low = 0;
    int high = n1;
    
    while(low <= high)
    {
        int mid1 = (low + high) >> 1;
        int mid2 = left - mid1;

        int l1 = INT_MIN;
        int l2 = INT_MIN;
        int r1 = INT_MAX;
        int r2 = INT_MAX;

        if(mid1 < n1) r1 = nums1[mid1];
        if(mid2 < n2) r2 = nums2[mid2];
        if(mid1 - 1 >= 0) l1 = nums1[mid1 - 1];
        if(mid2 - 1 >= 0) l2 = nums2[mid2 - 1];

        if(l1 <= r2 && l2 <= r1)
        {
        // 不懂为什么是比l1,l2。明明这里有四个数，不应该取中间的吗
            if(n % 2 == 1) return max(l1, l2);
            else return ((double)(max(l1 ,l2) + min(r1, r2))) / 2.0;
        }
        // 这两个条件就是二分
        else if(l1 > r2)
        {
            high = mid1 - 1;
        }
        else
        {
            low = mid1 + 1;
        }
    }
    return 0;
}

/*--------------------------33--------------------------*/
int SearchInRotatedSortedArray::search(vector<int>& nums, int target)
{
    int n = nums.size();
    int left{0};
    int right{n - 1};

    // 这里等于跳出来是因为会陷入无限循环，内部没有返回。
    while(right > left)
    {
        int mid = left + (right - left) / 2;
        // 我担心的情况就是陷入局部递增，但只要旋转了，第一次的取中间就不可能陷入局部递增。
        // 因为只会动一边，如果落在左边的局部递增。左边加，右边不变。还是能找到轴
        // 如果落在右边的局部递增。右边就保持在右边的局部递增，左边不变
        // 这就会变成，left和right永远都在左边的局部递增和右边的局部递增
        if(nums[mid] > nums[right]) left = mid + 1;
        else right = mid;
    }
    int rot = left;
    left = 0;
    right = n - 1;
    // 这里又要大于等于。因为那个数可能刚好是答案，但我们跳出来了
    while(right >= left)
    {
        int mid = left + (right - left) / 2;
        int realMid = (mid + rot) % n;
        if(nums[realMid] == target) return realMid;
        if(nums[realMid] < target) left = mid + 1;
        else right = mid - 1; 
    }
    return - 1;
}

/*--------------------------34--------------------------*/
vector<int> FindFirstAndLastPositionOfElementInSortedArray::searchRange(vector<int>& nums, int target)
{
    int n = nums.size();
    int l{0};
    int r{n - 1};
    vector<int> ans(2, -1);
    while(r >= l)
    {
        int mid = l + (r - l) / 2;
        // 从左边往右边加，l最终变成最右边
        // 反过来是不是也可以
        if(nums[mid] == target)
        {
            ans[1] = l;
            ++l; 
        }
        else if(nums[mid] < target) l = mid + 1;
        else r = mid - 1; 
    }
    l = 0;
    r = n - 1;
    while(r >= l)
    {
        int mid = l + (r - l) / 2;
        // 从右边往左边减，l最终变成最左边
        if(nums[mid] == target)
        {
            ans[0] = r;
            --r; 
        }
        else if(nums[mid] < target) l = mid + 1;
        else r = mid - 1; 
    }
    return ans;
 }

 /*--------------------------35--------------------------*/

int SearchInsertPosition::searchInsert(vector<int>& nums, int target)
{
    int n = nums.size();
    int l{0};
    int r{n - 1};
    // 内部有返回，需要等于
    while(r >= l)
    {
        int mid = l + (r - l) / 2;
        if(nums[mid] == target) return mid;
        if(nums[mid] < target) l = mid + 1;
        else r = mid - 1;
    }
    // 因为最后等于的那一次，肯定l和r相邻。相邻然后变成等于肯定是左边加1
    // 左边加1就会导致和r碰上。如果这时候r小于target，那么就会进入小的分支，l=mid+1得到答案。
    // 如果r大于target，那么l不会变，这也是最后一个大于的位置刚好插入
    return l;
}