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

/*--------------------------22--------------------------*/ //没写出来的
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

/*--------------------------131--------------------------*/  //没写出来的
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

/*--------------------------4--------------------------*/ //没写出来的
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

/*--------------------------33--------------------------*/ //没写出来的
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

/*--------------------------34--------------------------*/ //没写出来的
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

 /*--------------------------35--------------------------*/ //没写出来的

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
 /*--------------------------74--------------------------*/
bool SearchA2DMatrix::searchMatrix(vector<vector<int>>& matrix, int target)
{
    int m = matrix.size();
    int n = matrix[0].size();
    int l{0};
    int r{m * n - 1};
    while(r >= l)
    {
        int mid = l + (r - l) / 2;
        // 这里要搞清楚除的意思，n代表一行的个数，也就是列数。
        int col = mid % n; // 取余的话就代表第几列
        int row = mid / n; // 取整的话代表第几行
        if(matrix[row][col] == target) return true;
        if(matrix[row][col] < target) l = mid + 1;
        else r = mid - 1;
    }
    return false;
}

 /*--------------------------124--------------------------*/ //没写出来的

int BinaryTreeMaximumPathSum::maxGain(TreeNode* root)
{
    if(root == nullptr) return 0;
    // 把左边和右边的加完。然后加在一起。最后判断是否为最大。
    // 为负数的话那个分值不用考虑了，直接为0就行
    int leftGain = max(maxGain(root->left), 0);
    int rightGain = max(maxGain(root->right), 0);
    // 当这里是叶节点的时候，leftgain和rightgain就是0.得到的就是本身的值
    int currSum = root->val + leftGain + rightGain;
    // 更新最大值
    maxSum = max(maxSum, currSum);
    // 返回给更高一级的时候，只能从左右选一边不然又有分支了
    return root->val + max(leftGain, rightGain);

}
int BinaryTreeMaximumPathSum::maxPathSum(TreeNode* root)
{
    maxGain(root);
    return maxSum;
}

int FindMinimumInRotatedSortedArray::findMin(vector<int>& nums)
{
    int l = 0;
    int r = nums.size() - 1;
    while(r > l)
    {
        int mid = l + (r - l) / 2;
        if(nums[mid] > nums[r]) l = mid + 1;
        else r = mid;
    }
    return nums[l];
}


 /*--------------------------94--------------------------*/
void BinaryTreeInorderTraversal::dfs(TreeNode* root, vector<int>& ans)
{
    if(root == nullptr) return;
    dfs(root->left, ans);
    ans.push_back(root->val);
    dfs(root->right, ans);
}
vector<int> BinaryTreeInorderTraversal::inorderTraversal(TreeNode* root)
{
    vector<int> ans{};
    dfs(root, ans);
    return ans;
}

 /*--------------------------98--------------------------*/ //没写出来的
// 这种方法的错误之处在于，底层的右边有可能会大于爷爷节点
// bool ValidateBinarySearchTree::isValidBST(TreeNode* root)
// {
//     if(root == nullptr) return true;
//     if(root->left->val > root->val) return false;
//     if(root->right->val < root->val) return false;
//     return isValidBST(root->left) && isValidBST(root->right);
// }

void ValidateBinarySearchTree::dfs(TreeNode* root, TreeNode*& pre)
{
    if(root == nullptr) return;
    if(root->left != nullptr) dfs(root->left, pre);
    // 因为pre刚进来，这里root是第一个数的位置了，所以把自己变成pre
    if(pre == nullptr) pre = root;
    // binary search tree inorder一定是从小到大排好的。
    // 这里进入inorder，inorder要和前面一个比。满足的话自己就变成pre
    else
    {
        if(root->val <= pre->val) m_ans = false;
        pre = root;
    }
    if(root->right != nullptr) dfs(root->right, pre);
}

bool ValidateBinarySearchTree::isValidBST(TreeNode* root)
{
    TreeNode* pre = nullptr;
    dfs(root, pre);
    return m_ans;
}


 /*--------------------------101--------------------------*/
 // 没写出来的
bool SymmetricTree::equal(TreeNode* left, TreeNode* right)
{
    // 先判断有没有
    // 两个都是nullptr，证明当前位置是对的
    if(left == nullptr && right == nullptr) return true;
    // 有一个不是nullptr， 不是对称
    else if(left == nullptr || right == nullptr) return false;

    // 有没有之后判断是否相等
    // 不相等的情况也是不对称
    if(left->val != right->val) return false;
    // 这一层判断完了，就到下一层。下一层两个node对应的位置需要思考下
    // left的left和right的right。left的right和right的left对应
    return equal(left->left, right->right) && equal(left->right, right->left);
}

bool SymmetricTree::isSymmetric(TreeNode* root)
{
    return equal(root->left, root->right);
}

 /*--------------------------102--------------------------*/
 vector<vector<int>> BinaryTreeLevelOrderTraversal::levelOrder(TreeNode* root)
 {
    if(root == nullptr) return {};
    queue<TreeNode*> q{};
    vector<vector<int>> ans{};
    q.push(root);
    ans.push_back({root->val});
    while(q.size() > 0)
    {
        int s = q.size();
        vector<int> temp{};
        while(s > 0)
        {
            TreeNode* node = q.front();
            q.pop();
            temp.push_back(node->val);
            if(node->left) q.push(node->left);
            if(node->right) q.push(node->right);
            --s;
        }
        ans.push_back(temp);
    }
    return ans;
 }

 /*--------------------------104--------------------------*/
 int MaximumDepthOfBinaryTree::maxDepth(TreeNode* root)
 {
    if(root == nullptr) return 0;
    queue<TreeNode*> q{};
    int ans{0};
    q.push(root);
    while(q.size() > 0)
    {
        int s = q.size();
        while(s > 0)
        {
            TreeNode* node = q.front();
            q.pop();
            if(node->left) q.push(node->left);
            if(node->right) q.push(node->right);
            --s;
        }
        ++ans;
    }
    return ans;
 }

 /*--------------------------104--------------------------*/ //没写出来
TreeNode* ConstructBinaryTreeFromPreorderAndInorderTraversal::myTree(vector<int>& preorder, vector<int>& inorder, int p_left, int p_right, int in_left, int in_right)
{
    if(p_left > p_right) return nullptr;
    // preorder就是root最前面，root，left，right
    // 所以left就是root
    int p_root = p_left;
    // inorder里面root的位置
    int in_root = hash[preorder[p_root]];

    // 创建根节点
    TreeNode* root = new TreeNode(preorder[p_root]);
    // 左支的长度
    int size_left_subtree = in_root - in_left;
    // 递归的思想，我这一层分两支，分完之后交给下一层
    root->left = myTree(preorder, inorder, p_left + 1, p_left + size_left_subtree, in_left, in_root - 1);
    root->right = myTree(preorder, inorder, p_left + size_left_subtree + 1, p_right, in_root + 1, in_right);
}


TreeNode* ConstructBinaryTreeFromPreorderAndInorderTraversal::buildTree(vector<int>& preorder, vector<int>& inorder)
{
    int n = inorder.size();
    // 哈希表记录root的真正位置
    for(int i{0}; i < n; ++i)
    {
        hash[inorder[i]] = i;
    }
    return myTree(preorder, inorder, 0, n - 1, 0, n - 1);
}

 /*--------------------------108--------------------------*/ //没写出来
TreeNode* ConvertSortedArrayToBinarySearchTree::sortedArrayToBST(vector<int>& nums)
{
    if(nums.size() == 0) return nullptr;
    if(nums.size() == 1) return new TreeNode(nums[0]);

    int middle = nums.size() / 2;
    // BST顺序读取是inorder形式，
    // 每次都找中间那个，这样就满足binary search tree
    // 作为当前节点
    TreeNode* root = new TreeNode(nums[middle]);
    // 把左右分支确定好，然后就可以开始递归了
    vector<int> leftInts(nums.begin(), nums.begin() + middle);
    vector<int> rightIns(nums.begin() + middle + 1, nums.end());
    // 左右节点就靠递归了
    root->left = sortedArrayToBST(leftInts);
    root->right = sortedArrayToBST(rightIns);
    return root;
}

/*--------------------------114--------------------------*/ //没写出来
void FlattenBinaryTreeToLinkedList::flatten(TreeNode* root)
{
    // 后续遍历，但是记录下前一层的来当成pre。从而反向链接
    // 这里拉是拉前序遍历，就是反过来的后续遍历
    // 后续遍历才好记录前面的值
    if(root == nullptr) return;
    flatten(root->right);
    flatten(root->left);
    root->right = pre;
    root->left = nullptr;
    pre = root;
}

/*--------------------------114--------------------------*/ //push_back的位置出来点小问题
vector<int> BinaryTreeRightSideView::rightSideView(TreeNode* root)
{
    // 层次遍历
    if(root == nullptr) return {};
    queue<TreeNode*> q{};
    vector<int> ans;
    q.push(root);
    while(q.size() > 0)
    {
        int n = q.size();
        // 放到while后面会出错，最后一个会多push一次。但是不会报错，会把最后一个值再push进去一次
        ans.push_back(q.back()->val);
        while(n > 0)
        {
            TreeNode* temp = q.front();
            q.pop();
            if(temp->left) q.push(temp->left);
            if(temp->right) q.push(temp->right);
            --n;
        }
    }
    return ans;
}
/*--------------------------226--------------------------*/
TreeNode* InvertBinaryTree::invertTree(TreeNode* root)
{
    // 注意判断nullptr
    if(root == nullptr) return nullptr;
    // 把当前层的左右分支弄好
    TreeNode* temp = root->left;
    root->left = root->right;
    root->right = temp;
    // 递归交给下一层
    invertTree(root->left);
    invertTree(root->right);
    return root;      
}

/*--------------------------230--------------------------*/
void KthSmallestElementInABST::inorder(TreeNode* root, const int k, int i, int ans)
{
    // 需要确认是否为nullptr
    if(root == nullptr) return;
    if(root->left) inorder(root->left, k, i, ans);
    if(i < k) ++i;
    // 需要确保返回出去之后，不会再进来，要准确的值
    else if(k == i)
    {
        ++i;
        ans = root->val;
        return;
    }
    if(root->right) inorder(root->right, k, i, ans);
}   
int KthSmallestElementInABST::kthSmallest(TreeNode* root, int k)
{
    int i{1};
    int ans{};
    inorder(root->left, k, i, ans);
    return ans; 
}

/*--------------------------236--------------------------*/ // 没做出来
TreeNode* LowestCommonAncestorOfABinaryTree::lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q)
{
    // 找到了就返回，会返回给left或者right
    if(root == nullptr || root ==p || root == q) return root;
    // 当前节点没找到，就去找left和right
    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);
    // 都找完了，看left和right的情况，
    // 1. 都是null那就是没找到，返回null
    // 2. left为null，right里面肯定包含了
    // 3. right为null， left里面肯定包含了
    // 这题疑惑的点在于，我返回的时候找到了，怎么确保它是最近的公共祖先
    // 找到的时候肯定是一层层返回的，不会一下跳很高，你跳很多层想就会蒙了
    // 先想找到p或者q的节点
    // 然后就返回到left，或者right。如果两边都有那就直接结束了，当前root节点就是答案
    // 如果只有一边，那就只返回那一边，因为另一边为nullptr
    // 一直没找到，就会一直返回这个分支。知道两个分支都有值了。两个都有值就是答案了
    // 如果另一边一直没找到，那就说明它自己就是最近祖先
    
    if(left == nullptr && right == nullptr) return nullptr;
    if(left == nullptr) return right;
    if(right == nullptr) return left;
    // 然后当返回到一层，另一个值也找到的时候，那就是答案了。
    return root;
}


/*--------------------------437--------------------------*/
// 中文答案的前缀方法可以学一下
void PathSumIII::dfs(TreeNode* root, int targetSum, long long temp, int& ans)
{
    temp += root->val;
    if(temp == targetSum) ans += 1;
    // 如果这里不判断nullptr，会导致叶节点的左右分支，两个nullptr的结果判断两次。
    // 如果刚好叶节点是答案，这答案会加两次
    // 所以先加了，判断答案，再进行递归   
    if(root->left) dfs(root->left, targetSum, temp, ans);
    if(root->right) dfs(root->right, targetSum, temp, ans);
}

void PathSumIII::preOrder(TreeNode* root, int targetSum, int& ans)
{
    if(root == nullptr) return;
    dfs(root, targetSum, 0, ans);
    if(root->left) preOrder(root->left, targetSum, ans);
    if(root->right) preOrder(root->right, targetSum, ans);
    
}

int PathSumIII::pathSum(TreeNode* root, int targetSum)
{
    int ans{0};
    preOrder(root, targetSum, ans);
    return ans;
}

/*--------------------------437--------------------------*/ //没做出来
int DiameterOfBinaryTree::diameterOfBinaryTree(TreeNode* root)
{
    int d{0};
    dfs(root, d);
    return d;
}

int DiameterOfBinaryTree::dfs(TreeNode* root, int& d)
{
    if(root == nullptr) return 0;
    int ld = dfs(root->left, d);
    int rd = dfs(root->right, d);
    d = max(d, ld + rd);
    // 关键点是这个加1
    // 当我们递归到leaf node， dfs返回出来的都是0
    // ld, rd都是0
    // 但是我们需要保存当前节点往下的最长路径，一层就加1
    // 这里leaf最终返回给它的root node 长度1的路径
    return max(ld, rd) + 1;
}

/*--------------------------5--------------------------*/ //没做出来
string LongestPalindromicSubstring::longestPalindrome(string s)
{
    // 回文串涉及到左右，需要先考虑到二维dp
    int n = s.size();
    bool dp[n][n];
    // 二维dp初始化
    for(int i{0}; i < n; ++i)
    {
        for(int j{0}; j < n; ++j)
        {
            dp[i][j] = false;
        }
    }
    // 一个数必是回文，所以maxlen为1
    int maxLen = 1;
    int start = 0;
    int end = 0;
    // 以i结尾的回文串
    for(int i{0}; i < n; ++i)
    {
        dp[i][i] = true;
        // 想想一个字符串，从左边往内缩，因为右边是固定的
        for(int j{0}; j < i; ++j)
        {
            // 如果两个端点相等，就可以往内部考虑
            // 如果内部再缩一层发现是true，那么当前层就是正确的
            // 或者刚好3个数，那么肯定也是回文串了
            // i-1相关的已经在上一层全部判断过了
            if(s[i] == s[j] && (i - j <= 2 || dp[j + 1][i - 1]))
            {
                dp[j][i] = true;
                // 判断最大值
                if(i - j + 1 > maxLen)
                {
                    maxLen = i - j + 1;
                    start = j;
                    end = i;
                }
            }
        }
    }
    return s.substr(start, maxLen); 
}

/*--------------------------32--------------------------*/ //没做出来
int longestValidParentheses(string s)
{
    int maxans = 0;
    int n = s.length();
    vector<int> dp(n, 0);
    for(int i{1}; i < n; ++i)
    {
        if(s[i] == ')')
        {
            // i-1两种情况,左或者右
            // 刚好加一对括号
            if(s[i - 1] == '(')
            {
                dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
            }
            // 如果两个右括号，那就是把前一个当成dp，去判断它的最左端的左边一个是不是左括号
            // 是的话就可以扩充了
            else if(i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(')
            {
                // 内部都是 i - 1 不是i别写错了。要从上一层动态到这一层
                dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
            }
            maxans = max(maxans, dp[i]);
        }
    }
    return maxans;
}

/*--------------------------62--------------------------*/
int UniquePaths::uniquePaths(int m, int n)
{
    // 当前节点 = 上面+左边
    int dp[m][n];
    dp[0][0] = 1;
    // 边界条件处理一下就行
    for(int i{1}; i < m; ++i)
    {
        dp[i][0] = dp[i - 1][0];
    }
    for(int j{1}; j < n; ++j)
    {
        dp[0][j] = dp[0][j - 1];
    }
    for(int i{1}; i < m; ++i)
    {
        for(int j{1}; j < n; ++j)
        {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }
    return dp[m - 1][n - 1];
}

/*--------------------------64--------------------------*/
// 和62一样，不过这次要判断是否为最小
int MinimumPathSum::minPathSum(vector<vector<int>>& grid)
{
    int m = grid.size();
    int n = grid[0].size();
    int dp[m][n];
    dp[0][0] = grid[0][0];
    for(int i{1}; i < m; ++i)
    {
        dp[i][0] = dp[i - 1][0] + grid[i][0];
    }
    for(int j{1}; j < n; ++j)
    {
        dp[0][j] = dp[0][j - 1] + grid[0][j];
    }
    for(int i{1}; i < m; ++i)
    {
        for(int j{1}; j < n; ++j)
        {
            dp[i][j] = min(dp[i - 1][j], dp[i][ j - 1]) + grid[i][j];
        }
    }
    return dp[m - 1][n - 1];
}

/*--------------------------70--------------------------*/
int ClimbingStairs::climbStairs(int n)
{
    // 可以n + 1
    // 把0的时候也赋值1，就可以不用考虑边界了
    // 爬到第n阶的种数
    int dp[n];
    // 注意下边界条件
    if(n == 1) return 1;
    dp[0] = 1;
    dp[1] = 2;
    for(int i{2}; i < n; ++i)
    {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n - 1];   
}

/*--------------------------72--------------------------*/ //没做出来
int EditDistance::minDistance(string word1, string word2)
{
    int m = word1.size();
    int n = word2.size();
    // 由m位的word1变到word2的最小步数
    int dp[m + 1][n + 1];
    if(m == 0 && n == 0) return 0;
    // 初始化为0
    // dp[i][j] 表示从word1的0-i位，变化到word2的0-j位所需要的次数。
    // 变化有三种形式，A增，B增，改。对应前面一层为dp[i][j - 1]，dp[i - 1][j], dp[i - 1][j - 1];
    // dp[i][j - 1] --> dp[i][j],因为i到j-1的变化次数已经定了，所以当j-1变到j，
    // 只需再i后面再加一个相同的，其他的继续那么变就行
    // 其实也是一种贪心，一直选最小的那种变化
    // 合理性逆向思维来想更好，i 到 j，肯定上面三种情况变过来
    for(int i{0}; i < m; ++i)
    {
        for(int j{0}; j <= n; ++j)
        {
            dp[i][j] = 0;
        }
    }
    // 变到空，全部删掉就行
    for(int i{1}; i <= m; ++i) dp[i][0] = i;
    // 空变，全部加进来就可以
    for(int j{1}; j <= n; ++j) dp[0][j] = j;

    for(int i{1}; i <= m; ++i)
    {
        for(int j{1}; j <= n; ++j)
        {
            if(word1[i - 1] == word2[j - 1]) dp[i][j] = dp[i - 1][j - 1];
            // 
            else dp[i][j] = min(dp[i - 1][j], min(dp[i][j - 1], dp[i - 1][j - 1])) + 1;
        }
    }
    return dp[m][n];
}

/*--------------------------118--------------------------*/ //没做出来
vector<vector<int>> PascalTriangle::generate(int numRows)
{
    vector<vector<int>> ans{};
    // 一个行更特殊
    if(numRows == 1) return {{1}};
    // 把最初的两个特殊情况放进去
    ans.push_back({1});
    ans.push_back({1, 1});
    for(int i{2}; i < numRows; ++i)
    {
        // 元素个数比行数多1
        vector<int> temp(i + 1, 0);
        // 起点和终点都是特殊，直接赋值，因为只有一个数
        temp[0] = 1;
        temp[i] = 1;
        // 最开始的0号和最尾端的1号已经赋值，不用管
        for(int j{1}; j < i; ++j)
        {
            temp[j] = ans[i - 1][j - 1] + ans[i - 1][j];  
        }
        ans.push_back(temp);
    }
    return ans;     
}

/*--------------------------139--------------------------*/ //没做出来
bool wordBreak(string s, vector<string>& wordDict)
{
    unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
    int n = s.size();
    // 第i个位置能否完整的分隔为多个单词
    bool dp[n + 1];
    for(int i{0}; i < n + 1; ++i) dp[i] = false;
    dp[0] = true;
    // 以s的第i位结尾的字符串能否被分割
    for(int i{1}; i <= n; ++i)
    {
        for(int j{0}; j < i; ++j)
        {
                // 扩展了一个字符后，需要判断当前的j能否分割，
                // 剩下的j到i-j是否属于一个单词，属于就ok了。跳出来去下一个点
                // dp[j]的j其实是s中第j-1位，所以sub的时候可以直接从j开始
                // 同事i也是第 i - 1位，所以不能包括i，不能i-j+1
            if(dp[j] && wordSet.find(s.substr(j, i - j)) != wordSet.end())
            {
                dp[i] = true;
                break;
            }
        }
    }
    return dp[n];
}

/*--------------------------152--------------------------*/ //没做出来
int MaximumProductSubarray::maxProduct(vector<int>& nums)
{
    // 需要用double才能过
    int maxP = nums[0];
    int minP = nums[0];
    int n = nums.size();
    int ans = nums[0];
    // 构建最小值和最大值
    // 不需要管正负
    // 每一次的最大值就是，当前值，最大值乘以当前值，最小值。这三这中间取
    // 最小值也同理
    // 最后更新最大值即可
    for(int i{1}; i < n; ++i)
    {
        int mx = maxP;
        int mn = minP;
        maxP = max(mx * nums[i], max(nums[i], mn * nums[i]));
        minP = min(mn * nums[i], min(nums[i], mx * nums[i]));
        ans = max(ans, maxP);
    }
    return ans;
}

/*--------------------------198--------------------------*/ //没做出来
int HouseRobber::rob(vector<int>& nums)
{
    int n = nums.size();
    // 到第i个房子能抢到的最大值
    int dp[n];
    dp[0] = nums[0];
    if(n == 1) return dp[0];
    dp[1] = max(nums[0], nums[1]);
    // 抢上一家，这一家就不能抢
    // 抢上上一家，可以抢当前
    for(int i{2}; i < n; ++i)
    {
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);
    }
    return dp[n - 1];
}

/*--------------------------279--------------------------*/ //没做出来
int PerfectSquares::numSquares(int n)
{
    int dp[n + 1];
    dp[0] = 0;
    for(int i{1}; i <= n; ++i)
    {
        int minN = INT_MAX;
        // 关键在于这个 j * j <= i。这种条件设置要记住
        // for循环的条件可以是简单的表达式
        for(int j{1}; j * j <= i; ++j)
        {
            // 这里减去的是j平方，如果找到了最小的只需要加上j平方就行了。也就是后续的1。
            // 因为dp[0]一直是0，所以当i刚好为完全平方的时候也会找到
            minN = min(minN, dp[i - j * j]);
        }
        // 这里保证小于i的都有对应值了
        dp[i] = 1 + minN;
    }
    return dp[n];
}

int LongestIncreasingSubsequence::lengthOfLIS(vector<int>& nums)
{
    int n = nums.size();
    // 以i结尾的最大长度
    int dp[n];
    int ans{1};
    for(int i{0}; i < n; ++i) dp[i] = 1;
    // 一层不行，会错过很多可能的值。因为每次都是重新开始计数。但是这里是subsequence。
    // 跳过这个最大的也是可以的。要把前面的到i为止都要考虑一遍才是正确的
    for(int i{1}; i < n; ++i)
    {
        for(int j{0}; j < i; ++j)
        {
            // 这里不是相邻的比，而是与最后一位比
            if(nums[i] > nums[j]) dp[i] = max(dp[i], dp[j] + 1);
        }
    }
    for(int i{0}; i < n; ++i) ans = max(ans, dp[i]);
    return ans;
}

/*--------------------------322--------------------------*/ //没做出来
int CoinChange::coinChange(vector<int>& coins, int amount)
{
    // 第i个值，所需要的最小硬币数量
    int dp[amount + 1];
    sort(coins.begin(), coins.end());
    dp[0] = 0;

    for(int i{1}; i <= amount; ++i)
    {
        dp[i] = INT_MAX;
        // 这里用硬币来循环会减少很多次循环，一个个加很容易出问题
        for(int c: coins)
        {
            if(i - c < 0) break;
            if(dp[i - c] != INT_MAX) dp[i] = min(dp[i], dp[i - c] + 1);
        }
    }
    return dp[amount] == INT_MAX ? - 1 : dp[amount];
}

/*--------------------------416--------------------------*/ // 没做出啦，看中文官方答案
bool canPartition(vector<int>& nums)
{
    int n = nums.size();
    if( n < 2) return false;
    
    int sum = accumulate(nums.begin(), nums.end(), 0);
    int maxNum = *max_element(nums.begin(), nums.end());
    // 总和为奇数不能平分，所以不存在答案
    if(sum % 2 != 0) return false;
    // 单个数大于一半，不存在答案
    int target = sum / 2;
    if(maxNum > target) return false;

    // dp[i][j]，从0到第i号元素中选取，能否构成和为j
    vector<vector<int>> dp(n, vector<int>(target + 1, 0));
    for(int i{0}; i < n; ++i)
    {
        // 全都不取就可以构成0，所以为true
        dp[i][0] = true;
    }
    
    dp[0][nums[0]] = true;
    for(int i{1}; i < n; ++i)
    {
        for(int j{1}; j <= target; ++j)
        {
            // 目标值大于当前的num可以选或者不选，全都反应在上一层的j里面
            // 如果两种选择都不满足，那就是无法满足了
            if(nums[i] <= j) dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i]];
            // 目标值大于当前的num了，肯定不能选当前的num了。
            else dp[i][j] = dp[i - 1][j];
        }
    }
    return dp[n - 1][target];
}

int LongestCommonSubsequence::longestCommonSubsequence(string text1, string text2)
{
    // 有人说和Edit distance类似， 但这表达式和意义完全不一样吧
    // 只能说架构一样？
    int m = text1.size();
    int n = text2.size();
    // 可以这样初始化
    // 其中 dp[i][j] 表示 text1[0:i] 和 text2[0:j] 的最长公共子序列的长度。
    int dp[1001][1001] = {};

    for(int i{0}; i < m; ++i)
    {
        for(int j{0}; j < n; ++j)
        {
            // 相同要取找上一层加1
            if(text1[i] == text2[j]) dp[i + 1][j + 1] = dp[i][j] + 1;
            // 不相同就去找，减掉一个数的最大值
            else dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1]);
        }
    }
    return dp[m][n];
}

/*--------------------------200--------------------------*/
int NumberOfIslands::numIslands(vector<vector<char>>& grid)
{
    int ans{0};
    int r = grid.size();
    int c = grid[0].size();
    for(int i{0}; i < r; ++i)
    {
        for(int j{0}; j < c; ++j)
        {
            if(grid[i][j] == '1')
            {
                dfs(grid, i, j);
                ++ans;
            }
        }
    }
    return ans;
}

void NumberOfIslands::dfs(vector<vector<char>>& grid, int r, int c)
{
    grid[r][c] = '*';
    for(auto dir :DIR4)
    {
        int nextR = r + dir[0];
        int nextC = c + dir[1];
        if(nextR < grid.size() && nextR >= 0 && nextC < grid[0].size() && nextC >= 0 && grid[nextR][nextC] == '1')
        {
            dfs(grid, nextR, nextC);
        }
    }
}


/*--------------------------207--------------------------*/
bool CourseSchedule::canFinish(int numCourses, vector<vector<int>>& prerequisites)
{
    vector<vector<int>> G(numCourses);
    vector<int> degree(numCourses, 0), bfs;
    // 0是要上的课，1是前提。1-->0
    for(auto& pre : prerequisites)
    {
        G[pre[1]].push_back(pre[0]);
        // 第pre[0]堂课的入度加1
        ++degree[pre[0]];
    }
    // 关键是下面两个for循环
    // 1. 寻找开始的课程，也就是入度为0的课程
    // 2. 对入度为0的课程进行bfs，把和它连接的课程入度全部减1。
    // 3. 检查入度，入度为0的继续加入
    // 4. 能上完所有课程，bfs里面所有课程入度都变为0了
    for(int i{0}; i < numCourses; ++i)
    {
        // 入度为0的课先开始
        if(!degree[i]) bfs.push_back(i);
    }
    for(int i{0}; i < bfs.size(); ++i)
    {
        for(int j: G[bfs[i]])
        {
            --degree[j];
            if(degree[j] == 0) bfs.push_back(j);
        }
    }
    return bfs.size() == numCourses;
}


int RottingOranges::orangesRotting(vector<vector<int>>& grid)
{
    queue<pair<int,int>> q{};
    int m = grid.size();
    int n = grid[0].size();
    int orange{0};
    // 记录坏橘子队列，和新鲜橘子数量
    for(int i{0}; i < m; ++i)
    {
        for(int j{0}; j < n; ++j)
        {
            if(grid[i][j] == 2)
            {
                q.emplace(i, j);
            }
            else if(grid[i][j] == 1) ++orange;
        }
    }
    // 没有新鲜也没有坏的，直接返回
    if(orange == 0 && q.size() == 0) return 0;
    int ans{-1};
    // BFS形式一层层的往外扩张，
    while(q.size())
    {
        int s = q.size();
        for(int i{0}; i < s; ++i)
        {
            pair<int, int> pos = q.front();
            q.pop();
            for(auto dir : DIR4)
            {
                int nextR = pos.first + dir[0];
                int nextC = pos.second + dir[1];
                if(nextR < m && nextR >= 0 && nextC < n && nextC >= 0 && grid[nextR][nextC] == 1)
                {
                    --orange;
                    grid[nextR][nextC] = 2;
                    q.emplace(nextR, nextC);
                }
            }
        }
        ++ans;
    }
    return orange == 0 ? ans : -1;    
}

/*--------------------------45--------------------------*/ // 没做出来
int JumpGameII::jump(vector<int>& nums)
{
    int currMax{0}, nextMax{0}, i{0};
    int n = nums.size();
    int level{0};
    // bfs的方式，一层能找到的最远距离
    // 会一个个遍历，所以当前i的值代表的是遍历的末尾
    while(currMax - i + 1 > 0) // 当前层还能用的node数量
    {
        for(; i <= currMax; ++i) // currMax就是当前层的末尾
        {
            nextMax = max(nextMax, i + nums[i]);
            if(i >= n - 1) return level;
        }
        ++level;
        currMax = nextMax;
    }
    return 0;
}

bool JumpGame::canJump(vector<int>& nums)
{
    int i{0};
    int n = nums.size();
    // 一个个遍历（i）一直找能够到的最大值 reach
    // 这个最大值能比i大，就证明肯定可以到了。因为可以选择不走那么远
    // 当是当遍历的值要大于能够到的最大值时一定是不能跳完的
    for(int reach = 0; i < n && i <= reach; ++i)
    {
        reach = max(i + nums[i], reach);
    }
    return i == n;
}

int BestTimeToBuyAndSellStock::maxProfit(vector<int>& prices)
{
    int minNum = INT_MAX;
    int ans{0};
    // 不断的更新当前的最小值
    // 然后比较最小值和当前值之间能形成的利益，看是否大于上一次
    for(int i{0}; i < prices.size(); ++i)
    {
        minNum = min(minNum, prices[i]);
        ans = min(ans, prices[i] - minNum);
    }
    return ans;
}

vector<int> PartitionLabels::partitionLabels(string s)
{
    unordered_map<char, int> hash{};
    int n = s.size();
    // 把每个字符最后出现的位置记录下来
    for(int i{0}; i < n; ++i)
    {
        hash[s[i]] = i;
    }
    vector<int> ans{};
    int start{0};
    int end{0};
    // 需要一个个的进行索引
    for(int i{0}; i < n; ++i)
    {
        // 不断地更新i对应字符的最终位置，看是否超过了前面那个
        end = max(end, hash[s[i]]);
        // 到达了当前的最大位置，说明可以算成一个分割区间了
        if(i == end)
        {
            ans.push_back(end - start + 1);
            start = end + 1;
        }
    }
    return ans;
}

vector<int> TwoSum::twoSum(vector<int>& nums, int target)
{
    unordered_map<int, int> hash{};
    for(int i{0}; i < nums.size(); ++i)
    {
        if(hash.find(target - nums[i]) != hash.end()) return {hash[target - nums[i]], i};
        hash[nums[i]] = i;
    }
    return {};
}

/*--------------------------45--------------------------*/
vector<vector<string>> groupAnagrams(vector<string>& strs)
{
    vector<vector<string>> ans{};
    unordered_map<string, vector<string>> hash{};
    for(auto str : strs)
    {
        string temp = str;
        sort(temp.begin(), temp.end());
        hash[temp].push_back(str);
    }
    for(auto s : hash)
    {
        ans.push_back(s.second);
    }
    return ans;
}

/*--------------------------128--------------------------*/ // 没做出来
int longestConsecutive(vector<int>& nums)
{
    unordered_set<int> hash{};
    int best{0};
    int n = nums.size();
    for(int i{0}; i < n; ++i) hash.insert(nums[i]);
    for(auto num : hash)
    {
        int end{};
        // hash里面是一个个的找，这会导致有些数字已经用过了。
        // 如果这时候num的前面一个数是存在的，那就不应该管它。
        // 要么num已经被计算过一次了，要么后续会被计算到。
        if(!hash.count(num - 1))
        {
            end = num + 1;
            // 找了连续的开头，逐个往后加
            while(hash.count(end))
            {
                end += 1;
            }
            // 全部找到后判断是否要更新最大值
            // 最后一个end是不连续的，所以这里要去掉end，也就是不加1
            best = max(end - num, best);
        }    
    }
    return best;
}
/*--------------------------560--------------------------*/ // 没做出来
int subarraySum(vector<int>& nums, int k)
{
    unordered_map<int, int> hash{};
    int n = nums.size();
    int pre{};
    int ans{};
    // 以防前缀和刚好为k的时候
    hash[0] = 1;
    // 前缀和
    for(auto x : nums)
    {
        pre += x;
        // 因为有负数，所以同一个前缀和可能对应多个位置
        if(hash.find(pre - k) != hash.end()) ans += hash[pre - k];
        // 这里就是把对应多个位置的前缀和给加起来。这里加起来是为了给后面用
        // 大部分的值还是1。
        ++hash[pre];
    }
    return ans;
}

/*--------------------------215--------------------------*/ // 不是特别熟练
int KthLargestElementInAnArray::findKthLargest(vector<int>& nums, int k)
{
    // 默认大根堆
    // 这样处理变成小根堆，把最大的k元素存进来
    priority_queue<int, std::vector<int>, std::greater<int>> mq{};
    for(int i{0}; i < nums.size(); ++i)
    {
        if(mq.size() < k)
        {
            mq.push(nums[i]);
        }
        else
        {
            // 先push再pop，因为还要判断当前这个值和top的大小，直接pop会漏掉一次判断
            mq.push(nums[i]);
            mq.pop();
        }
    }
    return mq.top();
}

void MedianFinder::addNum(int num)
{
    // small来存小的那一段，最顶层是小段的最中间
    // large存大的那一段，最顶层是大段的最中间
    // 因为流式造成的中间值的变化问题，全部交给大根堆和小根堆去解决了

    // 先判断这个数是不是会属于小段，把最大的那个数挤出来
    small.push(num);
    // 挤出来之后取负数，就变成最小的了。
    // 越大的数越在底层（转成负数了），越难挤出来。
    large.push(-small.top());
    // 因为挤出来了一个，所以要pop掉
    small.pop();

    // 因为前面一直把small的存到large里面
    // 但我们为了保证两段相差最多为1，这里需要从large里面挤出来还给small
    // if放在后面是因为，新来的数需要在小段和大段里面都取比较一次
    // 两边的中心位置可能都要更新

    if(small.size() < large.size())
    {
        small.push(-large.top());
        large.pop();
    }
}
double MedianFinder::findMedian()
{
    // small 肯定是大于等于large的，从1个数开始考虑下addNum的流程就知道了
    return small.size() > large.size() ? small.top() : (small.top() - large.top()) / 2.0f;
}

vector<int> topKFrequent(vector<int>& nums, int k)
{
    unordered_map<int, int> hash;
    priority_queue<int> hash;
}

