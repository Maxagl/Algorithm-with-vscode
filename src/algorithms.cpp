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

/*--------------------------437--------------------------*/
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
