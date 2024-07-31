#include <unordered_map>
#include <vector>
#include <string>
#include <queue>
#include <unordered_set>

using namespace std;
const int DIR4[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
struct TreeNode {
     int val;
     TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 };


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

class PalindromePartition
{
private:
    vector<vector<int>> f{};
    vector<vector<string>> ret{};
    vector<string> ans{};
    int n{};

public:
    vector<vector<string>> partition(string s);
    void backTracking(const string& s, int i);

};

class MedianOfTwoSortedArray
{
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2);
};

class SearchInRotatedSortedArray
{
public:
    int search(vector<int>& nums, int target);
};

class FindFirstAndLastPositionOfElementInSortedArray
{
public:
    vector<int> searchRange(vector<int>& nums, int target);
};

class SearchInsertPosition
{
public:
    int searchInsert(vector<int>& nums, int target);
};

class SearchA2DMatrix
{
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target);
};

class BinaryTreeMaximumPathSum
{
private:
    int maxSum = INT_MIN;
public:
    int maxGain(TreeNode* root);
    int maxPathSum(TreeNode* root);
};

class FindMinimumInRotatedSortedArray
{
public:
    int findMin(vector<int>& nums);
};

class BinaryTreeInorderTraversal
{
public:
    void dfs(TreeNode* root, vector<int>& ans);
    vector<int> inorderTraversal(TreeNode* root);
};

class ValidateBinarySearchTree
{
private:
    bool m_ans{true};
public:
    void dfs(TreeNode* root, TreeNode*& pre);
    bool isValidBST(TreeNode* root);
};

class SymmetricTree
{
public:
    bool equal(TreeNode* left, TreeNode* right);
    bool isSymmetric(TreeNode* root);
};

class BinaryTreeLevelOrderTraversal
{
public:
    vector<vector<int>> levelOrder(TreeNode* root);
};

class MaximumDepthOfBinaryTree
{
public:
    int maxDepth(TreeNode* root);
};

class ConstructBinaryTreeFromPreorderAndInorderTraversal
{
private:
    unordered_map<int, int> hash{};
public:
    TreeNode* myTree(vector<int>& preorder, vector<int>& inorder, int p_left, int p_right, int in_left, int in_right);
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder);
};

class ConvertSortedArrayToBinarySearchTree
{
public:
    TreeNode* sortedArrayToBST(vector<int>& nums);
};

class FlattenBinaryTreeToLinkedList
{
private:
    TreeNode* pre = nullptr;
public:
    void flatten(TreeNode* root);
};

class BinaryTreeRightSideView
{
public:
    vector<int> rightSideView(TreeNode* root);
};

class InvertBinaryTree
{
public:
    TreeNode* invertTree(TreeNode* root);
};

class KthSmallestElementInABST
{
public:
    void inorder(TreeNode* root, const int k, int i, int ans);
    int kthSmallest(TreeNode* root, int k);
};

class LowestCommonAncestorOfABinaryTree
{
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q);
};

class  PathSumIII
{
public:
    void dfs(TreeNode* root, int targetSum, long long temp, int& ans);
    void preOrder(TreeNode* root, int targetSum, int& ans);
    int pathSum(TreeNode* root, int targetSum);
};

class DiameterOfBinaryTree
{
public:
    int dfs(TreeNode* root, int &d);
    int diameterOfBinaryTree(TreeNode* root);
};

class LongestPalindromicSubstring
{
public:
    string longestPalindrome(string s);
};

class LongestValidParentheses
{
public:
    int longestValidParentheses(string s);
};

class UniquePaths
{
public:
    int uniquePaths(int m, int n);
};

class MinimumPathSum
{
public:
    int minPathSum(vector<vector<int>>& grid);
};

class ClimbingStairs
{
public:
    int climbStairs(int n);
};

class EditDistance
{
public:
    int minDistance(string word1, string word2);
};

class PascalTriangle
{
public:
    vector<vector<int>> generate(int numRows);
};

class WordBreak
{
public:
    bool wordBreak(string s, vector<string>& wordDict);
};

class MaximumProductSubarray
{
public:
    int maxProduct(vector<int>& nums);
};

class HouseRobber
{
public:
    int rob(vector<int>& nums);
};

class PerfectSquares
{
public:
    int numSquares(int n);
};

class LongestIncreasingSubsequence
{
public:
    int lengthOfLIS(vector<int>& nums);
};