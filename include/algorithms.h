#include <unordered_map>
#include <vector>
#include <string>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <list>
#include <stack>
#include <ctype.h>
#include <algorithm>

// 1. 快排
// 2. 归并排序
// 3. 
using namespace std;
const int DIR4[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
struct TreeNode {
     int val;
     TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 };

class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
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

class CoinChange
{
public:
    int coinChange(vector<int>& coins, int amount);
};

class PartitionEqualSubsetSum
{
public:
    bool canPartition(vector<int>& nums);
};

class LongestCommonSubsequence
{
public:
    int longestCommonSubsequence(string text1, string text2);
};

class NumberOfIslands
{
public:
    int numIslands(vector<vector<char>>& grid);
    void dfs(vector<vector<char>>& grid, int r, int c);
};

class CourseSchedule
{
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites);
};

class RottingOranges
{
public:
    int orangesRotting(vector<vector<int>>& grid);
};

class JumpGameII
{
public:
    int jump(vector<int>& nums);
};

class JumpGame
{
public:
    bool canJump(vector<int>& nums);
};

class BestTimeToBuyAndSellStock
{
public:
    int maxProfit(vector<int>& prices);
};

class PartitionLabels
{
public:
    vector<int> partitionLabels(string s);
};

class TwoSum
{
public:
    vector<int> twoSum(vector<int>& nums, int target);
};

class GroupAnagrams
{
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs);
};

class LongestConsecutiveSequence
{
public:
    int longestConsecutive(vector<int>& nums);
};

class SubarraySumEqualsK
{
public:
    int subarraySum(vector<int>& nums, int k);
};

class KthLargestElementInAnArray
{
public:
    int findKthLargest(vector<int>& nums, int k);
};

class MedianFinder
{
public:
    priority_queue<int> small;
    priority_queue<int> large;
    MedianFinder();
    void addNum(int num);
    double findMedian();
};

class TopKFrequentElements
{
public:
    vector<int> topKFrequent(vector<int>& nums, int k);
};

class AddTwoNumbers
{
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2); 
};

class RemoveNthNodeFromEndofList
{
public:
    ListNode* removeNthFromEnd(ListNode* head, int n);

};

class MergeTwoSortedLists
{
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2);
};

class MergekSortedLists
{
public:
    ListNode* mergeKLists(vector<ListNode*>& lists);
    ListNode* merge(ListNode* list1, ListNode* list2);
};

class SwapNodesinPairs
{
public:
    ListNode* swapPairs(ListNode* head);
};

class ReverseNodesinkGroup
{
public:
    ListNode* reverseKGroup(ListNode* head, int k);
    ListNode* ReverseNode(ListNode* head);

};

class CopyListwithRandomPointer
{
public:
    Node* copyRandomList(Node* head);
};

class LinkedListCycle
{
public:
    bool hasCycle(ListNode *head);
};

class LinkedListCycleII
{
public:
    ListNode *detectCycle(ListNode *head);
};

class LRUCache
{
public:
    int size{};
    unordered_map<int, list<pair<int, int>>::iterator> m_map;
    list<pair<int, int>> m_list;
    LRUCache(int capacity);
    int get(int key);
    void put(int key, int value);
};

class SortList
{
public:
    ListNode* sortList(ListNode* head);
    ListNode* mergelist(ListNode *l1, ListNode *l2);   
};

class IntersectionofTwoLinkedLists
{
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB);
};

class PalindromeLinkedList
{
public:
    bool isPalindrome(ListNode* head);
};

class RotateImage
{
public:
    void rotate(vector<vector<int>>& matrix);
};

class SpiralMatrix
{
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix);
};

class SetMatrixZeroes
{
public:
    void setZeroes(vector<vector<int>>& matrix);
};

class SearchA2DMatrixII
{
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target);
};

class LongestSubstringWithoutRepeatingCharacters
{
public:
    int lengthOfLongestSubstring(string s);
};

class MinimumWindowSubstring
{
public:
    string minWindow(string s, string t);
};

class SlidingWindowMaximum
{
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k);
};

class FindAllAnagramsinaString
{
public:
    vector<int> findAnagrams(string s, string p);    
};

class ValidParentheses
{
public:
    bool isValid(string s);
};

class LargestRectangleinHistogram
{
public:
    int largestRectangleArea(vector<int>& heights);
};

class MinStack
{
public:
    stack<int> s1;
    stack<int> s2;
    MinStack();
    void push(int val);
    void pop();
    int top();
    int getMin();
};

class DecodeString
{
public:
    string decodeString(string s);
    string dfs(const string& s, int& i);
};

class DailyTemperatures
{
public:
    vector<int> dailyTemperatures(vector<int>& temperatures);
};

class ContainerWithMostWater
{
public:
    int maxArea(vector<int>& height);
};

class ThreeSum
{
public:
    vector<vector<int>> threeSum(vector<int>& nums);
};

class TrappingRainWater
{
public:
    int trap(vector<int>& height);
};

class MoveZeroes
{
public:
    void moveZeroes(vector<int>& nums);
};

class Trie
{
private:
    vector<Trie*> children{};
    bool isEnd{};
    Trie* searchPrefix(string prefix);
public:
    Trie();
    void insert(string word);
    bool search(string word);
    bool startsWith(string prefix);

};

class NextPermutation
{
public:
    void nextPermutation(vector<int>& nums);
};

class FirstMissingPositive
{
public:
    int firstMissingPositive(vector<int>& nums);
};