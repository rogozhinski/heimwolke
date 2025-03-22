#include <vector>
#include <iostream>
#include <memory>

using namespace std;

class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        
    }
};


int main () {
	// this code
	unique_ptr<Solution>  solution = make_unique<Solution> ();
	vector<int> height = {8,7,2,1};
	cout << (*solution).longestConsecutive(height);
	return 0;
};