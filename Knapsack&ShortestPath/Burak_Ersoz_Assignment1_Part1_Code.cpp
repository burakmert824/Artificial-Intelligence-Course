#include<iostream>
#include <fstream>
#include<vector>
#include<string>
#include<queue>
#include<stack>
#include<chrono>

using namespace std;
using namespace chrono;

class Item {
    public:
    int id;
    int benefit;
    int weight;
    Item(int newId, int newBenefit, int newWeight) 
        : id(newId), benefit(newBenefit), weight(newWeight) {
    }
};


pair<int,vector<Item> > readKnapsackItems(const string& filename) {
    vector<Item> items;
    ifstream inputFile(filename);

    if (!inputFile.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return make_pair(0,items);
    }
    int max_weight;
    inputFile>>max_weight;
    int id, benefit, weight;
    while (inputFile >> id >> benefit >> weight) {
        Item newItem(id, benefit, weight);
        items.push_back(newItem);
    }

    inputFile.close();
    return make_pair(max_weight,items);
}


struct Node{
    public:
    //holds the total weights of items in the bitItems
    int totalWeight;
    //holds the total benefits of the items in the bitItems
    int totalBenefit;
    //represents all items in bits according to their taken or not taken informations
    int bitItems;
    //represents which index is working on
    int index;
    Node(int totalWeight,int totalBenefit, int bitItems, int index) 
        : totalWeight(totalWeight),totalBenefit(totalBenefit), bitItems(bitItems), index(index) {
    }
};

Node bfs(vector<Item>&items, int max_weight){
    Node maximumNode = Node(0,0,0,0);

    queue<Node> qu;
    qu.push(Node(0,0,0,0));

    while(!qu.empty()){
        Node top = qu.front();
        qu.pop();
        
        if(top.totalWeight > max_weight) continue;

        if(top.totalBenefit > maximumNode.totalBenefit){
            maximumNode = top;
            
        }else if(top.totalBenefit == maximumNode.totalBenefit && top.totalWeight < maximumNode.totalWeight){
            maximumNode = top;
        }
        
        if(top.index == items.size()) continue; 


        Item curr_item = items[top.index];
        Node get_item = Node(
            top.totalWeight + curr_item.weight,
            top.totalBenefit + curr_item.benefit,
            top.bitItems | 1<<top.index,
            top.index+1
            );

        Node leave_item = Node(
            top.totalWeight,
            top.totalBenefit,
            top.bitItems,
            top.index+1
            );

        qu.push(get_item);
        qu.push(leave_item);
    }
    return maximumNode;
}


Node dfs(vector<Item>&items, int max_weight){
    Node maximumNode = Node(0,0,0,0);

    stack<Node> st;
    st.push(Node(0,0,0,0));

    while(!st.empty()){
        Node top = st.top();
        st.pop();
        
        if(top.totalWeight > max_weight) continue;

        if(top.totalBenefit > maximumNode.totalBenefit){
            maximumNode = top;
        }
        else if(top.totalBenefit == maximumNode.totalBenefit && top.totalWeight < maximumNode.totalWeight){
            maximumNode = top;
        }
        
        if(top.index == items.size()) continue; 


        Item curr_item = items[top.index];
        Node get_item = Node(
            top.totalWeight + curr_item.weight,
            top.totalBenefit + curr_item.benefit,
            top.bitItems | 1<<top.index,
            top.index+1
            );

        Node leave_item = Node(
            top.totalWeight,
            top.totalBenefit,
            top.bitItems,
            top.index+1
            );

        st.push(leave_item);
        st.push(get_item);
    }
    return maximumNode;
}

string binary_rep(vector<Item> items, int bits){
    string a = "";
    while(bits){
        if(bits%2) a = a + "1";
        else a = a + "0";
        bits/=2;
    }

    while(items.size() < a.length()) a = "0" + a;
    
    return a;
}

int main(){
    
    vector<Item> items;
    int max_weight;
    string fileName = "knapsack_items.txt";
    pair<int,vector<Item> > max_and_items = readKnapsackItems(fileName);
    max_weight = max_and_items.first;
    items = max_and_items.second;
    // Start measuring time
    auto start_all = high_resolution_clock::now();

    Node bfs_a = bfs(items, max_weight);

    auto end_bfs = high_resolution_clock::now();

    Node dfs_a = dfs(items, max_weight);

    auto end_all = high_resolution_clock::now();

    auto dfs_duration = duration_cast<microseconds>(end_bfs - start_all);
    auto bfs_duration = duration_cast<microseconds>(end_all - end_bfs);

    cout<<"bfs time(microseconds) : "<<bfs_duration.count()<<endl;
    cout<<bfs_a.totalBenefit<<" "<<bfs_a.totalWeight<<" "<<binary_rep(items,bfs_a.bitItems)<<endl;
    
    cout<<"dfs time(microseconds) : "<<dfs_duration.count()<<endl;
    cout<<dfs_a.totalBenefit<<" "<<dfs_a.totalWeight<<" "<<binary_rep(items,dfs_a.bitItems)<<endl;



    return 0;
}