#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <queue>
#include <stack>
#include <map>
#define mp make_pair

using namespace std;

void readSpainMap(const string &filename, map<string, int> &distances, map<string, vector<pair<string, int> > > &graph)
{
    distances.clear();
    graph.clear();

    ifstream inputFile(filename);
    if (!inputFile.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    int city_count, city_distance;
    string city_name;

    inputFile >> city_count;
    while (city_count)
    {
        inputFile >> city_name >> city_distance;
        
        distances[city_name] = city_distance;

        vector<pair<string, int> > temp;
        graph[city_name] = temp;

        city_count--;
    }

    string city_a, city_b;
    while (inputFile >> city_a >> city_b >> city_distance)
    {
        graph[city_a].push_back(mp(city_b, city_distance));
        graph[city_b].push_back(mp(city_a, city_distance));
    }

    // cout<<"\n----------Debug-------------\n";
    // cout<<"Distances:"<<endl;
    // for(pair<string,int> ds : distances){
    //     cout<<ds.first<<" : "<<ds.second<<endl;
    // }
    // int count = 0;
    // cout<<"\nGraph:"<<endl;
    // for(pair<string, vector<pair<string, int> > > tp : graph){
    //     count ++;
    //     cout<<tp.first<<":"<<endl;
    //     for(auto i :tp.second){
    //         cout<<"==>"<<i.first<<" - "<<i.second<<endl;
    //     }
    // }
    // cout<<"count : "<<count<<endl;

    inputFile.close();
    return;
}

// Greedy best-first search
// line_distance which is fn, graph, start, finish
void GBFS(map<string,int>&fn, map<string,vector<pair<string,int> > > &graph,string start, string finish){
    map<string,int> visited;
    //goes to - <coming from, path distance>
    map<string, pair<string,int> > parent;
    //h(n),edge, goes to, comes from (minimum priority queue)
    priority_queue<pair<int,pair<int,pair<string,string> > >,vector<pair<int,pair<int,pair<string,string> > > > ,greater<pair<int,pair<int,pair<string,string> > > > >  pri_qu;
    pri_qu.push(mp(0,mp(0,mp(start, start ) ) ) );

    while(!pri_qu.empty()){
        string city, comes_from;
        int line_distance,edge_distance;
        //f(city), edge, goes to, coming from
        pair<int,pair<int, pair<string, string> > > top = pri_qu.top();
        pri_qu.pop();
        
        line_distance = top.first;
        edge_distance = top.second.first;
        
        city = top.second.second.first;
        comes_from = top.second.second.second;

        // if it is visited before don't go there.
        if(visited[city] == 1){
            continue;
        }

        visited[city] = 1;
        parent[city] = mp(comes_from,edge_distance);

        // if it founds the finish break  
        if(city == finish){
            break;
        }

        for(int i = 0; i < graph[city].size(); i++){
            string goes_to = graph[city][i].first;
            int edge = graph[city][i].second;
            // if it is visited continue
            if(visited[goes_to] == 1) continue;
            //f(goes_to), edge, goes_to, city
            pair<int,pair<int, pair<string, string> > > temp;
            temp = mp(fn[goes_to], mp(edge, mp(goes_to, city)));
            pri_qu.push(temp);
        }
    }
    //create path
    vector<string> path;
    string temp_node = finish;
    int path_lenght = 0;
    while(temp_node != start){
    //going to, comes_from, edge
    //map<string, pair<string,int> > parent;
        path_lenght += parent[temp_node].second;
        path.push_back(temp_node);
        temp_node = parent[temp_node].first;
    }
    path.push_back(start);
    reverse(path.begin(),path.end());
    
    cout<<"Greedy Best First Search"<<endl;
    // printing the path lenght
    cout<<"Path lenght : "<<path_lenght<<endl;
    // printing the path
    cout<<"Path : ";
    for(auto i : path){
        cout<<i;
        if(i!=finish){
            cout<<"->";
        }
    }
    cout<<endl;
}

// A* search
// line_distance, graph, start, finish
void A_star_search(map<string,int>&line_dist_list, map<string,vector<pair<string,int> > > &graph,string start, string finish){
    map<string,int> visited;
    
    int inf=99999999;
    map<string,int> distance_from_start;
    for(pair<string,int> temp:line_dist_list){
        distance_from_start[temp.first] = inf;
    }

    //goes to - <coming from, path distance>
    map<string, pair<string,int> > parent;
    //h(n),edge, goes to, comes from (minimum priority queue)
    priority_queue<pair<int,pair<int,pair<string,string> > >,vector<pair<int,pair<int,pair<string,string> > > > ,greater<pair<int,pair<int,pair<string,string> > > > >  pri_qu;
    pri_qu.push(mp(0,mp(0,mp(start, start ) ) ) );

    while(!pri_qu.empty()){
        string city, comes_from;
        int hn_city,edge_distance;
        //f(city), edge, goes to, coming from
        pair<int,pair<int, pair<string, string> > > top = pri_qu.top();
        pri_qu.pop();
        
        hn_city = top.first;
        edge_distance = top.second.first;
        
        city = top.second.second.first;
        comes_from = top.second.second.second;

        // if it is visited before don't go there.
        if(visited[city] == 1){
            continue;
        }

        visited[city] = 1;
        parent[city] = mp(comes_from, edge_distance);

        // if it founds the finish break  
        if(city == finish){
            break;
        }

        for(int i = 0; i < graph[city].size(); i++){
            string goes_to = graph[city][i].first;
            int edge = graph[city][i].second;
            // if it is visited continue
            if(visited[goes_to] == 1) continue;
            //hn(goes_to), edge, goes_to, city
            pair<int,pair<int, pair<string, string> > > temp;
            int gn = distance_from_start[city] + edge;
            int hn = line_dist_list[goes_to] + gn;

            temp = mp(hn, mp(edge, mp(goes_to, city)));
            pri_qu.push(temp);
        }
    }
    //create path
    vector<string> path;
    string temp_node = finish;
    int path_lenght = 0;
    while(temp_node != start){
    //going to, comes_from, edge
    //map<string, pair<string,int> > parent;
        path_lenght += parent[temp_node].second;
        path.push_back(temp_node);
        temp_node = parent[temp_node].first;
    }
    path.push_back(start);
    reverse(path.begin(),path.end());
    
    cout<<"\nA* Search"<<endl;
    // printing the path lenght
    cout<<"Path lenght : "<<path_lenght<<endl;
    // printing the path
    cout<<"Path : ";
    for(auto i : path){
        cout<<i;
        if(i!=finish){
            cout<<"->";
        }
    }
    cout<<endl;
}

int main()
{
    string file_name = "spain_map.txt";
    string start = "Malaga";
    string finish = "Valladolid";
    // string file_name = "spain_map2.txt";
    // string start = "a";
    // string finish = "d";

    map<string, int> lineDistance;
    map<string, vector<pair<string, int> > > graph;
    readSpainMap(file_name,lineDistance,graph);
    
    GBFS(lineDistance,graph,start,finish);
    
    A_star_search(lineDistance,graph,start,finish);

    return 0;
}