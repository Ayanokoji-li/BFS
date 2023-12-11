#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <omp.h>

using namespace std;

#define START_NODE 2

string file_name[] = {"web-Stanford.txt", "soc-LiveJournal1.txt", "roadNet-CA.txt"}; 

// 图的数据结构
typedef unordered_map<unsigned long long, vector<unsigned long long>> Graph;

// 从文件中读取图
Graph readGraph(const string &filename, Graph& graph, unsigned long long& max_node) {
    ifstream file(filename);
    for(auto i = 0; i < 4; i++) file.ignore(1000, '\n'); // 跳过前四行
    unsigned long long from, to;
    while (file >> from >> to) {
        graph[from].push_back(to);
        max_node = max(max_node, from);
    }
    return graph;
}

// 广度优先搜索
void bfs(const Graph &graph, unsigned long long &edges,unsigned long long &max_node_index, unsigned long long start, std::vector<unsigned long long>&result) {
    vector<unsigned long long> visited(max_node_index, 0);
    queue<unsigned long long> q;
    q.push(start);
    visited[start] = 1;
    while (!q.empty()) {
        unsigned long long node = q.front();
        q.pop();
        result.push_back(node);
        if(graph.find(node) == graph.end()) continue; // 有些节点没有出度
        for (unsigned long long neighbor : graph.at(node)) {
            edges++;
            if (!visited[neighbor]) {
                q.push(neighbor);
                visited[neighbor] = 1;
            }
        }
    }
}

// 将结果写入文件
void writeResult(const string &filename, const vector<unsigned long long> &result, const unsigned long long &edges, double& time) {
    ofstream file(filename);
    file << "time:" << time << "s\n";
    file << "edges searched:" << edges << "\n";
    file << "nodes searched:\n";
    for (unsigned long long node : result) {
        file << node << "\n";
    }
}

int main() {
    Graph graph;
    cout << "which file to load\n";
    int index = 0;
    cin >> index;
    unsigned long long num_edges = 0;
    unsigned long long max_node_index = 0;
    readGraph(file_name[index], graph, max_node_index);
    vector<unsigned long long> result;
    auto start = omp_get_wtime();
    bfs(graph,num_edges, max_node_index, START_NODE, result);
    auto end = omp_get_wtime();
    auto time = end - start;
    string result_file = "_res.txt";
    result_file = file_name[index] + result_file;
    writeResult(result_file, result, num_edges, time);
    return 0;
}