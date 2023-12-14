#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <omp.h>
#include <cmath>
#include <set>

// using namespace std;

#define START_NODE 2

std::vector<std::string> file_name = {"test","web-Stanford", "soc-LiveJournal1", "roadNet-CA", "com-orkut"}; 
auto num_files = file_name.size();
bool is_parallel = false;

// 图的数据结构
typedef std::unordered_map<unsigned long long, std::vector<unsigned long long>> Graph;

// 从文件中读取图
Graph readGraph(const std::string &filename, Graph& graph, unsigned long long& max_node) {
    std::ifstream file(filename);
    for(auto i = 0; i < 4; i++) file.ignore(1000, '\n'); // 跳过前四行
    unsigned long long from, to;
    while (file >> from >> to) {
        graph[from].push_back(to);
        graph[to].push_back(from);
        max_node = std::max(max_node, from);
        max_node = std::max(max_node, to);
    }
    return graph;
}

// 广度优先搜索
void bfs(const Graph &graph, unsigned long long &edges, unsigned long long &max_node_index, unsigned long long start, std::vector<unsigned long long>&result, std::vector<unsigned long long>&parent, std::vector<unsigned long long>&distance) {

    if(is_parallel == false)
    {
        std::vector<unsigned long long> visited(max_node_index, 0);
        std::queue<unsigned long long> q;
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
                    parent[neighbor] = node;
                    distance[neighbor] = distance[node] + 1;
                }
            }
        }        
    }
    else
    {
        std::vector<unsigned long long> visited(max_node_index, 0);
        std::vector<unsigned long long> frontier;
        frontier.push_back(start);
        visited[start] = 1;
        while (!frontier.empty()) {
            std::unordered_set<unsigned long long> next_frontier;
            result.insert(result.end(), frontier.begin(), frontier.end());

            #pragma omp parallel
            {
                std::vector<unsigned long long> next_frontier_private;
                #pragma omp for reduction(+:edges) nowait
                for(auto i = 0; i < frontier.size(); i++) 
                {
                    unsigned long long node = frontier[i];
                    if(graph.find(node) == graph.end()) continue; // 有些节点没有出度
                    for (unsigned long long neighbor : graph.at(node)) 
                    {
                        edges++;
                        if (!visited[neighbor])
                        {
                            next_frontier_private.push_back(neighbor);
                            visited[neighbor] ++;
                            parent[neighbor] = node;
                            distance[neighbor] = distance[node] + 1;
                        }
                    }
                }
                #pragma omp critical
                {
                    next_frontier.insert(next_frontier_private.begin(), next_frontier_private.end());
                }
            }
            frontier = std::vector<unsigned long long>(next_frontier.begin(), next_frontier.end());
        }        
    
    }


}

// 将结果写入文件
void writeResult(const std::string &filename, const std::vector<unsigned long long> &result, const unsigned long long &edges, double& time, const std::vector<unsigned long long> &parent, const std::vector<unsigned long long> &distance) {
    std::ofstream file(filename);
    file << "number of nodes searched: " << result.size() << "\n";
    file << "time:" << time << "s\n";
    file << "number of edges searched:" << edges << "\n";
    file << "nodes searched:\n";
    for (unsigned long long i = 0; i < result.size(); i++) {
        file << result[i] << "\t" << distance[result[i]] << "\t" << parent[result[i]] << "\n";
    }
}

static int log_thread_num;

int main(int argc, char **argv) {
    Graph graph;
    omp_set_num_threads(20);
    int index = 0;
    std::vector<std::string> mode_name = {"sequencial", "parallel"};


    if(argc < 2) 
    {
        std::cout << "which file to load\n";
        for(auto i = 0; i < num_files; i++) {
            std::cout << i << ":" << file_name[i] << "\n";
        }
        std::cin >> index;
        if(index >= num_files) {
            std::cout << "invalid input\n";
            return 0;
        }


        std::cout << "which mode to use\n";

        for(auto i = 0; i < 2; i++) {
            std::cout << i << ":" << mode_name[i] << "\n";
        }

        std::cin >> is_parallel;
    }
    else
    {
        index = atoi(argv[1]);
        is_parallel = (bool)atoi(argv[2]);
    }
    unsigned long long num_edges = 0;
    unsigned long long max_node_index = 0;
    readGraph(file_name[index]+ ".txt" , graph, max_node_index);
    std::vector<unsigned long long> result;
    std::vector<unsigned long long> parent(max_node_index+1, 0);
    std::vector<unsigned long long> distance(max_node_index+1, 0);
    auto start = omp_get_wtime();
    bfs(graph,num_edges, max_node_index, START_NODE, result, parent, distance);
    auto end = omp_get_wtime();
    auto time = end - start;
    std::string result_file = "_res.txt";
    result_file = file_name[index] + "_" + mode_name[is_parallel] + result_file;
    writeResult(result_file, result, num_edges, time, parent, distance);
    return 0;
}