#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MAX_VERTICES 100

// Function to add an edge to the graph
void addEdge(int adjMatrix[MAX_VERTICES][MAX_VERTICES], int u, int v) {
    adjMatrix[u][v] = 1;
    adjMatrix[v][u] = 1;  // Assuming an undirected graph
}

void bfs(int rank, int size, int adjMatrix[MAX_VERTICES][MAX_VERTICES], int V, int start, int val) {
    int found = 0;
    int global_found = 0;
    int visited[MAX_VERTICES] = {0};
    int queue[MAX_VERTICES];
    int front = 0, rear = 0;

    // Initialize the queue with the start node
    if (rank == 0) {
        visited[start] = 1;
        queue[rear++] = start;
    }

    while (1) {
        int local_count = rear - front;
        int global_count;

        MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (global_count == 0) {
            break;
        }

        while (front < rear) {
            int node = queue[front++];
            if (node == val) {
                found = 1;
                global_found = 1;
                break;
            }

            for (int i = 0; i < V; i++) {
                if (adjMatrix[node][i] == 1 && !visited[i]) {
                    visited[i] = 1;
                    int target_rank = i % size;
                    if (target_rank == rank) {
                        queue[rear++] = i;
                    } else {
                        MPI_Send(&i, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }

        int flag;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
        while (flag) {
            int node;
            MPI_Recv(&node, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (!visited[node]) {
                visited[node] = 1;
                queue[rear++] = node;
            }
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        }

        MPI_Allreduce(&found, &global_found, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_found) {
            break;
        }
    }

    if (rank == 0) {
        if (global_found) {
            printf("Value %d found in the graph.\n", val);
        } else {
            printf("Value %d not found in the graph.\n", val);
        }
    }
}

int main() {
    int rank, size;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int adjMatrix[MAX_VERTICES][MAX_VERTICES] = {0};

    int V = 6;
    if (rank == 0) {
        // Adding edges to the adjacency matrix
        addEdge(adjMatrix, 0, 1);
        addEdge(adjMatrix, 0, 2);
        addEdge(adjMatrix, 1, 3);
        addEdge(adjMatrix, 1, 4);
        addEdge(adjMatrix, 4, 5);
    }

    // Broadcast the adjacency matrix to all processes
    MPI_Bcast(adjMatrix, MAX_VERTICES * MAX_VERTICES, MPI_INT, 0, MPI_COMM_WORLD);

    int start = 0;
    int search_value = 5;

    bfs(rank, size, adjMatrix, V, start, search_value);

    MPI_Finalize();
    return 0;
}
