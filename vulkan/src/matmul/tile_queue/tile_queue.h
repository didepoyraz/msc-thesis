#ifndef TILE_QUEUE_H
#define TILE_QUEUE_H

#include <pthread.h>
#include <stdbool.h>

#define TILE_QUEUE_CAPACITY 4096 // TODO: put all constants into their own file
 
typedef struct {
    int i;
    int j;
    int p;
} TileConfig;

typedef struct {
    float* A;
    float* B;
    float* C;
    int N;
    int BLOCK_SIZE;
} Mult;

typedef struct {
    TileConfig buffer[1024];
    int head, tail, size;
    bool done;
    
    pthread_cond_t not_empty;
    pthread_cond_t capacity_available;
    Mult matrix;
    
    pthread_mutex_t lock;
    pthread_mutex_t *C_locks; // for accessing tiles
} TileQueue;

bool enqueue_tile(TileQueue* q, TileConfig tile);
bool dequeue_tile(TileQueue* q, TileConfig* tile);

#endif