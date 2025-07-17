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
    TileConfig buffer[1024];
    int head, tail, size;
    bool done;
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
} TileQueue;

bool enqueue_tile(TileQueue* q, TileConfig tile);
bool dequeue_tile(TileQueue* q, TileConfig* tile);

#endif