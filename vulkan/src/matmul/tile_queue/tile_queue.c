#include <stdio.h>
#include <stdlib.h>
#include "tile_queue.h"

#define DEBUG_MODE 0

#if DEBUG_MODE 
    #define DEBUG_PRINT(...) prinft(__VA_ARGS__)
#else 
    # define DEBUG_PRINT(...) do {} while (0)
#endif

bool enqueue_tile(TileQueue* q, TileConfig tile){
    DEBUG_PRINT("Enqueuing tile!");
    
    pthread_mutex_lock(&q->lock);

    while(q->size == TILE_QUEUE_CAPACITY && !q->done){
        pthread_cond_wait(&q->capacity_available, &q->lock);
        DEBUG_PRINT("\nQueue size is at maximum capacity!");
    }

    if (q->done) {
        pthread_mutex_unlock(&q->lock);
        return false;
    }

    q->buffer[q->tail] = tile;
    q->tail = (q->tail + 1) % TILE_QUEUE_CAPACITY;
    q->size++;

    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->lock);

    DEBUG_PRINT("\nSuccessfully enqueued, queue size is: %i", q->size);

    return true;
}
bool dequeue_tile(TileQueue* q, TileConfig* tile){
    DEBUG_PRINT("\ndequeueing tile!");

    pthread_mutex_lock(&q->lock);

    while(q->size == 0 && !q->done){
        DEBUG_PRINT("\n Waiting for the queue to be populated! Queue size: %i\n", q->size);
        pthread_cond_wait(&q->not_empty, &q->lock);
    }

    DEBUG_PRINT("\n Woke up! Will start eating tiles\n");

    if(q->done && q->size == 0) {
        DEBUG_PRINT("\nQueue is already processed!");
        pthread_mutex_unlock(&q->lock);
        return false;  
    }

    *tile = q->buffer[q->head]; // dereference it so it saves it to the actual tile variables address
    q->head = (q->head + 1) % TILE_QUEUE_CAPACITY;
    q->size--;

    pthread_cond_broadcast(&q->capacity_available); 
    pthread_mutex_unlock(&q->lock);

    return true;
}
