#include <stdio.h>
#include <stdlib.h>
#include "tile_queue.h"

bool enqueue_tile(TileQueue* q, TileConfig tile){
    pthread_mutex_lock(&q->lock);

    while(q->size == TILE_QUEUE_CAPACITY){
        // TODO: busy wait here maybe
    }

    q->buffer[q->tail] = tile;
    q->tail = (q->tail + 1) % TILE_QUEUE_CAPACITY;
    q->size++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
}
bool dequeue_tile(TileQueue* q, TileConfig* tile){
    pthread_mutex_lock(&q->lock);
    
    while(q->size == 0 && !q->done){
        pthread_cond_wait(&q->not_empty, &q->lock);
    }

    if(q->done && q->size == 0) {
        pthread_mutex_unlock(&q->lock);
        return false;  
    }

    *tile = q->buffer[q->head]; // dereference it so it saves it to the actual tile variables address
    q->head = (q->head + 1) % TILE_QUEUE_CAPACITY;
    q->size--;
    pthread_mutex_lock(&q->lock);

    return true;
}
