import numpy as np

class Heap:
    def __init__(self, cmpFn, elems=None):
        """
        cmpFn: A user-supplied compare function for the binary heap.
        elems: A list of initial elements with their priorities.
               Each element must be in the form (Item, Priority).
        """
        self.cmpFn = cmpFn
        self.A = [42]  # The element at position 0 is trash.
        self.n = 0
        self.pos = {}
        if elems != None:
            self.construct_heap(elems)

    def construct_heap(self, elems):
        """
        Construct a heap from a list of elements with priorities.
        Each element of the list must be in the form (Item, Priority).
        """
        for e in elems:
            self.n += 1
            self.A.append(e)
            self.pos[e[0]] = self.n
        for i in range(self.n // 2, 0, -1):
            self.combine(i)

    def get_first(self):
        """
        Gets the first item of the heap (but doesn't remove it).
        """
        return self.A[1][0] if self.n > 0 else None

    def delete_first(self):
        """
        Gets the first item of the heap and removes it.
        """
        if self.n == 0:
            return None
        first = self.A[1]
        self.n -= 1
        last = self.A.pop()
        if self.n > 0:
            self.A[1] = last
            self.pos[last[0]] = 1
            self.combine(1)
        return first[0]

    def combine(self, i):
        l = 2*i
        r = l+1
        mp = i
        if (l <= self.n) and self.cmpFn(self.A[l][1], self.A[mp][1]):
            mp = l
        if (r <= self.n) and self.cmpFn(self.A[r][1], self.A[mp][1]):
            mp = r
        if mp != i:
            Ai, Amp = self.A[i], self.A[mp]
            self.pos[Ai[0]], self.pos[Amp[0]] = self.pos[Amp[0]], self.pos[Ai[0]] 
            self.A[i], self.A[mp] = Amp, Ai
            self.combine(mp)

    def insert(self, elem, prio):
        """
        Inserts the element elem with priority prio.
        """
        self.n += 1
        self.A.append( (e,w) )
        self.pos[e] = self.n
        i = self.n
        p = i // 2
        self.insert_loop(i, p)

    def insert_loop(self, i, p):
        while i > 1 and not self.cmpFn(self.A[p][1], self.A[i][1]):
            Ap, Ai = self.A[p], self.A[i]
            self.pos[Ai[0]], self.pos[Ap[0]] = self.pos[Ap[0]], self.pos[Ai[0]] 
            self.A[p], self.A[i] = Ai, Ap
            i = p
            p = i // 2

    def change_priority(self, elem, prio):
        """
        Changes the priority of the element elem to prio.
        """
        pos = self.pos[elem]
        currPrio = self.A[pos][1]
        self.A[pos] = (elem, prio)
        if self.cmpFn(prio, currPrio):
            self.insert_loop(pos, pos // 2)  # Up heapify
        else:
            self.combine(pos)  # Down heapify

    def get_priority(self, elem):
        """
        Gets the priority of an element.
        """
        pos = self.pos[elem]
        return self.A[pos][1]


class MinHeap(Heap):
    """
    A min heap.
    """
    def __init__(self, elems=None):
        Heap.__init__(self, lambda x,y: x < y, elems)

    def min(self):
        """
        Gets the minimum element of the heap.
        """
        return self.get_first()

    def take_min(self):
        """
        Gets the minimum element of the heap and removes it.
        """
        return self.get_first()

    def take_min(self):
        return self.delete_first()


class MaxHeap(Heap):
    """
    A max heap.
    """
    def __init__(self, elems=None):
        Heap.__init__(self, lambda x,y: x > y, elems)

    def max(self):
        """
        Gets the maximum element of the heap. 
        """
        return self.get_first()

    def take_max(self):
        """
        Gets the maximum element of the heap and removes it.
        """
        return self.delete_first()

import collections

def dijkstra(graph, root):
    vertices = graph.keys()
    n = len(vertices)

    # Initialize the priority queue
    inf = float("inf")
    pq = MinHeap([(v, inf) for v in graph.keys()]) 
    pq.change_priority(root, 0)
    # Other initializations
    parent = collections.defaultdict(lambda: None)
    selected = set()
    cost = collections.defaultdict(lambda: inf)
    cost[root] = 0

    while len(selected) < n:
        u = pq.min()
        du = cost[u] = pq.get_priority(u)
        selected.add(u)
        pq.take_min()
        for (v, w) in graph[u].items():
            if v not in selected and pq.get_priority(v) > du + w:
                pq.change_priority(v, du + w)
                parent[v] = u
    
    return cost, parent

def cost(props, i, j):
    return np.maximum(2 - props[i] - props[j], 0.1)

def add_neighbours(i, props, height, width):
    n = {}
        
    #not-last-column
    if (i % width) < (width - 1):
        n[i + 1] = cost(props, i, i + 1)
        
        #not-first row
        if i > width:
            n[i - width + 1] = cost(props, i, i - width + 1)

        #not-last row
        if i < (height-1)*width:
            n[i + width + 1] = cost(props, i, i + width + 1)
    else:
        n[width*height+1] = cost(props, i, width*height+1)
        
    return n
    
def build_graph(probs, height, width):
    values = [
        add_neighbours(i, probs, height, width)
        for i in range(height*width)
    ] + [
        dict(zip(list(np.arange(height)*width), [cost(probs, height*width, i) for i in np.arange(height)*width]))
    ] + [{}]
    
    return dict(zip(np.arange(height*width+2), values))
        
