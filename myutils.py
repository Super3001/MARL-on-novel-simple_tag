# utils.py

# import heapq 
# 不再用heapq，因为heapq不能够保证由小到大排序
import numpy as np
# priority queue
class PQ:
    def __init__(self, max_size, reverse=False):
        self.queue = []
        self.size = 0
        self.max_size = max_size
        self.reverse = reverse # 是否降序排列

    # item: np.ndarray, shape=(2,)
    def push(self, item: np.ndarray, priority):
        if self.reverse:
            priority = -priority
        # heapq.heappush(self.queue, (priority, self.size, item))
        insert_index = self.size
        for i in range(self.size): # 大于再插入，有观测的先后顺序
            if priority > self.queue[i][1]:
                insert_index = i
                break
        self.queue.insert(insert_index, (item, priority))
        self.size += 1
        if self.size > self.max_size:
            self.pop()
        
    def pop(self):
        self.size -= 1
        temp = self.queue[-1][0]
        del self.queue[-1]
        return temp # 只返回value，不返回priority
        # return heapq.heappop(self.queue)[-1]
    
    def top(self):
        # return self.queue[0][-1]
        return self.queue[0][0]
    
    # 不足max_size时用0填补
    def out(self):
        # return [x[-1] for x in self.queue] + [np.zeros(self.top().size)] * (self.max_size - self.size) # 保证queue中至少有一个元素
        return  [x[0] for x in self.queue] + [np.zeros(2)] * (self.max_size - self.size)
    
    def tolist(self):
        return [x[0] for x in self.queue]
    
    def empty(self):
        return self.size == 0
    
    def __len__(self):
        return self.size
    
    def __contains__(self, item):
        return item in self.queue
    
    def __getitem__(self, item):
        return self.queue[item][-1]
    
    def __repr__(self):
        return self.queue.__repr__()
    
    def __str__(self):
        return self.queue.__str__()
    
if __name__ == '__main__':
    priority_queue = PQ(3, reverse=True)
    print('out:',priority_queue.out())

    priority_queue.push(np.array([1, 2]), 1)
    print('out:',priority_queue.out())

    priority_queue.push(np.array([2, 3]), 2)
    print('out:',priority_queue.out())

    priority_queue.push(np.array([3, 4]), 3)
    print('out:',priority_queue.out())

    priority_queue.push(np.array([4, 5]), 4)
    print('out:',priority_queue.out())

    priority_queue.push(np.array([5, 6]), 5)
    print('out:',priority_queue.out())