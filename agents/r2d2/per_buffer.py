import numpy as np


class PrioritizedReplay:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity, self.alpha = capacity, alpha
        self.buf, self.pri, self.pos = [], [], 0

    def add(self, item, priority=1.0):
        if len(self.buf) < self.capacity:
            self.buf.append(item); self.pri.append(priority)
        else:
            self.buf[self.pos] = item; self.pri[self.pos] = priority; self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=32, beta=0.4):
        p = np.array(self.pri, dtype=np.float32) ** self.alpha
        p /= p.sum()
        idx = np.random.choice(len(self.buf), batch_size, p=p)
        w = (len(self.buf) * p[idx]) ** (-beta)
        w /= w.max()
        return [self.buf[i] for i in idx], idx, w

    def update(self, idx, td):
        for i, t in zip(idx, td):
            self.pri[i] = float(abs(t) + 1e-6)

    def __len__(self):
        return len(self.buf)
